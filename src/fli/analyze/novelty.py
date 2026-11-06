"""Novelty: how new is an insight for its entity, and for the whole stream?

  novelty_entity   1 - max cosine to insights of the same lab (same person when there is no lab)
                   published in the prior 90 days, excluding the insight's own development (its
                   corroborating coverage is not prior art); 1.0 when there is nothing to compare
  novelty_global   the same against every insight of the prior 30 days
  nearest_prior_insight_id   argmax of the entity comparison (the global one when no entity match)

"Prior" means an earlier event time, or the same time and a smaller id (feeds often carry dates
without times). Computed once per insight after clustering (requires development_id); the same
pass registers the insight's artifacts for first-seen detection. `recompute=True` rescores
everything, e.g. after a backfill added earlier insights.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from fli.analyze.artifacts import register_insight_artifacts
from fli.analyze.cluster import insight_time, insight_time_expr
from fli.analyze.embed import from_bytes
from fli.db.models import Insight
from fli.db.session import session_scope

log = logging.getLogger("fli.events.novelty")

ENTITY_WINDOW = timedelta(days=90)
GLOBAL_WINDOW = timedelta(days=30)
_EPOCH = datetime(1970, 1, 1)


def _secs(t: datetime) -> float:
    return (t - _EPOCH).total_seconds()


@dataclass
class _Corpus:
    """Embedded insights of the comparison window, sorted by (time, id)."""

    ids: np.ndarray
    times: np.ndarray
    labs: np.ndarray
    devs: np.ndarray
    vecs: np.ndarray
    person_rows: dict[str, list[int]]

    def person_mask(self, persons: set[str]) -> np.ndarray:
        mask = np.zeros(len(self.ids), dtype=bool)
        for p in persons:
            mask[self.person_rows.get(p, [])] = True
        return mask


def _load_corpus(session: Session, lo: datetime, hi: datetime, dim: int) -> _Corpus:
    rows = session.execute(
        select(Insight.id, Insight.lab_id, Insight.person_ids, Insight.development_id, Insight.published_at,
               Insight.created_at, Insight.embedding)
        .where(Insight.embedding.is_not(None), insight_time_expr() >= lo, insight_time_expr() <= hi)
    ).all()
    items = []
    for r in rows:
        vec = from_bytes(r.embedding)
        if vec.shape[0] == dim:
            items.append((_secs(r.published_at or r.created_at), r.id, r, vec))
    items.sort(key=lambda x: (x[0], x[1]))
    person_rows: dict[str, list[int]] = {}
    for idx, (_, _, r, _) in enumerate(items):
        for p in r.person_ids or []:
            if isinstance(p, str):
                person_rows.setdefault(p, []).append(idx)
    return _Corpus(
        ids=np.array([x[1] for x in items], dtype=np.int64),
        times=np.array([x[0] for x in items], dtype=np.float64),
        labs=np.array([x[2].lab_id for x in items], dtype=object),
        devs=np.array([x[2].development_id or -1 for x in items], dtype=np.int64),
        vecs=np.vstack([x[3] for x in items]) if items else np.zeros((0, dim), dtype=np.float32),
        person_rows=person_rows,
    )


def _novelty(sims: np.ndarray, mask: np.ndarray | None) -> tuple[float, int | None]:
    """(1 - max cosine over mask, index of the max) or (1.0, None) when the mask is empty."""
    if mask is None or not mask.any():
        return 1.0, None
    masked = np.where(mask, sims, -np.inf)
    idx = int(np.argmax(masked))
    return round(float(min(1.0, max(0.0, 1.0 - masked[idx]))), 4), idx


def score_insight(corpus: _Corpus, ins: Insight, vec: np.ndarray) -> tuple[float, float, int | None]:
    """(novelty_entity, novelty_global, nearest_prior_insight_id) for one insight."""
    t = _secs(insight_time(ins))
    lo = int(np.searchsorted(corpus.times, t - ENTITY_WINDOW.total_seconds(), side="left"))
    hi = int(np.searchsorted(corpus.times, t, side="right"))
    ids, times = corpus.ids[lo:hi], corpus.times[lo:hi]
    # rows in the slice have time <= t: earlier, or same time and smaller id
    base = ((times < t) | (ids < ins.id)) & (corpus.devs[lo:hi] != (ins.development_id or -2))
    sims = corpus.vecs[lo:hi] @ vec
    persons = {p for p in ins.person_ids or [] if isinstance(p, str)}
    if ins.lab_id:
        entity_mask = base & (corpus.labs[lo:hi] == ins.lab_id)
    elif persons:
        entity_mask = base & corpus.person_mask(persons)[lo:hi]
    else:
        entity_mask = None
    global_mask = base & (times >= t - GLOBAL_WINDOW.total_seconds())
    n_entity, i_entity = _novelty(sims, entity_mask)
    n_global, i_global = _novelty(sims, global_mask)
    nearest = i_entity if i_entity is not None else i_global
    return n_entity, n_global, (int(ids[nearest]) if nearest is not None else None)


def run_novelty(limit: int | None = None, recompute: bool = False, **_: object) -> dict:
    """Score novelty (and register artifacts) for clustered insights not yet scored."""
    with session_scope() as s:
        q = select(Insight).where(Insight.embedding.is_not(None), Insight.development_id.is_not(None))
        if not recompute:
            q = q.where(or_(Insight.novelty_entity.is_(None), Insight.novelty_global.is_(None)))
        q = q.order_by(insight_time_expr(), Insight.id)
        pending = list(s.scalars(q.limit(limit) if limit else q))
        if not pending:
            return {"scored": 0, "no_prior_entity": 0, "artifacts_new": 0, "artifacts_moved": 0}
        art = register_insight_artifacts(s, pending)
        vecs = {i.id: from_bytes(i.embedding) for i in pending}
        dim = Counter(v.shape[0] for v in vecs.values()).most_common(1)[0][0]
        times = [insight_time(i) for i in pending]
        corpus = _load_corpus(s, min(times) - ENTITY_WINDOW, max(times), dim)
        no_prior = 0
        for ins in pending:
            vec = vecs[ins.id]
            if vec.shape[0] != dim:
                n_entity, n_global, nearest = 1.0, 1.0, None
            else:
                n_entity, n_global, nearest = score_insight(corpus, ins, vec)
            ins.novelty_entity, ins.novelty_global, ins.nearest_prior_insight_id = n_entity, n_global, nearest
            no_prior += n_entity == 1.0
    log.info("novelty: %d insights scored (%d with no prior entity match), %d new artifacts",
             len(pending), no_prior, art["artifacts_new"])
    return {"scored": len(pending), "no_prior_entity": int(no_prior), **art}
