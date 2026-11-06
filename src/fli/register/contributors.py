"""Contributor score: how strongly the market values a tracked person, predicted from public signals.

Features per person (tier is deliberately NOT a feature - it is our own judgement, and including it
would make the validation circular):
  papers_12m         papers credited to the person (Document.person_id or meta.people_matched /
                     meta.tracked_people) published in the last 365 days
  insight_mentions   insights naming the person, excluding talent_move / new_venture / funding_ipo
                     events - those describe the label event itself
  tracked_coauthors  distinct other tracked people on those papers
  leadership_roles   meta.roles entries that are leadership roles; people.yaml excludes the role that
                     *is* the revealed-preference event, so this cannot encode the label
  log_citations, h_index  author citation stats: Semantic Scholar (unauthenticated, <= 1 request/s)
                     with OpenAlex as fallback when S2's shared pool is rate-limited; <= 120 calls
                     per run, cached in Person.contributor_features for 30 days

Label (a defensible proxy, no human ratings needed): meta.revealed_preference - in 2024-2026 a rival
lab poached the person with a reported offer, or investors funded their alumni startup (sources in
config/people.yaml). Model: L2 logistic regression on standardised features, class-balanced, fixed
C=1 (nothing tuned on the labels). Validation: leave-one-out AUC and precision@20 against a
citations-only baseline. With fewer than MIN_CLASS labels per class the score falls back to a
transparent rank average and says so.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fli.config import DATA_DIR, Settings, get_settings
from fli.db.models import Document, Identity, Insight, Person
from fli.db.session import session_scope
from fli.register.entity_resolution import norm_name
from fli.timeutil import to_naive_utc, utcnow

log = logging.getLogger("fli.register.contributors")

MODEL_VERSION = "contrib-lr-v1"
FEATURES = ("papers_12m", "insight_mentions", "tracked_coauthors", "leadership_roles", "log_citations",
            "h_index")
LEADERSHIP_ROLES = {
    "founder", "ceo", "president", "cto", "chief_scientist", "research_lead", "pretraining_lead",
    "post_training_lead", "reasoning_lead", "infra_lead", "safety_lead", "interpretability_lead",
    "agents_lead", "product_lead",
}
LABEL_EVENT_TYPES = {"talent_move", "new_venture", "funding_ipo"}
MIN_CLASS = 8  # below this many positives or negatives, a fitted model is not worth reporting
TOP_K = 20
CITE_CAP, CITE_INTERVAL_S, CITE_REFRESH = 120, 1.05, timedelta(days=30)
METRICS_PATH = DATA_DIR / "eval" / "contributors_metrics.json"


# ---------------------------------------------------------------------------- feature extraction
def credited(doc: Document) -> set[str]:
    """Every tracked person credited on a document, not only the most senior one."""
    meta = doc.meta or {}
    ids = {m.get("person_id") for m in meta.get("people_matched") or [] if isinstance(m, dict)}
    return ({doc.person_id} | ids | set(meta.get("tracked_people") or [])) - {None}


def is_paper(doc: Document) -> bool:
    meta = doc.meta or {}
    return bool(meta.get("arxiv_id") or meta.get("authors")) or doc.source_class == "research"


def db_features(s: Session, people: list[Person], now: datetime) -> dict[str, dict[str, float]]:
    """papers_12m, tracked_coauthors, insight_mentions and leadership_roles for every person."""
    known = {p.id for p in people}
    papers: dict[str, int] = defaultdict(int)
    coauthors: dict[str, set[str]] = defaultdict(set)
    since = now - timedelta(days=365)
    recent = select(Document).where(Document.published_at >= since, Document.status != "duplicate")
    for doc in s.scalars(recent):
        if not is_paper(doc):
            continue
        team = credited(doc) & known
        for pid in team:
            papers[pid] += 1
            coauthors[pid] |= team - {pid}
    mentions: dict[str, int] = defaultdict(int)
    for event_type, person_ids in s.execute(select(Insight.event_type, Insight.person_ids)):
        if event_type in LABEL_EVENT_TYPES:
            continue
        for pid in set(person_ids or []) & known:
            mentions[pid] += 1
    return {p.id: {"papers_12m": float(papers[p.id]), "tracked_coauthors": float(len(coauthors[p.id])),
                   "insight_mentions": float(mentions[p.id]), "leadership_roles": float(leadership_count(p))}
            for p in people}


def leadership_count(p: Person) -> int:
    return len({r for r in (p.meta or {}).get("roles") or [] if r in LEADERSHIP_ROLES})


def label_of(p: Person) -> int | None:
    value = (p.meta or {}).get("revealed_preference")
    return None if value is None else int(bool(value))


# ------------------------------------------------------------------------------ citation stats
@dataclass(frozen=True)
class CitationSource:
    """An author-search API normalised to {id, names, citations, h_index, works} candidates."""

    name: str
    url: str
    params: Callable[[str], dict]
    candidates: Callable[[dict], list[dict]]


S2 = CitationSource(
    "s2", "https://api.semanticscholar.org/graph/v1/author/search",
    lambda q: {"query": q, "limit": 10, "fields": "name,paperCount,citationCount,hIndex"},
    lambda d: [{"id": a.get("authorId"), "names": [a.get("name") or ""],
                "citations": a.get("citationCount"), "h_index": a.get("hIndex"), "works": a.get("paperCount")}
               for a in d.get("data") or []])
OPENALEX = CitationSource(
    "openalex", "https://api.openalex.org/authors",
    lambda q: {"search": q, "per-page": 10, "select": "id,display_name,display_name_alternatives,"
                                                      "works_count,cited_by_count,summary_stats"},
    lambda d: [{"id": str(a.get("id") or "").rsplit("/", 1)[-1],
                "names": [a.get("display_name") or "", *(a.get("display_name_alternatives") or [])],
                "citations": a.get("cited_by_count"), "works": a.get("works_count"),
                "h_index": (a.get("summary_stats") or {}).get("h_index")} for a in d.get("results") or []])


def _name_key(name: str) -> str:
    return " ".join(sorted(norm_name(name).split()))


def query_name(p: Person, arxiv_names: list[str]) -> str:
    return arxiv_names[0] if arxiv_names else re.sub(r"\s*\(.*?\)", "", p.name).strip()


def pick_author(candidates: list[dict], names: list[str]) -> dict:
    """Highest-cited candidate with one of the person's names (word order ignored). cite_ambiguous
    flags a runner-up with >= 20% of the top citations: a likely namesake or a split profile."""
    wanted = {_name_key(n) for n in names}
    same = [c for c in candidates if any(_name_key(n) in wanted for n in c["names"])]
    same.sort(key=lambda c: c.get("citations") or 0, reverse=True)
    out: dict[str, Any] = {"cite_found": bool(same), "cite_candidates": len(same)}
    if same:
        top, runner = same[0], (same[1].get("citations") or 0) if len(same) > 1 else 0
        out.update({"cite_author_id": str(top["id"]), "cite_citations": int(top.get("citations") or 0),
                    "cite_h_index": int(top.get("h_index") or 0), "cite_works": int(top.get("works") or 0),
                    "cite_ambiguous": runner >= 0.2 * max(1, top.get("citations") or 0)})
    return out


class PoliteSearch:
    """>= CITE_INTERVAL_S between requests, one shared call cap, backoff on 429. A source that stays
    rate-limited (S2's shared unauthenticated pool often is) hands over to the next source."""

    def __init__(self, client: httpx.Client, sources: list[CitationSource], max_calls: int,
                 sleep: Callable[[float], None] = time.sleep):
        self.client, self.sources, self.max_calls, self.sleep = client, list(sources), max_calls, sleep
        self.last = 0.0
        self.stats: dict[str, Any] = {"cite_calls": 0, "cite_found": 0, "cite_errors": 0,
                                      "cite_rate_limited": 0}

    def search(self, query: str) -> tuple[str, list[dict]] | None:
        while self.sources and self.stats["cite_calls"] < self.max_calls:
            source = self.sources[0]
            for attempt in range(2):
                if (wait := CITE_INTERVAL_S - (time.monotonic() - self.last)) > 0:
                    self.sleep(wait)
                self.last = time.monotonic()
                self.stats["cite_calls"] += 1
                try:
                    resp = self.client.get(source.url, params=source.params(query))
                except httpx.HTTPError as exc:
                    log.warning("%s search %r failed: %s", source.name, query, exc)
                    self.stats["cite_errors"] += 1
                    return None
                if resp.status_code != 429:
                    break
                self.stats["cite_rate_limited"] += 1
                self.sleep(5.0 * (attempt + 1))
            if resp.status_code == 200:
                return source.name, source.candidates(resp.json())
            if resp.status_code != 429:
                log.warning("%s search %r: HTTP %s", source.name, query, resp.status_code)
                self.stats["cite_errors"] += 1
                return None
            log.warning("%s stays rate-limited; switching citation source", source.name)
            self.sources.pop(0)
        return None


def refresh_citations(people: list[Person], arxiv: dict[str, list[str]], settings: Settings, *,
                      client: httpx.Client | None = None, max_calls: int = CITE_CAP,
                      sleep: Callable[[float], None] = time.sleep) -> dict:
    """Fetch author stats for people whose cache is missing or older than 30 days (S2 first, OpenAlex
    as fallback). Mutates Person.contributor_features in place; the caller commits."""
    now = utcnow()
    due = [p for p in people if _cite_due(p, now)]
    own = client is None and bool(due)
    search = PoliteSearch(client or _client(settings), [S2, OPENALEX], max_calls, sleep)
    try:
        for p in due:
            names = arxiv.get(p.id, [])
            hit = search.search(query_name(p, names))
            if hit is None:
                if not search.sources or search.stats["cite_calls"] >= max_calls:
                    break
                continue
            found = pick_author(hit[1], [p.name, *names])
            search.stats["cite_found"] += found["cite_found"]
            p.contributor_features = {**_cache_only(p.contributor_features), **found, "cite_source": hit[0],
                                      "cite_fetched_at": now.isoformat(timespec="seconds")}
    finally:
        if own:
            search.client.close()
    return search.stats


def _client(settings: Settings) -> httpx.Client:
    """One client for both hosts; the S2 key (if any) is attached to S2 requests only."""

    def add_key(request: httpx.Request) -> None:
        if settings.s2_api_key and request.url.host == "api.semanticscholar.org":
            request.headers["x-api-key"] = settings.s2_api_key

    return httpx.Client(timeout=settings.http_timeout_s, headers={"User-Agent": settings.user_agent},
                        event_hooks={"request": [add_key]})


def _cache_only(features: dict | None) -> dict:
    return {k: v for k, v in (features or {}).items() if k.startswith("cite_")}


def _cite_due(p: Person, now: datetime) -> bool:
    fetched = to_naive_utc((p.contributor_features or {}).get("cite_fetched_at"))
    return fetched is None or now - fetched >= CITE_REFRESH


# --------------------------------------------------------------------------------- model + metrics
def feature_matrix(rows: list[dict[str, float | None]]) -> tuple[np.ndarray, dict[str, float]]:
    """log1p the counts, impute missing citation stats with the median, keep columns in FEATURES order."""
    medians: dict[str, float] = {}
    cols = []
    for name in FEATURES:
        raw = [r.get(name) for r in rows]
        known = [float(v) for v in raw if v is not None]
        medians[name] = float(np.median(known)) if known else 0.0
        cols.append([float(v) if v is not None else medians[name] for v in raw])
    x = np.array(cols, dtype=float).T if rows else np.zeros((0, len(FEATURES)))
    for j, name in enumerate(FEATURES):
        if name != "log_citations":  # log_citations is already a log
            x[:, j] = np.log1p(x[:, j])
    return x, medians


def _model() -> tuple[StandardScaler, LogisticRegression]:
    return StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)


def loo_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out predicted probabilities: every person is scored by a model that never saw them."""
    out = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        scaler, model = _model()
        model.fit(scaler.fit_transform(x[mask]), y[mask])
        out[i] = model.predict_proba(scaler.transform(x[i : i + 1]))[0, 1]
    return out


def precision_at(scores: np.ndarray, y: np.ndarray, k: int = TOP_K) -> float:
    order = np.argsort(-scores, kind="stable")[: min(k, len(y))]
    return float(y[order].mean()) if len(order) else 0.0


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else None


def rank_average(x: np.ndarray) -> np.ndarray:
    """Mean percentile rank over informative (non-constant) features - the transparent fallback."""
    cols = [j for j in range(x.shape[1]) if np.ptp(x[:, j]) > 0]
    if not cols or len(x) < 2:
        return np.full(len(x), 0.5)
    ranks = [np.argsort(np.argsort(x[:, j], kind="stable"), kind="stable") / (len(x) - 1) for j in cols]
    return np.mean(ranks, axis=0)


def score_people(x: np.ndarray, y_all: list[int | None], groups: list[str] | None = None
                 ) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """(scores, per-feature contributions or None, metrics); `groups` (people's profile) adds
    per-group LOO metrics, because executives' value barely shows in public research output."""
    idx = [i for i, v in enumerate(y_all) if v is not None]
    y, xl = np.array([y_all[i] for i in idx], dtype=int), x[idx]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    citations = xl[:, FEATURES.index("log_citations")]
    metrics: dict[str, Any] = {
        "n_people": len(y_all), "n_labelled": len(y), "n_positive": n_pos, "n_negative": n_neg,
        "baseline": {"name": "citations_only", "auc": _r(safe_auc(y, citations)),
                     "precision_at_20": round(precision_at(citations, y), 3)},
        "base_rate": round(n_pos / len(y), 3) if len(y) else None,
    }
    if min(n_pos, n_neg) < MIN_CLASS:
        scores = rank_average(x)
        metrics.update({"method": "rank_average", "fallback_reason":
                        f"only {n_pos} positive / {n_neg} negative labels (< {MIN_CLASS} per class)",
                        "auc_in_sample": _r(safe_auc(y, scores[idx])), "precision_at_20": None})
        return scores, None, metrics
    active = informative_columns(xl)
    loo = loo_scores(xl[:, active], y)
    scaler, model = _model()
    model.fit(scaler.fit_transform(xl[:, active]), y)
    coef = np.zeros(len(FEATURES))
    coef[active] = model.coef_[0]
    z = np.zeros_like(x)
    z[:, active] = scaler.transform(x[:, active])
    metrics.update({
        "method": "logistic_regression", "model_version": MODEL_VERSION,
        "loo_auc": _r(safe_auc(y, loo)), "precision_at_20": round(precision_at(loo, y), 3),
        "coefficients": {FEATURES[j]: round(float(coef[j]), 3) for j in active},
        "intercept": round(float(model.intercept_[0]), 3),
        "dropped_sparse_features": [f for j, f in enumerate(FEATURES) if j not in active],
    })
    if groups:
        metrics["by_profile"] = group_metrics([groups[i] for i in idx], y, loo, citations)
    return model.predict_proba(z[:, active])[:, 1], z * coef, metrics


def informative_columns(x: np.ndarray) -> list[int]:
    """Features that differ from their median for >= max(5, 10%) of the labelled people. Decided
    without looking at the labels: a coefficient fitted on a handful of non-zero values is noise."""
    need = max(5, int(0.1 * len(x)))
    cols = [j for j in range(x.shape[1]) if int((x[:, j] != np.median(x[:, j])).sum()) >= need]
    return cols or list(range(x.shape[1]))


def group_metrics(groups: list[str], y: np.ndarray, loo: np.ndarray, baseline: np.ndarray) -> dict:
    out = {}
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        out[g] = {"n": int(m.sum()), "n_positive": int(y[m].sum()), "loo_auc": _r(safe_auc(y[m], loo[m])),
                  "baseline_auc": _r(safe_auc(y[m], baseline[m]))}
    return out


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


# ---------------------------------------------------------------------------------- entry point
def model_row(db: dict[str, float], cached: dict | None, common_name: bool = False
              ) -> dict[str, float | None]:
    """One person's raw feature values. Citation stats are None (imputed later) when no author was
    found, or when a common name had several same-name candidates: a namesake's 243k citations
    ("Daniel Levy") would otherwise dominate the feature."""
    cached = cached or {}
    found = bool(cached.get("cite_found")) and not (common_name and (cached.get("cite_candidates") or 0) > 1)
    cites = cached.get("cite_citations") if found else None
    return {**db, "log_citations": math.log1p(cites) if cites is not None else None,
            "h_index": cached.get("cite_h_index") if found else None}


def run_contributor_scoring(limit: int | None = None, *, use_citations: bool = True,
                            http_client: httpx.Client | None = None, metrics_path: Path | str | None = None,
                            **_: Any) -> dict:
    """Refresh S2 stats, compute features, fit + validate, write scores and the metrics file.
    `limit` caps the citation-API calls of this run (the stage contract's limit)."""
    settings, now = get_settings(), utcnow()
    stamp = now.isoformat(timespec="seconds")
    with session_scope() as s:
        people = list(s.scalars(select(Person).order_by(Person.id)))
        arxiv: dict[str, list[str]] = defaultdict(list)
        names = select(Identity.person_id, Identity.handle).where(
            Identity.platform == "arxiv_name", Identity.person_id.is_not(None)).order_by(Identity.id)
        for pid, handle in s.execute(names):
            arxiv[pid].append(handle)
        cite_stats: dict[str, Any] = {"cite_calls": 0}
        if use_citations:
            cap = min(CITE_CAP, limit or CITE_CAP)
            cite_stats = refresh_citations(people, arxiv, settings, client=http_client, max_calls=cap)
        feats = db_features(s, people, now)
        corpus = {"documents": s.scalar(select(func.count(Document.id))) or 0,
                  "insights": s.scalar(select(func.count(Insight.id))) or 0,
                  "people_with_papers_12m": sum(f["papers_12m"] > 0 for f in feats.values()),
                  "people_with_insight_mentions": sum(f["insight_mentions"] > 0 for f in feats.values())}
        rows = [model_row(feats[p.id], p.contributor_features, bool((p.meta or {}).get("common_name")))
                for p in people]
        x, medians = feature_matrix(rows)
        profiles = [str((p.meta or {}).get("profile") or "unknown") for p in people]
        scores, contrib, metrics = score_people(x, [label_of(p) for p in people], profiles)
        for i, p in enumerate(people):
            row = {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[i].items() if v is not None}
            why = {} if contrib is None else {f"contrib_{f}": round(float(contrib[i, j]), 3)
                                              for j, f in enumerate(FEATURES)}
            p.contributor_score = round(float(scores[i]), 4)
            p.contributor_features = {**_cache_only(p.contributor_features), **row, **why,
                                      "score_method": metrics["method"], "scored_at": stamp}
    imputed = {k: round(v, 3) for k, v in medians.items()}
    metrics.update({"features": list(FEATURES), "imputed_medians": imputed, "corpus": corpus,
                    "label": "meta.revealed_preference (config/people.yaml)", "created_at": stamp,
                    **cite_stats})
    path = Path(metrics_path) if metrics_path else METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    log.info("contributor scores: %s", {k: metrics.get(k) for k in ("method", "loo_auc", "precision_at_20")})
    return metrics
