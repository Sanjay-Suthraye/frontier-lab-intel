"""Clustering insights into developments: one real-world event seen across sources (lab blog + X
post + repo + press). Developments, not insights, are the unit of scoring and alerting.

Online, single pass in published_at order (an insight is assigned once and never moved), against an
in-memory index of the developments active in the batch's time range:

  candidates  developments whose [first_seen, last_seen] is within 7 days of the insight and that
              - belong to the same lab, or
              - share an artifact or a person, when the insight or the development has no lab, or
              - share a person, for people events (talent_move / new_venture): a researcher leaving
                OpenAI for Meta is one event reported from two labs
  join        similarity >= 0.78, or a structural match: a shared artifact (people events: a shared
              person) plus a compatible event family
  conflict    both sides name different subjects (models/products, e.g. Claude Opus 5.5 vs Claude
              Haiku 5.5): only similarity >= 0.92 joins. bge-small puts same-week sibling launches at
              0.87-0.88, above the 0.78 join bar, so cosine alone would merge them. A shared versioned
              name counts as a shared subject ("Beam 1.0" is kind 'other' only because guess_kind does
              not know the name, while "Gemini 4" is a model)
  late join   a launch is seen across sources over weeks (blog, then Hugging Face, then OpenRouter, then
              the paper: Reflection's Beam became 4 developments and 4 alerts in 11 days). Up to 21 days
              after a release-type development of the same lab started, a launch-type insight
              (release, paper, benchmark claim) joins it when it shares a *versioned* artifact
              ("beam 1.0 501b a 23b", "gemini 4 argon", "timestral 209m") with the development's
              founding insight and names no versioned artifact the development does not know.
              Embedding-only joins keep the 7-day window. Each guard is measured: re-clustering the
              683 insights of the 2026-11 corpus with a bare "shared versioned name within 21 days"
              rule merged 19 groups, 15 of them different events (Gemini 4 Carbon into Argon because it
              names Argon, the Opus 4.8 fast-mode deprecation into the Opus 5.5 fast-mode post, a
              GPT-5.6 Sol price cut into a fast-mode post, DeepSeek-V4 items into two snowballs, two
              unrelated Google DeepMind papers that both used Qwen3-8B); with the guards 5 groups merge,
              all one launch each (Beam, Timestral and its paper, Gemini 4 Argon preview and GA,
              Qwen3.8-27B and its listing, two Muse Spark 1.2 listings), and nothing else moves
  choose      structural matches first, then the highest similarity; otherwise open a new development

Similarity is the mean cosine to the development's members (the dot product with the unnormalised
mean), which equals cosine to the centroid for a one-member development. Cosine to the renormalised
centroid inflates as members diversify: bge vectors share a large lab/topic component, so a
development holding two different same-lab events has a centroid ~0.78 from *any* insight of that
lab and snowballs (synthetic run with the measured cosine structure: 14 developments swallowed 415
of 1200 events). The stored `centroid` is still the renormalised mean.

Artifacts match on normalised names, where a versioned family name matches its variants
("Qwen3.5" ~ "Qwen3.5-72B") and benchmarks never link. Each touched development is then recomputed
from all its members (`recompute_development`), so stored aggregates never drift from the insights.

Event type of a development: when any official member is a model_release or open_weight_release, that
wins (a vote among those); otherwise a vote among the official members, then among all members, with
'other' ignored. A plain majority let Claude Fable/Mythos 5.1, a new model generation, become a
product_launch because two same-day feature posts outvoted the release post, and release-only
materiality rules then never fired for it.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fli.analyze.artifacts import guess_kind, link_keys, related, unique_normalized, version_prefixes
from fli.analyze.embed import from_bytes, normalize, to_bytes
from fli.db.models import Development, Document, Insight
from fli.db.session import session_scope
from fli.timeutil import utcnow

log = logging.getLogger("fli.events.cluster")

JOIN_COSINE = 0.78
CONFLICT_COSINE = 0.92
WINDOW = timedelta(days=7)  # embedding joins and plain structural joins
LATE_WINDOW = timedelta(days=21)  # after a release-type development starts: its launch tail (see "late join")
WILDCARD_EVENT = "other"  # compatible with every family: "other" must not block an artifact match
RELEASE_EVENTS = frozenset({"model_release", "open_weight_release"})
# What may join a launch late: the weights, a listing, the paper, a benchmark claim. Not pricing changes,
# feature posts or deprecations: those are developments of their own even when they name the model.
LATE_EVENTS = RELEASE_EVENTS | {"research_paper", "capability_claim"}
EVENT_FAMILIES: dict[str, frozenset[str]] = {
    "release": frozenset({"model_release", "open_weight_release", "product_launch", "pricing_change",
                          "capability_claim", "research_paper", "technique"}),
    "people": frozenset({"talent_move", "new_venture"}),
    "capital": frozenset({"funding_ipo", "new_venture"}),
    "infrastructure": frozenset({"compute_deal", "energy_deal", "custom_silicon", "partnership"}),
    "strategy": frozenset({"strategy_shift", "partnership", "safety_policy", "hiring_signal"}),
}
PEOPLE_EVENTS = EVENT_FAMILIES["people"]


def compatible_events(a: str | None, b: str | None) -> bool:
    """Two event types can describe the same development when they share a family."""
    a, b = a or WILDCARD_EVENT, b or WILDCARD_EVENT
    if a == b or WILDCARD_EVENT in (a, b):
        return True
    return any(a in fam and b in fam for fam in EVENT_FAMILIES.values())


def insight_time(ins: Insight) -> datetime:
    """Event time of an insight (published_at comes from source metadata)."""
    return ins.published_at or ins.created_at or utcnow()


def insight_time_expr():
    return func.coalesce(Insight.published_at, Insight.created_at)


def _persons(values: Iterable | None) -> set[str]:
    return {p for p in values or [] if isinstance(p, str) and p}


def versioned_keys(artifacts: Iterable[str] | None, linkable: set[str] | None = None) -> set[str]:
    """Link keys that name one specific artifact: a version number or a size in a non-umbrella name
    ("gemini 4 carbon", "deepseek v 4.1 flash", "beam 1.0", "qwen 3.8 27b"), unlike a brand ("Beam").
    Chips are left out ("H100", "TPU v7" are named across unrelated deals and papers), as are the
    benchmarks and umbrella brands that are not link keys at all."""
    linkable = link_keys(artifacts)[0] if linkable is None else linkable
    return {key for key, display in unique_normalized(artifacts)
            if key in linkable and any(c.isdigit() for c in key) and guess_kind(display) != "chip"}


# ------------------------------------------------------------------------------ in-memory index
@dataclass
class Signals:
    """What one insight contributes to matching."""

    vec: np.ndarray | None
    linkable: set[str]
    subjects: set[str]
    persons: set[str]
    event_type: str
    versioned: set[str] = field(default_factory=set)  # subset of linkable, see versioned_keys

    @classmethod
    def of(cls, embedding: bytes | None, artifacts, person_ids, event_type: str | None) -> Signals:
        linkable, subjects = link_keys(artifacts)
        return cls(from_bytes(embedding) if embedding else None, linkable, subjects, _persons(person_ids),
                   event_type or WILDCARD_EVENT, versioned_keys(artifacts, linkable))


@dataclass(eq=False)
class DevState:
    """What the matcher needs to know about one development."""

    dev_id: int
    lab_id: str | None
    first_seen: datetime
    last_seen: datetime
    vec_sum: np.ndarray | None = None
    n_vecs: int = 0
    linkable: set[str] = field(default_factory=set)
    subjects: set[str] = field(default_factory=set)
    persons: set[str] = field(default_factory=set)
    event_types: set[str] = field(default_factory=set)
    versioned: set[str] = field(default_factory=set)
    # versioned names of the earliest member that has any: what the development is about, as opposed
    # to names its later members mention in passing (an OpenRouter listing of GPT-6 Sol names Astra)
    core: set[str] = field(default_factory=set)
    core_time: datetime | None = None

    def similarity(self, vec: np.ndarray | None) -> float:
        """Mean cosine between `vec` and the member vectors (-1 when not comparable)."""
        if self.vec_sum is None or vec is None or self.vec_sum.shape != vec.shape or not self.n_vecs:
            return -1.0
        return float(self.vec_sum @ vec) / self.n_vecs

    def absorb(self, sig: Signals, t: datetime | None = None, lab_id: str | None = None) -> None:
        """Add one member; absorb members in (time, id) order so that `core` is the founder's."""
        if sig.vec is not None:
            if self.vec_sum is None:
                self.vec_sum, self.n_vecs = sig.vec.astype(np.float32).copy(), 1
            elif self.vec_sum.shape == sig.vec.shape:
                self.vec_sum, self.n_vecs = self.vec_sum + sig.vec, self.n_vecs + 1
        self.linkable |= sig.linkable
        self.subjects |= sig.subjects
        self.persons |= sig.persons
        self.versioned |= sig.versioned
        self.event_types.add(sig.event_type)
        earlier = t is not None and self.core_time is not None and t < self.core_time  # a backfilled founder
        if sig.versioned and (not self.core or earlier):
            self.core, self.core_time = set(sig.versioned), t
        if t is not None:
            self.first_seen, self.last_seen = min(self.first_seen, t), max(self.last_seen, t)
        if self.lab_id is None and lab_id:
            self.lab_id = lab_id

    def active_at(self, t: datetime) -> bool:
        return self.last_seen >= t - WINDOW and self.first_seen <= t + WINDOW

    def late_open_at(self, t: datetime) -> bool:
        """Not active, but `t` falls in the 21 days after the development started (forward only)."""
        return not self.active_at(t) and self.first_seen <= t <= self.first_seen + LATE_WINDOW


def _lookup_keys(keys: Iterable[str]) -> set[str]:
    return {k for key in keys for k in (key, *version_prefixes(key))}


class DevIndex:
    """Developments keyed by lab, artifact (and versioned family prefix) and person."""

    def __init__(self) -> None:
        self.by_lab: dict[str, dict[int, DevState]] = defaultdict(dict)
        self.by_artifact: dict[str, dict[int, DevState]] = defaultdict(dict)
        self.by_person: dict[str, dict[int, DevState]] = defaultdict(dict)

    def register(self, st: DevState) -> None:
        if st.lab_id:
            self.by_lab[st.lab_id][st.dev_id] = st
        for key in _lookup_keys(st.linkable):
            self.by_artifact[key][st.dev_id] = st
        for p in st.persons:
            self.by_person[p][st.dev_id] = st

    def candidates(self, lab_id: str | None, sig: Signals, t: datetime) -> list[DevState]:
        found: dict[int, DevState] = {}
        if lab_id:
            found.update({k: st for k, st in self.by_lab.get(lab_id, {}).items() if st.active_at(t)})
        for key in _lookup_keys(sig.linkable):
            for k, st in self.by_artifact.get(key, {}).items():
                if (st.active_at(t) and (lab_id is None or st.lab_id in (None, lab_id))
                        and related(sig.linkable, st.linkable)):
                    found[k] = st
        for p in sig.persons:
            for k, st in self.by_person.get(p, {}).items():
                cross_lab_people = sig.event_type in PEOPLE_EVENTS and bool(st.event_types & PEOPLE_EVENTS)
                if st.active_at(t) and (lab_id is None or st.lab_id in (None, lab_id) or cross_lab_people):
                    found[k] = st
        return [found[k] for k in sorted(found)]

    def late_candidates(self, lab_id: str | None, sig: Signals, t: datetime) -> list[DevState]:
        """Same-lab developments past their 7-day window that started at most 21 days before `t`, for a
        launch-type insight naming a versioned artifact. No lab, no late join: the lab is what tells a
        launch's own paper apart from a paper that merely used the model."""
        if not lab_id or not sig.versioned or sig.event_type not in LATE_EVENTS:
            return []
        return [st for _, st in sorted(self.by_lab.get(lab_id, {}).items()) if st.late_open_at(t)]


def structural_match(st: DevState, sig: Signals) -> bool:
    """Shared artifact (or, for people events, shared person) with a compatible event family."""
    if not any(compatible_events(sig.event_type, e) for e in st.event_types or {WILDCARD_EVENT}):
        return False
    if related(sig.linkable, st.linkable):
        return True
    people = sig.event_type in PEOPLE_EVENTS and bool(st.event_types & PEOPLE_EVENTS)
    return people and bool(sig.persons & st.persons)


def subject_conflict(st: DevState, sig: Signals) -> bool:
    """Both sides name subjects (models/products) and share none. A shared versioned name counts as a
    shared subject: "Beam-1.0-DFlash2, Beam 1.0" names the Beam 1.0 family exactly as "Gemini 4" does,
    but guess_kind files "Beam 1.0" under 'other', so it was no subject and the drafter of the model
    released 6 days earlier hit a conflict instead of a match."""
    return bool(sig.subjects and st.subjects) and not related(sig.subjects | sig.versioned,
                                                              st.subjects | st.versioned)


def late_match(st: DevState, sig: Signals) -> bool:
    """A launch seen again 7-21 days after its development started (weights, listing, paper).

    The candidate is a same-lab development (see DevIndex.late_candidates) holding a release; the
    insight is launch-type, shares a versioned artifact with the development's founding member (`core`),
    and names no versioned artifact the development does not know: "Gemini 4 Carbon ... after Gemini 4
    Argon" is the next model, not the Argon launch, and "deprecates fast mode for Opus 4.8, migrate to
    Opus 5.5" is a deprecation, not the Opus 5.5 fast-mode post.
    """
    return (sig.event_type in LATE_EVENTS and bool(st.event_types & RELEASE_EVENTS)
            and related(sig.versioned, st.core)
            and all(related({key}, st.linkable) for key in sig.versioned)
            and not subject_conflict(st, sig))


def choose(cands: Sequence[DevState], sig: Signals,
           late: Sequence[DevState] = ()) -> tuple[DevState | None, float]:
    """Best qualifying candidate (structural first, then similarity) and its similarity. `late`
    candidates (7-21 days, see late_match) only join structurally, never on similarity alone."""
    best, best_key = None, None
    for st in cands:
        sim = st.similarity(sig.vec)
        conflict = subject_conflict(st, sig)
        structural = not conflict and structural_match(st, sig)
        if not structural and sim < (CONFLICT_COSINE if conflict else JOIN_COSINE):
            continue
        key = (structural, sim)
        if best_key is None or key > best_key:
            best, best_key = st, key
    for st in late:
        if late_match(st, sig):
            key = (True, st.similarity(sig.vec))
            if best_key is None or key > best_key:
                best, best_key = st, key
    return best, (best_key[1] if best_key else 0.0)


def find_development(index: DevIndex, lab_id: str | None, sig: Signals,
                     t: datetime) -> tuple[DevState | None, float]:
    """Where an insight goes: the best near (7-day) or late (21-day launch) candidate, or None."""
    return choose(index.candidates(lab_id, sig, t), sig, index.late_candidates(lab_id, sig, t))


class MemberSignal(NamedTuple):
    development_id: int
    insight_id: int
    time: datetime
    lab_id: str | None
    sig: Signals


def member_signals(session: Session, dev_ids: Iterable[int]) -> list[MemberSignal]:
    """Matching signals of the developments' member insights, in (time, id) order."""
    ids = sorted(set(dev_ids))
    rows = []
    for i in range(0, len(ids), 500):
        rows += session.execute(
            select(Insight.id, Insight.development_id, Insight.lab_id, Insight.embedding, Insight.artifacts,
                   Insight.person_ids, Insight.event_type, insight_time_expr().label("seen_at"))
            .where(Insight.development_id.in_(ids[i : i + 500]))
        ).all()
    rows.sort(key=lambda r: (r.seen_at, r.id))
    return [MemberSignal(r.development_id, r.id, r.seen_at, r.lab_id,
                         Signals.of(r.embedding, r.artifacts, r.person_ids, r.event_type)) for r in rows]


def _load_index(session: Session, lo: datetime, hi: datetime) -> DevIndex:
    """Developments overlapping [lo, hi], rebuilt from their member insights (in time order, so that
    each development's `core` is its founder's)."""
    devs = session.execute(
        select(Development.id, Development.lab_id, Development.first_seen_at, Development.last_seen_at)
        .where(Development.last_seen_at >= lo, Development.first_seen_at <= hi)
    ).all()
    states = {d.id: DevState(d.id, d.lab_id, d.first_seen_at, d.last_seen_at) for d in devs}
    for m in member_signals(session, states):
        states[m.development_id].absorb(m.sig, m.time)
    index = DevIndex()
    for st in states.values():
        index.register(st)
    return index


def _open_development(session: Session, ins: Insight, t: datetime) -> DevState:
    dev = Development(title=(ins.title or "untitled")[:400], event_type=ins.event_type or WILDCARD_EVENT,
                      lab_id=ins.lab_id, person_ids=[], artifacts=[], first_seen_at=t, last_seen_at=t,
                      insight_count=0)
    session.add(dev)
    session.flush()
    return DevState(dev.id, ins.lab_id, t, t)


def _stale_developments(session: Session) -> list[int]:
    """Developments whose stored insight_count no longer matches their members (insights deleted or
    re-extracted since the last run): recomputing them keeps aggregates self-healing."""
    counts = dict(session.execute(
        select(Insight.development_id, func.count()).where(Insight.development_id.is_not(None))
        .group_by(Insight.development_id)
    ).all())
    rows = session.execute(select(Development.id, Development.insight_count)).all()
    return [r.id for r in rows if counts.get(r.id, 0) != (r.insight_count or 0)]


def run_clustering(limit: int | None = None, **_: object) -> dict:
    """Assign embedded, unassigned insights to developments (idempotent, incremental)."""
    with session_scope() as s:
        repaired = _stale_developments(s)
        for dev_id in repaired:
            recompute_development(s, dev_id)
        q = (select(Insight).where(Insight.embedding.is_not(None), Insight.development_id.is_(None))
             .order_by(insight_time_expr(), Insight.id))
        pending = list(s.scalars(q.limit(limit) if limit else q))
        if not pending:
            return {"assigned": 0, "joined": 0, "created": 0, "developments_touched": 0,
                    "repaired": len(repaired)}
        times = [insight_time(i) for i in pending]
        index = _load_index(s, min(times) - LATE_WINDOW, max(times) + WINDOW)
        touched: set[int] = set()
        joined = created = 0
        for ins, t in zip(pending, times, strict=True):
            sig = Signals.of(ins.embedding, ins.artifacts, ins.person_ids, ins.event_type)
            st, sim = find_development(index, ins.lab_id, sig, t)
            if st is None:
                st = _open_development(s, ins, t)
                created += 1
            else:
                joined += 1
                log.debug("insight %s joins development %s (similarity %.3f%s)", ins.id, st.dev_id, sim,
                          "" if st.active_at(t) else ", late launch join")
            ins.development_id = st.dev_id
            st.absorb(sig, t, ins.lab_id)
            index.register(st)
            touched.add(st.dev_id)
        s.flush()
        for dev_id in sorted(touched):
            recompute_development(s, dev_id)
    log.info("clustering: %d insights, %d joined, %d new developments", len(pending), joined, created)
    return {"assigned": len(pending), "joined": joined, "created": created,
            "developments_touched": len(touched), "repaired": len(repaired)}


# ------------------------------------------------------------------------------ aggregates
@dataclass
class _Member:
    ins: Insight
    source_id: str
    source_class: str
    time: datetime

    @property
    def official(self) -> bool:
        return bool(self.ins.is_official) or self.source_class == "official"


def _vote(values: Sequence[str | None], ignore: str | None = None) -> str | None:
    """Most common value; ties go to the earliest (values are in time order)."""
    vals = [v for v in values if v]
    if ignore:
        vals = [v for v in vals if v != ignore] or vals
    if not vals:
        return None
    counts = Counter(vals)
    top = max(counts.values())
    return next(v for v in vals if counts[v] == top)


def development_event_type(members: Sequence[tuple[str | None, bool]]) -> str:
    """Event type from the members' (event_type, official) in time order: an official model_release /
    open_weight_release wins (vote among those); otherwise a vote among the official members, then
    among all members; 'other' only when nothing else is typed. Vote ties go to the earliest."""
    official = [e for e, is_official in members if is_official]
    releases = [e for e in official if e in RELEASE_EVENTS]
    if releases:
        return _vote(releases) or WILDCARD_EVENT
    for pool in (official, [e for e, _ in members]):
        typed = [e for e in pool if e and e != WILDCARD_EVENT]
        if typed:
            return _vote(typed) or WILDCARD_EVENT
    return WILDCARD_EVENT


def _centroid(members: Sequence[_Member]) -> bytes | None:
    vecs = [from_bytes(m.ins.embedding) for m in members if m.ins.embedding]
    if not vecs:
        return None
    dim = Counter(v.shape[0] for v in vecs).most_common(1)[0][0]
    return to_bytes(normalize(np.mean([v for v in vecs if v.shape[0] == dim], axis=0)))


def recompute_development(session: Session, dev_id: int) -> Development | None:
    """Recompute every aggregate of a development from its member insights (idempotent)."""
    dev = session.get(Development, dev_id)
    if dev is None:
        return None
    rows = session.execute(
        select(Insight, Document.source_id, Document.source_class)
        .join(Document, Document.id == Insight.document_id)
        .where(Insight.development_id == dev_id)
    ).all()
    if not rows:
        dev.insight_count, dev.source_count, dev.status = 0, 0, "empty"
        return dev
    members = sorted((_Member(i, sid, scls, insight_time(i)) for i, sid, scls in rows),
                     key=lambda m: (m.time, m.ins.id))
    pool = [m for m in members if m.official] or members  # official statements outrank coverage
    lead = min(members, key=lambda m: (not m.official, not m.ins.quotes_verified, m.time, m.ins.id))
    dev.title = (lead.ins.title or dev.title)[:400]
    dev.event_type = development_event_type([(m.ins.event_type, m.official) for m in members])
    dev.lab_id = _vote([m.ins.lab_id for m in pool]) or _vote([m.ins.lab_id for m in members])
    dev.person_ids = sorted(set().union(*(_persons(m.ins.person_ids) for m in members)))
    artifacts: dict[str, str] = {}
    for m in members:
        for key, display in unique_normalized(m.ins.artifacts):
            artifacts.setdefault(key, display)
    dev.artifacts = list(artifacts.values())
    dev.first_seen_at, dev.last_seen_at = members[0].time, max(m.time for m in members)
    dev.insight_count = len(members)
    dev.source_count = len({m.source_id for m in members})
    dev.source_classes = sorted({m.source_class for m in members if m.source_class})
    dev.has_official = any(m.official for m in members)
    dev.has_verified = any(bool(m.ins.quotes_verified) for m in members)
    dev.centroid = _centroid(members)
    if dev.status == "empty":
        dev.status = "open"
    return dev
