"""Merge developments that the 21-day launch rule would have clustered together, and re-type them all.

Clustering is online and never moves an insight once it is assigned, so developments created before the
late-join rule in fli.analyze.cluster stay split until they are merged here. Under that rule a launch's
weights, listing or paper joins the launch up to 21 days after it started when it shares a versioned
artifact with it. Examples from data/fli.sqlite (2026-11-05):
  295 + 322 + 339  Reflection Beam-1.0 501B on Hugging Face, its drafter 6 days later, the OpenRouter listing
  326 + 411        Mistral Timestral release and its arXiv paper 12 days later
  270 + 319        Gemini 4 Argon for trusted cyber defenders, then generally available 10 days later

"Would merge": development B merges into an earlier development A when B's founding insight (its
earliest member) would have joined A through a shared *versioned* artifact under the current rule, with
A as the online clusterer saw it (A's members before that insight):
  - A active (within 7 days): a structural match on a versioned name with no subject conflict
    (cluster.structural_match, cluster.subject_conflict), from a candidate lab (same lab, or no lab)
  - 7-21 days after A started: cluster.late_match (same lab, release-type A, launch-type insight)
Plain cosine and unversioned names are not replayed: those joins were possible when B was created and
were rejected then, and replaying them would undo the 2026-11-04 repairs of scripts/fixes/split_insights.py
(insights 519, 594, 590, 639 stay apart). A split that contradicts the versioned rule itself is merged
again, as the clusterer would (nothing records a split); --exclude keeps named developments as they are.
With several qualifying A, the most similar wins (as cluster.choose among structural matches). Chains end
at the earliest development, the survivor; the search repeats until nothing more merges, so a second
run finds nothing.

For each group: move the insights to the survivor and recompute it; delete the absorbed developments'
score, lens, exposure_link and alert rows (the pipeline regenerates what the survivor needs); move each
label to the survivor unless it already has one for that (audience, annotator), in which case the
survivor's is kept and the other stays on the absorbed row; mark the absorbed development 'merged'
with a summary naming the survivor. Absorbed rows are kept, not deleted, so old alert and report links
still resolve; every stage skips status 'merged' (and labels on it are not used for training).

Then every live development is recomputed, which applies the event-type rule (an official
model_release / open_weight_release member wins over the majority: Claude Fable/Mythos 5.1 was a
product_launch). Recompute only writes what changed, so unchanged developments are not touched.
On a copy of data/fli.sqlite (2026-11-05): 6 developments merged into 5 survivors, 7 re-typed, and the
result equals a from-scratch re-clustering with the new rule except the 2 manual splits of 2026-11-04.

Afterwards re-run the deterministic stages, so survivors are re-linked and re-scored, then the lens and
alert stages: a survivor's lens was written from its own first insights (Gemini 4 Argon's says 'hold'
for the gated preview, which vetoes the generally-available launch merged into it), and the lens stage
refreshes it because the development gained insights and source classes (LLM calls):
    uv run fli pipeline --stages cluster,novelty,exposure,score
    uv run fli pipeline --stages lens,alerts

Usage:
    uv run python scripts/fixes/merge_developments.py --dry-run   # list the groups and re-typed developments
    uv run python scripts/fixes/merge_developments.py [--exclude DEV_ID ...]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import delete, func, select

from fli.analyze.artifacts import same_artifact
from fli.analyze.cluster import (
    DevState,
    MemberSignal,
    development_event_type,
    insight_time_expr,
    late_match,
    member_signals,
    recompute_development,
    structural_match,
    subject_conflict,
)
from fli.analyze.exposure import SKIP_STATUSES
from fli.db.models import Alert, Development, Document, ExposureLink, Insight, Label, Lens, Score
from fli.db.session import session_scope
from fli.timeutil import utcnow

MERGED = "merged"
DERIVED = (Score, Lens, ExposureLink, Alert)  # per-development rows the pipeline regenerates for the survivor


@dataclass
class Live:
    """A live development while planning: its stored lab and its members in (time, id) order."""

    dev_id: int
    lab_id: str | None
    members: list[MemberSignal]

    @property
    def start(self) -> tuple[datetime, int]:
        return self.members[0].time, self.dev_id

    def state_before(self, m: MemberSignal) -> DevState | None:
        """The development as the online clusterer saw it when member `m` of another one arrived."""
        prior = [p for p in self.members if (p.time, p.insight_id) < (m.time, m.insight_id)]
        if not prior:
            return None
        st = DevState(self.dev_id, self.lab_id, prior[0].time, prior[0].time)
        for p in prior:
            st.absorb(p.sig, p.time, p.lab_id)
        return st


@dataclass
class Edge:
    absorbed: int
    into: int
    how: str  # "within 7 days" | "late launch join"
    names: list[str]  # the shared versioned names
    gap_days: float  # from the target's last member before the founding insight
    similarity: float


@dataclass
class Group:
    survivor: int
    absorbed: list[int] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def shared_versioned(xs: set[str], ys: set[str]) -> list[str]:
    """Versioned names shared by xs and ys: equal, or one the versioned family of the other (the shorter)."""
    return sorted({min(x, y, key=len) for x in xs for y in ys if same_artifact(x, y)})


def match(a: Live, b: Live) -> Edge | None:
    """Would b's founding insight join a through a shared versioned artifact (see the module docstring)?"""
    f = b.members[0]
    if not f.sig.versioned:
        return None
    st = a.state_before(f)
    if st is None:
        return None
    if st.active_at(f.time):
        lab_ok = f.lab_id is None or a.lab_id is None or a.lab_id == f.lab_id
        names = shared_versioned(f.sig.versioned, st.versioned)
        ok = lab_ok and bool(names) and structural_match(st, f.sig) and not subject_conflict(st, f.sig)
        how = "within 7 days"
    else:
        ok = st.late_open_at(f.time) and bool(f.lab_id) and a.lab_id == f.lab_id and late_match(st, f.sig)
        names, how = shared_versioned(f.sig.versioned, st.core), "late launch join"
    if not ok:
        return None
    gap = (f.time - st.last_seen).total_seconds() / 86400
    return Edge(b.dev_id, a.dev_id, how, names, round(max(gap, 0.0), 1), round(st.similarity(f.sig.vec), 3))


def _load_live(session) -> dict[int, Live]:
    devs = session.execute(select(Development.id, Development.lab_id)
                           .where(Development.status.not_in(SKIP_STATUSES))).all()
    live = {d.id: Live(d.id, d.lab_id, []) for d in devs}
    for m in member_signals(session, live):
        live[m.development_id].members.append(m)
    return {k: v for k, v in live.items() if v.members}


def plan(session, exclude: Iterable[int] = ()) -> list[Group]:
    """Groups to merge, survivor first (read-only). `exclude`: developments kept out of every group."""
    skip = set(exclude)
    live = {k: v for k, v in _load_live(session).items() if k not in skip}
    edges: dict[int, Edge] = {}
    while True:  # to a fixpoint: a grown survivor can reach a development neither part reached alone
        order = sorted((d for d in live.values() if d.dev_id not in edges), key=lambda d: d.start)
        moved = False
        for i, b in enumerate(order):
            if b.dev_id in edges:
                continue
            found = [e for a in order[:i] if a.dev_id not in edges and (e := match(a, b)) is not None]
            if not found:
                continue
            best = max(found, key=lambda e: e.similarity)  # all structural; ties -> earliest (max is stable)
            edges[b.dev_id] = best
            target = live[best.into]
            target.members = sorted(target.members + b.members, key=lambda m: (m.time, m.insight_id))
            moved = True
        if not moved:
            break

    def root(dev_id: int) -> int:
        while dev_id in edges:
            dev_id = edges[dev_id].into
        return dev_id

    groups: dict[int, Group] = {}
    for dev_id in sorted(edges, key=lambda d: live[d].start):
        g = groups.setdefault(root(dev_id), Group(root(dev_id)))
        g.absorbed.append(dev_id)
        g.edges.append(edges[dev_id])
    return sorted(groups.values(), key=lambda g: live[g.survivor].start)


def labels_plan(session, survivor: int, absorbed: Sequence[int]) -> tuple[list[Label], list[Label]]:
    """(labels to move to the survivor, labels that stay because the survivor has that annotator)."""
    taken = {(lb.audience, lb.annotator) for lb in session.scalars(select(Label).where(Label.development_id == survivor))}
    move, keep = [], []
    for dev_id in absorbed:  # earliest absorbed first: its label wins a tie between two absorbed rows
        for lb in session.scalars(select(Label).where(Label.development_id == dev_id).order_by(Label.id)):
            key = (lb.audience, lb.annotator)
            (keep if key in taken else move).append(lb)
            taken.add(key)
    return move, keep


def retype_preview(session, groups: Sequence[Group]) -> list[tuple[int, str, str, str]]:
    """(id, title, stored event type, type after merging and recomputing) for live developments that change."""
    into = {d: g.survivor for g in groups for d in g.absorbed}
    rows = session.execute(
        select(Insight.development_id, Insight.event_type, Insight.is_official, Document.source_class)
        .join(Document, Document.id == Insight.document_id)
        .join(Development, Development.id == Insight.development_id)
        .where(Development.status.not_in(SKIP_STATUSES))
        .order_by(insight_time_expr(), Insight.id)).all()
    members: dict[int, list[tuple[str | None, bool]]] = defaultdict(list)
    for dev_id, event_type, is_official, source_class in rows:
        members[into.get(dev_id, dev_id)].append((event_type, bool(is_official) or source_class == "official"))
    out = []
    for dev in session.scalars(select(Development).where(Development.id.in_(sorted(members))).order_by(Development.id)):
        new = development_event_type(members[dev.id])
        if new != dev.event_type:
            out.append((dev.id, dev.title or "", dev.event_type, new))
    return out


def describe(session, groups: Sequence[Group], retyped: Sequence[tuple[int, str, str, str]]) -> list[str]:
    ids = [x for g in groups for x in (g.survivor, *g.absorbed)]
    devs = {d.id: d for d in session.scalars(select(Development).where(Development.id.in_(ids)))}
    lines = []
    for g in groups:
        s = devs[g.survivor]
        lines.append(f"group: survivor {s.id} [{s.event_type}] {s.first_seen_at:%Y-%m-%d} {s.title[:80]!r}")
        for e in g.edges:
            d = devs[e.absorbed]
            lines.append(f"  absorb {d.id} [{d.event_type}] {d.first_seen_at:%Y-%m-%d} {d.title[:70]!r} into {e.into}: "
                         f"{e.how}, {e.gap_days} d after its last member, similarity {e.similarity}, "
                         f"shared {', '.join(e.names)}")
        counts = {m.__tablename__: session.scalar(select(func.count()).select_from(m)
                                                  .where(m.development_id.in_(g.absorbed))) for m in DERIVED}
        move, keep = labels_plan(session, g.survivor, g.absorbed)
        lines.append("  delete " + ", ".join(f"{n} {k}" for k, n in counts.items())
                     + f"; labels: move {len(move)}, keep the survivor's {len(keep)}")
    for dev_id, title, old, new in retyped:
        lines.append(f"retype {dev_id} {title[:80]!r}: {old} -> {new}")
    return lines


def apply(session, groups: Sequence[Group]) -> None:
    today = f"{utcnow():%Y-%m-%d}"
    for g in groups:
        move, _ = labels_plan(session, g.survivor, g.absorbed)
        for lb in move:
            lb.development_id = g.survivor
        for ins in session.scalars(select(Insight).where(Insight.development_id.in_(g.absorbed))):
            ins.development_id = g.survivor
        session.flush()
        for model in DERIVED:
            session.execute(delete(model).where(model.development_id.in_(g.absorbed)))
        recompute_development(session, g.survivor)
        for dev_id in g.absorbed:
            dev = session.get(Development, dev_id)
            dev.status, dev.insight_count, dev.source_count = MERGED, 0, 0
            dev.summary = (f"Merged into development {g.survivor} on {today} by scripts/fixes/merge_developments.py "
                           "(the same launch, seen again later).")
        session.flush()


def _live_types(session) -> dict[int, str]:
    return dict(session.execute(select(Development.id, Development.event_type)
                                .where(Development.status.not_in(SKIP_STATUSES))).all())


def retype_all(session) -> None:
    """Recompute every live development, which applies the event-type rule (unchanged rows are not written)."""
    for dev_id in sorted(_live_types(session)):
        recompute_development(session, dev_id)
    session.flush()


def merge_developments(dry_run: bool = False, exclude: Iterable[int] = ()) -> tuple[list[Group], int, list[str]]:
    """Plan, describe and (unless dry_run) apply, in one transaction -> (groups, re-typed count, report)."""
    with session_scope() as s:
        groups = plan(s, exclude)
        retyped = retype_preview(s, groups)
        lines = describe(s, groups, retyped)
        n_retyped = len(retyped)
        if not dry_run:
            before = _live_types(s)
            apply(s, groups)
            retype_all(s)
            n_retyped = sum(before[k] != v for k, v in _live_types(s).items())
    return groups, n_retyped, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge developments split by the old 7-day clustering window.")
    ap.add_argument("--dry-run", action="store_true", help="list the groups and re-typed developments; write nothing")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], metavar="DEV_ID",
                    help="developments kept out of every group (still re-typed), e.g. after reviewing a dry run")
    args = ap.parse_args(argv)
    groups, n_retyped, lines = merge_developments(dry_run=args.dry_run, exclude=args.exclude)
    for line in lines:
        print(line)
    n_absorbed = sum(len(g.absorbed) for g in groups)
    survivors = ", ".join(str(g.survivor) for g in groups) or "none"
    if args.dry_run:
        print(f"dry run: would merge {n_absorbed} developments into {len(groups)} (survivors: {survivors}); "
              f"would re-type {n_retyped}")
    else:
        print(f"merged {n_absorbed} developments into {len(groups)} (survivors: {survivors}); re-typed {n_retyped}")
        if groups or n_retyped:
            print("next: uv run fli pipeline --stages cluster,novelty,exposure,score, then "
                  "uv run fli pipeline --stages lens,alerts (refreshes the survivors' lenses: LLM calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
