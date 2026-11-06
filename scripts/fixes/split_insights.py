"""Detach wrongly clustered insights, each into a development of its own.

Clustering is online and never moves an insight once it is assigned, so a wrong merge found in a
review stays until it is repaired here. Examples from the 2026-11-04 review of data/fli.sqlite:
  519  Meta's Prometheus supercluster milestone, inside development 312 (Muse Wonder release)
  594  the OpenAI Signals work-usage data post, inside development 377 (ChatGPT Ads expansion)
  590  an arXiv sycophancy paper that used Gemini, inside development 402 (PlanAssist)
  639  a ChatGPT violence-prevention post, inside development 416 (OpenAI-Oracle/ServiceNow)

For each insight id: open a new development, move the insight into it, then recompute both the old
and the new development from their members (fli.analyze.cluster.recompute_development), so title,
event type, lab, artifacts, dates and counts never disagree with the insights.

Idempotent: an insight that already is the only member of its development is left alone, so running
the same command twice changes nothing the second time.

Scores, lenses, exposure links, labels and alerts stay attached to the old development. Afterwards
re-run the deterministic stages so both developments are re-linked and re-scored:
    uv run fli pipeline --stages novelty,exposure,score
(the lens and alerts stages may call the LLM for new lenses).

Usage:
    uv run python scripts/fixes/split_insights.py 519 594 590 639 [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import func, select

from fli.analyze.cluster import WILDCARD_EVENT, insight_time, recompute_development
from fli.db.models import Development, Insight
from fli.db.session import session_scope


@dataclass
class Outcome:
    insight_id: int
    action: str  # split | would_split | already_alone | unclustered | not_found
    old_development_id: int | None = None
    new_development_id: int | None = None
    detail: str = ""

    def describe(self) -> str:
        head = f"insight {self.insight_id}:"
        if self.action == "split":
            return (f"{head} split from development {self.old_development_id} into new development "
                    f"{self.new_development_id}; {self.detail}")
        if self.action == "would_split":
            return f"{head} would split from development {self.old_development_id} (dry run); {self.detail}"
        if self.action == "already_alone":
            return f"{head} already the only member of development {self.old_development_id}; nothing to do"
        if self.action == "unclustered":
            return f"{head} not assigned to a development yet (the cluster stage places it); skipped"
        return f"{head} not found; skipped"


def _members(session, dev_id: int) -> int:
    return int(session.scalar(select(func.count()).select_from(Insight).where(Insight.development_id == dev_id)) or 0)


def split_insights(insight_ids: Iterable[int], dry_run: bool = False) -> list[Outcome]:
    """Detach each insight into a new development (see module docstring). One transaction for all."""
    outcomes: list[Outcome] = []
    with session_scope() as s:
        for iid in dict.fromkeys(int(i) for i in insight_ids):  # de-duplicated, order kept
            ins = s.get(Insight, iid)
            if ins is None:
                outcomes.append(Outcome(iid, "not_found"))
                continue
            old_id = ins.development_id
            if old_id is None:
                outcomes.append(Outcome(iid, "unclustered"))
                continue
            n_old = _members(s, old_id)
            if n_old <= 1:
                outcomes.append(Outcome(iid, "already_alone", old_id, old_id))
                continue
            old = s.get(Development, old_id)
            before = f"development {old_id} {old.title[:70]!r} had {n_old} insights"
            if dry_run:
                outcomes.append(Outcome(iid, "would_split", old_id, None, f"{before}; insight {ins.title[:70]!r}"))
                continue
            t = insight_time(ins)
            new = Development(title=(ins.title or "untitled")[:400], event_type=ins.event_type or WILDCARD_EVENT,
                              lab_id=ins.lab_id, person_ids=[], artifacts=[], first_seen_at=t, last_seen_at=t,
                              insight_count=0)
            s.add(new)
            s.flush()
            ins.development_id = new.id
            s.flush()
            recompute_development(s, old_id)
            recompute_development(s, new.id)
            s.flush()
            outcomes.append(Outcome(
                iid, "split", old_id, new.id,
                f"{before}, now {old.insight_count} titled {old.title[:70]!r}; new development "
                f"{new.title[:70]!r} ({new.event_type}, first seen {new.first_seen_at:%Y-%m-%d %H:%M})"))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detach wrongly clustered insights into developments of their own.")
    ap.add_argument("insight_ids", nargs="+", type=int, help="insight ids to detach")
    ap.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = ap.parse_args(argv)
    outcomes = split_insights(args.insight_ids, dry_run=args.dry_run)
    for o in outcomes:
        print(o.describe())
    done = sum(o.action == "split" for o in outcomes)
    print(f"split {done}, would split {sum(o.action == 'would_split' for o in outcomes)}, "
          f"unchanged {sum(o.action in ('already_alone', 'unclustered') for o in outcomes)}, "
          f"not found {sum(o.action == 'not_found' for o in outcomes)}")
    if done:
        print("next: uv run fli pipeline --stages novelty,exposure,score")
    return 1 if any(o.action == "not_found" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
