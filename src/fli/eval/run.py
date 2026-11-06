"""Evaluation harness: `fli eval [--suite ...]` -> data/eval/report.json + data/eval/report.md

Suites
  known_events  recall of independently listed real events (data/eval/known_events.yaml), their rank
                within their week, whether an alert fired or was pushed, and detection delay measured
                from the source date of the insight that reported the event
  extraction    hallucination control: quote-verification rate on 100% of insights, citation validity
                (every insight resolves to a stored document with a URL and a metadata date), golden-set
                field accuracy when data/eval/golden_extraction.yaml exists
  noise         what triage suppressed and why; exports a random sample of suppressed items for audit
  alerts        pushed alerts (P1/P2) per audience: volume per week and per active day, precision and
                label-2 recall against labels, overall and on the scorer's hold-out window
  scoring       metrics written by `fli train` (time-split AUC, P@k, calibration, baselines, ablation)
  er            entity-resolution precision/recall on hand-labelled pairs
  costs         dollars and tokens per workflow from the llm_call ledger
  lens          lens faithfulness from the deterministic checks stored on every lens (Lens.checks): share
                grounded, unsupported figures by field, replaced headlines, official-wording flags, per
                audience and for the lenses behind pushed alerts
  holdings      coverage by BIT holding (config/exposure.yaml): documents in the window that name the
                company or a brand, developments with an event-specific exposure link, pushed alerts
                listing it; a holding no document names has no source coverage (silence is not news)
  digests       weekly digest quality per audience: written by the model vs the deterministic fallback,
                cited coverage of the selected items, items left to 'Also this week', missing weeks
  sources       documents per source class, kind and lab; registered sources that never produced a
                document, and why (disabled, never run, failing, ran without items)

Running a subset (`--suite alerts`) updates those suites in the existing report and keeps the rest.
The report (and the noise audit sample) is written to data/eval, or to $FLI_EVAL_OUT_DIR when set
(for runs on a database copy); the inputs (known_events.yaml, scoring_metrics.json, er_pairs.yaml)
are always read from data/eval.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from sqlalchemy import func, select

from fli.config import CONFIG_DIR, DATA_DIR, get_settings
from fli.db.models import (
    Alert,
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Lens,
    LLMCall,
    Report,
    Score,
    Source,
)
from fli.db.session import session_scope
from fli.taxonomy import AUDIENCES, DIRECTIONS
from fli.timeutil import utcnow, week_start

log = logging.getLogger("fli.eval")
EVAL_DIR = DATA_DIR / "eval"
OUT_DIR_ENV = "FLI_EVAL_OUT_DIR"  # write the report elsewhere (a run on a database copy must not overwrite data/eval)

PUSHED = ("P1", "P2")  # severities that reach a person (in-app / Slack); P3 is digest-only
_SEVERITY_RANK = {"P3": 1, "P2": 2, "P1": 3}
# Date-only sources are stamped at midnight, so an alert dated up to a day before the evidence can
# still be about it; one older than that was raised for an earlier event of the same development.
EVIDENCE_SLACK = timedelta(days=1)
# Structurally dead developments (as fli.analyze.exposure.SKIP_STATUSES): emptied or merged by the
# cluster stage, they keep a title but no members, so they are neither matched nor counted.
DEAD_STATUSES = ("merged", "duplicate", "empty")


# ------------------------------------------------------------------------------------ helpers
def _latest_scores(s) -> dict[tuple[int, str], float]:
    """(development_id, audience) -> p_material from the newest score row."""
    out: dict[tuple[int, str], tuple[datetime, float]] = {}
    for dev_id, aud, p, created in s.execute(
        select(Score.development_id, Score.audience, Score.p_material, Score.created_at)
    ):
        key = (dev_id, aud)
        if key not in out or created > out[key][0]:
            out[key] = (created, p)
    return {k: v[1] for k, v in out.items()}


def _has_term(text: str, term: str) -> bool:
    """Whole-word/phrase match: 'accord' must not match 'according', 's-1' must not match 'cs-1'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _groups_hit(text: str, groups: list[list[str]]) -> int:
    """How many `match_all` groups have at least one term in `text` (lowercase)."""
    return sum(any(_has_term(text, t) for t in g) for g in groups)


def _insight_text(ins: Insight) -> str:
    return " ".join([ins.title or "", ins.claim or "", " ".join(ins.artifacts or [])]).lower()


def _text_of(dev: Development, insights: Iterable[Insight]) -> str:
    """What known-event terms are matched against: the development's title and artifacts plus every
    member insight's title, claim and artifacts."""
    return " ".join([dev.title or "", " ".join(dev.artifacts or []), *(_insight_text(i) for i in insights)]).lower()


def _dev_text(s, dev: Development) -> str:
    """Matched text of one development, loaded on its own (for ad-hoc audits of the matching)."""
    return _text_of(dev, s.scalars(select(Insight).where(Insight.development_id == dev.id)))


def _candidate_rank(dev: Development, ev: dict, groups: list[list[str]], date: datetime, p: float) -> tuple:
    """Which matching development *is* the event: same event type first, then the most match groups in
    its title, the highest primary-audience score, the closest first sighting, the oldest id. Taking
    the earliest match instead credited the Anthropic S-1 to an API changelog that said 'listing'."""
    return (bool(ev.get("event_type")) and dev.event_type == ev.get("event_type"),
            _groups_hit((dev.title or "").lower(), groups), p,
            -abs((dev.first_seen_at - date).total_seconds()), -dev.id)


def _evidence(members: list[tuple[Insight, datetime | None]],
              groups: list[list[str]]) -> tuple[Insight | None, datetime | None]:
    """The member insight that reports the event and when its source published it: most match groups
    in the insight's own title/claim/artifacts, then the earliest document published_at. That is when
    the system first held the news; the development can be older (a ChatGPT-for-Financial-Services
    launch folded into the GPT-6 Astra development first seen nine days earlier)."""
    best: tuple[tuple, Insight, datetime] | None = None
    for ins, published in members:
        when = published or ins.published_at or ins.created_at
        if when is None:
            continue
        key = (-_groups_hit(_insight_text(ins), groups), when, ins.id)
        if best is None or key < best[0]:
            best = (key, ins, when)
    return (best[1], best[2]) if best else (None, None)


# ------------------------------------------------------------------------------------ suites
def eval_known_events(path: Path | None = None) -> dict:
    spec = yaml.safe_load((path or EVAL_DIR / "known_events.yaml").read_text(encoding="utf-8"))
    results = []
    with session_scope() as s:
        scores = _latest_scores(s)
        devs = list(s.scalars(select(Development).where(Development.status.not_in(DEAD_STATUSES))
                              .order_by(Development.id)))
        members: dict[int, list[tuple[Insight, datetime | None]]] = defaultdict(list)
        for ins, published in s.execute(
            select(Insight, Document.published_at).outerjoin(Document, Document.id == Insight.document_id)
            .where(Insight.development_id.is_not(None)).order_by(Insight.id)
        ):
            members[ins.development_id].append((ins, published))
        texts = {d.id: _text_of(d, [i for i, _ in members[d.id]]) for d in devs}
        alerts: dict[tuple[int, str], Alert] = {}
        for a in s.scalars(select(Alert).order_by(Alert.id)):
            key = (a.development_id, a.audience)
            if key not in alerts or _SEVERITY_RANK.get(a.severity, 0) > _SEVERITY_RANK.get(alerts[key].severity, 0):
                alerts[key] = a
        by_week: dict[tuple[datetime, str], list[tuple[float, int]]] = defaultdict(list)
        for d in devs:
            for aud in AUDIENCES:
                if (d.id, aud) in scores:
                    by_week[(week_start(d.first_seen_at), aud)].append((scores[(d.id, aud)], d.id))
        for ev in spec["events"]:
            date = datetime.fromisoformat(ev["date"])
            win = timedelta(days=int(ev.get("window_days", 7)))
            labs = set(ev.get("labs") or [])
            groups = [[t.lower() for t in g] for g in ev.get("match_all", [])]
            primary = (ev.get("audiences") or ["investment"])[0]
            cands = [
                d for d in devs
                if (not labs or d.lab_id in labs)
                and abs(d.first_seen_at - date) <= win
                and all(any(_has_term(texts[d.id], t) for t in g) for g in groups)
            ]
            row = {"id": ev["id"], "title": ev["title"], "confidence": ev.get("confidence"), "found": bool(cands)}
            if cands:
                best = max(cands, key=lambda d: _candidate_rank(d, ev, groups, date, scores.get((d.id, primary), 0.0)))
                ins, seen = _evidence(members[best.id], groups)
                seen = seen or best.first_seen_at
                row.update({
                    "development_id": best.id,
                    "development_title": best.title,
                    "first_seen_at": best.first_seen_at.isoformat(),
                    "evidence_insight_id": ins.id if ins else None,
                    "evidence_title": ins.title if ins else None,
                    "evidence_published_at": seen.isoformat(),
                    "delay_days": round((seen - date).total_seconds() / 86400, 2),
                    "matched_developments": len(cands),
                })
                for aud in ev.get("audiences", []):
                    ranked = sorted(by_week.get((week_start(best.first_seen_at), aud), []), reverse=True)
                    ids = [i for _, i in ranked]
                    row[f"rank_{aud}"] = ids.index(best.id) + 1 if best.id in ids else None
                    row[f"p_{aud}"] = round(scores.get((best.id, aud), 0.0), 3)
                    alert = alerts.get((best.id, aud))
                    row[f"alert_{aud}"] = alert.severity if alert else None
                    if alert is not None and alert.created_at < seen - EVIDENCE_SLACK:
                        row[f"alert_{aud}_predates_evidence"] = True
            results.append(row)
    n = len(results)
    found = [r for r in results if r["found"]]

    def share(pred) -> float:
        return round(sum(1 for r in results if pred(r)) / n, 3) if n else 0.0

    primary = {r["id"]: (spec_ev.get("audiences") or ["investment"])[0]
               for r, spec_ev in zip(results, spec["events"], strict=True)}
    return {
        "events": n,
        "recall": share(lambda r: r["found"]),
        "recall_high_confidence": round(
            sum(1 for r, e in zip(results, spec["events"], strict=True) if r["found"] and e.get("confidence") == "high")
            / max(1, sum(1 for e in spec["events"] if e.get("confidence") == "high")), 3),
        "top10_in_week_primary_audience": share(
            lambda r: r["found"] and (r.get(f"rank_{primary[r['id']]}") or 99) <= 10),
        "alerted_primary_audience": share(lambda r: r["found"] and r.get(f"alert_{primary[r['id']]}") is not None),
        "pushed_primary_audience": share(lambda r: r["found"] and r.get(f"alert_{primary[r['id']]}") in PUSHED),
        # alerted above, but the alert was raised for an earlier event of the same development
        "alert_predates_evidence": [r["id"] for r in found if r.get(f"alert_{primary[r['id']]}_predates_evidence")],
        "median_delay_days": statistics.median([r["delay_days"] for r in found]) if found else None,
        "matching": "whole-word terms, every match_all group; best candidate by event type, title hits, "
                    "primary-audience p, closeness in time; delay from the reporting insight's document date",
        "details": results,
    }


def eval_extraction() -> dict:
    with session_scope() as s:
        insights = list(s.scalars(select(Insight)))
        n = len(insights)
        quotes = [q for i in insights for q in (i.verification or {}).get("quotes", [])]
        verified_insights = sum(1 for i in insights if i.quotes_verified)
        any_verified = sum(1 for i in insights if any(q.get("verified") for q in (i.verification or {}).get("quotes", [])))
        doc_ids = {i.document_id for i in insights}
        docs = {d.id: d for d in s.scalars(select(Document).where(Document.id.in_(doc_ids)))} if doc_ids else {}
        citation_ok = sum(1 for i in insights if docs.get(i.document_id) is not None
                          and docs[i.document_id].url and docs[i.document_id].published_at is not None)
        by_type = Counter(i.event_type for i in insights)
        by_class = Counter(docs[i.document_id].source_class for i in insights if i.document_id in docs)
        unverified_examples = [
            {"insight_id": i.id, "claim": i.claim, "quotes": i.evidence_quotes[:2]}
            for i in insights if not i.quotes_verified
        ][:15]
    out = {
        "insights": n,
        "quotes": len(quotes),
        "quote_verification_rate": round(sum(1 for q in quotes if q.get("verified")) / len(quotes), 4) if quotes else None,
        "insights_all_quotes_verified": round(verified_insights / n, 4) if n else None,
        "insights_at_least_one_verified": round(any_verified / n, 4) if n else None,
        "citation_validity": round(citation_ok / n, 4) if n else None,
        "by_event_type": dict(by_type.most_common()),
        "by_source_class": dict(by_class.most_common()),
        "unverified_examples": unverified_examples,
    }
    golden = EVAL_DIR / "golden_extraction.yaml"
    if golden.exists():
        out["golden"] = _eval_golden(golden)
    return out


def _eval_golden(path: Path) -> dict:
    """Golden set: per document the human-annotated expected insights (event_type + key phrase).
    An expected insight is recovered when an extracted insight of that document has the same
    event_type and contains the key phrase; precision counts extracted insights that match any
    expected insight of their document."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    docs = spec.get("documents", [])
    exp_total = exp_hit = ext_total = ext_hit = 0
    with session_scope() as s:
        for d in docs:
            got = list(s.scalars(select(Insight).where(Insight.document_id == d["document_id"])))
            expected = d.get("expected", [])
            exp_total += len(expected)
            ext_total += len(got)

            def hit(ins: Insight, e: dict) -> bool:
                text = f"{ins.title} {ins.claim} {' '.join(ins.artifacts or [])}".lower()
                return ins.event_type == e["event_type"] and e["key"].lower() in text

            exp_hit += sum(1 for e in expected if any(hit(i, e) for i in got))
            ext_hit += sum(1 for i in got if any(hit(i, e) for e in expected))
    return {
        "documents": len(docs),
        "expected_insights": exp_total,
        "recall": round(exp_hit / exp_total, 3) if exp_total else None,
        "precision": round(ext_hit / ext_total, 3) if ext_total else None,
    }

def eval_noise(sample_size: int = 40, out_dir: Path | None = None) -> dict:
    """What triage suppressed and why; exports a seeded random sample of suppressed items for a
    human false-negative audit (noise_audit_sample.csv in `out_dir`, default data/eval)."""
    sample_dir = Path(out_dir) if out_dir else EVAL_DIR
    with session_scope() as s:
        status = dict(s.execute(select(Document.status, func.count()).group_by(Document.status)).all())
        skipped = list(s.scalars(select(Document).where(Document.status == "skipped")))
        reasons = Counter((d.triage or {}).get("noise_reason", "unknown") for d in skipped)
        by_class: dict[str, Counter] = defaultdict(Counter)
        for cls, st, n in s.execute(
            select(Document.source_class, Document.status, func.count()).group_by(Document.source_class, Document.status)
        ):
            by_class[cls][st] += n
        rng = random.Random(20261104)
        sample = rng.sample(skipped, min(sample_size, len(skipped)))
        sample_dir.mkdir(parents=True, exist_ok=True)
        with open(sample_dir / "noise_audit_sample.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["document_id", "source_id", "title", "noise_reason", "triage_reason", "url", "should_have_surfaced"])
            for d in sample:
                t = d.triage or {}
                w.writerow([d.id, d.source_id, (d.title or "")[:160], t.get("noise_reason"), t.get("reason"), d.url, ""])
    total_triaged = sum(v for k, v in status.items() if k in {"skipped", "relevant", "extracted"})
    out = {
        "documents_by_status": status,
        "suppressed_share": round(status.get("skipped", 0) / total_triaged, 3) if total_triaged else None,
        "noise_reasons": dict(reasons.most_common()),
        "relevant_share_by_source_class": {
            cls: round((c["relevant"] + c["extracted"]) / max(1, c["relevant"] + c["extracted"] + c["skipped"]), 3)
            for cls, c in by_class.items()
        },
        "audit_sample": str(sample_dir / "noise_audit_sample.csv"),
    }
    audited = EVAL_DIR / "noise_audit_labels.csv"  # same columns, should_have_surfaced filled with 0/1
    if audited.exists():
        with open(audited, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        judged = [r for r in rows if r.get("should_have_surfaced") in {"0", "1"}]
        if judged:
            out["audit_false_negative_rate"] = round(sum(r["should_have_surfaced"] == "1" for r in judged) / len(judged), 3)
            out["audit_n"] = len(judged)
    return out


ALERT_DEFINITIONS = {
    "pushed": "P1 or P2 (reaches a person in-app or on Slack); P3 is digest-only and not counted",
    "per_week": "pushed alerts / weeks from the window start to the newest development (at least one week)",
    "per_active_day": "pushed alerts / days on which at least one new development was first seen",
    "per_push_day": "pushed alerts / days with at least one push (burst size, never below 1)",
    "precision": "share of labelled pushed alerts whose development label is >= 1 (worth knowing) or == 2 "
                 "(material); label_0 counts pushes labelled noise",
    "label2_recall": "label-2 developments of the audience (in the window) that got a P1/P2 alert",
    "holdout": "developments first seen on/after the scorer's time-split date (scoring_metrics.json "
               "split.test_from)",
    "lens_dismissed": "pushed alerts whose own audience lens says no_action (investment) or hold (ai_team)",
}
LENS_DISMISS = {"investment": "no_action", "ai_team": "hold"}  # the lens verdict for "nothing to do"
LABEL_NOTE = ("One label per development: human annotators (majority) when present, else the latest LLM judge. "
              "LLM-judge precision is in-sample: `fli train` refits the deployed scorers on all labels after the "
              "time-split evaluation (hold-out window included), so treat it as optimistic until human labels exist.")


def _resolved_labels(s) -> dict[tuple[int, str], tuple[str, int]]:
    """(development_id, audience) -> (source 'human' | 'llm', label 0/1/2) under the scorer's policy."""
    from fli.analyze.labeling import training_labels

    return {(t.dev.id, aud): (t.source, 2 if t.y2 else t.y) for aud in AUDIENCES for t in training_labels(s, aud)}


def _holdout_starts(path: Path | None = None) -> dict[str, datetime]:
    """Audience -> first_seen_at where the scorer's time-split hold-out begins (written by `fli train`)."""
    try:
        spec = json.loads((path or EVAL_DIR / "scoring_metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, datetime] = {}
    for aud, rep in (spec.get("audiences") or {}).items():
        start = ((rep.get("split") or {}) if isinstance(rep, dict) else {}).get("test_from")
        try:
            if start:
                out[aud] = datetime.fromisoformat(str(start))
        except ValueError:
            log.warning("scoring_metrics.json: unreadable split.test_from %r for %s", start, aud)
    return out


def _precision(alerts: list[Alert], labels: dict[tuple[int, str], tuple[str, int]], aud: str) -> dict:
    vals = [labels[(a.development_id, aud)][1] for a in alerts if (a.development_id, aud) in labels]
    n = len(vals)
    return {"n": len(alerts), "labelled": n,
            "label_ge1": round(sum(v >= 1 for v in vals) / n, 3) if n else None,
            "label_eq2": round(sum(v == 2 for v in vals) / n, 3) if n else None,
            "label_0": sum(v == 0 for v in vals)}


def _pushed_metrics(alerts: list[Alert], first_seen: dict[int, datetime], labels: dict[tuple[int, str], tuple[str, int]],
                    aud: str, lenses: dict | None = None, start: datetime | None = None) -> dict:
    """Pushed (P1/P2) alerts of one audience on developments first seen at/after `start`."""
    window = {d: t for d, t in first_seen.items() if start is None or t >= start}
    pushed = [a for a in alerts if a.audience == aud and a.severity in PUSHED and a.development_id in window]
    lo = start or (min(window.values()) if window else None)
    hi = max(window.values()) if window else lo
    weeks = max((hi - lo).total_seconds() / 86400, 7.0) / 7 if lo and hi else 1.0
    active_days = {t.date() for t in window.values()}
    per_day = Counter(a.created_at.date() for a in pushed)
    label2 = {d for (d, au), (_, v) in labels.items() if au == aud and v == 2 and d in window}
    hit = label2 & {a.development_id for a in pushed}
    sources = Counter(labels[(a.development_id, aud)][0] for a in pushed if (a.development_id, aud) in labels)
    out = {
        "from": lo.isoformat(sep=" ", timespec="seconds") if lo else None,
        "to": hi.isoformat(sep=" ", timespec="seconds") if hi else None,
        "count": len(pushed),
        "by_severity": {sev: sum(a.severity == sev for a in pushed) for sev in PUSHED},
        "per_week": round(len(pushed) / weeks, 2),
        "per_active_day": round(len(pushed) / len(active_days), 2) if active_days else None,
        "per_push_day": round(len(pushed) / len(per_day), 2) if per_day else None,
        "max_per_day": max(per_day.values(), default=0),
        "precision": {"P1+P2": _precision(pushed, labels, aud),
                      "P1": _precision([a for a in pushed if a.severity == "P1"], labels, aud)},
        "label2_recall": {"label2_developments": len(label2), "pushed": len(hit),
                          "share": round(len(hit) / len(label2), 3) if label2 else None},
        "label_sources": dict(sources),
        "in_sample": bool(sources.get("llm")),
    }
    if lenses is not None:
        complete = [a for a in pushed if (lz := lenses.get((a.development_id, aud))) and lz.what_to_do
                    and (aud != "investment" or lz.tickers)]
        out["complete_actionable_share"] = round(len(complete) / len(pushed), 3) if pushed else None
        dismissed = [a for a in pushed if (lz := lenses.get((a.development_id, aud)))
                     and lz.action == LENS_DISMISS.get(aud)]
        out["lens_dismissed"] = {"n": len(dismissed), "share": round(len(dismissed) / len(pushed), 3) if pushed else None,
                                 "P1": sum(a.severity == "P1" for a in dismissed)}
    return out


def eval_alerts() -> dict:
    """Alert noise as a person sees it: pushed alerts (P1/P2) only, volume per week and per active day,
    precision and label-2 recall against labels, overall and on the scorer's hold-out window."""
    holdout = _holdout_starts()
    with session_scope() as s:
        alerts = list(s.scalars(select(Alert).order_by(Alert.id)))
        lenses = {(lz.development_id, lz.audience): lz for lz in s.scalars(select(Lens))}
        labels = _resolved_labels(s)
        first_seen = dict(s.execute(select(Development.id, Development.first_seen_at)
                                    .where(Development.status.not_in(DEAD_STATUSES))).all())
        out: dict = {"alerts": len(alerts), "definitions": ALERT_DEFINITIONS, "label_note": LABEL_NOTE}
        for aud in AUDIENCES:
            mine = [a for a in alerts if a.audience == aud]
            start = holdout.get(aud)
            out[aud] = {
                "count": len(mine),  # every alert row, digest-only P3 included
                "by_severity": dict(Counter(a.severity for a in mine)),
                "duplicates": len(mine) - len({a.development_id for a in mine}),
                "pushed": _pushed_metrics(alerts, first_seen, labels, aud, lenses),
                "holdout": (_pushed_metrics(alerts, first_seen, labels, aud, lenses, start) if start
                            else {"status": "no scorer split yet (run fli train)"}),
            }
    return out


def eval_scoring() -> dict:
    path = EVAL_DIR / "scoring_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "not trained yet (run fli train)"}


def eval_er() -> dict:
    pairs = EVAL_DIR / "er_pairs.yaml"
    if not pairs.exists():
        return {"status": "no labelled pairs"}
    from fli.register.entity_resolution import evaluate_er

    return evaluate_er(pairs)


def eval_costs() -> dict:
    from fli.db.models import ApiUsage

    with session_scope() as s:
        rows = s.execute(
            select(LLMCall.purpose, LLMCall.model, LLMCall.status, func.count(), func.sum(LLMCall.input_tokens),
                   func.sum(LLMCall.output_tokens), func.sum(LLMCall.cost_usd))
            .group_by(LLMCall.purpose, LLMCall.model, LLMCall.status)
        ).all()
        n_docs = s.scalar(select(func.count()).select_from(Document).where(Document.triage.is_not(None))) or 0
        n_extracted = s.scalar(select(func.count()).select_from(Document).where(Document.status == "extracted")) or 0
        x_usd = s.scalar(select(func.coalesce(func.sum(ApiUsage.cost_usd), 0.0)).where(ApiUsage.provider == "x")) or 0.0
    per: dict[str, dict] = defaultdict(lambda: {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "models": set()})
    for purpose, model, status, n, i, o, usd in rows:
        p = per[purpose]
        p["calls"] += n
        p["cached"] += n if status == "cached" else 0
        p["input_tokens"] += i or 0
        p["output_tokens"] += o or 0
        p["usd"] += usd or 0.0
        p["models"].add(model)
    table = {k: {**v, "usd": round(v["usd"], 4), "models": sorted(v["models"])} for k, v in per.items()}
    total = sum(v["usd"] for v in per.values())
    return {
        "by_workflow": table,
        "llm_total_usd": round(total, 4),
        "x_api_usd": round(x_usd, 4),
        "usd_per_triaged_document": round(table.get("triage", {}).get("usd", 0) / n_docs, 5) if n_docs else None,
        "usd_per_extracted_document": round(table.get("extract", {}).get("usd", 0) / n_extracted, 5) if n_extracted else None,
    }


# ------------------------------------------------------------------------------------ helpers (new suites)
RELEVANT_STATUSES = ("relevant", "extracted")  # documents that passed triage


def _window_start() -> datetime:
    """Start of the evaluation window: settings.backfill_start as naive UTC midnight."""
    d = get_settings().backfill_start
    return datetime(d.year, d.month, d.day)


def _dicts(value: object) -> list[dict]:
    """The dict entries of a stored JSON list (stored JSON is read defensively)."""
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _share(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _pushed_severity(s) -> dict[tuple[int, str], str]:
    """(development_id, audience) -> the highest pushed severity (P1/P2) among its alerts."""
    out: dict[tuple[int, str], str] = {}
    for dev_id, aud, sev in s.execute(select(Alert.development_id, Alert.audience, Alert.severity)
                                      .where(Alert.severity.in_(PUSHED))):
        if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(out.get((dev_id, aud), ""), 0):
            out[(dev_id, aud)] = sev
    return out


# ------------------------------------------------------------------------------------ lens faithfulness
LENS_DEFINITIONS = {
    "checked": "lenses on live developments whose Lens.checks carry a 'grounded' verdict (written by the lens stage, "
               "refreshed by `fli lens-recheck`)",
    "grounded": "no figure in the headline, so-what or a ticker rationale that the lens input does not contain, and no "
                "official-wording flag (fli.deliver.lens.faithfulness_checks); a code check of figures and provenance "
                "wording, not a semantic entailment judge",
    "unsupported_numbers": "figures (numbers, percentages, multiples, amounts) not written in the lens input, by field; "
                           "figures in what_to_do are reported but do not make a lens ungrounded",
    "headlines_replaced": "lenses whose model headline stated an unsupported figure or official wording and was replaced "
                          "by the development title (the original is kept in the checks)",
    "official_wording": "claims worded as official without an official source, or whose figures are not in the official "
                        "evidence",
    "tickers_dropped": "investment lenses that abstained (no_action) and had their tickers cleared",
    "pushed": "the same counts for the lenses behind a P1/P2 alert of their own audience (the text a person is sent)",
}


def _lens_checks(lens: Lens) -> dict | None:
    """Lens.checks when the faithfulness checks ran (they always record 'grounded'), else None."""
    return lens.checks if isinstance(lens.checks, dict) and "grounded" in lens.checks else None


def _lens_block(lenses: list[Lens]) -> dict:
    checked = [c for lz in lenses if (c := _lens_checks(lz)) is not None]
    grounded = sum(bool(c["grounded"]) for c in checked)
    numbers = [u for c in checked for u in _dicts(c.get("unsupported_numbers"))]
    official = [o for c in checked for o in _dicts(c.get("official_wording"))]
    return {
        "lenses": len(lenses),
        "checked": len(checked),
        "grounded": grounded,
        "grounded_share": _share(grounded, len(checked)),
        "with_unsupported_numbers": sum(1 for c in checked if _dicts(c.get("unsupported_numbers"))),
        "unsupported_numbers": len(numbers),
        "unsupported_by_field": dict(Counter(str(u.get("field")) for u in numbers).most_common()),
        "headlines_replaced": sum(1 for c in checked if c.get("headline_replaced")),
        "with_official_wording_flags": sum(1 for c in checked if _dicts(c.get("official_wording"))),
        "official_wording_by_issue": dict(Counter(str(o.get("issue")) for o in official).most_common()),
        "tickers_dropped": sum(1 for c in checked if c.get("tickers_dropped")),
    }


def _lens_example(lens: Lens, checks: dict, alert: str | None) -> dict:
    replaced = checks.get("headline_replaced")
    return {"development_id": lens.development_id, "audience": lens.audience, "alert": alert,
            "headline": (lens.headline or "")[:200],
            "replaced_headline": str(replaced.get("original") or "")[:200] if isinstance(replaced, dict) else None,
            "unsupported_numbers": [f"{u.get('field')}: {u.get('text')}" for u in _dicts(checks.get("unsupported_numbers"))],
            "official_wording": [str(o.get("issue")) for o in _dicts(checks.get("official_wording"))]}


def eval_lens(max_examples: int = 12) -> dict:
    """Lens faithfulness from the deterministic checks stored on every lens (no LLM call), per audience,
    overall and for the lenses behind pushed alerts, with examples of flagged lenses (pushed first)."""
    with session_scope() as s:
        live = set(s.scalars(select(Development.id).where(Development.status.not_in(DEAD_STATUSES))))
        lenses = [lz for lz in s.scalars(select(Lens).order_by(Lens.id)) if lz.development_id in live]
        pushed = _pushed_severity(s)
        out: dict = {"definitions": LENS_DEFINITIONS}
        for aud in AUDIENCES:
            mine = [lz for lz in lenses if lz.audience == aud]
            out[aud] = _lens_block(mine) | {
                "pushed": _lens_block([lz for lz in mine if (lz.development_id, aud) in pushed])}
        out["all"] = _lens_block(lenses)
        out["checks_versions"] = dict(Counter(str(c.get("version")) for lz in lenses if (c := _lens_checks(lz))))
        flagged = [(lz, c) for lz in lenses if (c := _lens_checks(lz)) is not None
                   and (not c["grounded"] or c.get("headline_replaced"))]
        flagged.sort(key=lambda x: (-_SEVERITY_RANK.get(pushed.get((x[0].development_id, x[0].audience), ""), 0),
                                    x[0].development_id, x[0].audience))
        out["flagged"] = len(flagged)
        out["examples"] = [_lens_example(lz, c, pushed.get((lz.development_id, lz.audience)))
                           for lz, c in flagged[:max_examples]]
    return out


# ------------------------------------------------------------------------------------ coverage by BIT holding
NAMED_LINK = "named in development text"  # rationale of a link from a company/brand mention (fli.analyze.exposure)
_ALERT_TICKERS = re.compile(r"^- \*\*Tickers:\*\*(?P<line>.*)$", re.MULTILINE)
_LISTED_TICKER = re.compile(r"(?:^|;\s)(?P<sym>[A-Z0-9][A-Z0-9.]*) \((?P<kind>"
                            + "|".join(map(re.escape, [*DIRECTIONS, "lab-level link"])) + r")(?:, BIT [\d.]+%)?\)")
HOLDINGS_DEFINITIONS = {
    "window": "documents published (else fetched) and developments first seen on/after settings.backfill_start",
    "documents": "collected documents in the window, duplicates excluded, whose title or text names the company or a "
                 "brand it owns (the exposure rules' whole-word, case-sensitive names: fli.analyze.exposure."
                 "mention_names); documents_relevant passed triage",
    "developments_event_specific": "live developments with an exposure link to the holding that is not lab-level "
                                   "(method != 'lab'); named = the development's own text names the company (the rest "
                                   "are theme-keyword links); at_policy_floor = confidence >= the materiality policy's "
                                   "exposure_min_confidence, i.e. what the alert rules count as exposure",
    "developments_lab_only": "developments linked to the holding only through their lab's curated relations (context, "
                             "not exposure)",
    "pushed_alerts": "investment P1/P2 alerts whose Tickers line lists the holding (investment is the only audience whose "
                     "alerts list tickers); pushed_lab_level = listed only as a lab-level link",
    "coverage": "direct: at least one document names it; indirect: none does, but theme-keyword links exist; none: neither",
}


def listed_tickers(body: str | None) -> dict[str, str]:
    """Symbols an alert body lists on its '- **Tickers:**' line, as fli.deliver.alerts renders it ('SYM
    (direction[, BIT w%])[: why]' entries joined by '; ') -> 'lab' when listed as a lab-level link, else 'event'."""
    m = _ALERT_TICKERS.search(body or "")
    out: dict[str, str] = {}
    for x in _LISTED_TICKER.finditer(m.group("line").strip() if m else ""):
        if out.get(x.group("sym")) != "event":
            out[x.group("sym")] = "lab" if x.group("kind") == "lab-level link" else "event"
    return out


def bit_holdings(path: Path | None = None) -> list[tuple[str, float, str]]:
    """(symbol, weight %, company) of each BIT holding in config/exposure.yaml, largest weight first."""
    cfg = yaml.safe_load((path or CONFIG_DIR / "exposure.yaml").read_text(encoding="utf-8")) or {}
    company = {str(t["symbol"]): str(t.get("company") or t["symbol"])
               for t in cfg.get("tickers") or [] if isinstance(t, dict) and t.get("symbol")}
    rows = [(str(h["symbol"]), float(h.get("weight") or 0.0), company.get(str(h["symbol"]), str(h["symbol"])))
            for h in cfg.get("bit_holdings") or [] if isinstance(h, dict) and h.get("symbol")]
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def _mention_pattern(symbol: str, company: str, weight: float) -> tuple[list[str], re.Pattern | None]:
    """The names the exposure rules accept for a company (its name and brands; whole word, case sensitive)
    and one pattern matching any of them."""
    from fli.analyze.exposure import TickerInfo, mention_names

    names = list(mention_names(TickerInfo(symbol, company, (), weight)))
    if not names:
        return names, None
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return names, re.compile(rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])")


def eval_holdings(path: Path | None = None) -> dict:
    """Coverage by BIT holding over the window: does the corpus see the company at all (documents that name
    it), did developments get event-specific exposure to it, and did a pushed alert list it. A holding no
    document names has no source coverage: the system cannot surface its news, however it ranks."""
    from fli.deliver.materiality import LAB_LEVEL_METHODS, load_policy

    holdings = bit_holdings(path)
    floor = float(load_policy().exposure_min_confidence)
    start = _window_start()
    patterns = {sym: _mention_pattern(sym, company, weight) for sym, weight, company in holdings}
    doc_classes: dict[str, Counter] = {sym: Counter() for sym in patterns}
    relevant: Counter = Counter()
    links: dict[str, dict[str, set[int]]] = {sym: defaultdict(set) for sym in patterns}
    listed: dict[str, Counter] = {sym: Counter() for sym in patterns}
    n_docs = 0
    with session_scope() as s:
        when = func.coalesce(Document.published_at, Document.fetched_at)
        for title, text, cls, status in s.execute(
            select(Document.title, Document.content_text, Document.source_class, Document.status)
            .where(when >= start, Document.status != "duplicate")
        ):
            n_docs += 1
            blob = f"{title or ''}\n{text or ''}"
            for sym, (_, pattern) in patterns.items():
                if pattern is not None and pattern.search(blob):
                    doc_classes[sym][cls or "unknown"] += 1
                    relevant[sym] += status in RELEVANT_STATUSES
        live = set(s.scalars(select(Development.id).where(Development.status.not_in(DEAD_STATUSES),
                                                         Development.first_seen_at >= start)))
        for dev_id, sym, method, conf, why in s.execute(
            select(ExposureLink.development_id, ExposureLink.ticker, ExposureLink.method, ExposureLink.confidence,
                   ExposureLink.rationale).where(ExposureLink.ticker.in_(sorted(patterns)))
        ):
            if dev_id not in live:
                continue
            if (method or "rule") in LAB_LEVEL_METHODS:
                links[sym]["lab"].add(dev_id)
                continue
            links[sym]["event"].add(dev_id)
            if NAMED_LINK in (why or ""):
                links[sym]["named"].add(dev_id)
            if float(conf or 0.0) >= floor:
                links[sym]["floor"].add(dev_id)
        for dev_id, sev, body in s.execute(select(Alert.development_id, Alert.severity, Alert.body_md)
                                           .where(Alert.audience == "investment", Alert.severity.in_(PUSHED))):
            if dev_id not in live:
                continue
            for sym, kind in listed_tickers(body).items():
                if sym in listed:
                    listed[sym]["pushed"] += 1
                    listed[sym]["P1"] += int(sev == "P1")
                    listed[sym]["lab_level"] += int(kind == "lab")
    rows = []
    for sym, weight, company in holdings:
        docs, event = sum(doc_classes[sym].values()), links[sym]["event"]
        rows.append({
            "symbol": sym, "company": company, "weight": weight, "names": patterns[sym][0],
            "documents": docs, "documents_relevant": relevant[sym],
            "documents_by_class": dict(doc_classes[sym].most_common()),
            "developments_event_specific": len(event),
            "developments_named": len(links[sym]["named"]),
            "developments_at_policy_floor": len(links[sym]["floor"]),
            "developments_lab_only": len(links[sym]["lab"] - event),
            "pushed_alerts": listed[sym]["pushed"], "pushed_p1": listed[sym]["P1"],
            "pushed_lab_level": listed[sym]["lab_level"],
            "coverage": "direct" if docs else ("indirect" if event else "none"),
        })
    gap = [r for r in rows if not r["documents"]]
    return {
        "definitions": HOLDINGS_DEFINITIONS,
        "window": {"from": start.date().isoformat(), "documents": n_docs},
        "policy_exposure_min_confidence": floor,
        "holdings": rows,
        "weight_total": round(sum(r["weight"] for r in rows), 2),
        "no_direct_coverage": [r["symbol"] for r in gap],
        "weight_no_direct_coverage": round(sum(r["weight"] for r in gap), 2),
        "never_pushed": [r["symbol"] for r in rows if not r["pushed_alerts"]],
        # pushes that list a holding no document names rest on theme-keyword or lab-level inference alone
        "pushed_without_naming_document": {r["symbol"]: r["pushed_alerts"] for r in gap if r["pushed_alerts"]},
    }


# ------------------------------------------------------------------------------------ digest quality
DIGEST_DEFINITIONS = {
    "writer": "model: the digest model wrote the sections; fallback: the deterministic renderer (no usable model "
              "output; provisional, regenerated on the next run); empty: no item reached the bar that week",
    "cited": "selected items the published report cites anywhere (written sections, 'Also this week', holdings check)",
    "written": "selected items cited by the written sections; the others were appended under 'Also this week'",
    "model_written": "the same over model-written reports; the digest rejects model output citing fewer than "
                     "min_coverage of the selected items",
    "flags_missing": "items stored without the cited / also_this_week flags (legacy rows), left out of the shares",
    "missing_weeks": "weeks from the backfill start to the last full week without a report",
    "preview_reports": "reports generated before their week ended (regenerated on the next run)",
}


def _report_row(rep: Report, fallback: str) -> dict:
    items = _dicts(rep.items)
    flagged = [i for i in items if "cited" in i and "also_this_week" in i]
    writer = "empty" if not items else "fallback" if rep.model == fallback else "model" if rep.model else "unknown"
    return {"period_start": rep.period_start.date().isoformat(), "writer": writer, "model": rep.model,
            "items": len(items), "flags_missing": len(items) - len(flagged),
            "cited": sum(bool(i["cited"]) for i in flagged),
            "written": sum(bool(i["cited"]) and not i["also_this_week"] for i in flagged),
            "also_this_week": sum(bool(i["also_this_week"]) for i in flagged),
            "preview": bool(rep.created_at and rep.period_end and rep.created_at < rep.period_end)}


def _digest_coverage(rows: list[dict]) -> dict:
    flagged = sum(r["items"] - r["flags_missing"] for r in rows)
    cited, written, also = (sum(r[k] for r in rows) for k in ("cited", "written", "also_this_week"))
    return {"reports": len(rows), "items": sum(r["items"] for r in rows),
            "flags_missing": sum(r["flags_missing"] for r in rows),
            "cited": cited, "cited_share": _share(cited, flagged),
            "written": written, "written_share": _share(written, flagged),
            "also_this_week": also, "reports_with_also_this_week": sum(1 for r in rows if r["also_this_week"])}


def eval_digests(now: datetime | None = None) -> dict:
    """Weekly digest quality per audience from the stored reports (Report.model and the items' cited /
    also_this_week flags): who wrote them, how much of the selection the written sections cover, gaps."""
    from fli.deliver.digest import FALLBACK, MIN_COVERAGE

    first, last = week_start(_window_start()), week_start(now or utcnow()) - timedelta(days=7)
    expected = [(first + timedelta(days=7 * k)).date().isoformat() for k in range((last - first).days // 7 + 1)]
    out: dict = {"definitions": DIGEST_DEFINITIONS, "min_coverage": MIN_COVERAGE}
    with session_scope() as s:
        for aud in AUDIENCES:
            rows = [_report_row(r, FALLBACK) for r in s.scalars(select(Report).where(Report.audience == aud)
                                                               .order_by(Report.period_start))]
            writers = Counter(r["writer"] for r in rows)
            model_rows = [r for r in rows if r["writer"] == "model"]
            shares = [r["written"] / (r["items"] - r["flags_missing"]) for r in model_rows if r["items"] > r["flags_missing"]]
            seen = {r["period_start"] for r in rows}
            out[aud] = {
                **_digest_coverage(rows),
                "by_writer": {k: writers[k] for k in ("model", "fallback", "empty", "unknown")},
                "model_share": _share(writers["model"], writers["model"] + writers["fallback"]),
                "models": dict(Counter(str(r["model"]) for r in model_rows).most_common()),
                "fallback_weeks": [r["period_start"] for r in rows if r["writer"] == "fallback"],
                "preview_reports": sum(r["preview"] for r in rows),
                "missing_weeks": [w for w in expected if w not in seen],
                "model_written": _digest_coverage(model_rows) | {
                    "min_report_written_share": round(min(shares), 3) if shares else None,
                    "reports_below_min_coverage": sum(x < MIN_COVERAGE for x in shares)},
                "weeks": rows,
            }
    return out


# ------------------------------------------------------------------------------------ source coverage
SOURCE_DEFINITIONS = {
    "documents": "every collected document, duplicates included, by the document's source class, its source's kind and "
                 "its lab (the document's, else its source's); relevant = passed triage",
    "silent_sources": "registered sources that never produced a document; reason: disabled, never_run (no run "
                      "recorded), failing (the last run failed), no_items (ran without collecting anything)",
    "active_weeks": "weeks of the window (backfill start to the newest document) with at least one document",
}
UNATTRIBUTED = "unattributed"


def _silent_reason(src: Source) -> str:
    if not src.enabled:
        return "disabled"
    if src.last_run_at is None:
        return "never_run"
    if (src.consecutive_failures or 0) > 0 or (src.last_error and src.last_success_at is None):
        return "failing"
    return "no_items"


def _coverage_table(sources: Iterable[str], docs: Counter, relevant: Counter, with_docs: Counter) -> dict[str, dict]:
    """key -> {sources, with_documents, documents, relevant} for every key with a source or a document."""
    n_sources = Counter(sources)
    keys = sorted(set(n_sources) | set(docs), key=lambda k: (-docs[k], k))
    return {k: {"sources": n_sources[k], "with_documents": with_docs[k], "documents": docs[k], "relevant": relevant[k]}
            for k in keys}


def eval_sources() -> dict:
    """Source coverage: documents per source class, source kind and lab, tracked labs without documents,
    and the registered sources that never produced a document, with the reason."""
    start = _window_start()
    with session_scope() as s:
        sources = {src.id: src for src in s.scalars(select(Source).order_by(Source.id))}
        labs = [(lab.id, lab.name, lab.tier) for lab in s.scalars(select(Lab).order_by(Lab.tier, Lab.id))]
        when = func.coalesce(Document.published_at, Document.fetched_at)
        docs = s.execute(select(Document.source_id, Document.source_class, Document.lab_id, Document.status,
                                when.label("at"))).all()
        per_source: Counter = Counter()
        tallies = {k: (Counter(), Counter()) for k in ("class", "kind", "lab")}  # documents, relevant
        last: dict[str, datetime] = {}
        weeks: dict[str, set[datetime]] = defaultdict(set)
        newest, before, duplicates = None, 0, 0
        for source_id, cls, doc_lab, status, at in docs:
            src = sources.get(source_id)
            keys = {"class": cls or "unknown", "kind": src.kind if src else "unknown",
                    "lab": doc_lab or (src.lab_id if src else None) or UNATTRIBUTED}
            per_source[source_id] += 1
            for k, key in keys.items():
                tallies[k][0][key] += 1
                tallies[k][1][key] += status in RELEVANT_STATUSES
            duplicates += status == "duplicate"
            if at is None:
                continue
            newest = at if newest is None or at > newest else newest
            last[keys["lab"]] = max(last.get(keys["lab"], at), at)
            if at >= start:
                weeks[keys["lab"]].add(week_start(at))
            else:
                before += 1
        silent = [src for src in sources.values() if not per_source[src.id]]
        producing = [src for src in sources.values() if per_source[src.id]]
        silent_rows = [{"id": src.id, "kind": src.kind, "source_class": src.source_class, "lab_id": src.lab_id,
                        "enabled": bool(src.enabled), "reason": _silent_reason(src),
                        "last_error": (src.last_error or "")[:160] or None} for src in silent]
        lab_sources = Counter(src.lab_id or UNATTRIBUTED for src in sources.values())
        lab_silent = Counter(src.lab_id or UNATTRIBUTED for src in silent)
        n_weeks = (week_start(newest) - week_start(start)).days // 7 + 1 if newest and newest >= start else 0
        lab_docs, lab_relevant = tallies["lab"]
        by_lab = {lab_id: {"name": name, "tier": tier} for lab_id, name, tier in labs}
        if lab_docs[UNATTRIBUTED] or lab_sources[UNATTRIBUTED]:
            by_lab[UNATTRIBUTED] = {"name": UNATTRIBUTED, "tier": None}
        for lab_id, row in by_lab.items():
            row.update({"sources": lab_sources[lab_id], "silent_sources": lab_silent[lab_id],
                        "documents": lab_docs[lab_id], "relevant": lab_relevant[lab_id],
                        "active_weeks": len(weeks[lab_id]),
                        "last_document": last[lab_id].isoformat(sep=" ", timespec="seconds") if lab_id in last else None})
        return {
            "definitions": SOURCE_DEFINITIONS,
            "window": {"from": start.date().isoformat(),
                       "to": newest.isoformat(sep=" ", timespec="seconds") if newest else None, "weeks": n_weeks},
            "documents": len(docs), "before_window": before, "duplicates": duplicates,
            "sources": len(sources), "enabled": sum(1 for src in sources.values() if src.enabled),
            "with_documents": len(producing), "silent": len(silent),
            "silent_by_reason": dict(Counter(r["reason"] for r in silent_rows).most_common()),
            "silent_by_kind": dict(Counter(r["kind"] for r in silent_rows).most_common()),
            "by_source_class": _coverage_table((src.source_class for src in sources.values()), *tallies["class"],
                                               Counter(src.source_class for src in producing)),
            "by_kind": _coverage_table((src.kind for src in sources.values()), *tallies["kind"],
                                       Counter(src.kind for src in producing)),
            "by_lab": by_lab,
            "labs_without_documents": [lab_id for lab_id, *_ in labs if not lab_docs[lab_id]],
            "silent_sources": silent_rows,
        }


SUITES = {
    "known_events": eval_known_events,
    "extraction": eval_extraction,
    "noise": eval_noise,
    "alerts": eval_alerts,
    "scoring": eval_scoring,
    "er": eval_er,
    "costs": eval_costs,
    "lens": eval_lens,
    "holdings": eval_holdings,
    "digests": eval_digests,
    "sources": eval_sources,
}
WRITES_FILES = {"noise"}  # suites that write a file next to the report (they take out_dir)


def _previous_report(path: Path) -> dict:
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return old if isinstance(old, dict) else {}


def run_eval(suite: str = "all", out_dir: Path | None = None) -> dict:
    """Run suites and write report.json/report.md to `out_dir` (default $FLI_EVAL_OUT_DIR, else data/eval).
    A subset updates those suites in the existing report.json; the others are kept with their own timestamps."""
    out_dir = Path(out_dir or os.environ.get(OUT_DIR_ENV) or EVAL_DIR)
    names = list(SUITES) if suite == "all" else [n.strip() for n in suite.split(",") if n.strip()]
    now = utcnow().isoformat(timespec="seconds")
    report: dict = {} if suite == "all" else _previous_report(out_dir / "report.json")
    # suites kept from an older run keep the time they were generated
    stamps = {k: report.get("generated_at") for k in SUITES if k in report} | dict(report.get("suite_generated_at") or {})
    report["generated_at"] = now
    for name in names:
        try:
            report[name] = SUITES[name](out_dir=out_dir) if name in WRITES_FILES else SUITES[name]()
        except Exception as exc:  # one broken suite must not hide the others
            log.exception("eval suite %s failed", name)
            report[name] = {"error": f"{exc.__class__.__name__}: {exc}"}
        stamps[name] = now
    report["suite_generated_at"] = stamps
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return {k: (v if k in {"generated_at", "suite_generated_at"} else _safe_headline(k, v)) for k, v in report.items()}


def _safe_headline(name: str, res):
    """_headline, falling back to the scalar fields when a stored result has an unexpected layout."""
    try:
        return _headline(name, res)
    except (KeyError, TypeError, ValueError, AttributeError):
        log.warning("eval report: unexpected layout for %s; showing its scalar fields", name)
        return {k: v for k, v in res.items() if not isinstance(v, list | dict)} if isinstance(res, dict) else res


def _headline(name: str, res: dict) -> dict:
    """Compact summary for the CLI; full detail stays in report.json."""
    if not isinstance(res, dict):
        return res
    if name in _HEADLINES and "error" not in res:
        return _HEADLINES[name](res)
    if name == "alerts" and "error" not in res:
        return {"alerts": res.get("alerts"),
                **{aud: _alert_headline(res[aud]) for aud in AUDIENCES if isinstance(res.get(aud), dict)}}
    if name == "scoring" and isinstance(res.get("audiences"), dict):
        return {"generated_at": res.get("generated_at"), "target": res.get("target"),
                **{aud: {k: (r.get("model") or {}).get(k) for k in ("roc_auc", "roc_auc_material", "material_n",
                                                                   "precision_at_10", "precision_at_20")}
                       | {"label_source": r.get("label_source"), "deployed": r.get("deployed")}
                   for aud, r in res["audiences"].items() if isinstance(r, dict)}}
    keep = {"by_severity", "noise_reasons", "alert_predates_evidence"}
    return {k: v for k, v in res.items() if not isinstance(v, list | dict) or k in keep}


def _alert_headline(block: dict) -> dict:
    out: dict = {"count": block.get("count"), "by_severity": block.get("by_severity")}
    for label, m in (("pushed", block.get("pushed")), ("holdout", block.get("holdout"))):
        if isinstance(m, dict) and "count" in m:
            prec, p1, rec = m["precision"]["P1+P2"], m["precision"]["P1"], m["label2_recall"]
            out[label] = {"count": m["count"], "per_week": m["per_week"], "per_active_day": m["per_active_day"],
                          "precision_label_ge1": prec["label_ge1"], "precision_label_eq2": prec["label_eq2"],
                          "p1_precision_label_eq2": p1["label_eq2"], "label2_recall": rec["share"],
                          "lens_dismissed_share": (m.get("lens_dismissed") or {}).get("share"),
                          "in_sample": m["in_sample"]}
    return out


def _lens_headline(res: dict) -> dict:
    out: dict = {}
    for key in (*AUDIENCES, "all"):
        b = res.get(key)
        if not isinstance(b, dict):
            continue
        out[key] = {k: b.get(k) for k in ("lenses", "grounded_share", "with_unsupported_numbers", "unsupported_numbers",
                                          "headlines_replaced", "with_official_wording_flags")}
        if isinstance(b.get("pushed"), dict):
            out[key] |= {"pushed_lenses": b["pushed"].get("checked"), "pushed_grounded_share": b["pushed"].get("grounded_share")}
    return out


def _holdings_headline(res: dict) -> dict:
    return {k: res.get(k) for k in ("no_direct_coverage", "weight_no_direct_coverage", "never_pushed")} | {
        "by_holding": {r["symbol"]: {"documents": r["documents"], "event_specific": r["developments_event_specific"],
                                     "named": r["developments_named"], "pushed": r["pushed_alerts"]}
                       for r in res.get("holdings") or [] if isinstance(r, dict)}}


def _digests_headline(res: dict) -> dict:
    out: dict = {}
    for aud in AUDIENCES:
        b = res.get(aud)
        if isinstance(b, dict):
            out[aud] = {"reports": b.get("reports"), **(b.get("by_writer") or {}), "model_share": b.get("model_share"),
                        "cited_share": b.get("cited_share"),
                        "model_written_share": (b.get("model_written") or {}).get("written_share"),
                        "also_this_week": b.get("also_this_week"), "missing_weeks": len(b.get("missing_weeks") or [])}
    return out


def _sources_headline(res: dict) -> dict:
    return {k: res.get(k) for k in ("documents", "sources", "with_documents", "silent", "silent_by_reason",
                                    "labs_without_documents")} | {
        "documents_by_class": {c: b.get("documents") for c, b in (res.get("by_source_class") or {}).items()}}


_HEADLINES: dict[str, Callable[[dict], dict]] = {
    "lens": _lens_headline, "holdings": _holdings_headline, "digests": _digests_headline, "sources": _sources_headline}


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _alerts_markdown(res: dict) -> list[str]:
    lines = ["## Alerts\n",
             f"{res.get('alerts', 0)} alert rows. Only pushed alerts (P1/P2, they reach a person) are measured below; "
             f"digest-only P3 rows are not. Precision and recall use labels: {LABEL_NOTE}\n",
             "| audience | window | pushed (P1/P2) | per week | per active day | max/day | labelled (source) "
             "| P(label>=1) | P(label=2) | P1: P(label=2) (labelled) | label-0 pushes | label-2 recall "
             "| own lens says no_action / hold |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for aud in AUDIENCES:
        block = res.get(aud)
        if not isinstance(block, dict):
            continue
        for label, m in (("all", block.get("pushed")), ("hold-out", block.get("holdout"))):
            if not isinstance(m, dict) or "count" not in m:
                continue
            prec, p1, rec = m["precision"]["P1+P2"], m["precision"]["P1"], m["label2_recall"]
            src = ", ".join(f"{k}{' in-sample' if k == 'llm' else ''}" for k in m["label_sources"]) or "none"
            dismissed = m.get("lens_dismissed") or {}
            lines.append(
                f"| {aud} | {label} from {(m['from'] or '')[:10]} | {m['count']} ({m['by_severity']['P1']}/"
                f"{m['by_severity']['P2']}) | {_fmt(m['per_week'], 2)} | {_fmt(m['per_active_day'], 2)} | "
                f"{m['max_per_day']} | {prec['labelled']} ({src}) | {_fmt(prec['label_ge1'])} | {_fmt(prec['label_eq2'])} | "
                f"{_fmt(p1['label_eq2'])} ({p1['labelled']}) | {prec['label_0']} | "
                f"{rec['pushed']}/{rec['label2_developments']} | {dismissed.get('n', '')} ({dismissed.get('P1', '')} P1) |")
    lines.append("")
    return lines


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _one_line(text: str | None, n: int) -> str:
    text = " ".join(str(text or "").split()).replace("|", "/")
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _lens_markdown(res: dict) -> list[str]:
    lines = ["## Lens faithfulness\n",
             "Deterministic checks stored on every lens (`Lens.checks`, refreshed by `fli lens-recheck`): every figure in "
             "the headline, so-what, what-to-do and ticker rationales must be written in the lens input, and wording "
             "that presents facts as official needs an official source. Grounded = no unsupported figure in the "
             "headline, so-what or a ticker rationale and no official-wording flag (figures in what-to-do are reported, "
             "not gating). A code check of figures and provenance wording, not a semantic entailment judge.\n",
             "| audience | lenses (checked) | grounded | lenses with unsupported figures | unsupported figures by field "
             "| headlines replaced | official-wording flags | tickers dropped (no_action) | behind P1/P2 alerts: grounded |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in (*AUDIENCES, "all"):
        b = res.get(key)
        if not isinstance(b, dict):
            continue
        fields = ", ".join(f"{k} {v}" for k, v in (b.get("unsupported_by_field") or {}).items()) or "none"
        issues = ", ".join(f"{k} {v}" for k, v in (b.get("official_wording_by_issue") or {}).items())
        p = b.get("pushed")
        pushed = f"{p['grounded']}/{p['checked']} ({_pct(p['grounded_share'])})" if isinstance(p, dict) else ""
        lines.append(f"| {key} | {b['lenses']} ({b['checked']}) | {b['grounded']} ({_pct(b['grounded_share'])}) | "
                     f"{b['with_unsupported_numbers']} | {fields} | {b['headlines_replaced']} | "
                     f"{b['with_official_wording_flags']}{f' ({issues})' if issues else ''} | {b['tickers_dropped']} | "
                     f"{pushed} |")
    examples = res.get("examples") or []
    if examples:
        lines += ["", f"Flagged lenses ({res.get('flagged', len(examples))}, ungrounded or headline replaced; pushed "
                      f"first, {len(examples)} shown):", ""]
        for e in examples:
            parts = []
            if e.get("replaced_headline"):
                parts.append(f"headline replaced (was \"{_one_line(e['replaced_headline'], 140)}\")")
            if e.get("unsupported_numbers"):
                parts.append("unsupported figures: " + ", ".join(e["unsupported_numbers"]))
            if e.get("official_wording"):
                parts.append("official wording: " + ", ".join(e["official_wording"]))
            alert = f", {e['alert']} alert" if e.get("alert") else ""
            lines.append(f"- development {e['development_id']} ({e['audience']}{alert}): " + "; ".join(parts))
    lines.append("")
    return lines


def _holdings_markdown(res: dict) -> list[str]:
    window, rows = res.get("window") or {}, [r for r in res.get("holdings") or [] if isinstance(r, dict)]
    lines = ["## Coverage by BIT holding\n",
             f"Window from {window.get('from')}: {window.get('documents')} documents, duplicates excluded. Documents "
             "naming it: title or text names the company or a brand it owns (the exposure rules' whole-word names). "
             "Event-specific developments: an exposure link that is not lab-level; named = the development's own text "
             "names the company, the rest are theme-keyword links; >= floor = confidence >= "
             f"{res.get('policy_exposure_min_confidence')}, what the alert rules count as exposure. Lab-level only: "
             "linked through the lab's curated relations alone (context, not exposure). Pushed alerts: investment "
             "P1/P2 alerts whose Tickers line lists the holding.\n",
             "| holding | BIT weight | documents naming it (triaged relevant) | event-specific developments "
             "(named / >= floor) | lab-level only | pushed alerts listing it (P1) | coverage |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        name = r["symbol"] + (f" ({r['company']})" if r.get("company") not in (None, r["symbol"]) else "")
        lines.append(f"| {name} | {r['weight']:.1f}% | {r['documents']} ({r['documents_relevant']}) | "
                     f"{r['developments_event_specific']} ({r['developments_named']} / {r['developments_at_policy_floor']}) | "
                     f"{r['developments_lab_only']} | {r['pushed_alerts']} ({r['pushed_p1']}) | {r['coverage']} |")
    gap = [r for r in rows if not r["documents"]]
    if gap:
        named = ", ".join(f"{r['symbol']} ({r['weight']:.1f}%)" for r in gap)
        lines.append(f"\nNo collected document names {named}: {res.get('weight_no_direct_coverage', 0):.1f} of the "
                     f"{res.get('weight_total', 0):.1f} percentage points of fund weight in the tracked holdings sit in "
                     "positions without direct source coverage, so silence on them is not evidence that nothing happened.")
    elif rows:
        lines.append("\nEvery tracked holding is named by at least one collected document.")
    if res.get("pushed_without_naming_document"):
        listed = ", ".join(f"{sym} {n}" for sym, n in res["pushed_without_naming_document"].items())
        lines.append(f"Listed by pushed alerts although no document names them (theme-keyword or lab-level links "
                     f"only): {listed}.")
    if res.get("never_pushed"):
        lines.append(f"Never listed by a pushed investment alert: {', '.join(res['never_pushed'])}.")
    lines.append("")
    return lines


def _digests_markdown(res: dict) -> list[str]:
    lines = ["## Digest quality\n",
             "Stored weekly reports. Cited: selected items the report cites anywhere; in written sections: cited by the "
             "model's (or the fallback's) sections rather than appended under 'Also this week'. The digest rejects "
             f"model output citing fewer than {_pct(res.get('min_coverage'))} of the selected items, retries once on a different model, then "
             "renders a deterministic fallback.\n",
             "| audience | reports | model / fallback / empty | items | cited | in written sections | 'Also this week' "
             "items (reports) | model-written: lowest share in written sections | fallback weeks | missing weeks |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    legacy = []
    for aud in AUDIENCES:
        b = res.get(aud)
        if not isinstance(b, dict):
            continue
        w, mw = b.get("by_writer") or {}, b.get("model_written") or {}
        lines.append(f"| {aud} | {b['reports']} | {w.get('model', 0)} / {w.get('fallback', 0)} / {w.get('empty', 0)} | "
                     f"{b['items']} | {b['cited']} ({_pct(b['cited_share'])}) | {b['written']} ({_pct(b['written_share'])}) | "
                     f"{b['also_this_week']} ({b['reports_with_also_this_week']}) | "
                     f"{_pct(mw.get('min_report_written_share'))} | {', '.join(b.get('fallback_weeks') or []) or 'none'} | "
                     f"{', '.join(b.get('missing_weeks') or []) or 'none'} |")
        if b.get("flags_missing") or w.get("unknown"):
            legacy.append(f"{aud}: {b.get('flags_missing', 0)} items without flags, {w.get('unknown', 0)} reports "
                          "without a model")
        if b.get("preview_reports"):
            legacy.append(f"{aud}: {b['preview_reports']} report(s) generated before their week ended (provisional)")
    if legacy:
        lines.append("\nNotes: " + "; ".join(legacy) + ".")
    lines.append("")
    return lines


def _sources_markdown(res: dict) -> list[str]:
    window = res.get("window") or {}
    reasons = ", ".join(f"{k} {v}" for k, v in (res.get("silent_by_reason") or {}).items()) or "none"
    lines = ["## Source coverage\n",
             f"{res['sources']} registered sources ({res['enabled']} enabled): {res['with_documents']} produced at least "
             f"one document, {res['silent']} never did ({reasons}). {res['documents']} documents ({res['duplicates']} "
             f"duplicates, {res['before_window']} dated before {window.get('from')}); newest "
             f"{(window.get('to') or '')[:10] or 'none'}.\n",
             "| source class | sources (with documents) | documents | triaged relevant |",
             "|---|---|---|---|"]
    for cls, b in (res.get("by_source_class") or {}).items():
        lines.append(f"| {cls} | {b['sources']} ({b['with_documents']}) | {b['documents']} | {b['relevant']} |")
    lines += ["", f"| lab | tier | sources (never produced) | documents | triaged relevant | weeks with documents "
                  f"(of {window.get('weeks')}) | newest document |",
              "|---|---|---|---|---|---|---|"]
    for lab_id, b in (res.get("by_lab") or {}).items():
        lines.append(f"| {_one_line(b.get('name') or lab_id, 60)} | {b.get('tier') or ''} | {b['sources']} "
                     f"({b['silent_sources']}) | {b['documents']} | {b['relevant']} | {b['active_weeks']} | "
                     f"{(b.get('last_document') or '')[:10]} |")
    no_docs = res.get("labs_without_documents") or []
    lines.append("\nTracked labs without a single document: " + (", ".join(no_docs) or "none") + ".")
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for x in res.get("silent_sources") or []:
        groups[(str(x.get("kind")), str(x.get("source_class")), str(x.get("reason")))].append(str(x.get("id")))
    if groups:
        lines += ["", "Sources that never produced a document:", "",
                  "| kind | class | reason | sources | examples |", "|---|---|---|---|---|"]
        for (kind, cls, reason), ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            more = f" (+{len(ids) - 4} more)" if len(ids) > 4 else ""
            lines.append(f"| {kind} | {cls} | {reason} | {len(ids)} | {', '.join(ids[:4])}{more} |")
    lines.append("")
    return lines


_SECTIONS: dict[str, Callable[[dict], list[str]]] = {
    "alerts": _alerts_markdown, "lens": _lens_markdown, "holdings": _holdings_markdown,
    "digests": _digests_markdown, "sources": _sources_markdown}


def _markdown(report: dict) -> str:
    lines = [f"# Evaluation report\n\nGenerated {report['generated_at']} by `fli eval`.\n"]
    ke = report.get("known_events")
    if isinstance(ke, dict) and "details" in ke:
        late = ", ".join(ke.get("alert_predates_evidence") or []) or "none"
        lines.append(f"## Known events\n\nRecall {ke['recall']} ({ke['events']} events); in weekly top-10 for the primary "
                     f"audience {ke['top10_in_week_primary_audience']}; alerted {ke['alerted_primary_audience']}; "
                     f"pushed (P1/P2) {ke.get('pushed_primary_audience')}; median detection delay "
                     f"{ke['median_delay_days']} days, measured from the source date of the insight that reported "
                     f"the event. Alert raised for an earlier event of the same development: {late}.\n")
        lines.append("| event | found | development | rank inv | rank ai | alert inv | alert ai | delay (d) |\n"
                     "|---|---|---|---|---|---|---|---|")
        for r in ke["details"]:
            lines.append(f"| {r['title'][:70]} | {'yes' if r['found'] else 'no'} | {r.get('development_id') or ''} | "
                         f"{r.get('rank_investment', '')} | {r.get('rank_ai_team', '')} | {r.get('alert_investment') or ''} | "
                         f"{r.get('alert_ai_team') or ''} | {r.get('delay_days', '')} |")
        lines.append("")
    rendered = set()
    for name, render in _SECTIONS.items():
        res = report.get(name)
        if isinstance(res, dict) and "error" not in res:
            try:
                lines += render(res)
                rendered.add(name)
            except (KeyError, TypeError, ValueError, AttributeError):  # e.g. a stored result of an older layout
                log.exception("eval report: section %s could not be rendered; showing its JSON headline", name)
    for name in ("extraction", "noise", *_SECTIONS, "scoring", "er", "costs"):
        res = report.get(name)
        if res is None or name in rendered:
            continue
        lines.append(f"## {name}\n\n```json\n{json.dumps(_safe_headline(name, res), indent=2, default=str)[:4000]}\n```\n")
    return "\n".join(lines)
