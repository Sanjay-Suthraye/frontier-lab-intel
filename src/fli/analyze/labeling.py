"""Ground truth for the materiality models: LLM-judge silver labels, a human review round-trip, agreement.

  silver  stratified sample of developments without an LLM label -> the judge (settings.model_judge,
          effort 'medium', prompts/judge_materiality.md) labels BOTH audiences 0 noise / 1 worth knowing /
          2 material with a <= 25-word reason -> Label rows 'llm:<model>', rubric_version 'r1'
  export  data/eval/labels_export.csv for human review (human columns survive re-exports)
  import  human labels back from that CSV -> annotator 'human:<name>'
  stats   counts per annotator/audience, Cohen's kappa human vs LLM where both exist

The judge sees what a reviewer would see (titles, claims, quotes, sources, people, the lab's curated
public exposure) but none of the scoring features derived from it (magnitude, confidence, novelty,
tiers, rule exposure links), so its labels can be used to test whether those features carry signal.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from fli.analyze.exposure import SKIP_STATUSES
from fli.analyze.features import FeatureContext, development_features, event_group
from fli.config import CONFIG_DIR, DATA_DIR, get_settings
from fli.db.models import Development, Document, Insight, Lab, Label, Person
from fli.db.session import session_scope
from fli.llm.client import BudgetExceeded, LLMResult, run_async
from fli.llm.prompts import Prompt, load_prompt
from fli.taxonomy import AUDIENCES, EVENT_TYPES
from fli.textutil import slugify

log = logging.getLogger("fli.labeling")

RUBRIC_VERSION = "r1"
JUDGE_PROMPT = "judge_materiality"
JUDGE_EFFORT, JUDGE_MAX_TOKENS = "medium", 4096
REASON_MAX_WORDS = 25
MAX_INSIGHTS, MAX_CLAIM_CHARS, MAX_QUOTE_CHARS = 8, 500, 240
EXPORT_PATH = DATA_DIR / "eval" / "labels_export.csv"
SOURCE_PRIORITY = ("official", "filing", "research", "personal", "press", "aggregator", "market_data")
HUMAN_COLUMNS = ["human_name", "human_investment", "human_ai_team", "human_notes"]
EXPORT_COLUMNS = ["development_id", "first_seen", "lab", "event_type", "title", "sources", "urls",
                  "llm_investment", "llm_investment_reason", "llm_ai_team", "llm_ai_team_reason", *HUMAN_COLUMNS]

_AUDIENCE_LABEL = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "label": {"type": "integer", "enum": [0, 1, 2]}},
    "required": ["reason", "label"], "additionalProperties": False,
}
# reason before label: the model states the deciding fact, then commits (no maxLength: unsupported)
JUDGE_SCHEMA = {"type": "object", "properties": {aud: _AUDIENCE_LABEL for aud in AUDIENCES},
                "required": list(AUDIENCES), "additionalProperties": False}


# --------------------------------------------------------------------------------------- training view


@dataclass
class TrainingLabel:
    dev: Development
    y: int  # label >= 1 (majority of the chosen annotators)
    y2: int  # label == 2
    source: str  # human | llm


def training_labels(session: Session, audience: str) -> list[TrainingLabel]:
    """One label per development: human annotators (majority) if any, else the latest LLM judge."""
    grouped: dict[int, dict] = defaultdict(lambda: {"dev": None, "human": [], "llm": []})
    rows = session.execute(select(Label, Development).join(Development, Development.id == Label.development_id)
                           .where(Label.audience == audience, Development.status.not_in(SKIP_STATUSES)))
    for label, dev in rows:
        kind = label.annotator.split(":", 1)[0]
        if kind in ("human", "llm"):
            grouped[dev.id]["dev"] = dev
            grouped[dev.id][kind].append(label)
    out = []
    for g in grouped.values():
        source = "human" if g["human"] else "llm"
        chosen = g["human"] or [max(g["llm"], key=lambda lb: (lb.created_at, lb.id))]
        vals = [lb.label for lb in chosen]
        out.append(TrainingLabel(g["dev"], int(2 * sum(v >= 1 for v in vals) >= len(vals)),
                                 int(2 * sum(v == 2 for v in vals) > len(vals)), source))
    return out


# --------------------------------------------------------------------------------------- sampling


def primary_source_class(dev: Development) -> str:
    classes = set(dev.source_classes or [])
    return next((c for c in SOURCE_PRIORITY if c in classes), "unknown")


def stratum(dev: Development, lab_tiers: dict[str, int]) -> tuple[str, str, int]:
    return event_group(dev.event_type) or "other", primary_source_class(dev), lab_tiers.get(dev.lab_id or "", 0)


def allocate(sizes: dict, n: int) -> dict:
    """One seat per stratum first (largest strata first if n is short), then Webster seats on sqrt(size):
    between proportional allocation (big strata swamp the sample) and equal allocation (tiny strata
    over-weighted)."""
    alloc = dict.fromkeys(sizes, 0)
    n = min(n, sum(sizes.values()))
    for h in sorted((h for h in sizes if sizes[h] > 0), key=lambda h: (-sizes[h], str(h)))[:n]:
        alloc[h] = 1
    for _ in range(n - sum(alloc.values())):
        h = max((h for h in sizes if alloc[h] < sizes[h]),
                key=lambda h: (math.sqrt(sizes[h]) / (alloc[h] + 0.5), str(h)))
        alloc[h] += 1
    return alloc


def systematic(items: list, k: int, rng: random.Random) -> list:
    """k items evenly spaced along `items` (sorted by prior score) from a random start, so low-prior
    items are labelled as well as top-ranked ones."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    step = len(items) / k
    start = rng.random() * step
    return [items[int(start + i * step)] for i in range(k)]


def sample_for_labeling(session: Session, n: int, seed: int = 0) -> list[Development]:
    """Up to n developments lacking an LLM label, stratified by event group x source class x lab tier."""
    from fli.analyze.scoring import load_model  # scoring imports this module lazily

    have: dict[int, set] = defaultdict(set)
    for dev_id, aud in session.execute(select(Label.development_id, Label.audience)
                                       .where(Label.annotator.like("llm:%"))):
        have[dev_id].add(aud)
    devs = [d for d in session.scalars(select(Development).where(Development.status.not_in(SKIP_STATUSES))
                                       .order_by(Development.id)) if len(have[d.id]) < len(AUDIENCES)]
    ctx = FeatureContext.load(session)
    models = [load_model(aud) for aud in AUDIENCES]
    prior = {d.id: max(m.predict(f) for m in models)
             for d in devs for f in [development_features(session, d, ctx)]}
    strata: dict[tuple, list[Development]] = defaultdict(list)
    for d in devs:
        strata[stratum(d, ctx.lab_tiers)].append(d)
    rng = random.Random(seed)
    picked: list[Development] = []
    for h, k in allocate({h: len(v) for h, v in strata.items()}, n).items():
        picked += systematic(sorted(strata[h], key=lambda d: (prior[d.id], d.id)), k, rng)
    return sorted(picked, key=lambda d: d.id)


# --------------------------------------------------------------------------------------- judge


def holdings_text() -> str:
    cfg = yaml.safe_load((CONFIG_DIR / "exposure.yaml").read_text(encoding="utf-8")) or {}
    names = {t["symbol"]: t["company"] for t in cfg.get("tickers", [])}
    lines = [f"- {h['symbol']} ({names.get(h['symbol'], h['symbol'])}): {h.get('weight')}% of the fund"
             for h in cfg.get("bit_holdings", [])]
    return f"(as of {cfg.get('as_of', 'n/a')})\n" + "\n".join(lines)


def judge_prompt() -> Prompt:
    """Rubric with the current holdings filled in; the text (hence prompt version) changes with them."""
    base = load_prompt(JUDGE_PROMPT)
    return dataclasses.replace(base, system=base.system.replace("{{holdings}}", holdings_text()))


def _clip(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def judge_input(session: Session, dev: Development) -> str:
    """The evidence for one development, deterministic so the response cache hits on re-runs."""
    lab = session.get(Lab, dev.lab_id) if dev.lab_id else None
    insights = list(session.scalars(select(Insight).where(Insight.development_id == dev.id)))
    docs = {d.id: d for d in session.scalars(select(Document).where(
        Document.id.in_({i.document_id for i in insights})))} if insights else {}
    person_ids = sorted(set(dev.person_ids or []) | {p for i in insights for p in (i.person_ids or [])})
    people = list(session.scalars(select(Person).where(Person.id.in_(person_ids)))) if person_ids else []
    rank = {c: i for i, c in enumerate(SOURCE_PRIORITY)}
    insights.sort(key=lambda i: (rank.get(docs[i.document_id].source_class if i.document_id in docs else "", 99),
                                 i.published_at or dev.first_seen_at, i.id))
    classes = ", ".join(sorted(dev.source_classes or [])) or "unknown"
    lines = [f"Development: {dev.title}",
             f"Event type: {dev.event_type} ({EVENT_TYPES.get(dev.event_type, 'unknown')})",
             f"First seen {dev.first_seen_at:%Y-%m-%d}, last seen {dev.last_seen_at:%Y-%m-%d}; "
             + f"{dev.source_count} source(s): {classes}"]
    if lab is not None:
        lines.append(f"Lab: {lab.name} ({lab.kind}) - {_clip(lab.description, 200)}")
        if lab.public_exposure:
            rel = "; ".join(f"{e.get('ticker')} {e.get('relation')}" + (f" ({e['note']})" if e.get("note") else "")
                            for e in lab.public_exposure)
            lines.append(f"Lab's public-market links (curated register): {rel}")
    for p in people:
        lines.append(f"Person: {p.name}, {p.title or 'role unknown'}" + (f" - {_clip(p.why_key, 160)}" if p.why_key else ""))
    if dev.summary:
        lines.append(f"Summary: {_clip(dev.summary, MAX_CLAIM_CHARS)}")
    lines.append("")
    for n, ins in enumerate(insights[:MAX_INSIGHTS], 1):
        doc = docs.get(ins.document_id)
        when = ins.published_at or (doc.published_at if doc else None)
        lines.append(f"[{n}] {doc.source_class if doc else 'unknown'} source" + (f", {when:%Y-%m-%d}" if when else ""))
        lines.append(f"    {_clip(ins.title, 200)}: {_clip(ins.claim, MAX_CLAIM_CHARS)}")
        for q in (ins.evidence_quotes or [])[:2]:
            quote = q.get("quote") or q.get("text") if isinstance(q, dict) else q
            if quote:
                lines.append(f'    quote{" (verified)" if ins.quotes_verified else ""}: "{_clip(str(quote), MAX_QUOTE_CHARS)}"')
        if ins.is_rumor:
            lines.append("    (reported as rumour / unconfirmed)")
        if doc is not None:
            lines.append(f"    {doc.url}")
    if len(insights) > MAX_INSIGHTS:
        lines.append(f"(+{len(insights) - MAX_INSIGHTS} more items from the same sources)")
    return "\n".join(lines)


def parse_judgement(data: dict | None) -> dict[str, tuple[int, str]] | None:
    """Validate the judge output (labels 0-2, reason clipped to 25 words); None if unusable."""
    if not isinstance(data, dict):
        return None
    out = {}
    for aud in AUDIENCES:
        item = data.get(aud)
        if not isinstance(item, dict) or item.get("label") not in (0, 1, 2) or isinstance(item.get("label"), bool):
            return None
        out[aud] = (int(item["label"]), " ".join(str(item.get("reason") or "").split()[:REASON_MAX_WORDS]))
    return out


def _gateway():
    from fli.llm.client import LLMGateway

    return LLMGateway()


def _calls(prompt: Prompt, part: list[tuple[int, str]], model: str) -> list[dict]:
    return [{"purpose": "judge", "prompt": prompt, "user": user, "schema": JUDGE_SCHEMA, "model": model,
             "max_tokens": JUDGE_MAX_TOKENS, "effort": JUDGE_EFFORT, "ref": {"development_id": d}} for d, user in part]


def _store(results: list[tuple[int, LLMResult | None]], annotator: str) -> tuple[int, int]:
    written = failed = 0
    with session_scope() as s:
        for dev_id, res in results:
            parsed = parse_judgement(res.data) if res is not None and res.ok else None
            if parsed is None:
                failed += 1
                continue
            for aud, (label, reason) in parsed.items():
                key = (Label.development_id == dev_id, Label.audience == aud, Label.annotator == annotator)
                if s.scalar(select(Label.id).where(*key)) is None:
                    s.add(Label(development_id=dev_id, audience=aud, label=label, reason=reason,
                                annotator=annotator, rubric_version=RUBRIC_VERSION))
                    written += 1
    return written, failed


def run_silver(n: int, use_batch: bool = False, chunk: int = 25, seed: int = 0) -> dict:
    """Judge-label a stratified sample. Saved chunk by chunk so a budget stop keeps finished work."""
    model = get_settings().model_judge
    prompt = judge_prompt()
    with session_scope() as s:
        sample = sample_for_labeling(s, n, seed=seed)
        lab_tiers = FeatureContext.load(s).lab_tiers
        strata = Counter("/".join(map(str, stratum(d, lab_tiers))) for d in sample)
        payloads = [(d.id, judge_input(s, d)) for d in sample]
    out = {"sampled": len(payloads), "strata": len(strata), "labels_written": 0, "failed": 0,
           "annotator": f"llm:{model}", "rubric_version": RUBRIC_VERSION}
    if not payloads:
        return out
    gw = _gateway()

    def save(results: list[tuple[int, LLMResult | None]]) -> None:
        written, failed = _store(results, out["annotator"])
        out["labels_written"] += written
        out["failed"] += failed
        out["cost_usd"] = round(out.get("cost_usd", 0.0) + sum(r.cost_usd for _, r in results if r), 4)

    async def chunks() -> None:
        # one event loop for every chunk: the gateway's semaphore and HTTP pool bind to the first loop
        for i in range(0, len(payloads), chunk):
            part = payloads[i:i + chunk]
            save(list(zip([d for d, _ in part], await gw.structured_many(_calls(prompt, part, model)), strict=True)))

    try:
        if use_batch:  # Message Batches: 50% cheaper, one submission for the whole sample
            res = gw.structured_batch(purpose="judge", prompt=prompt, schema=JUDGE_SCHEMA, model=model,
                                      max_tokens=JUDGE_MAX_TOKENS, effort=JUDGE_EFFORT,
                                      items=[(f"dev-{d}", user, {"development_id": d}) for d, user in payloads])
            save([(d, res.get(f"dev-{d}")) for d, _ in payloads])
        else:
            run_async(chunks())
    except BudgetExceeded as exc:  # chunks already saved stay saved; cached responses make a re-run free
        log.error("silver labelling stopped: %s", exc)
        out["stopped"] = str(exc)
    out["sample_by_stratum"] = dict(strata.most_common())
    return out


# --------------------------------------------------------------------------------------- review round-trip


def _read_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["development_id"]: row for row in csv.DictReader(fh) if row.get("development_id")}


def export_labels(path: Path | None = None, blind: bool = False) -> dict:
    """Write the review sheet. blind=True hides the LLM labels (use it when measuring agreement)."""
    path = Path(path or EXPORT_PATH)
    previous = _read_csv(path) if path.exists() else {}
    with session_scope() as s:
        labels = list(s.scalars(select(Label).order_by(Label.created_at, Label.id)))
        llm = {(lb.development_id, lb.audience): lb for lb in labels if lb.annotator.startswith("llm:")}
        human = {(lb.development_id, lb.audience): lb for lb in labels if lb.annotator.startswith("human:")}
        ids = {d for d, _ in llm} | {d for d, _ in human}
        devs = list(s.scalars(select(Development).where(Development.id.in_(ids)).order_by(Development.first_seen_at)))
        labs = {lab.id: lab.name for lab in s.scalars(select(Lab))}
        rows = []
        for dev in devs:
            urls = list(s.scalars(select(Document.url).join(Insight, Insight.document_id == Document.id)
                                  .where(Insight.development_id == dev.id).distinct().limit(3)))
            row = {"development_id": str(dev.id), "first_seen": f"{dev.first_seen_at:%Y-%m-%d}",
                   "lab": labs.get(dev.lab_id or "", dev.lab_id or ""), "event_type": dev.event_type,
                   "title": dev.title, "sources": f"{dev.source_count} ({', '.join(sorted(dev.source_classes or []))})",
                   "urls": " | ".join(urls), **dict.fromkeys(EXPORT_COLUMNS[7:], "")}
            for aud in AUDIENCES:
                if (lb := llm.get((dev.id, aud))) is not None and not blind:
                    row[f"llm_{aud}"], row[f"llm_{aud}_reason"] = str(lb.label), lb.reason or ""
                if (hb := human.get((dev.id, aud))) is not None:  # pre-fill from imported labels
                    row[f"human_{aud}"], row["human_name"] = str(hb.label), hb.annotator.split(":", 1)[1]
            prev = previous.get(str(dev.id), {})
            if any((prev.get(c) or "").strip() for c in HUMAN_COLUMNS):  # never overwrite reviewer edits
                row.update({c: prev.get(c, "") for c in HUMAN_COLUMNS})
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path), "rows": len(rows), "blind": blind,
            "rows_with_human": sum(any(r[f"human_{a}"] for a in AUDIENCES) for r in rows)}


def import_labels(path: Path | None = None) -> dict:
    """Read human labels (0/1/2) from the review sheet; upserts annotator 'human:<name>'."""
    path = Path(path or EXPORT_PATH)
    counts = Counter()
    with session_scope() as s:
        for dev_id, row in _read_csv(path).items():
            if not dev_id.isdigit() or s.get(Development, int(dev_id)) is None:
                counts["unknown_development"] += 1
                continue
            annotator = f"human:{slugify((row.get('human_name') or '').strip() or 'reviewer')}"
            for aud in AUDIENCES:
                raw = (row.get(f"human_{aud}") or "").strip()
                if not raw:
                    continue
                if raw not in {"0", "1", "2"}:
                    counts["invalid"] += 1
                    continue
                lb = s.scalar(select(Label).where(Label.development_id == int(dev_id), Label.audience == aud,
                                                  Label.annotator == annotator))
                if lb is None:
                    s.add(Label(development_id=int(dev_id), audience=aud, label=int(raw), annotator=annotator,
                                reason=(row.get("human_notes") or None), rubric_version=RUBRIC_VERSION))
                    counts["created"] += 1
                elif lb.label != int(raw):
                    lb.label, lb.reason = int(raw), row.get("human_notes") or lb.reason
                    counts["updated"] += 1
                else:
                    counts["unchanged"] += 1
    return {"path": str(path), **counts}


def _kappa(a: list[int], b: list[int], **kw) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return None
    value = cohen_kappa_score(a, b, **kw)
    return None if math.isnan(value) else round(float(value), 3)


def label_stats() -> dict:
    """Counts per annotator/audience, and human-vs-LLM agreement (Cohen's kappa) on shared items."""
    with session_scope() as s:
        labels = list(s.scalars(select(Label).order_by(Label.created_at, Label.id)))
    counts: dict[str, dict] = defaultdict(dict)
    for lb in labels:
        c = counts[lb.annotator].setdefault(lb.audience, {"n": 0, "0": 0, "1": 0, "2": 0})
        c["n"] += 1
        c[str(lb.label)] += 1
    agreement: dict[str, dict] = {}
    for aud in AUDIENCES:
        llm = {lb.development_id: lb.label for lb in labels if lb.audience == aud and lb.annotator.startswith("llm:")}
        humans = sorted({lb.annotator for lb in labels if lb.audience == aud and lb.annotator.startswith("human:")})
        for h in humans:
            pairs = [(lb.label, llm[lb.development_id]) for lb in labels
                     if lb.audience == aud and lb.annotator == h and lb.development_id in llm]
            hv, lv = [p[0] for p in pairs], [p[1] for p in pairs]
            agreement[f"{aud}:{h}"] = {
                "n": len(pairs), "exact_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 3) if pairs else None,
                "kappa": _kappa(hv, lv, labels=[0, 1, 2]),
                "kappa_quadratic": _kappa(hv, lv, labels=[0, 1, 2], weights="quadratic"),
                "kappa_binary_ge1": _kappa([int(x >= 1) for x in hv], [int(y >= 1) for y in lv], labels=[0, 1]),
            }
    return {"counts": dict(counts), "human_vs_llm": agreement}


def run_labels(action: str, n: int = 300, path: str | None = None, use_batch: bool = False,
               blind: bool = False, **_: object) -> dict:
    """`fli labels <silver|export|import|stats>`."""
    actions = {
        "silver": lambda: run_silver(n, use_batch=use_batch),
        "export": lambda: export_labels(Path(path) if path else None, blind=blind),
        "import": lambda: import_labels(Path(path) if path else None),
        "stats": label_stats,
    }
    if action not in actions:
        raise ValueError(f"unknown action {action!r}; choose from {', '.join(actions)}")
    return actions[action]()
