"""Audience lenses: the audience-specific last mile on top of the shared core.

One Sonnet call per (development, audience) turns a scored development into "what it means and
what to do" for that reader:

* investment: so-what for positions and theses, tickers restricted to the development's exposure
  candidates = ExposureLink rows (closed Ticker list) + the lab's curated public_exposure, a
  direction, horizon and an INVESTMENT_ACTIONS action. The symbol enum is the union of the
  candidates of the developments sent together (the schema is part of the cached prompt prefix),
  the user message names each development's own allowed symbols, and stored tickers are filtered
  to them. An abstaining lens (no_action) keeps no tickers (schemas.clean_lens).
* ai_team: technical so-what, a tech-radar action and artifacts to try

Which developments get one (run_lens): per audience, latest Score (newest created_at) with
p_material >= LENS_MIN_P, or top TOP_PER_WEEK of its ISO week (by first_seen_at), and no Lens for
that audience yet. ensure_lens() generates missing lenses on demand (digests call it).

Refresh: a lens records the evidence it was written from in Lens.checks['basis'] = {insight_count,
source_classes, has_official, newest_insight_id}. run_lens also regenerates, in place, lenses whose
development changed materially since (refresh_reasons: an official source arrived, a new source
class, or >= REFRESH_MIN_NEW_INSIGHTS newer insights), on developments that still warrant a lens or
carry an alert, at most REFRESH_PER_RUN per run: official confirmations first, then alerted ones.
So a rumor confirmed by the lab gets confirmed-stage analysis before the alert stage re-renders it.
ensure_lens(..., refresh=True) regenerates any lens older than its development's evidence (alerts
call it before pushing). backfill_lens_basis() records the basis of lenses written before bases
existed, inferred from timestamps, without an LLM call.

Storage (AI team): Lens.tickers stays [] and artifacts_to_try is stored as structured data in
Lens.artifacts (models / repos / tools to try), rendered next to what_to_do.

Faithfulness (no LLM): the lens text is what PMs read, so every stored lens carries deterministic
checks in Lens.checks (faithfulness_checks): each figure (number, percentage, multiple, currency
amount) in headline, so_what and what_to_do must be written in the lens input, and wording that
presents facts as official must have an official source behind them. A headline with an unsupported
figure (or official wording without an official source) is replaced by the development title, and
the original is kept in the checks. recheck_lenses() applies the checks and the no_action ticker
rule to lenses already stored.

Failures are not retried forever: a (development, audience) whose lens call failed MAX_FAILURES
times (counted from the llm_call ledger) is skipped by run_lens.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from functools import lru_cache

import yaml
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from fli.config import CONFIG_DIR, get_settings
from fli.db.models import (
    Alert,
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Lens,
    LLMCall,
    Person,
    Score,
    Ticker,
)
from fli.db.session import session_scope
from fli.deliver.materiality import INACTIVE_STATUSES
from fli.extract import context as ctx
from fli.extract.schemas import (
    clean_lens,
    clean_str,
    clip_chars,
    clip_words,
    lens_ai_team_schema,
    lens_investment_schema,
)
from fli.extract.verify import verified_quotes
from fli.llm.client import BudgetExceeded, LLMResult
from fli.taxonomy import AUDIENCES, EVENT_TYPES
from fli.timeutil import utcnow

log = logging.getLogger("fli.deliver.lens")

LENS_MIN_P = 0.35
TOP_PER_WEEK = 15
MAX_FAILURES = 3
PURPOSE = "lens"
MAX_TOKENS = 6_000
EFFORT = "medium"
CHUNK = 8
MAX_EVIDENCE = 8
MAX_CANDIDATES = 20
REFRESH_PER_RUN = 30  # lens refreshes per run_lens call, all audiences together (one Sonnet call each)
REFRESH_MIN_NEW_INSIGHTS = 2  # newer insights that make a lens stale without a new source class


@dataclass
class LensJob:
    dev_id: int
    audience: str
    user: str
    symbols: list[str]
    # for the faithfulness checks (not sent to the model beyond what `user` already contains)
    title: str = ""  # development title: the safe fallback headline
    has_official: bool = False  # the input says "official source: yes"
    official_text: str = ""  # date, claim, quotes and artifacts of the official evidence items
    context: str = ""  # facts in the system prompt a lens may cite (investment: holdings, themes)
    lab_names: tuple[str, ...] = ()  # how lens text names the lab ("confirmed by OpenAI")
    basis: dict = dataclasses.field(default_factory=dict)  # the evidence the input describes (Lens.checks['basis'])


# ------------------------------------------------------------------------------- selection
def latest_scores(session: Session, audience: str) -> dict[int, float]:
    """p_material of the newest Score per development (ties broken by id)."""
    best: dict[int, tuple[tuple[datetime, int], float]] = {}
    rows = session.execute(select(Score.development_id, Score.p_material, Score.created_at, Score.id)
                           .where(Score.audience == audience))
    for dev_id, p, created, sid in rows:
        key = (created, sid)
        if dev_id not in best or key > best[dev_id][0]:
            best[dev_id] = (key, float(p))
    return {dev_id: p for dev_id, (_, p) in best.items()}


def failed_counts(session: Session, audience: str) -> Counter:
    rows = session.scalars(select(LLMCall.ref).where(LLMCall.purpose == PURPOSE,
                                                     LLMCall.status.notin_(("ok", "cached"))))
    return Counter(r.get("development_id") for r in rows if isinstance(r, dict) and r.get("audience") == audience)


def _eligible(session: Session, scores: dict[int, float], min_p: float, top_n: int) -> set[int]:
    """Developments that warrant a lens: p_material >= min_p, or top `top_n` of their ISO week."""
    first_seen = dict(session.execute(select(Development.id, Development.first_seen_at)
                                      .where(Development.status != "merged")).all())
    weeks: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for dev_id, p in scores.items():
        if first_seen.get(dev_id) is not None:
            weeks[tuple(first_seen[dev_id].isocalendar()[:2])].append((p, dev_id))
    eligible = {d for d, p in scores.items() if p >= min_p and d in first_seen}
    for ranked in weeks.values():
        ranked.sort(key=lambda t: (-t[0], t[1]))
        eligible.update(d for _, d in ranked[:top_n])
    return eligible


def select_for_lens(session: Session, audience: str, min_p: float = LENS_MIN_P,
                    top_n: int = TOP_PER_WEEK) -> list[int]:
    """Development ids that need a lens for `audience`, highest p_material first."""
    scores = latest_scores(session, audience)
    if not scores:
        return []
    eligible = _eligible(session, scores, min_p, top_n)
    done = set(session.scalars(select(Lens.development_id).where(Lens.audience == audience)))
    dead = {d for d, n in failed_counts(session, audience).items() if n >= MAX_FAILURES}
    return sorted(eligible - done - dead, key=lambda d: (-scores[d], d))


# ------------------------------------------------------------------------------- basis and refresh
@dataclass(frozen=True)
class Member:
    """One insight of a development, as far as a lens basis is concerned."""

    id: int
    created_at: datetime | None
    source_class: str | None
    official: bool


def members_by_development(session: Session, dev_ids: Iterable[int] | None = None) -> dict[int, list[Member]]:
    """Insights per development (all developments when `dev_ids` is None)."""
    q = (select(Insight.development_id, Insight.id, Insight.created_at, Insight.is_official, Document.source_class)
         .join(Document, Document.id == Insight.document_id).where(Insight.development_id.is_not(None)))
    if dev_ids is None:
        chunks: list[list[int] | None] = [None]
    else:
        ids = sorted(set(dev_ids))
        chunks = [ids[i:i + 500] for i in range(0, len(ids), 500)]
    out: dict[int, list[Member]] = defaultdict(list)
    for chunk in chunks:
        for dev_id, iid, created, official, source_class in session.execute(
                q if chunk is None else q.where(Insight.development_id.in_(chunk))):
            out[dev_id].append(Member(iid, created, source_class, bool(official) or source_class == "official"))
    return out


def basis_of(members: Sequence[Member]) -> dict:
    """What a lens written from these insights saw: the aggregates clustering stores on the
    development (cluster.recompute_development), plus the newest insight id."""
    return {"insight_count": len(members),
            "source_classes": sorted({m.source_class for m in members if m.source_class}),
            "has_official": any(m.official for m in members),
            "newest_insight_id": max((m.id for m in members), default=None)}


def inferred_basis(lens: Lens, members: Sequence[Member]) -> dict:
    """Basis of a lens written before bases were recorded: the insights that existed when it was
    created (Insight.created_at <= Lens.created_at). When none qualifies (clock skew) the lens is
    assumed to have seen the current state, so a legacy lens is never refreshed on a guess."""
    seen = [m for m in members if m.created_at is None or lens.created_at is None or m.created_at <= lens.created_at]
    return {**basis_of(seen or members), "backfilled": True}


def lens_basis(lens: Lens, members: Sequence[Member]) -> dict:
    basis = (lens.checks or {}).get("basis")
    return basis if isinstance(basis, dict) and basis else inferred_basis(lens, members)


def new_insights(basis: dict, members: Sequence[Member]) -> int:
    newest = basis.get("newest_insight_id")
    return sum(1 for m in members if newest is None or m.id > newest)


def refresh_reasons(basis: dict, members: Sequence[Member], strict: bool = False) -> list[str]:
    """Why a lens written from `basis` no longer matches the development's insights (`members`):
    'official_source' (has_official flipped to true), 'new_source_class:<classes>', and
    'new_insights:<n>' for n >= REFRESH_MIN_NEW_INSIGHTS newer insights (any newer one with
    `strict`). An empty list: the lens is current enough to keep."""
    now = basis_of(members)
    out = []
    if now["has_official"] and not basis.get("has_official"):
        out.append("official_source")
    if added := sorted(set(now["source_classes"]) - set(basis.get("source_classes") or [])):
        out.append("new_source_class:" + ",".join(added))
    if (n := new_insights(basis, members)) >= (1 if strict else REFRESH_MIN_NEW_INSIGHTS):
        out.append(f"new_insights:{n}")
    return out


@dataclass(frozen=True)
class Refresh:
    development_id: int
    audience: str
    reasons: tuple[str, ...]
    priority: tuple  # ascending: official confirmations, then alerted, new source class, more news, p


def select_for_refresh(session: Session, audience: str, min_p: float = LENS_MIN_P,
                       top_n: int = TOP_PER_WEEK) -> list[Refresh]:
    """Stored lenses of `audience` whose development changed materially since the lens was written
    (refresh_reasons), on active developments that still warrant a lens (as in select_for_lens) or
    carry an alert for the audience; most urgent first. Lenses without a recorded basis are judged
    by their inferred one. Developments whose lens calls keep failing (MAX_FAILURES) are skipped."""
    lenses = list(session.scalars(select(Lens).where(Lens.audience == audience).order_by(Lens.id)))
    if not lenses:
        return []
    scores = latest_scores(session, audience)
    eligible = _eligible(session, scores, min_p, top_n) if scores else set()
    alerted = set(session.scalars(select(Alert.development_id).where(
        Alert.audience == audience, or_(Alert.status.is_(None), Alert.status != "suppressed"))))
    active = set(session.scalars(select(Development.id).where(Development.status.not_in(INACTIVE_STATUSES))))
    dead = {d for d, n in failed_counts(session, audience).items() if n >= MAX_FAILURES}
    members = members_by_development(session)
    out = []
    for lens in lenses:
        dev_id = lens.development_id
        if dev_id not in active or dev_id in dead or (dev_id not in eligible and dev_id not in alerted):
            continue
        mine = members.get(dev_id, [])
        basis = lens_basis(lens, mine)
        if reasons := refresh_reasons(basis, mine):
            priority = ("official_source" not in reasons, dev_id not in alerted,
                        not any(r.startswith("new_source_class") for r in reasons),
                        -new_insights(basis, mine), -scores.get(dev_id, 0.0), dev_id)
            out.append(Refresh(dev_id, audience, tuple(reasons), priority))
    return sorted(out, key=lambda r: r.priority)


# ------------------------------------------------------------------------------- prompt context
@lru_cache(maxsize=1)
def _themes() -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    path = CONFIG_DIR / "exposure.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return tuple((t["id"], t.get("name", ""), t.get("description", ""), tuple(t.get("tickers", [])))
                 for t in (cfg or {}).get("themes", []))


def holdings_block(session: Session) -> str:
    rows = session.scalars(select(Ticker).where(Ticker.bit_weight.is_not(None)).order_by(Ticker.bit_weight.desc()))
    lines = [f"- {t.symbol} ({t.company}): {t.bit_weight:.1f}%" for t in rows]
    return "\n".join(lines) or "- (holdings not loaded)"


def themes_block() -> str:
    return "\n".join(f"- {tid}: {name}. {desc} ({', '.join(syms)})" for tid, name, desc, syms in _themes()) \
        or "- (no themes configured)"


def build_prompt(session: Session, audience: str):
    if audience == "investment":
        return ctx.render_prompt("lens_investment", template_schema=lens_investment_schema(None),
                                 holdings=holdings_block(session), themes=themes_block())
    return ctx.render_prompt("lens_ai_team")


def _development_block(session: Session, dev: Development, lab: Lab | None, p: float | None, audience: str) -> str:
    names = dict(session.execute(select(Person.id, Person.name).where(Person.id.in_(dev.person_ids or []))).all())
    lines = [
        f"id: {dev.id}",
        f"title: {dev.title}",
        f"event_type: {dev.event_type} ({EVENT_TYPES.get(dev.event_type, 'unknown')})",
        f"lab: {lab.id} ({lab.name}; {lab.kind}; tier {lab.tier})" if lab else "lab: none (not a register lab)",
        f"first_seen: {dev.first_seen_at:%Y-%m-%d} | last_seen: {dev.last_seen_at:%Y-%m-%d}",
        (f"sources: {dev.source_count} distinct ({', '.join(dev.source_classes or []) or 'unknown'}) | "
         f"insights: {dev.insight_count} | official source: {'yes' if dev.has_official else 'no'} | "
         f"verified quotes: {'yes' if dev.has_verified else 'no'}"),
    ]
    if dev.person_ids:
        lines.append("people: " + ", ".join(f"{pid} ({names.get(pid, '?')})" for pid in dev.person_ids))
    if dev.artifacts:
        lines.append("artifacts: " + ", ".join(map(str, dev.artifacts[:12])))
    if p is not None:
        lines.append(f"system materiality estimate ({audience}): p={p:.2f}")
    if dev.summary:
        lines.append("summary: " + dev.summary[:800])
    return "\n".join(lines)


def _evidence_rows(session: Session, dev_id: int) -> list:
    return list(session.execute(select(Insight, Document.url, Document.source_class)
                                .join(Document, Insight.document_id == Document.id)
                                .where(Insight.development_id == dev_id)).all())


def _evidence(session: Session, dev_id: int, rows: list | None = None) -> tuple[str, str]:
    """(evidence block for the prompt, the official items' date, claim, quotes and artifacts)."""
    rows = list(_evidence_rows(session, dev_id) if rows is None else rows)
    rows.sort(key=lambda r: (not r[0].quotes_verified, not r[0].is_official, -(r[0].magnitude or 0),
                             r[0].published_at or datetime.max))
    parts, official = [], []
    for i, (ins, url, source_class) in enumerate(rows[:MAX_EVIDENCE], 1):
        status = (ins.verification or {}).get("status") or ("verified" if ins.quotes_verified else "unverified")
        flags = [source_class, f"quotes {status}"] + (["rumor"] if ins.is_rumor else [])
        when = f"{ins.published_at:%Y-%m-%d}" if ins.published_at else "undated"
        lines = [f"claim: {ins.claim}"]
        lines += [f'quote: "{q}"' for q in verified_quotes(ins.verification)[:2]]
        if ins.artifacts:
            lines.append("artifacts: " + ", ".join(map(str, ins.artifacts[:8])))
        parts.append("\n".join([f"[{i}] {' | '.join(flags)} | {when} | {url}", *lines]))
        if source_class == "official":
            official.append("\n".join([when, *lines]))
    return "\n\n".join(parts) or "(no insights linked)", "\n\n".join(official)


def _evidence_block(session: Session, dev_id: int) -> str:
    return _evidence(session, dev_id)[0]


def lab_aliases(lab: Lab | None) -> tuple[str, ...]:
    """'Meta AI (Superintelligence Labs / FAIR)' -> Meta AI, Meta, Superintelligence Labs, FAIR."""
    if lab is None or not lab.name:
        return ()
    main, _, rest = lab.name.partition("(")
    names = [main.strip(), *(p.strip() for p in re.split(r"[/,]", rest.rstrip(") ")) if p.strip())]
    if len(main.split()) > 1:
        names.append(main.split()[0])
    return tuple(dict.fromkeys(n for n in names if len(n) >= 3))


def exposure_candidates(session: Session, dev: Development, lab: Lab | None) -> list[dict]:
    """ExposureLink rows (rule/llm) plus the lab's curated relations, restricted to the Ticker table."""
    cands: dict[str, dict] = {}

    def add(t: Ticker, note: str) -> None:
        c = cands.setdefault(t.symbol, {"symbol": t.symbol, "company": t.company, "sector": t.sector,
                                        "bit_weight": t.bit_weight, "themes": list(t.themes or []), "links": []})
        c["links"].append(note)

    links = session.execute(select(ExposureLink, Ticker).join(Ticker, ExposureLink.ticker == Ticker.symbol)
                            .where(ExposureLink.development_id == dev.id)
                            .order_by(ExposureLink.confidence.desc(), ExposureLink.id)).all()
    for link, ticker in links:
        add(ticker, f"{link.method} link" + (f" [{link.theme}]" if link.theme else "")
            + (f": {link.rationale}" if link.rationale else "") + f" (hint {link.direction})")
    for rel in (lab.public_exposure if lab else None) or []:
        ticker = session.get(Ticker, rel.get("ticker")) if isinstance(rel, dict) and rel.get("ticker") else None
        if ticker is not None:
            add(ticker, f"lab relation: {rel.get('relation', 'related')}" + (f" ({rel['note']})" if rel.get("note") else ""))
    ordered = sorted(cands.values(), key=lambda c: -(c["bit_weight"] or 0.0))  # holdings first, stable otherwise
    return ordered[:MAX_CANDIDATES]


def _candidates_block(cands: list[dict]) -> str:
    if not cands:
        return "allowed symbols: none (return tickers: [])"
    lines = ["allowed symbols: " + ", ".join(c["symbol"] for c in cands) + " (only these may appear in tickers)"]
    for c in cands:
        held = f"BIT holding {c['bit_weight']:.1f}%" if c["bit_weight"] else "not held"
        themes = f" | themes: {', '.join(c['themes'])}" if c["themes"] else ""
        lines.append(f"- {c['symbol']} | {c['company']} | {c['sector'] or '-'} | {held}{themes} | " + "; ".join(c["links"]))
    return "\n".join(lines)


def build_job(session: Session, dev: Development, audience: str, p: float | None) -> LensJob:
    lab = session.get(Lab, dev.lab_id) if dev.lab_id else None
    rows = _evidence_rows(session, dev.id)
    evidence, official = _evidence(session, dev.id, rows)
    body = (f"<development>\n{ctx.fence(_development_block(session, dev, lab, p, audience))}\n</development>\n"
            f"<evidence>\n{ctx.fence(evidence)}\n</evidence>")
    basis = basis_of([Member(ins.id, ins.created_at, source_class, bool(ins.is_official) or source_class == "official")
                      for ins, _, source_class in rows])
    facts = {"title": dev.title or "", "has_official": bool(dev.has_official), "official_text": official,
             "lab_names": lab_aliases(lab), "basis": basis}
    if audience == "ai_team":
        return LensJob(dev.id, audience, "Write the AI-team lens for this development.\n\n" + body, [], **facts)
    cands = exposure_candidates(session, dev, lab)
    symbols = [c["symbol"] for c in cands]
    user = ("Write the investment lens for this development.\n\n" + body
            + f"\n<exposure_candidates>\n{_candidates_block(cands)}\n</exposure_candidates>")
    return LensJob(dev.id, audience, user, symbols, context=holdings_block(session) + "\n" + themes_block(), **facts)


def group_schema(audience: str, jobs: list[LensJob]) -> dict:
    if audience == "ai_team":
        return lens_ai_team_schema()
    return lens_investment_schema(sorted({sym for j in jobs for sym in j.symbols}))


# ------------------------------------------------------------------------------- faithfulness checks
CHECKS_VERSION = 1
TEXT_FIELDS = ("headline", "so_what", "what_to_do")
CLAIM_FIELDS = ("headline", "so_what")  # what_to_do recommends (sample sizes, next quarter): reported, not gating
GATING_FIELDS = (*CLAIM_FIELDS, "tickers")  # an unsupported figure here makes the lens ungrounded
AUDIT_KEYS = ("headline_replaced", "tickers_dropped")  # kept across rechecks
GENERATION_KEYS = ("basis", "refresh")  # recorded when the text is written; rechecks keep them too

_FIGURE = re.compile(
    r"(?P<cur>[$€£¥]\s?)?(?<![\d.,])(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:[ -]?(?P<unit>%|(?i:percent|bn|mn|tn|thousand|million|billion|trillion)\b"
    r"|[kKMBT](?![A-Za-z0-9])|[x×](?![A-Za-z0-9])))?")
_SCALE = {"k": 3, "K": 3, "thousand": 3, "M": 6, "mn": 6, "million": 6, "B": 9, "bn": 9, "billion": 9,
          "T": 12, "tn": 12, "trillion": 12}
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())} | {
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "dozen": 12}
_WORD_FIGURE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")(?:[ -](thousand|million|billion|trillion))?\b",
                          re.IGNORECASE)
_APPROX = re.compile(r"(?:[~≈]|\b(?:about|around|roughly|approximately|approx\.|nearly|almost|circa))\s*$",
                     re.IGNORECASE)
APPROX_TOLERANCE = Decimal("0.1")  # "~50B" is grounded by 49B or 52B in the input
_PER_UNIT = re.compile(r"\bper\s*$", re.IGNORECASE)  # "cost per 1M tokens": a unit, not a figure
_GLUED_UNIT = re.compile(r"(?:[kMGT]Wh?|[KMGTP]B|[GTP]bps|ms|nm)(?![A-Za-z])")  # 10GW, 128GB, 3nm are figures
_PROVENANCE = re.compile(r"\b(?P<neg>un|non-?)?(?P<word>official(?:ly)?|confirm(?:s|ed|ing)?)\b", re.IGNORECASE)
_HEDGED = re.compile(  # a negation or condition shortly before the word: "no official source", "if confirmed"
    r"\b(?:no|not|nor|never|without|lacks?|lacking|absent|yet|await(?:s|ing)?|pending|until|unless|if|whether|"
    r"once|before|needs?|requires?|look(?:s|ing)?\s+for|check|verify|seek)\b[^.;:!?()]{0,30}$", re.IGNORECASE)
# a sentence that also names a non-official channel attributes jointly ("official notes and OpenRouter")
_OTHER_CHANNEL = re.compile(r"\b(?:openrouter|listings?|listed|market[ _]data|papers?|arxiv|preprint|press|media|"
                            r"reported|third[- ]party)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Figure:
    text: str  # as written: "$11.6B", "80%", "1.05M"
    value: Decimal  # multipliers applied: 11.6B -> 11600000000
    exp: int  # the figure is written to a precision of 10**exp (11.6B -> 8), for rounding tolerance
    approx: bool = False  # hedged as approximate ("~50B", "roughly 10")


def _figure(m: re.Match, approx: bool = False) -> Figure | None:
    num, unit = m.group("num"), m.group("unit") or ""
    try:
        value = Decimal(num.replace(",", ""))
    except InvalidOperation:
        return None
    scale = _SCALE.get(unit if len(unit) == 1 else unit.lower(), 0)
    return Figure(m.group(0).strip(), value.scaleb(scale), scale - len(num.partition(".")[2]), approx)


def figures(text: str | None) -> list[Figure]:
    """Figures stated in lens text: numbers, percentages, multiples and currency amounts. Digits
    inside names are identifiers, not figures, and are skipped (GPT-6.1, H100, p95, Q3, 3D, 4o), as
    are unit denominators ("per 1M tokens")."""
    text = text or ""
    out = []
    for m in _FIGURE.finditer(text):
        start, end = m.start(), m.end()
        before = text[start - 1] if start else ""
        if not m.group("cur") and (before.isalpha() or (before in "-_./" and start > 1 and text[start - 2].isalpha())):
            continue  # a currency sign makes it an amount even after letters ("US$2")
        if not m.group("unit") and end < len(text) and text[end].isalpha() and not _GLUED_UNIT.match(text, end):
            continue  # 3D, 4o, 1st: part of a name
        prefix = text[max(0, start - 20):start]
        if _PER_UNIT.search(prefix):
            continue
        if (fig := _figure(m, approx=bool(_APPROX.search(prefix)))) is not None:
            out.append(fig)
    return out


def support_values(text: str) -> list[Decimal]:
    """Sorted values written anywhere in `text`: every number as written and with its multiplier
    applied (identifiers included: the input side is lenient), and number words ('five', 'one million')."""
    vals: set[Decimal] = set()
    for m in _FIGURE.finditer(text):
        if (fig := _figure(m)) is not None:
            vals.update((fig.value, Decimal(m.group("num").replace(",", ""))))
    for m in _WORD_FIGURE.finditer(text):
        base = Decimal(_NUMBER_WORDS[m.group(1).lower()])
        vals.add(base)
        if m.group(2):
            vals.add(base.scaleb(_SCALE[m.group(2).lower()]))
    return sorted(vals)


def is_grounded(fig: Figure, support: list[Decimal]) -> bool:
    """The figure is in the input exactly, or as the rounding or truncation of a more precise input
    figure at the precision it is written to (16x for 16.7x, 10 for 10.2, 1M for 1,048,576); a figure
    hedged as approximate may be within APPROX_TOLERANCE of an input figure (~50B for 49B)."""
    if fig.approx and fig.value:
        tol = abs(fig.value) * APPROX_TOLERANCE
        if bisect_right(support, fig.value + tol) > bisect_left(support, fig.value - tol):
            return True
    q = Decimal(1).scaleb(fig.exp)
    for x in support[bisect_left(support, fig.value - q):bisect_right(support, fig.value + q)]:
        if x == fig.value:
            return True
        try:
            if fig.value in (x.quantize(q, ROUND_HALF_UP), x.quantize(q, ROUND_DOWN)):
                return True
        except InvalidOperation:  # quantize beyond the context precision
            continue
    return False


def official_assertions(text: str | None, lab_names: Sequence[str] = ()) -> list[tuple[str, str]]:
    """Sentences that present facts as official ("Confirmed by official sources: ...", "OpenAI
    confirmed ..."); negated or conditional uses ("no official source", "if confirmed") do not count.
    Returns (sentence, scope): a bare lead-in ("Confirmed by official release notes.") also covers the
    next sentence."""
    sentences = [s for s in _SENTENCE_END.split(clean_str(text)) if s]
    labs = [re.compile(rf"(?<!\w){re.escape(n)}(?!\w)") for n in lab_names]
    out = []
    for i, sent in enumerate(sentences):
        for m in _PROVENANCE.finditer(sent):
            if m.group("neg") or _HEDGED.search(sent[:m.start()]):
                continue
            if m.group("word").lower().startswith("official") or any(
                    p.search(sent[max(0, m.start() - 40):m.end() + 40]) for p in labs):
                scope = sent
                if not figures(sent) and len(sent.split()) <= 8 and i + 1 < len(sentences):
                    scope = f"{sent} {sentences[i + 1]}"
                out.append((sent, scope))
                break
    return out


def faithfulness_checks(fields: dict[str, str], job: LensJob, extra: str = "",
                        tickers: Sequence[dict] | None = None) -> dict:
    """Deterministic checks of lens text against the lens input (job.user, plus the system-prompt
    facts in job.context and `extra`):

    * unsupported_numbers: figures in headline / so_what / what_to_do / ticker why that are not
      written in the input ({field, text} and, for tickers, symbol)
    * official_wording: claims worded as official when the development has no official source, or
      whose figures come only from non-official evidence (e.g. an OpenRouter context window)
    * grounded: no unsupported figure in headline, so_what or a ticker why, and no official_wording flag
    """
    support = support_values("\n".join((job.user, job.context, extra)))
    unsupported: list[dict] = []
    for field in TEXT_FIELDS:
        for text in dict.fromkeys(f.text for f in figures(fields.get(field)) if not is_grounded(f, support)):
            unsupported.append({"field": field, "text": text})
    for t in tickers or []:
        why = str(t.get("why") or "") if isinstance(t, dict) else ""
        for text in dict.fromkeys(f.text for f in figures(why) if not is_grounded(f, support)):
            unsupported.append({"field": "tickers", "symbol": str(t.get("symbol")), "text": text})
    official_support = support_values(job.official_text)
    portfolio = set(support_values(job.context))  # "Meta (4.6%)": a holding weight, not a sourced fact
    official = []
    for field in CLAIM_FIELDS:
        for sent, scope in official_assertions(fields.get(field), job.lab_names):
            if not job.has_official:
                official.append({"field": field, "issue": "no_official_source", "text": clip_chars(sent, 200)})
                continue
            if _OTHER_CHANNEL.search(scope):
                continue
            figs = [f.text for f in figures(scope) if f.value not in portfolio and is_grounded(f, support)
                    and not is_grounded(f, official_support)]
            if figs:
                official.append({"field": field, "issue": "figures_not_in_official_evidence",
                                 "figures": list(dict.fromkeys(figs)), "text": clip_chars(sent, 200)})
    return {"version": CHECKS_VERSION, "checked_at": utcnow().isoformat(timespec="seconds"),
            "unsupported_numbers": unsupported, "official_wording": official,
            "grounded": not official and not any(u["field"] in GATING_FIELDS for u in unsupported)}


def safe_headline(job: LensJob) -> str:
    """Fallback headline: the development title, which is part of the lens input."""
    return clip_chars(clip_words(job.title, 20), 400) or f"Development {job.dev_id}"


def apply_checks(data: dict, job: LensJob, extra: str = "") -> tuple[dict, dict]:
    """(data with a safe headline if needed, Lens.checks). A headline stating a figure the input does
    not contain, or calling facts official without an official source, is replaced by the development
    title; the original headline is kept under checks['headline_replaced']."""
    fields = {f: str(data.get(f) or "") for f in TEXT_FIELDS}
    checks = faithfulness_checks(fields, job, extra, data.get("tickers"))
    numbers = [u["text"] for u in checks["unsupported_numbers"] if u["field"] == "headline"]
    wording = [o["issue"] for o in checks["official_wording"] if o["field"] == "headline"]
    if (numbers or wording) and (safe := safe_headline(job)) != fields["headline"]:
        replaced = {"original": fields["headline"], "unsupported_numbers": numbers, "official_wording": wording}
        data = {**data, "headline": safe}
        checks = faithfulness_checks({**fields, "headline": safe}, job, extra, data.get("tickers"))
        checks["headline_replaced"] = {k: v for k, v in replaced.items() if v}
    if data.get("dropped_tickers"):
        checks["tickers_dropped"] = list(data["dropped_tickers"])
    return data, checks


# ------------------------------------------------------------------------------- generation
def _store(session: Session, job: LensJob, res: LLMResult | None, refresh: bool = False) -> tuple[str, dict | None]:
    """Write a lens: 'created', or with `refresh` 'refreshed' (the existing row is rewritten in
    place); 'exists' when a row exists and this is not a refresh; 'failed' without a usable result.
    Either way the row gets the deterministic faithfulness checks of its new text and the basis it
    was written from; a refresh also records why it ran and what the lens said before."""
    if res is None or not res.ok:
        return "failed", None
    row = session.scalar(select(Lens).where(Lens.development_id == job.dev_id, Lens.audience == job.audience))
    if row is not None and not refresh:
        return "exists", None
    data, checks = apply_checks(clean_lens(res.data, job.audience, job.symbols), job)
    checks["basis"] = dict(job.basis)
    fields = {"headline": data["headline"], "so_what": data["so_what"], "what_to_do": data["what_to_do"],
              "tickers": data["tickers"], "direction": data["direction"], "horizon": data["horizon"],
              "action": data["action"], "confidence": data["confidence"],
              "artifacts": list(data.get("artifacts_to_try") or []), "model": res.model,
              "prompt_version": res.prompt_version}
    if row is None:
        session.add(Lens(development_id=job.dev_id, audience=job.audience, checks=checks, **fields))
        status = "created"
    else:
        old = dict(row.checks or {})
        prior = old.get("refresh") if isinstance(old.get("refresh"), dict) else {}
        before = old.get("basis") if isinstance(old.get("basis"), dict) else {}
        checks["refresh"] = {"count": int(prior.get("count") or 0) + 1, "at": checks["checked_at"],
                             "reasons": _basis_changes(before, job.basis) if before else ["no_recorded_basis"],
                             "previous": {"headline": row.headline, "action": row.action, "basis": before or None}}
        for name, value in fields.items():
            setattr(row, name, value)
        row.checks = checks
        status = "refreshed"
    if not checks["grounded"] or "headline_replaced" in checks:
        log.info("lens %s/%s flagged: unsupported %s, official wording %s, headline replaced %s", job.dev_id,
                 job.audience, [u["text"] for u in checks["unsupported_numbers"]],
                 [o["issue"] for o in checks["official_wording"]], "headline_replaced" in checks)
    return status, checks


def _basis_changes(before: dict, after: dict) -> list[str]:
    """refresh_reasons between two recorded bases (any newer insight counts)."""
    out = []
    if after.get("has_official") and not before.get("has_official"):
        out.append("official_source")
    if added := sorted(set(after.get("source_classes") or []) - set(before.get("source_classes") or [])):
        out.append("new_source_class:" + ",".join(added))
    old_newest, new_newest = before.get("newest_insight_id"), after.get("newest_insight_id")
    if new_newest is not None and (old_newest is None or new_newest > old_newest):
        out.append("new_insights")
    return out or ["regenerated"]


def store_lens(session: Session, job: LensJob, res: LLMResult | None, refresh: bool = False) -> str:
    return _store(session, job, res, refresh)[0]


async def _generate(gw, dev_ids: list[int], audience: str, summary: Counter, refresh: bool = False) -> None:
    model = get_settings().model_lens
    with session_scope() as s:
        prompt = build_prompt(s, audience)
        scores = latest_scores(s, audience)
    prefix = ctx.PrefixTracker()
    for chunk in ctx.chunked(dev_ids, CHUNK):
        with session_scope() as s:
            devs = s.scalars(select(Development).where(Development.id.in_(chunk))).all()
            jobs = [build_job(s, d, audience, scores.get(d.id)) for d in sorted(devs, key=lambda d: chunk.index(d.id))]
        schema = group_schema(audience, jobs)
        ref = {"refresh": True} if refresh else {}
        calls = [{"purpose": PURPOSE, "prompt": prompt, "user": j.user, "schema": schema, "model": model,
                  "max_tokens": MAX_TOKENS, "effort": EFFORT,
                  "ref": {"development_id": j.dev_id, "audience": audience, **ref}}
                 for j in jobs]
        results = await ctx.call_many(gw, calls, warm=prefix.changed(schema))
        with session_scope() as s:
            for job, res in zip(jobs, results, strict=True):
                status, checks = _store(s, job, res, refresh)
                summary[status] += 1
                if checks is not None and not checks["grounded"]:
                    summary["ungrounded"] += 1
                if checks is not None and "headline_replaced" in checks:
                    summary["headline_replaced"] += 1
                summary["cost_usd"] += res.cost_usd if res is not None else 0.0


async def _run_all(gw, plan: dict[str, list[int]], out: dict[str, Counter],
                   refresh: dict[str, list[int]] | None = None) -> None:
    for audience, ids in plan.items():  # one event loop for the gateway's whole life
        await _generate(gw, ids, audience, out[audience])
    for audience, ids in (refresh or {}).items():  # new lenses first: a missing lens is worse than a stale one
        await _generate(gw, ids, audience, out[audience], refresh=True)


def plan_refresh(session: Session, max_refresh: int | None = None) -> tuple[list[Refresh], list[Refresh]]:
    """(refreshes for this run, deferred ones): every audience's select_for_refresh merged by
    priority and cut at `max_refresh` (default REFRESH_PER_RUN)."""
    cap = REFRESH_PER_RUN if max_refresh is None else max(0, int(max_refresh))
    ranked = sorted((r for a in AUDIENCES for r in select_for_refresh(session, a)), key=lambda r: r.priority)
    return ranked[:cap], ranked[cap:]


def run_lens(limit: int | None = None, gateway=None, use_batch: bool = False, max_refresh: int | None = None,
             **_: object) -> dict:
    """Pipeline stage: new lenses (`limit` caps them per audience), then in-place refreshes of lenses
    whose development changed materially (`max_refresh` per run, default REFRESH_PER_RUN, across
    audiences). Budget exhaustion propagates. Lenses are few and Sonnet's refusal fallbacks are not
    available on the Batches API, so `use_batch` is accepted (stage contract) but ignored."""
    if use_batch:
        log.info("lens: use_batch ignored (interactive calls keep server-side refusal fallbacks)")
    with session_scope() as s:
        plan = {a: select_for_lens(s, a) for a in AUDIENCES}
        chosen, deferred = plan_refresh(s, max_refresh)
    if limit is not None:
        plan = {a: ids[:limit] for a, ids in plan.items()}
    refresh = {a: [r.development_id for r in chosen if r.audience == a] for a in AUDIENCES}
    out: dict[str, Counter] = {a: Counter() for a in AUDIENCES}
    if any(plan.values()) or any(refresh.values()):
        ctx.run_coro(_run_all(gateway or ctx.make_gateway(), {a: ids for a, ids in plan.items() if ids}, out,
                              {a: ids for a, ids in refresh.items() if ids}))
    result = {a: {"todo": len(plan[a]), "refresh_todo": len(refresh[a]),
                  "refresh_deferred": sum(r.audience == a for r in deferred),
                  **{k: (round(v, 5) if k == "cost_usd" else v) for k, v in out[a].items()}}
              for a in AUDIENCES}
    if chosen:
        log.info("lens refresh: %s", [(r.development_id, r.audience, r.reasons) for r in chosen])
    log.info("lens: %s", result)
    return result


def ensure_lens(development_ids: list[int], audience: str, gateway=None, refresh: bool = False) -> int:
    """Generate missing lenses for these developments (no eligibility filter); with `refresh`, also
    regenerate in place every lens of theirs that is older than the development's evidence (any
    newer insight, an official source or a new source class: refresh_reasons with strict=True), as
    alerts do before pushing. Returns how many were created or refreshed. Never raises for budget
    exhaustion or a missing API key: the caller (digest, alerts) can still render without lenses,
    and the reason is logged."""
    if audience not in AUDIENCES:
        raise ValueError(f"unknown audience {audience!r}; expected one of {AUDIENCES}")
    ids = list(dict.fromkeys(int(d) for d in development_ids))
    if not ids:
        return 0
    stale: list[int] = []
    with session_scope() as s:
        have = {lz.development_id: lz for lz in s.scalars(select(Lens).where(Lens.audience == audience,
                                                                            Lens.development_id.in_(ids)))}
        known = dict(s.execute(select(Development.id, Development.status).where(Development.id.in_(ids))).all())
        if refresh and have:
            members = members_by_development(s, list(have))
            stale = [d for d in ids if d in have and known.get(d) not in INACTIVE_STATUSES
                     and refresh_reasons(lens_basis(have[d], members.get(d, [])), members.get(d, []), strict=True)]
    todo = [d for d in ids if d in known and d not in have]
    if not todo and not stale:
        return 0
    return _generate_now(gateway, todo, stale, audience, "ensure_lens")


def regenerate_lenses(development_ids: list[int], audience: str, gateway=None) -> int:
    """Regenerate these lenses in place whatever their basis says (creating missing ones), e.g. after
    a data repair such as scripts/fixes/split_insights.py moved evidence out of a development. One
    LLM call per lens; returns how many were written."""
    if audience not in AUDIENCES:
        raise ValueError(f"unknown audience {audience!r}; expected one of {AUDIENCES}")
    ids = list(dict.fromkeys(int(d) for d in development_ids))
    with session_scope() as s:
        have = set(s.scalars(select(Lens.development_id).where(Lens.audience == audience, Lens.development_id.in_(ids))))
        known = set(s.scalars(select(Development.id).where(Development.id.in_(ids))))
    todo, stale = [d for d in ids if d in known and d not in have], [d for d in ids if d in have]
    if not todo and not stale:
        return 0
    return _generate_now(gateway, todo, stale, audience, "regenerate_lenses")


def _generate_now(gateway, todo: list[int], stale: list[int], audience: str, caller: str) -> int:
    try:
        gw = gateway or ctx.make_gateway()
    except RuntimeError as exc:  # ANTHROPIC_API_KEY missing
        log.warning("%s unavailable: %s", caller, exc)
        return 0
    summary: Counter = Counter()

    async def both() -> None:  # one event loop for the gateway's whole life
        if todo:
            await _generate(gw, todo, audience, summary)
        if stale:
            await _generate(gw, stale, audience, summary, refresh=True)

    try:
        ctx.run_coro(both())
    except BudgetExceeded as exc:
        log.warning("%s stopped early (%d created, %d refreshed): %s", caller, summary["created"],
                    summary["refreshed"], exc)
    return summary["created"] + summary["refreshed"]


def backfill_lens_basis(limit: int | None = None, dry_run: bool = False) -> dict:
    """Record Lens.checks['basis'] for lenses written before bases existed, inferred from
    timestamps (inferred_basis: the insights created before the lens). No LLM call. Idempotent:
    lenses that already have a basis are left alone. Returns counts, plus what the next run_lens
    would refresh (`would_refresh` before the per-run cap, `next_run` after it) and those lenses."""
    per: dict[str, Counter] = defaultdict(Counter)
    with session_scope() as s:
        members = members_by_development(s)
        for lens in list(s.scalars(select(Lens).order_by(Lens.id)))[:limit]:
            c = per[lens.audience]
            c["checked"] += 1
            basis = (lens.checks or {}).get("basis")
            if isinstance(basis, dict) and basis:
                c["had_basis"] += 1
                continue
            c["backfilled"] += 1
            if not dry_run:
                lens.checks = {**(lens.checks or {}), "basis": inferred_basis(lens, members.get(lens.development_id, []))}
        s.flush()
        chosen, deferred = plan_refresh(s)
        if dry_run:
            s.rollback()
    ranked = chosen + deferred
    total: Counter = Counter()
    for c in per.values():
        total.update(c)
    result = {"dry_run": dry_run, **{k: total[k] for k in ("checked", "backfilled", "had_basis")},
              "would_refresh": len(ranked), "next_run": len(chosen),
              "by_audience": {a: {**{k: per[a][k] for k in ("checked", "backfilled", "had_basis")},
                                  "would_refresh": sum(r.audience == a for r in ranked)} for a in AUDIENCES},
              "refresh": [{"development_id": r.development_id, "audience": r.audience, "reasons": list(r.reasons)}
                          for r in ranked[:50]]}
    log.info("backfill_lens_basis: %s", {k: v for k, v in result.items() if k != "refresh"})
    return result


# ------------------------------------------------------------------------------- recheck (no LLM)
def _retitle_alerts(session: Session, lens: Lens, old: str, new: str) -> int:
    """Put a replaced headline into the alerts already rendered from it (title and bold first line)."""
    n = 0
    for alert in session.scalars(select(Alert).where(Alert.development_id == lens.development_id,
                                                     Alert.audience == lens.audience)):
        title, body = alert.title.replace(old, new)[:400], (alert.body_md or "").replace(f"**{old}**", f"**{new}**")
        if (title, body) != (alert.title, alert.body_md or ""):
            alert.title, alert.body_md = title, body
            n += 1
    return n


def recheck_lenses(limit: int | None = None, sync_alerts: bool = True) -> dict:
    """Apply the faithfulness checks and the no_action ticker rule to stored lenses. No LLM call.

    Each lens input is rebuilt with build_job from the current database; every materiality estimate
    the development has had is added to the support, since the lens may quote the one it saw.
    Idempotent: a lens whose text and checks (ignoring checked_at) are unchanged is not written, and
    earlier audit records (replaced headline, dropped tickers) and the generation records (basis,
    refresh) are kept, so a recheck never makes a stale lens look current. With `sync_alerts`, alerts
    rendered from a replaced headline get the new headline too. Returns counts (overall and per
    audience); the unsupported / ungrounded / official counts describe the stored text after the run.
    """
    per: dict[str, Counter] = defaultdict(Counter)
    with session_scope() as s:
        ids = list(s.scalars(select(Lens.id).order_by(Lens.id)))[:limit]
        latest = {a: latest_scores(s, a) for a in AUDIENCES}
        history: dict[tuple[int, str], set[str]] = defaultdict(set)
        for dev_id, aud, p in s.execute(select(Score.development_id, Score.audience, Score.p_material)):
            history[(dev_id, aud)].add(f"p={float(p):.2f}")
        for lens_id in ids:
            row = s.get(Lens, lens_id)
            c = per[row.audience]
            dev = s.get(Development, row.development_id)
            if dev is None or dev.status == "merged":  # a merged development's insights moved to the survivor
                c["skipped_merged_or_missing"] += 1
                continue
            job = build_job(s, dev, row.audience, latest.get(row.audience, {}).get(dev.id))
            tickers, direction, dropped = list(row.tickers or []), row.direction, []
            if row.audience == "investment" and row.action == "no_action":  # same rule as clean_lens
                dropped = [str(t.get("symbol")) for t in tickers if isinstance(t, dict) and t.get("symbol")]
                tickers, direction = [], "unclear"
            data = {f: getattr(row, f) for f in TEXT_FIELDS} | {"tickers": tickers, "dropped_tickers": dropped}
            data, checks = apply_checks(data, job, " ".join(sorted(history[(dev.id, row.audience)])))
            old = dict(row.checks or {})
            for key in (*AUDIT_KEYS, *GENERATION_KEYS):  # the basis stays what the text was written from
                if key in old and key not in checks:
                    checks[key] = old[key]
            c["checked"] += 1
            c["tickers_cleared"] += bool(dropped)
            if data["headline"] != row.headline:
                c["headlines_replaced"] += 1
                if sync_alerts:
                    c["alerts_retitled"] += _retitle_alerts(s, row, row.headline, data["headline"])
            same = {k: v for k, v in checks.items() if k != "checked_at"} == \
                {k: v for k, v in old.items() if k != "checked_at"}
            if same and data["headline"] == row.headline and tickers == list(row.tickers or []) \
                    and direction == row.direction:
                c["unchanged"] += 1
            else:
                row.headline, row.tickers, row.direction, row.checks = data["headline"], tickers, direction, checks
                c["updated"] += 1
            c["ungrounded"] += not checks["grounded"]
            c["with_unsupported_numbers"] += bool(checks["unsupported_numbers"])
            c["with_unsupported_claim_numbers"] += any(u["field"] in GATING_FIELDS for u in checks["unsupported_numbers"])
            c["with_official_wording_flags"] += bool(checks["official_wording"])
    total: Counter = Counter()
    for c in per.values():
        total.update(c)
    keys = ("checked", "updated", "unchanged", "tickers_cleared", "headlines_replaced", "alerts_retitled",
            "ungrounded", "with_unsupported_numbers", "with_unsupported_claim_numbers",
            "with_official_wording_flags", "skipped_merged_or_missing")
    result = {k: total[k] for k in keys} | {"by_audience": {a: {k: per[a][k] for k in keys} for a in sorted(per)}}
    log.info("recheck_lenses: %s", result)
    return result
