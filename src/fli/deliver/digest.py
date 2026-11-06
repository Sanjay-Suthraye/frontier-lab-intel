"""Weekly digests per audience (ISO weeks, Monday 00:00 UTC to the next Monday, exclusive).

Selection is code (select_items): developments that got a P1/P2 alert for the audience that week are
always included; developments the audience lens dismissed (investment no_action, ai_team hold) are
left out unless alerted; the rest of the top N is filled by the audience's latest p_material
(p >= min_p, verified only; config/materiality.yaml). Writing is the model: settings.model_digest at
effort 'medium' turns ONLY those numbered items into the audience's sections with [n] citations.

Model output is validated before it is used (output_issues): at least one entry with its own text
that cites a valid item, and cited coverage >= MIN_COVERAGE of the selected items. A rejected output
is never written to the response cache (LLMGateway.structured(validate=...)), and the call is
retried once with use_cache=False. Without a usable output (no key, budget stop, error, two rejected
outputs) a deterministic renderer fills the same sections from the lenses (model='fallback'); such
reports are provisional and regenerated on the next run.

Citations are then enforced by code: numbers outside the provided set are dropped, an entry that
cites nothing valid is discarded, links the model writes are stripped, and the reference list that
maps [n] to source URLs is built from database metadata. Unconfirmed items cannot sit in "Top of
mind". Every selected item the written sections leave uncited is appended deterministically under
"Also this week", so a selected item is never silently dropped, and only items a report actually
cites count as reported (never repeated in a later digest). The investment digest ends with a
deterministic "Holdings check" (each BIT holding: net lens direction and top item, or 'no new
development' / 'no source coverage') and separates quiet labs (documents that week, nothing material)
from labs that were not monitored (no documents that week).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import yaml
from sqlalchemy import and_, func, or_, select

from fli.analyze.exposure import TickerInfo, mention_names
from fli.config import CONFIG_DIR, Settings, get_settings
from fli.db.models import Alert, Development, Document, Lab, Report, Source, Ticker
from fli.db.session import session_scope
from fli.deliver.alerts import PUSH, md_label, source_ref
from fli.deliver.materiality import (
    DIRECTIONAL,
    INACTIVE_STATUSES,
    SEVERITY_RANK,
    DevContext,
    Policy,
    evaluate_facts,
    facts_for,
    is_lab_level,
    load_contexts,
    load_policy,
)
from fli.llm.client import LLMGateway
from fli.llm.prompts import load_prompt
from fli.taxonomy import AI_TEAM_ACTIONS, AUDIENCES, INVESTMENT_ACTIONS
from fli.timeutil import to_naive_utc, utcnow, week_start

log = logging.getLogger("fli.deliver.digest")

FALLBACK = "fallback"
NOTHING = "Nothing material this week."
LABELS = {"investment": "Investment", "ai_team": "AI team"}
CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
MD_LINK = re.compile(r"\[([^\]]*)\]\((?:https?://|www\.)[^)]*\)")
BARE_URL = re.compile(r"\(?\bhttps?://\S+\)?")
MIN_COVERAGE = 0.6  # a model digest must cite at least this share of the selected items
DISMISSED = {"investment": ("no_action",), "ai_team": ("hold",)}  # lens verdicts kept out unless alerted
# Effort sweep on a failing week: medium returned a near-empty structure in 7/14 AI-team weeks; high
# produced complete, fully cited digests (17 entries, 12/12 items cited) for ~$0.11 per digest.
DIGEST_EFFORT = "high"


def _entry(text: str, extra: dict | None = None) -> dict:
    props = {"title": {"type": "string", "description": "short name or headline"},
             "text": {"type": "string", "description": f"never empty: {text}; cite items as [n]"},
             "items": {"type": "array", "items": {"type": "integer"}, "description": "item numbers this entry uses"}}
    props.update(extra or {})
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _schema(sections: dict[str, dict]) -> dict:
    # headline last: the entries (the substance) are written before the one-line summary
    props = {**{k: {"type": "array", "items": v} for k, v in sections.items()}, "headline": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


SCHEMAS = {
    "investment": _schema({
        "top_of_mind": _entry("2-4 sentences: what happened, why it matters for named holdings or theses, direction",
                              {"tickers": {"type": "array", "items": {"type": "string"}},
                               "action": {"type": "string", "enum": list(INVESTMENT_ACTIONS)}}),
        "by_theme": _entry("1-3 sentences on the theme or holding"),
        "watchlist": _entry("1-2 sentences: the signal and what would confirm or kill it")}),
    "ai_team": _schema({
        "radar": _entry("1-3 sentences: what it is, what it changes for us, the first concrete step",
                        {"ring": {"type": "string", "enum": list(AI_TEAM_ACTIONS)}}),
        "papers": _entry("the core idea and why it changes how we build or evaluate"),
        "tools_techniques": _entry("the concrete engineering takeaway")}),
}
SECTIONS = {aud: [k for k in schema["properties"] if k != "headline"] for aud, schema in SCHEMAS.items()}


@dataclass
class Item:
    n: int
    ctx: DevContext
    p: float
    unconfirmed: bool
    alert: str | None = None  # severity of this week's alert for the audience (P1 / P2 / P3)
    facts: dict = field(default_factory=dict)  # materiality facts (held_tickers, lens_action ...)


@dataclass
class Plan:
    audience: str
    start: datetime
    items: list[Item]
    quiet_labs: list[tuple[int, str]] = field(default_factory=list)  # (tier, name): documents, nothing material
    unmonitored_labs: list[tuple[int, str]] = field(default_factory=list)  # (tier, name): no documents
    tickers: set[str] = field(default_factory=set)  # closed list
    holdings: dict[str, float] = field(default_factory=dict)
    covered: set[str] = field(default_factory=set)  # holdings the week's documents could see
    output: dict | None = None
    model: str | None = None
    note: str | None = None
    cost: float = 0.0
    rejected: str | None = None  # why the gateway's validate hook refused the last output

    @property
    def unconfirmed(self) -> set[int]:
        return {it.n for it in self.items if it.unconfirmed}


# --------------------------------------------------------------------------------- periods
def last_full_week(now: datetime) -> datetime:
    return week_start(now) - timedelta(days=7)


def week_of(value: str | date | datetime) -> datetime:
    parsed = value if isinstance(value, date) else to_naive_utc(value)
    if parsed is None:
        raise ValueError(f"cannot parse week {value!r}; pass any date in the week as YYYY-MM-DD")
    return week_start(parsed)


def periods(week: str | date | None, backfill: bool, settings: Settings, now: datetime) -> list[datetime]:
    last = last_full_week(now)
    if not backfill:
        return [week_of(week) if week else last]
    out, cur = [], week_start(settings.backfill_start)
    while cur <= last:
        out.append(cur)
        cur += timedelta(days=7)
    return out


# --------------------------------------------------------------------------------- selection
def week_alerts(s, audience: str, start: datetime) -> dict[int, str]:
    """Severity per development of the audience's alerts raised (or escalated) in the week; alerts a
    reader suppressed do not count."""
    end = start + timedelta(days=7)
    out: dict[int, str] = {}
    rows = s.execute(select(Alert.development_id, Alert.severity).where(
        Alert.audience == audience, Alert.created_at >= start, Alert.created_at < end, Alert.status != "suppressed"))
    for dev_id, severity in rows:
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(out.get(dev_id), 0):
            out[dev_id] = severity
    return out


def select_items(s, audience: str, start: datetime, policy: Policy, reported: set[int]) -> list[int]:
    """The week's developments (first seen in it, newly active in it, or alerted in it) not yet
    reported: every P1/P2-alerted one, then up to top_n by p among the rest. Developments the
    audience lens dismissed (DISMISSED) are left out unless alerted; P3 (digest-only) alerts rank
    ahead of unalerted items and skip the min_p floor."""
    end = start + timedelta(days=7)
    alerted = week_alerts(s, audience, start)
    q = select(Development.id).where(Development.status.not_in(INACTIVE_STATUSES), or_(
        and_(Development.first_seen_at >= start, Development.first_seen_at < end),
        and_(Development.last_seen_at >= start, Development.last_seen_at < end, Development.first_seen_at < start)))
    ids = sorted(i for i in set(s.scalars(q)) | set(alerted) if i not in reported)
    must, rest = [], []
    for ctx in load_contexts(s, ids).values():
        if ctx.dev.status in INACTIVE_STATUSES:
            continue
        f = facts_for(ctx, audience, policy)
        if not f["verified"]:
            continue
        severity = alerted.get(ctx.dev.id)
        key = (-SEVERITY_RANK.get(severity, 0), -(f["p"] or 0.0), -ctx.dev.first_seen_at.toordinal(), ctx.dev.id)
        if severity in PUSH:
            must.append(key)
        elif f["lens_action"] in DISMISSED[audience]:
            continue
        elif severity is not None or (f["p"] is not None and f["p"] >= policy.digest_min_p):
            rest.append(key)
    chosen = sorted(must) + sorted(rest)[: max(0, policy.digest_top_n - len(must))]
    return [dev_id for *_, dev_id in chosen]


def labs_with_documents(s, start: datetime, end: datetime) -> set[str]:
    """Labs that published at least one collected document in [start, end) (by published date)."""
    when = func.coalesce(Document.published_at, Document.fetched_at)
    q = (select(func.coalesce(Document.lab_id, Source.lab_id)).join(Source, Source.id == Document.source_id)
         .where(when >= start, when < end).distinct())
    return {lab_id for lab_id in s.scalars(q) if lab_id}


def holdings_coverage(s, start: datetime, end: datetime, holdings: dict[str, float]) -> set[str]:
    """BIT holdings the week's collected documents could see: a document that names the company (or
    a brand it owns), or one from a lab whose curated public exposure lists the holding. A holding
    outside this set had no source coverage: silence there is not evidence of no news."""
    if not holdings:
        return set()
    names = {t.symbol: [re.compile(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])")
                        for n in mention_names(TickerInfo(t.symbol, t.company or t.symbol, (), float(t.bit_weight or 0)))]
             for t in s.scalars(select(Ticker).where(Ticker.symbol.in_(sorted(holdings))))}
    lab_links = {lab.id: {str(e.get("ticker", "")) for e in (lab.public_exposure or []) if isinstance(e, dict)}
                 for lab in s.scalars(select(Lab))}
    when = func.coalesce(Document.published_at, Document.fetched_at)
    rows = s.execute(select(Document.title, Document.content_text, func.coalesce(Document.lab_id, Source.lab_id))
                     .join(Source, Source.id == Document.source_id).where(when >= start, when < end))
    covered: set[str] = set()
    for title, text, lab_id in rows:
        covered |= lab_links.get(lab_id, set()) & set(holdings)
        blob = f"{title or ''}\n{text or ''}"
        covered |= {sym for sym, pats in names.items() if sym not in covered and any(p.search(blob) for p in pats)}
    return covered


def build_plan(s, audience: str, start: datetime, ids: list[int], policy: Policy) -> Plan:
    end = start + timedelta(days=7)
    ctxs = load_contexts(s, ids)
    alerted = week_alerts(s, audience, start)
    items = []
    for n, dev_id in enumerate(ids, 1):
        ev = evaluate_facts(facts_for(ctxs[dev_id], audience, policy), audience, policy)  # same guard as alerts
        items.append(Item(n=n, ctx=ctxs[dev_id], p=float(ev.facts["p"] or 0.0), unconfirmed=ev.unconfirmed,
                          alert=alerted.get(dev_id), facts=ev.facts))
    active, seen = {it.ctx.dev.lab_id for it in items}, labs_with_documents(s, start, end)
    labs = [lab for lab in s.scalars(select(Lab).order_by(Lab.tier, Lab.name)) if lab.id not in active]
    tickers = set(s.scalars(select(Ticker.symbol)))
    holdings = {t.symbol: float(t.bit_weight) for t in s.scalars(select(Ticker).where(Ticker.bit_weight.is_not(None)))}
    return Plan(audience=audience, start=start, items=items, tickers=tickers, holdings=holdings,
                quiet_labs=[(lab.tier, lab.name) for lab in labs if lab.id in seen],
                unmonitored_labs=[(lab.tier, lab.name) for lab in labs if lab.id not in seen],
                covered=holdings_coverage(s, start, end, holdings) if audience == "investment" else set())


def _ensure_lenses(ids: list[int], audience: str) -> None:
    try:
        from fli.deliver.lens import ensure_lens
    except ImportError as exc:
        log.warning("lens module unavailable (%s); digest items use development summaries", exc)
        return
    try:
        ensure_lens(list(ids), audience)
    except Exception as exc:  # noqa: BLE001 - budget stop or model error: the digest still renders from summaries
        log.warning("ensure_lens(%s, %d items) failed: %s", audience, len(ids), exc)


# --------------------------------------------------------------------------------- model input
@lru_cache(maxsize=1)
def theme_names() -> dict[str, str]:
    path = CONFIG_DIR / "exposure.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {t["id"]: t.get("name", t["id"]) for t in (raw or {}).get("themes", [])}


def item_tickers(plan: Plan, it: Item) -> list[tuple[str, str]]:
    """(symbol, description) restricted to the closed ticker list: the lens's tickers when a lens
    exists (none when it abstained), exposure links only when there is no lens (event-specific first;
    lab-level relations marked as such)."""
    out: dict[str, str] = {}
    lens = it.ctx.lenses.get(plan.audience)
    if lens is not None:
        for t in (lens.tickers or []) if lens.action != "no_action" else []:
            if isinstance(t, dict) and str(t.get("symbol", "")).upper() in plan.tickers:
                out[str(t["symbol"]).upper()] = f"{t.get('direction') or 'unclear'}: {t.get('why') or ''}".rstrip(": ")
    else:
        for x in sorted(it.ctx.exposures, key=lambda x: (is_lab_level(x), -(x.confidence or 0.0), x.ticker)):
            if x.ticker in plan.tickers and x.ticker not in out:
                theme = theme_names().get(x.theme or "", x.theme or "")
                out[x.ticker] = (("lab-level link" if is_lab_level(x) else f"{x.direction or 'unclear'}")
                                 + (f", theme {theme}" if theme else ""))
    return [(sym, desc + (f", BIT holding {plan.holdings[sym]:.1f}%" if sym in plan.holdings else ""))
            for sym, desc in out.items()]


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _lab_groups(labs: list[tuple[int, str]]) -> str:
    groups: dict[int, list[str]] = {}
    for tier, name in labs:
        groups.setdefault(tier, []).append(name)
    names = {1: "tier 1", 2: "tier 2", 3: "alumni startups"}
    return "; ".join(f"{', '.join(v)} ({names.get(t, f'tier {t}')})" for t, v in sorted(groups.items()))


def render_payload(plan: Plan) -> str:
    """The user message: deterministic text (stable ordering keeps the response cache effective)."""
    end = plan.start + timedelta(days=6)
    y, w, _ = plan.start.isocalendar()
    lines = [f"Week: {plan.start:%Y-%m-%d} to {end:%Y-%m-%d} (ISO {y}-W{w:02d})"]
    if plan.audience == "investment" and plan.holdings:
        held = sorted(plan.holdings.items(), key=lambda kv: -kv[1])
        lines.append("BIT Capital holdings (weight %): " + ", ".join(f"{s} {v:.1f}" for s, v in held))
    if plan.quiet_labs:
        lines.append("Quiet labs (documents collected, nothing material): " + _lab_groups(plan.quiet_labs))
    if plan.unmonitored_labs:
        lines.append("Not monitored this week (no documents collected, so their silence is not evidence): "
                     + _lab_groups(plan.unmonitored_labs))
    lines += ["", "Items:"]
    for it in plan.items:
        ctx, dev = it.ctx, it.ctx.dev
        lens = ctx.lenses.get(plan.audience)
        lab = f"{ctx.lab.name} (tier {ctx.lab.tier})" if ctx.lab is not None else "unattributed"
        classes = ", ".join(sorted({d.source_class for d in ctx.documents.values()})) or "unknown"
        status = "UNCONFIRMED" if it.unconfirmed else "confirmed"
        alert = (f" | alert: {it.alert} " + ("(pushed)" if it.alert in PUSH else "(digest only)")) if it.alert else ""
        lines += ["", f"[{it.n}] {dev.title}",
                  (f"type: {dev.event_type} | lab: {lab} | first seen: {dev.first_seen_at:%Y-%m-%d} | "
                   f"relevance: {it.p:.2f} | {status} | sources: {classes} ({len(ctx.documents)}){alert}")]
        if ctx.people:
            lines.append("people: " + ", ".join(f"{p.name} (tier {p.tier})" for p in ctx.people))
        if plan.audience == "investment" and (ticks := item_tickers(plan, it)):
            lines.append("tickers: " + "; ".join(f"{sym} ({desc})" for sym, desc in ticks))
        if plan.audience == "ai_team" and dev.artifacts:
            lines.append("artifacts: " + ", ".join(_clip(str(a.get("name", a)) if isinstance(a, dict) else str(a), 80)
                                                 for a in dev.artifacts[:6]))
        if lens is not None:
            lines += [f"lens: {lens.headline}", f"so what: {lens.so_what}",
                      f"suggested: {lens.what_to_do}" + (f" (action: {lens.action})" if lens.action else "")]
        elif dev.summary:
            lines.append(f"summary: {_clip(dev.summary, 600)}")
        for quote, doc in ctx.evidence(2):
            lines.append(f'evidence: "{_clip(quote, 300)}"' + (f" ({doc.source_class})" if doc is not None else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------------- generation
def output_issues(plan: Plan, data: Any) -> str | None:
    """Why a model output cannot be published, or None when it can: it needs at least one entry
    with its own text citing a valid item, and must cite >= MIN_COVERAGE of the selected items
    (seen live: a headline plus one empty entry, and a digest citing 2 of 12 items)."""
    if not isinstance(data, dict):
        return "output is not an object"
    valid = {it.n for it in plan.items}
    entries = [e for key in SECTIONS[plan.audience] for e in _entries(data.get(key), valid, plan, backfill=False)]
    if not entries:
        return "no usable entry (empty texts or no valid citations)"
    cited = {n for e in entries for n in e["cited"]}
    if len(cited) < MIN_COVERAGE * len(valid):
        return f"cites {len(cited)} of {len(valid)} selected items (minimum {MIN_COVERAGE:.0%})"
    return None


def _validator(plan: Plan) -> Callable[[dict], bool]:
    """LLMGateway.structured(validate=...): a rejected output is never cached; the reason is kept."""
    def check(data: dict) -> bool:
        plan.rejected = output_issues(plan, data)
        return plan.rejected is None
    return check


async def _gather(gateway: LLMGateway, calls: list[dict]) -> list[Any]:
    return await asyncio.gather(*(gateway.structured(**c) for c in calls), return_exceptions=True)


def _batch(gateway: LLMGateway, todo: list[Plan], calls: list[dict]) -> list[Any]:
    results: dict[int, Any] = {}
    for audience in AUDIENCES:
        idx = [i for i, p in enumerate(todo) if p.audience == audience]
        if not idx:
            continue
        c0 = calls[idx[0]]
        items = [(f"{audience}-{todo[i].start:%Y%m%d}", calls[i]["user"], calls[i]["ref"]) for i in idx]
        try:
            out = gateway.structured_batch(purpose="digest", prompt=c0["prompt"], items=items, schema=c0["schema"],
                                           model=c0["model"], max_tokens=c0["max_tokens"], effort=c0["effort"])
        except Exception as exc:  # noqa: BLE001 - whole batch failed: every plan in it falls back
            out = {cid: exc for cid, _, _ in items}
        for i, (cid, _, _) in zip(idx, items, strict=True):
            results[i] = out.get(cid, RuntimeError("no batch result"))
    return [results[i] for i in range(len(todo))]


def _accept(plan: Plan, res: Any) -> str | None:
    """Take a usable result into the plan; otherwise return why it was not usable. Results are
    re-validated here because a cached response comes back without passing the gateway's hook."""
    if isinstance(res, BaseException):
        return f"{res.__class__.__name__}: {res}"
    plan.cost += res.cost_usd or 0.0
    if not res.ok:
        if res.status == "invalid" and plan.rejected:  # refused by our own validate hook
            why, plan.rejected = f"rejected: {plan.rejected}", None
            return why
        return f"{res.status}: {res.error}"
    if (issue := output_issues(plan, res.data)) is not None:
        return f"rejected: {issue}"
    plan.output, plan.model = res.data, res.model
    return None


def generate(plans: list[Plan], settings: Settings, use_batch: bool = False) -> None:
    """Fill plan.output from the model; an unusable output is retried once on a different model
    (settings.model_digest_retry, uncached), and failures leave output None (fallback renderer).
    An identical resend is pointless: the failure seen live (the model drafts the digest in its
    thinking, then emits one empty entry) repeats deterministically for a given week."""
    todo = [p for p in plans if p.items]
    if not todo:
        return
    try:
        gateway = LLMGateway(settings)
    except Exception as exc:  # noqa: BLE001 - e.g. no API key
        for p in todo:
            p.note = f"model unavailable: {exc}"
        return
    calls = [{"purpose": "digest", "prompt": load_prompt(f"digest_{p.audience}"), "user": render_payload(p),
              "schema": SCHEMAS[p.audience], "model": settings.model_digest, "max_tokens": 12_000, "effort": DIGEST_EFFORT,
              "ref": {"audience": p.audience, "period_start": p.start.date().isoformat()},
              "validate": _validator(p)} for p in todo]
    results = _batch(gateway, todo, calls) if use_batch else asyncio.run(_gather(gateway, calls))
    retry: list[tuple[Plan, dict]] = []
    for plan, call, res in zip(todo, calls, results, strict=True):
        if (why := _accept(plan, res)) is not None:
            plan.note = why
            if not isinstance(res, BaseException):  # an exception (budget stop) would only repeat
                retry.append((plan, {**call, "model": settings.model_digest_retry, "use_cache": False}))
    if retry:
        again = asyncio.run(_gather(gateway, [c for _, c in retry]))
        for (plan, _), res in zip(retry, again, strict=True):
            if (why := _accept(plan, res)) is None:
                plan.note = f"first output unusable ({plan.note}); retry on {plan.model} succeeded"
            else:
                plan.note = f"{plan.note}; retry: {why}"
    for plan in todo:
        if plan.output is None:
            log.warning("digest %s %s: model output unusable (%s); using fallback", plan.audience,
                        plan.start.date(), plan.note)


def _fallback_entry(plan: Plan, it: Item, **extra: Any) -> dict:
    lens = it.ctx.lenses.get(plan.audience)
    text = f"{lens.so_what} {lens.what_to_do}" if lens is not None else (it.ctx.dev.summary or it.ctx.dev.title)
    return {"items": [it.n], "title": lens.headline if lens is not None else it.ctx.dev.title,
            "text": f"{text} [{it.n}]", **extra}


def fallback_output(plan: Plan) -> dict:
    """Deterministic sections from lenses: same shape as the model output, same citation checks."""
    if plan.audience == "investment":
        confirmed = [it for it in plan.items if not it.unconfirmed]
        themes: dict[str, list[Item]] = {}
        for it in confirmed[3:]:
            theme = next((x.theme for x in it.ctx.exposures if x.theme), None)
            themes.setdefault(theme_names().get(theme or "", theme or "Other developments"), []).append(it)
        return {"headline": "",
                "top_of_mind": [_fallback_entry(plan, it, tickers=[s for s, _ in item_tickers(plan, it)][:4],
                                                action=getattr(it.ctx.lenses.get("investment"), "action", None)
                                                or "watch")
                                for it in confirmed[:3]],
                "by_theme": [{"items": [i.n for i in its], "title": theme,
                              "text": " ".join(f"{_clip(i.ctx.dev.title, 160)} [{i.n}]." for i in its)}
                             for theme, its in themes.items()],
                "watchlist": [_fallback_entry(plan, it) for it in plan.items if it.unconfirmed]}
    radar, papers, tools = [], [], []
    for it in plan.items:
        if it.ctx.dev.event_type == "research_paper":
            papers.append(_fallback_entry(plan, it))
        elif it.ctx.dev.event_type == "technique":
            tools.append(_fallback_entry(plan, it))
        else:
            action = getattr(it.ctx.lenses.get("ai_team"), "action", None)
            ring = action if action in AI_TEAM_ACTIONS else ("hold" if it.unconfirmed else "assess")
            radar.append(_fallback_entry(plan, it, ring=ring))
    return {"headline": "", "radar": radar, "papers": papers, "tools_techniques": tools}


# --------------------------------------------------------------------------------- rendering
def clean_citations(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Drop citation numbers outside `valid`, strip model-written links, normalise to [n][m]."""
    used: list[int] = []

    def repl(m: re.Match) -> str:
        keep = [int(x) for x in re.split(r"\s*[,;]\s*", m.group(1)) if int(x) in valid]
        used.extend(keep)
        return "".join(f"[{n}]" for n in dict.fromkeys(keep))

    text = BARE_URL.sub("", MD_LINK.sub(r"\1", text or ""))
    text = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", CITE.sub(repl, text))  # [3] [3] -> [3]
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text))
    return text.strip(), list(dict.fromkeys(used))


def _lens_text(plan: Plan | None, n: int) -> str:
    """Deterministic text for item n (lens so-what + action, else the development summary)."""
    it = next((i for i in (plan.items if plan else []) if i.n == n), None)
    if it is None:
        return ""
    lens = it.ctx.lenses.get(plan.audience)
    return f"{lens.so_what} {lens.what_to_do}" if lens is not None else (it.ctx.dev.summary or "")


def _entries(raw: Any, valid: set[int], plan: Plan | None = None, *, backfill: bool = True) -> list[dict]:
    """Entries that cite at least one valid item, with cleaned citations. An entry without its own
    text (fewer than three words) is backfilled from the lens, or skipped when backfill=False: for
    validation such an entry is not the model's work."""
    out = []
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        text, used = clean_citations(str(e.get("text", "")), valid)
        declared = [n for n in (e.get("items") or []) if isinstance(n, int) and n in valid]
        if len(re.findall(r"[A-Za-z]{2,}", CITE.sub("", text))) < 3 and (declared or used):
            if not backfill:
                continue
            text = _clip(_lens_text(plan, (used or declared)[0]), 600) + " " + text  # empty text: lens backfill
        if not used and declared:  # the model listed its items but forgot inline citations
            text, used = f"{text.strip()} " + "".join(f"[{n}]" for n in declared), declared
        if used:  # an entry that cites nothing valid is an unsupported claim: dropped
            out.append({**e, "title": clean_citations(str(e.get("title", "")), valid)[0], "text": text.strip(),
                        "cited": used})
    return out


def _investment_sections(plan: Plan, out: dict, valid: set[int]) -> list[str]:
    top, themes, watch = (_entries(out.get(k), valid, plan) for k in ("top_of_mind", "by_theme", "watchlist"))
    moved = [e for e in top if set(e["cited"]) <= plan.unconfirmed]  # unconfirmed never tops the digest
    top = [e for e in top if e not in moved]
    themes += [{**e, "title": e["title"] or "Other"} for e in top[3:]]
    watch, top = moved + watch, top[:3]
    allowed = {it.n: {s for s, _ in item_tickers(plan, it)} for it in plan.items}
    lines = ["## Top of mind", ""]
    for e in top:
        ticks = [t.upper() for t in e.get("tickers") or [] if any(t.upper() in allowed[n] for n in e["cited"])]
        action = e.get("action") if e.get("action") in INVESTMENT_ACTIONS else None
        meta = "; ".join(x for x in (f"Tickers: {', '.join(dict.fromkeys(ticks))}" if ticks else "",
                                     f"Suggested action: {action.replace('_', ' ')}" if action else "") if x)
        lines += [f"### {e['title'] or 'Development'}", "", e["text"], ""] + ([f"_{meta}_", ""] if meta else [])
    lines += [] if top else [NOTHING, ""]
    lines += ["## By theme / holding", ""]
    for e in themes:
        lines += [f"**{e['title'] or 'Other'}**: {e['text']}", ""]
    lines += [] if themes else [NOTHING, ""]
    lines += ["## Watchlist (unconfirmed)", ""] + ([f"- **{e['title']}**: {e['text']}" for e in watch] or [NOTHING])
    return lines + [""]


def _ring(entry: dict, unconfirmed: set[int]) -> str:
    """Radar ring from the model, defaulting to assess; unconfirmed items never land in adopt/trial."""
    ring = entry.get("ring") if entry.get("ring") in AI_TEAM_ACTIONS else "assess"
    return "assess" if ring in ("adopt", "trial") and set(entry["cited"]) <= unconfirmed else ring


def _ai_sections(plan: Plan, out: dict, valid: set[int]) -> list[str]:
    radar = _entries(out.get("radar"), valid, plan)
    lines = ["## Radar", ""]
    for ring in AI_TEAM_ACTIONS:
        entries = [e for e in radar if _ring(e, plan.unconfirmed) == ring]
        if entries:
            lines += [f"**{ring.capitalize()}**", ""] + [f"- **{e['title']}**: {e['text']}" for e in entries] + [""]
    lines += [] if radar else [NOTHING, ""]
    for key, heading in (("papers", "Papers worth reading"), ("tools_techniques", "Tools and techniques")):
        entries = _entries(out.get(key), valid, plan)
        lines += [f"## {heading}", ""] + ([f"- **{e['title']}**: {e['text']}" for e in entries] or [NOTHING]) + [""]
    return lines


def _also_lines(plan: Plan, items: list[Item]) -> list[str]:
    """Selected items the written sections left uncited, listed deterministically from the lens."""
    if not items:
        return []
    lines = ["## Also this week", ""]
    for it in items:
        lens, dev = it.ctx.lenses.get(plan.audience), it.ctx.dev
        title = lens.headline if lens is not None else dev.title
        text = f"{lens.so_what} {lens.what_to_do}" if lens is not None else (dev.summary or dev.title)
        text, _ = clean_citations(text, set())  # lens text carries no citations or links of its own
        tags = [t for t in ("unconfirmed" if it.unconfirmed else "",
                            f"{it.alert} alert" if it.alert else "",
                            (lens.action or "").replace("_", " ") if lens is not None else "") if t]
        meta = f" ({'; '.join(tags)})" if tags else ""
        lines.append(f"- **{md_label(_clip(title, 160))}**{meta}: {_clip(text, 600)} [{it.n}]")
    return lines + [""]


def _directions(plan: Plan, it: Item, symbol: str) -> set[str]:
    """Directions the item's lens commits to for `symbol` (none when the lens abstained)."""
    lens = it.ctx.lenses.get(plan.audience)
    if lens is None or lens.action == "no_action":
        return set()
    return {str(t["direction"]) for t in lens.tickers or [] if isinstance(t, dict)
            and str(t.get("symbol", "")).upper() == symbol and t.get("direction") in DIRECTIONAL}


def holdings_check(plan: Plan) -> list[str]:
    """One line per BIT holding (largest first): net direction of the lens tickers on the items that
    expose it and the top item, else 'no new development' or 'no source coverage'."""
    lines = []
    for sym, weight in sorted(plan.holdings.items(), key=lambda kv: (-kv[1], kv[0])):
        touched = [it for it in plan.items if sym in (it.facts.get("held_tickers") or [])]
        if touched:
            directions = set().union(*(_directions(plan, it, sym) for it in touched))
            net = (next(iter(directions)) if len(directions) == 1 else "mixed") if directions else "direction not assessed"
            top = max(touched, key=lambda it: (not it.unconfirmed, it.p, -it.n))
            lens = top.ctx.lenses.get(plan.audience)
            title = md_label(_clip(lens.headline if lens is not None else top.ctx.dev.title, 140))
            others = "".join(f"[{it.n}]" for it in touched if it is not top)
            text = (f"{net}. Top item: {title}{' (unconfirmed)' if top.unconfirmed else ''} [{top.n}]"
                    + (f"; also {others}" if others else "") + ".")
        elif sym in plan.covered:
            text = "no new development."
        else:
            text = "no source coverage (no document collected this week names it or comes from a lab linked to it)."
        lines.append(f"- **{sym}** ({weight:.1f}%): {text}")
    return lines


def _holdings_lines(plan: Plan) -> list[str]:
    if not plan.holdings:
        return []
    return ["## Holdings check", "", *holdings_check(plan), ""]


def _labs_lines(plan: Plan) -> list[str]:
    quiet = ("No material developments this week from: " + _lab_groups(plan.quiet_labs) + "." if plan.quiet_labs
             else "Every tracked lab with documents this week had at least one material development.")
    lines = ["## Quiet labs", "", quiet]
    if plan.unmonitored_labs:
        lines += ["", "Not monitored this week (no documents collected, so silence is not evidence): "
                  + _lab_groups(plan.unmonitored_labs) + "."]
    return lines + [""]


def _numbers(lines: list[str], valid: set[int]) -> set[int]:
    return {int(m) for m in re.findall(r"\[(\d+)\]", "\n".join(lines)) if int(m) in valid}


def compose(plan: Plan) -> tuple[str, str, set[int], str | None, set[int]]:
    """(title, markdown body, cited item numbers, model, items appended under 'Also this week')."""
    end = plan.start + timedelta(days=6)
    y, w, _ = plan.start.isocalendar()
    title = f"{LABELS[plan.audience]} digest: {plan.start:%d %b} to {end:%d %b %Y} ({y}-W{w:02d})"
    investment = plan.audience == "investment"
    if not plan.items:
        body = [f"# {title}", "", f"{NOTHING} No verified development reached the materiality bar for this audience.", ""]
        if investment:
            body += _holdings_lines(plan) + _labs_lines(plan)
        return title, "\n".join(body).rstrip() + "\n", set(), None, set()
    valid, model = {it.n for it in plan.items}, plan.model
    out = plan.output
    if out is None or output_issues(plan, out) is not None:
        out, model = fallback_output(plan), FALLBACK  # no usable model output: never publish an empty digest
    render = _investment_sections if investment else _ai_sections
    headline, _ = clean_citations(str(out.get("headline", "")), valid)
    lines = [f"# {title}", ""] + ([f"*{headline}*", ""] if headline else []) + render(plan, out, valid)
    written = _numbers(lines, valid)
    also = [it for it in plan.items if it.n not in written]  # never silently dropped
    lines += _also_lines(plan, also)
    if investment:
        lines += _holdings_lines(plan) + _labs_lines(plan)
    body = "\n".join(lines)
    cited = _numbers(lines, valid)
    refs = [f"- [{it.n}] {md_label(it.ctx.dev.title)}{' (unconfirmed)' if it.unconfirmed else ''} "
            f"({it.ctx.lab.name if it.ctx.lab else 'unattributed'}, {it.ctx.dev.first_seen_at:%Y-%m-%d}): "
            + ("; ".join(source_ref(d) for d in it.ctx.links(3)) or "no source link")
            for it in plan.items if it.n in cited]
    body = body.rstrip() + "\n\n## Sources\n\n" + "\n".join(refs) + "\n"
    return title, body, cited, model, {it.n for it in also}


# --------------------------------------------------------------------------------- stage
def _save(s, plan: Plan, now: datetime) -> dict:
    title, body, cited, model, also = compose(plan)
    rep = s.scalar(select(Report).where(Report.audience == plan.audience, Report.period_start == plan.start))
    status = "created" if rep is None else "updated"
    if rep is None:
        rep = Report(audience=plan.audience, period_start=plan.start)
        s.add(rep)
    rep.period_end, rep.title, rep.body_md, rep.model, rep.created_at = (
        plan.start + timedelta(days=7), title, body, model, now)
    rep.items = [{"development_id": it.ctx.dev.id, "rank": it.n, "p_material": round(it.p, 4),
                  "cited": it.n in cited, "also_this_week": it.n in also, "alert": it.alert,
                  "unconfirmed": it.unconfirmed, "title": it.ctx.dev.title[:200]}
                 for it in plan.items]
    log.info("digest %s %s: %s (%d items, %d cited, %d also-this-week, model=%s)", plan.audience, plan.start.date(),
             status, len(plan.items), len(cited), len(also), model)
    return {"audience": plan.audience, "period_start": plan.start.date().isoformat(), "status": status,
            "items": len(plan.items), "cited": len(cited), "also_this_week": len(also), "model": model,
            **({"note": plan.note} if plan.note else {})}


def _shown(rep: Report) -> set[int]:
    """Developments a report actually cited (legacy rows without the flag count as cited)."""
    return {int(i["development_id"]) for i in rep.items or []
            if isinstance(i, dict) and str(i.get("development_id", "")).isdigit() and i.get("cited", True)}


def run_digest(audience: str = "all", week: str | date | None = None, backfill: bool = False, force: bool = False,
               use_batch: bool = False, now: datetime | None = None, **_kwargs: Any) -> dict:
    """Generate the weekly report(s). Idempotent: one Report per (audience, week). A final
    report (model-written, generated after its week ended) is kept unless force=True; fallback
    reports and mid-week previews are provisional and regenerated."""
    settings, policy, now = get_settings(), load_policy(), now or utcnow()
    audiences = list(AUDIENCES) if audience in (None, "", "all") else [audience]
    if any(a not in AUDIENCES for a in audiences):
        raise ValueError(f"unknown audience {audience!r}; use investment, ai_team or all")
    weeks = periods(week, backfill, settings, now)
    plans, results = [], []
    with session_scope() as s:  # developments already cited by earlier reports are not repeated
        reported = {a: {d for r in s.scalars(select(Report).where(Report.audience == a, Report.period_start < weeks[0]))
                        for d in _shown(r)} for a in audiences}
    for start in weeks:
        for aud in audiences:
            with session_scope() as s:
                rep = s.scalar(select(Report).where(Report.audience == aud, Report.period_start == start))
                if rep is not None and not force and rep.model != FALLBACK and rep.created_at >= rep.period_end:
                    reported[aud] |= _shown(rep)
                    results.append({"audience": aud, "period_start": start.date().isoformat(), "status": "exists"})
                    continue
                ids = select_items(s, aud, start, policy, reported[aud])
            if ids:
                _ensure_lenses(ids, aud)
            with session_scope() as s:
                plans.append(build_plan(s, aud, start, ids, policy))
            # every selected item is cited by the composed report (written sections or "Also this
            # week"), so the later weeks of this run can treat them as reported before generation
            reported[aud] |= set(ids)
    generate(plans, settings, use_batch=use_batch)
    with session_scope() as s:
        results += [_save(s, plan, now) for plan in plans]
    return {"weeks": len(weeks), "reports": results, "cost_usd": round(sum(p.cost for p in plans), 4)}
