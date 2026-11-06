"""Read models for the web UI and the JSON API.

All SQL the web layer runs lives here; templates only format. Functions take an open Session and
return plain dicts (datetimes stay naive UTC), so nothing lazy-loads after the request's session
closes and the same shapes feed the HTML pages and the JSON endpoints. Related rows for a page of
developments are loaded with a fixed number of IN-queries, never one query per card.

Shapes written by other stages (score contributions, evidence verification, alert rules, lens
tickers, lens checks, delivery state) are read defensively: unknown layouts degrade to a plainer
display instead of a 500.

The score shown is 'relevance': the scorer estimates P(the audience would call it worth knowing),
not P(material). Per (development, audience) the web reads the row of the live model version (the
one alerts, digests and the lens read), falling back to the newest row by created_at when that
version has not scored the development yet or the live version cannot be determined. The live
version is read from the 'version' field of data/models/scoring_<audience>.json (cached by file
mtime), never by importing fli.analyze.scoring, which would load scikit-learn and scipy into every
web process.

Holding exposure has one definition, the alert rules' own (fli.deliver.materiality): a ticker counts
when an event-specific exposure link (`event_specific_links`: not a lab-level relation, at least
`exposure_min_confidence`) or the investment lens's committed direction (`directional_lens_tickers`)
names it. The BIT-holdings filter, the holdings tiles, the ticker chips and the alert routing fields
all use it; a lab's curated relations (OpenAI -> MSFT) appear only as 'lab relation' context.

'Why flagged' is materiality.explain() per audience (fired rules with the facts they matched, guards),
plus every rule that did not fire with its first failing condition and the top score contributions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import String, and_, case, cast, false, func, or_, select
from sqlalchemy.orm import Session, aliased, defer

from fli.config import DATA_DIR, get_settings
from fli.db.models import (
    Affiliation,
    Alert,
    ApiUsage,
    Development,
    Document,
    ExposureLink,
    Identity,
    Insight,
    JobRun,
    Lab,
    Label,
    Lens,
    LLMCall,
    OpsEvent,
    Person,
    Report,
    Score,
    Source,
    Ticker,
)
from fli.deliver import materiality as mat
from fli.taxonomy import AUDIENCES
from fli.timeutil import utcnow, week_start

log = logging.getLogger("fli.web")

Row = dict[str, Any]
HIDDEN_DEV_STATUSES = ("merged", "duplicate", "deleted")
# Curated register facts: analyst-maintained entries (config/*.yaml), each citing a source URL. They
# are shown as such and linked to what they cite, never as an official lab source.
CURATED_CLASS = "curated"
# A development counts as confirmed once something other than personal posts / aggregators carries it
# (a curated register entry cites its source, so it confirms; it is still never 'official').
CONFIRMING_CLASSES = frozenset({"official", "research", "filing", "press", "market_data", CURATED_CLASS})
X_PROVIDERS = ("x", "twitter", "x_api")
RUBRIC_VERSION = "r1"
FAILING_AFTER = 2  # consecutive failures before a source counts as failing (one blip is noise)
STALE_INGEST_HOURS = 24  # the scheduler polls every few hours; a day of silence means it is down
LLM_ERROR_STATUSES = ("error", "invalid", "refusal", "max_tokens")
PRIOR_VERSION = "prior-v0"  # = fli.analyze.scoring.PRIOR_VERSION: hand-set day-one weights, never fitted
MODELS_DIR = DATA_DIR / "models"  # = fli.analyze.scoring.MODELS_DIR (not imported: it loads scikit-learn)
FACT_HELP = mat.FACTS  # what each materiality fact means (the 'why flagged' facts table)
# Feed defaults. Investment relevance below 0.35 is mostly SDK bumps and grants, and no lens is written
# under it (deliver.lens.LENS_MIN_P); the AI-team score sits in 0.5-0.9 anyway, so it gets no floor.
DEFAULT_MIN_P = {"investment": 0.35}
DISMISSED = {"investment": ("no_action",), "ai_team": ("hold",)}  # the lens's own "ignore this"
ACTION_RANK = {"investment": {"review_position": 4, "update_thesis": 4, "ask_company": 3, "watch": 2,
                              "no_action": 0},
               "ai_team": {"adopt": 4, "trial": 3, "assess": 2, "hold": 0}}
NO_LENS_RANK = 1  # not analysed yet: below anything the lens recommends, above what it dismissed
PUSH_SEVERITIES = ("P1", "P2")
DIRECTIONAL = ("positive", "negative", "mixed")


# ------------------------------------------------------------------------------------ helpers
def _visible():
    return or_(Development.status.is_(None), Development.status.not_in(HIDDEN_DEV_STATUSES))


def _like_json(col, value: str):
    """Portable 'JSON value contains this string' (SQLite TEXT and Postgres JSONB::text)."""
    esc = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return cast(col, String).like(f'%"{esc}"%', escape="\\")


@lru_cache(maxsize=16)
def _model_file_version(path: str, _mtime_ns: int, audience: str) -> str | None:
    """The 'version' field of a deployed model file, or None when the file is not a usable model for
    `audience` (unreadable, invalid JSON, another audience's model, no version). Cached per path and
    modification time: `fli train` replaces the file atomically, which changes the mtime."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or str(data.get("audience", "")) != audience:
        return None
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, str | int) or not str(version):
        return None
    return str(version)


def deployed_version(audience: str, models_dir: Path | None = None) -> str | None:
    """Version of the scoring model deployed for `audience` (data/models/scoring_<audience>.json):
    PRIOR_VERSION when none is deployed (no file: scoring uses the hand-set prior), None when a file
    exists but cannot be used (scoring halts for that audience; see fli.analyze.scoring.load_model)."""
    path = Path(models_dir or MODELS_DIR) / f"scoring_{audience}.json"
    try:
        mtime_ns = path.stat().st_mtime_ns
    except FileNotFoundError:
        return PRIOR_VERSION
    except OSError:
        return None
    return _model_file_version(str(path), mtime_ns, audience)


def live_versions(s: Session | None = None, models_dir: Path | None = None) -> dict[str, str]:
    """Live scoring model version per audience: the rule of fli.analyze.scoring.current_versions()
    (the deployed model's version once it has Score rows, else the version of the audience's newest
    row, else prior-v0), with the deployed version read from the model file's 'version' field. That
    keeps scikit-learn and scipy out of the web process. Without a session only the files are read.
    Model-file problems never raise: the page shows the newest rows instead."""
    out: dict[str, str] = {}
    for aud in AUDIENCES:
        deployed = deployed_version(aud, models_dir)
        if s is None:
            if deployed:
                out[aud] = deployed
            continue
        if deployed and s.scalar(select(Score.id).where(Score.audience == aud, Score.model_version == deployed)
                                 .limit(1)) is not None:
            out[aud] = deployed
            continue
        newest = s.scalar(select(Score.model_version).where(Score.audience == aud)
                          .order_by(Score.created_at.desc(), Score.id.desc()).limit(1))
        out[aud] = newest or deployed or PRIOR_VERSION
    return out


def _ranked_scores(live: dict[str, str], audience: str | None = None, dev_ids=None):
    """Score rows numbered per (development, audience): rn = 1 is the one to show, i.e. the live
    model version's row if there is one, else the newest by created_at (then id)."""
    s2 = aliased(Score)
    order = [s2.created_at.desc(), s2.id.desc()]
    live = {a: v for a, v in live.items() if audience is None or a == audience}
    if live:
        is_live = case(*[(and_(s2.audience == a, s2.model_version == v), 1) for a, v in live.items()], else_=0)
        order.insert(0, is_live.desc())
    rn = func.row_number().over(partition_by=(s2.development_id, s2.audience), order_by=order)
    q = select(s2.id.label("sid"), s2.development_id.label("dev_id"), rn.label("rn"))
    if audience:
        q = q.where(s2.audience == audience)
    if dev_ids is not None:
        q = q.where(s2.development_id.in_(dev_ids))
    return q.subquery("ranked_score")


def _latest_score_ids(live: dict[str, str], audience: str | None = None, dev_ids=None):
    """Ids of the score rows to show, one per (development, audience) (see _ranked_scores)."""
    ranked = _ranked_scores(live, audience, dev_ids)
    return select(ranked.c.sid).where(ranked.c.rn == 1)


def _with_latest_score(q, audience: str, live: dict[str, str]):
    """LEFT JOIN each development to its shown score row for one audience.

    Joins a one-row-per-development subquery (materialised once), then `score` by primary key.
    An `id IN (subquery)` inside the ON clause instead made SQLite probe the whole id list per
    development: 2.4 s vs ~0.1 s for a 3,000-development feed."""
    ranked = _ranked_scores(live, audience)
    latest = select(ranked.c.dev_id, ranked.c.sid).where(ranked.c.rn == 1).subquery("latest_score")
    return q.outerjoin(latest, latest.c.dev_id == Development.id).outerjoin(Score, Score.id == latest.c.sid)


def _names(values) -> list[str]:
    out = []
    for v in values or []:
        name = (v.get("display_name") or v.get("name")) if isinstance(v, dict) else v
        if name:
            out.append(str(name))
    return out


def span(td: timedelta) -> str:
    hours = int(td.total_seconds() // 3600)
    return f"{hours // 24}d {hours % 24}h" if hours >= 24 else f"{max(hours, 0)}h"


def dev_titles(s: Session, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    return dict(s.execute(select(Development.id, Development.title).where(Development.id.in_(ids))).all())


def lab_names(s: Session) -> dict[str, str]:
    return dict(s.execute(select(Lab.id, Lab.name)).all())


def lab_choices(s: Session) -> list[tuple[str, str]]:
    return [(i, n) for i, n in s.execute(select(Lab.id, Lab.name).order_by(Lab.tier, Lab.name))]


def bit_holdings(s: Session) -> dict[str, float]:
    """BIT holdings, symbol -> fund weight % (read as materiality.load_contexts reads them)."""
    return {sym: float(w) for sym, w in s.execute(select(Ticker.symbol, Ticker.bit_weight)
                                                  .where(Ticker.bit_weight.is_not(None)))}


def _chunked_in(col, ids):
    """`col IN ids` for any number of ids (an OR of bounded IN lists: SQLite caps bound parameters)."""
    ids = sorted(set(ids))
    if not ids:
        return false()
    return or_(*[col.in_(ids[i : i + 500]) for i in range(0, len(ids), 500)])


# ------------------------------------------------------------------------------------ holding exposure
_policy_warned = False


def materiality_policy() -> tuple[mat.Policy, str | None]:
    """The policy the alert stage uses (config/materiality.yaml, cached by mtime) and, when it cannot be
    loaded, the error: pages then explain no rules and count every non-lab link instead of failing."""
    global _policy_warned
    try:
        return mat.load_policy(), None
    except Exception as exc:  # noqa: BLE001 - PolicyError, YAML or I/O error: degrade, never a 500
        if not _policy_warned:
            log.warning("materiality policy unavailable (%s: %s)", exc.__class__.__name__, exc)
            _policy_warned = True
        return mat.Policy(rules={}), f"{exc.__class__.__name__}: {exc}"


@dataclass
class Exposure:
    """A development's tickers as the materiality rules count them (facts held_exposure / held_tickers)."""

    event: set[str] = field(default_factory=set)  # event-specific exposure links (event_specific_links)
    lens: set[str] = field(default_factory=set)  # committed by the investment lens (directional_lens_tickers)
    held: list[str] = field(default_factory=list)  # BIT holdings among them = the rules' held_tickers
    lab: list[ExposureLink] = field(default_factory=list)  # the lab's curated relations: context only

    @property
    def counted(self) -> set[str]:
        return self.event | self.lens


def exposure_of(dev: Development | None, links: list[ExposureLink], inv_lens: Lens | None,
                holdings: dict[str, float], policy: mat.Policy) -> Exposure:
    """Exposure computed by the alert rules' own functions, so a page cannot disagree with an alert."""
    ctx = mat.DevContext(dev=dev, exposures=list(links), holdings=holdings,
                         lenses={"investment": inv_lens} if inv_lens is not None else {})
    event = {x.ticker for x in mat.event_specific_links(ctx, policy)}
    lens = mat.directional_lens_tickers(ctx.lenses.get("investment"))
    return Exposure(event=event, lens=lens, held=sorted((event | lens) & set(holdings)),
                    lab=[x for x in ctx.exposures if mat.is_lab_level(x)])


def exposures(s: Session, dev_ids, policy: mat.Policy | None = None) -> dict[int, Exposure]:
    """Exposure of many developments with a fixed number of queries (holdings, developments, links,
    investment lenses), whatever the count."""
    ids = sorted(set(dev_ids))
    if not ids:
        return {}
    policy = policy or materiality_policy()[0]
    holdings = bit_holdings(s)
    devs = {d.id: d for d in s.scalars(select(Development).options(defer(Development.centroid))
                                       .where(_chunked_in(Development.id, ids)))}
    links: dict[int, list[ExposureLink]] = defaultdict(list)
    for x in s.scalars(select(ExposureLink).where(_chunked_in(ExposureLink.development_id, ids))
                       .order_by(ExposureLink.confidence.desc().nulls_last(), ExposureLink.id)):
        links[x.development_id].append(x)
    lenses = {x.development_id: x for x in s.scalars(select(Lens).where(
        Lens.audience == "investment", _chunked_in(Lens.development_id, ids)))}
    return {i: exposure_of(d, links.get(i, []), lenses.get(i), holdings, policy) for i, d in devs.items()}


def _changed_since(t: datetime):
    """The development row, or a score / lens / alert about it, is newer than t (API pollers)."""
    return or_(Development.updated_at >= t,
               Development.id.in_(select(Score.development_id).where(Score.created_at >= t)),
               Development.id.in_(select(Lens.development_id).where(Lens.created_at >= t)),
               Development.id.in_(select(Alert.development_id).where(Alert.created_at >= t)))


# ------------------------------------------------------------------------------------ feeds
@dataclass
class FeedFilters:
    audience: str = "investment"
    week: str = "7d"  # 7d | 30d | all | a Monday as YYYY-MM-DD
    lab: str | None = None
    event_type: str | None = None
    ticker: str | None = None
    text: str | None = None
    min_p: float = 0.0
    bit_only: bool = False
    since: datetime | None = None  # explicit bounds (API, overview) override `week`
    until: datetime | None = None
    updated_since: datetime | None = None
    hide_dismissed: bool = False  # drop what the audience lens dismissed (DISMISSED); counted, not lost
    order: str = "relevance"  # relevance | triage (pushed alert, then lens action, then relevance)
    limit: int = 50
    offset: int = 0


def window_bounds(week: str, now: datetime) -> tuple[datetime | None, datetime | None]:
    """'Nd' rolling window, 'all', or one calendar week starting on the given Monday."""
    if week == "all":
        return None, None
    if week.endswith("d") and week[:-1].isdigit():
        return now - timedelta(days=int(week[:-1])), None
    try:
        start = week_start(datetime.fromisoformat(week))
    except ValueError:
        return now - timedelta(days=7), None
    return start, start + timedelta(days=7)


def week_options(now: datetime) -> list[tuple[str, str]]:
    first = week_start(datetime.combine(get_settings().backfill_start, time()))
    opts = [("7d", "Last 7 days"), ("30d", "Last 30 days"), ("all", "Whole window")]
    w = week_start(now)
    while w >= first:
        opts.append((w.date().isoformat(), f"Week of {w:%d %b %Y}"))
        w -= timedelta(days=7)
    return opts


def _triage_order(audience: str, lens) -> list:
    """Pushed (P1/P2, not suppressed) alerts first, then the lens's recommended action, then relevance."""
    pushed = (select(Alert.development_id)
              .where(Alert.audience == audience, Alert.severity.in_(PUSH_SEVERITIES),
                     or_(Alert.status.is_(None), Alert.status != "suppressed")))
    keys = [case((Development.id.in_(pushed), 1), else_=0).desc()]
    if ranks := ACTION_RANK.get(audience):
        keys.append(case(ranks, value=lens.action, else_=NO_LENS_RANK).desc())
    return keys


def feed(s: Session, f: FeedFilters, now: datetime | None = None) -> Row:
    """Ranked developments for one audience (shown score per development, unscored last).

    `dismissed` counts what hide_dismissed left out, so the page can say so instead of hiding it."""
    now = now or utcnow()
    lo, hi = (f.since, f.until) if (f.since or f.until) else window_bounds(f.week, now)
    live = live_versions(s)
    lens = aliased(Lens, name="feed_lens")  # unique per (development, audience): at most one row
    q = (_with_latest_score(select(Development.id), f.audience, live)
         .outerjoin(lens, and_(lens.development_id == Development.id, lens.audience == f.audience))
         .where(_visible()))
    if lo is not None:
        q = q.where(Development.first_seen_at >= lo)
    if hi is not None:
        q = q.where(Development.first_seen_at < hi)
    if f.updated_since is not None:
        q = q.where(_changed_since(f.updated_since))
    if f.lab:
        q = q.where(Development.lab_id == f.lab)
    if f.event_type:
        q = q.where(Development.event_type == f.event_type)
    if f.text:
        q = q.where(or_(Development.title.ilike(f"%{f.text}%"), Development.summary.ilike(f"%{f.text}%")))
    if f.min_p > 0:
        q = q.where(Score.p_material >= f.min_p)
    if f.bit_only or f.ticker:  # the alert rules' exposure definition, over what the other filters keep
        sym = f.ticker.upper() if f.ticker else None
        keep = [i for i, e in exposures(s, s.scalars(q)).items()
                if (not f.bit_only or e.held) and (sym is None or sym in e.counted)]
        q = q.where(_chunked_in(Development.id, keep))
    dismissed, gone = DISMISSED.get(f.audience, ()), None
    if f.hide_dismissed and dismissed:
        gone = s.scalar(select(func.count()).select_from(q.where(lens.action.in_(dismissed)).subquery())) or 0
        q = q.where(or_(lens.action.is_(None), lens.action.not_in(dismissed)))
    total = s.scalar(select(func.count()).select_from(q.subquery())) or 0
    order = [Score.p_material.desc().nulls_last(), Development.first_seen_at.desc(), Development.id.desc()]
    if f.order == "triage":
        order = _triage_order(f.audience, lens) + order
    ids = list(s.scalars(q.order_by(*order).limit(f.limit).offset(f.offset)))
    return {"items": build_cards(s, ids, f.audience, live=live), "total": total, "since": lo, "until": hi,
            "dismissed": gone, "live_versions": live}


def latest_scores(s: Session, dev_ids, live: dict[str, str] | None = None) -> dict[tuple[int, str], Score]:
    live = live_versions(s) if live is None else live
    rows = s.scalars(select(Score).where(Score.id.in_(_latest_score_ids(live, dev_ids=dev_ids))))
    return {(r.development_id, r.audience): r for r in rows}


def _score_brief(x: Score | None, live: dict[str, str] | None = None) -> Row | None:
    """`p_material` is kept for API compatibility; `relevance` is the same number under its real name."""
    if x is None:
        return None
    return {"p_material": x.p_material, "relevance": x.p_material, "model_version": x.model_version,
            "scored_at": x.created_at, "prior": x.model_version == PRIOR_VERSION,
            "live": live.get(x.audience) == x.model_version if live else None}


def lens_warnings(checks: Any) -> list[str]:
    """Reader-facing findings of the lens stage's deterministic faithfulness checks (Lens.checks):
    numbers the lens wrote that its evidence does not contain, and a headline the check rewrote.
    Unknown layouts give no warning rather than an error."""
    if not isinstance(checks, dict):
        return []
    out = []
    nums = checks.get("unsupported_numbers")
    if isinstance(nums, dict):  # per field: {"headline": ["80%"], "so_what": [...]}
        nums = [n for v in nums.values() for n in (v if isinstance(v, list | tuple) else [v])]
    elif isinstance(nums, str | int | float) and not isinstance(nums, bool):
        nums = [nums]
    if isinstance(nums, list | tuple) and (nums := [str(n) for n in nums if n not in (None, "")]):
        shown = ", ".join(nums[:6]) + (" …" if len(nums) > 6 else "")
        out.append(f"Numbers not found in the evidence the lens was given: {shown}. Check the cited "
                   "sources before relying on them.")
    head = checks.get("headline") if isinstance(checks.get("headline"), dict) else {}
    original = next((v for v in (checks.get("original_headline"), checks.get("headline_original"),
                                 head.get("original")) if isinstance(v, str) and v.strip()), None)
    corrected = (original or checks.get("headline_corrected") or checks.get("corrected_headline")
                 or head.get("corrected"))
    if corrected:
        msg = "The headline was rewritten by the faithfulness check"
        out.append(f'{msg}; the model\'s original: "{original.strip()[:240]}"' if original else f"{msg}.")
    return out


def lens_tickers(raw) -> list[Row]:
    """Lens tickers as [{symbol, direction, why}] whatever key names the lens stage used."""
    out = []
    for t in raw or []:
        if isinstance(t, str):
            t = {"symbol": t}
        if isinstance(t, dict) and (t.get("symbol") or t.get("ticker")):
            out.append({"symbol": str(t.get("symbol") or t.get("ticker")), "direction": t.get("direction"),
                        "why": t.get("why") or t.get("rationale")})
    return out


def _lens_ticker_rows(x: Lens) -> list[Row]:
    """Lens tickers, each marked with whether the alert rules count it as exposure: for the investment
    lens, a positive / negative / mixed direction from a lens that did not abstain."""
    committed = mat.directional_lens_tickers(x) if x.audience == "investment" else set()
    rows = []
    for t in lens_tickers(x.tickers):
        counted = t["symbol"].upper() in committed
        reason = None
        if not counted and x.audience == "investment":
            reason = ("the lens abstained (no action)" if x.action == "no_action"
                      else "no committed direction" if t["direction"] not in DIRECTIONAL else "not counted")
        rows.append({**t, "counted": counted, "not_counted": reason})
    return rows


def _lens_row(x: Lens | None) -> Row | None:
    if x is None:
        return None
    return {"headline": x.headline, "so_what": x.so_what, "what_to_do": x.what_to_do, "action": x.action,
            "direction": x.direction, "horizon": x.horizon, "confidence": x.confidence,
            "tickers": _lens_ticker_rows(x), "model": x.model, "prompt_version": x.prompt_version,
            "created_at": x.created_at, "dismissed": x.action in DISMISSED.get(x.audience, ()),
            "warnings": lens_warnings(x.checks)}


def _insight_flags(s: Session, dev_ids) -> dict[int, Row]:
    flags: dict[int, Row] = defaultdict(lambda: {"rumor": False, "official": False, "classes": set()})
    rows = s.execute(select(Insight.development_id, Insight.is_rumor, Insight.is_official,
                            Document.source_class)
                     .join(Document, Document.id == Insight.document_id)
                     .where(Insight.development_id.in_(dev_ids)))
    for dev_id, rumor, official, cls in rows:
        f = flags[dev_id]
        f["rumor"] |= bool(rumor)
        f["official"] |= bool(official)
        if cls:
            f["classes"].add(cls)
    return dict(flags)


def _unconfirmed(d: Development, flag: Row | None) -> tuple[bool, str]:
    """Rumours and personal-only items get a badge: nobody should act on a single tweet."""
    flag = flag or {}
    if d.has_official or flag.get("official"):
        return False, ""
    if flag.get("rumor"):
        return True, "reported or rumoured; not confirmed by the lab"
    classes = {c for c in d.source_classes or [] if c} | flag.get("classes", set())
    if classes and not classes & CONFIRMING_CLASSES:
        return True, "only personal posts or aggregators so far"
    return False, ""


def _ticker_row(sym: str, t: Ticker | None, direction: str | None, why: str | None, theme: str | None,
                via: str, basis: str) -> Row:
    return {"symbol": sym, "company": t.company if t else None, "bit_weight": t.bit_weight if t else None,
            "direction": direction or "unclear", "why": why, "theme": theme, "via": via, "basis": basis}


def ticker_chips(e: Exposure, inv_lens: Lens | None, links: list[ExposureLink],
                 tickers: dict[str, Ticker]) -> tuple[list[Row], list[Row]]:
    """(exposure, lab relations) for display. Exposure is exactly what the alert rules count: the
    investment lens's committed tickers (with direction and why), then event-specific links, BIT holdings
    first. Lab relations are the lab's curated links to tickers not otherwise exposed: context only."""
    entries = [(t["symbol"].upper(), t["direction"], t["why"], None, "lens", "lens")
               for t in lens_tickers(inv_lens.tickers if inv_lens is not None else None)
               if t["symbol"].upper() in e.lens]
    entries += [(x.ticker, x.direction, x.rationale, x.theme, x.method or "rule", "event")
                for x in links if x.ticker in e.event]
    exposure, seen = [], set()
    for sym, direction, why, theme, via, basis in entries:
        if sym not in seen:
            seen.add(sym)
            exposure.append(_ticker_row(sym, tickers.get(sym), direction, why, theme, via, basis))
    lab = []
    for x in e.lab:
        if x.ticker not in seen:
            seen.add(x.ticker)
            lab.append(_ticker_row(x.ticker, tickers.get(x.ticker), None, x.rationale, x.theme, "lab",
                                   "lab_relation"))
    for rows in (exposure, lab):  # stable: non-holdings keep lens / confidence order
        rows.sort(key=lambda r: -(r["bit_weight"] or 0.0))
    return exposure, lab


def cited_url(url: str | None, meta: Any) -> str | None:
    """The source a curated register entry cites: meta 'source_url' (as the register collector writes
    it; or 'cited_url', or the first of 'sources' / 'source_urls'), else the document's own URL."""
    m = meta if isinstance(meta, dict) else {}
    for key in ("source_url", "cited_url"):
        if isinstance(m.get(key), str) and m[key].strip():
            return m[key].strip()
    for key in ("sources", "source_urls"):
        values = m.get(key) if isinstance(m.get(key), list) else []
        if first := next((u.strip() for u in values if isinstance(u, str) and u.strip()), None):
            return first
    return url or None


def _curated_sources(s: Session, dev_ids: list[int]) -> dict[int, list[Row]]:
    """Curated register entries behind each development, with the URL each one cites."""
    out: dict[int, list[Row]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    rows = s.execute(select(Insight.development_id, Document.id, Document.url, Document.title, Document.meta)
                     .join(Document, Document.id == Insight.document_id)
                     .where(Insight.development_id.in_(dev_ids), Document.source_class == CURATED_CLASS)
                     .order_by(Document.id))
    for dev_id, doc_id, url, title, meta in rows:
        if (dev_id, doc_id) not in seen:
            seen.add((dev_id, doc_id))
            out[dev_id].append({"title": title, "cited_url": cited_url(url, meta)})
    return dict(out)


def build_cards(s: Session, dev_ids: list[int], audience: str | None = None,
                live: dict[str, str] | None = None) -> list[Row]:
    """Card / API view of developments, in the given order."""
    if not dev_ids:
        return []
    devs = {d.id: d for d in s.scalars(select(Development).where(Development.id.in_(dev_ids)))}
    labs = lab_names(s)
    tickers = {t.symbol: t for t in s.scalars(select(Ticker))}
    holdings = {sym: float(t.bit_weight) for sym, t in tickers.items() if t.bit_weight is not None}
    live = live_versions(s) if live is None else live
    scores = latest_scores(s, dev_ids, live)
    lenses = {(x.development_id, x.audience): x
              for x in s.scalars(select(Lens).where(Lens.development_id.in_(dev_ids)))}
    links: dict[int, list[ExposureLink]] = defaultdict(list)
    for e in s.scalars(select(ExposureLink).where(ExposureLink.development_id.in_(dev_ids))
                       .order_by(ExposureLink.confidence.desc().nulls_last(), ExposureLink.id)):
        links[e.development_id].append(e)
    alerts: dict[int, list[Row]] = defaultdict(list)
    for a in s.scalars(select(Alert).where(Alert.development_id.in_(dev_ids)).order_by(Alert.severity)):
        alerts[a.development_id].append({"id": a.id, "severity": a.severity, "audience": a.audience,
                                         "status": a.status})
    flags = _insight_flags(s, dev_ids)
    curated = _curated_sources(s, dev_ids)
    policy = materiality_policy()[0]
    out = []
    for i in dev_ids:
        d = devs.get(i)
        if d is None:
            continue
        brief = {aud: _score_brief(scores.get((i, aud)), live) for aud in AUDIENCES}
        # the score a card shows: its audience's, or (API, both audiences) the higher of the two
        shown = brief[audience] if audience else max((b for b in brief.values() if b), default=None,
                                                         key=lambda b: b["p_material"] or 0.0)
        lens_by_aud = {aud: _lens_row(lenses.get((i, aud))) for aud in AUDIENCES}
        unconfirmed, why = _unconfirmed(d, flags.get(i))
        classes = {c for c in d.source_classes or [] if c} | flags.get(i, {}).get("classes", set())
        inv = lenses.get((i, "investment"))
        exp = exposure_of(d, links.get(i, []), inv, holdings, policy)
        chips, lab_relations = ticker_chips(exp, inv, links.get(i, []), tickers)
        out.append({
            "id": d.id, "title": d.title, "event_type": d.event_type, "lab_id": d.lab_id,
            "lab_name": labs.get(d.lab_id, d.lab_id) if d.lab_id else None,
            "first_seen_at": d.first_seen_at, "last_seen_at": d.last_seen_at, "updated_at": d.updated_at,
            "insight_count": d.insight_count, "source_count": d.source_count,
            "source_classes": sorted(classes),
            "has_official": d.has_official, "has_verified": d.has_verified, "summary": d.summary,
            "status": d.status, "artifacts": _names(d.artifacts)[:8], "person_ids": list(d.person_ids or []),
            "audience": audience,
            "p": shown["p_material"] if shown else None, "p_prior": bool(shown and shown["prior"]),
            "scores": brief, "lens": lens_by_aud.get(audience) if audience else None, "lenses": lens_by_aud,
            # exposure as the alert rules count it; lab relations are context, never exposure
            "tickers": chips, "held_tickers": exp.held, "lab_relations": lab_relations,
            "curated_sources": curated.get(i, []),
            "unconfirmed": unconfirmed, "unconfirmed_reason": why, "alerts": alerts.get(i, []),
        })
    return out


def dev_documents(s: Session, dev_ids: list[int], per_dev: int = 8) -> dict[int, list[Row]]:
    """Primary sources (citations) per development, oldest first."""
    out: dict[int, list[Row]] = defaultdict(list)
    if not dev_ids:
        return out
    seen: set[tuple[int, int]] = set()
    rows = s.execute(select(Insight.development_id, Document)
                     .join(Document, Document.id == Insight.document_id)
                     .where(Insight.development_id.in_(dev_ids))
                     .order_by(Document.published_at.asc().nulls_last(), Document.id))
    for dev_id, doc in rows:
        if (dev_id, doc.id) not in seen and len(out[dev_id]) < per_dev:
            seen.add((dev_id, doc.id))
            out[dev_id].append(_doc_row(doc))
    return out


def _doc_row(doc: Document, kind: str | None = None) -> Row:
    curated = doc.source_class == CURATED_CLASS
    return {"id": doc.id, "title": doc.title or doc.url, "url": doc.url, "source_id": doc.source_id,
            "source_class": doc.source_class, "source_kind": kind, "published_at": doc.published_at,
            "author": doc.author, "status": doc.status, "curated": curated,
            "cited_url": cited_url(doc.url, doc.meta) if curated else None}


# ------------------------------------------------------------------------------------ overview
def overview(s: Session, now: datetime | None = None, per_audience: int = 8) -> Row:
    """'This week' is a rolling 7 days; an idle week falls back to the latest week with data."""
    now = now or utcnow()
    since, until, note = now - timedelta(days=7), None, None
    recent = select(func.count(Development.id)).where(Development.first_seen_at >= since, _visible())
    if not s.scalar(recent):
        latest = s.scalar(select(func.max(Development.first_seen_at)).where(_visible()))
        if latest is not None:
            since, until = latest - timedelta(days=7), latest + timedelta(seconds=1)
            note = ("Nothing new in the last 7 days; showing the latest week with data "
                    f"(to {latest:%d %b %Y}).")
    # same defaults as the feed pages: what the lens dismissed stays off the front page
    top = {aud: feed(s, FeedFilters(audience=aud, since=since, until=until, limit=per_audience,
                                    min_p=DEFAULT_MIN_P.get(aud, 0.0), hide_dismissed=True, order="triage"),
                     now)
           for aud in AUDIENCES}
    return {"since": since, "until": until, "note": note, "top": top,
            "alerts": alert_rows(s, status="open", severities=("P1", "P2"), limit=12),
            "reports": report_rows(s, limit=6), "health": health_summary(s, now),
            "holdings": holdings_radar(s, since, until)}


def holdings_radar(s: Session, since: datetime, until: datetime | None) -> list[Row]:
    """Per BIT holding: developments in the window exposing it (as the alert rules count exposure),
    the most relevant one, and how many more touch it only through a lab-level relation (context)."""
    holds = list(s.scalars(select(Ticker).where(Ticker.bit_weight.is_not(None))
                           .order_by(Ticker.bit_weight.desc())))
    window = select(Development.id).where(_visible(), Development.first_seen_at >= since)
    if until is not None:
        window = window.where(Development.first_seen_at < until)
    hits: dict[str, set[int]] = defaultdict(set)
    lab_only: dict[str, set[int]] = defaultdict(set)
    for dev_id, e in exposures(s, s.scalars(window)).items():
        for sym in e.held:
            hits[sym].add(dev_id)
        for x in e.lab:
            if x.ticker not in e.counted:
                lab_only[x.ticker].add(dev_id)
    ids = sorted({i for t in holds for i in hits.get(t.symbol, ())})
    p = {k[0]: v.p_material for k, v in latest_scores(s, ids).items() if k[1] == "investment"} if ids else {}
    titles = dev_titles(s, ids)
    out = []
    for t in holds:
        devs = sorted(hits.get(t.symbol, ()), key=lambda i: (-(p.get(i) or 0.0), -i))
        top = devs[0] if devs else None
        out.append({"symbol": t.symbol, "company": t.company, "weight": t.bit_weight, "count": len(devs),
                    "lab_only": len(lab_only.get(t.symbol, ())), "max_p": p.get(top) if top else None,
                    "top_id": top, "top_title": titles.get(top)})
    return out


# ------------------------------------------------------------------------------------ development
def contribution_rows(x: Score | None) -> list[Row]:
    """'Why flagged': numeric contributions sorted by magnitude; tolerant of nested shapes."""
    if x is None:
        return []
    feats = x.features if isinstance(x.features, dict) else {}
    rows = []
    for name, val in (x.contributions or {}).items():
        if isinstance(val, dict):
            c = val.get("contribution", val.get("value"))
            v, w = val.get("feature_value", val.get("x", feats.get(name))), val.get("weight", val.get("coef"))
        else:
            c, v, w = val, feats.get(name), None
        if isinstance(c, bool) or not isinstance(c, int | float):
            continue
        rows.append({"feature": str(name), "value": v, "weight": w, "contribution": float(c)})
    rows.sort(key=lambda r: -abs(r["contribution"]))
    peak = max((abs(r["contribution"]) for r in rows), default=0.0) or 1.0
    for r in rows:
        r["width"] = round(100 * abs(r["contribution"]) / peak)
    return rows


def quote_rows(ins: Insight) -> list[Row]:
    """Evidence quotes with a per-quote verified mark (falls back to the insight-level flag)."""
    ver = ins.verification if isinstance(ins.verification, dict) else {}
    per = ver.get("quotes") or ver.get("results") or ver.get("per_quote") or []
    out = []
    for i, q in enumerate(ins.evidence_quotes or []):
        text = str(q.get("quote") or q.get("text") or "") if isinstance(q, dict) else str(q)
        mark = q.get("verified", q.get("found")) if isinstance(q, dict) else None
        if mark is None and isinstance(per, list) and i < len(per):
            item = per[i]
            mark = item if isinstance(item, bool) else (
                item.get("verified", item.get("found", item.get("ok"))) if isinstance(item, dict) else None)
        elif mark is None and isinstance(per, dict):
            mark = per.get(text, per.get(str(i)))
            mark = mark.get("verified", mark.get("found")) if isinstance(mark, dict) else mark
        out.append({"text": text, "verified": bool(ins.quotes_verified) if mark is None else bool(mark)})
    return out


def _insight_row(ins: Insight, doc_row: Row, people: dict[str, str]) -> Row:
    return {"id": ins.id, "title": ins.title, "claim": ins.claim, "event_type": ins.event_type,
            "quotes": quote_rows(ins), "quotes_verified": ins.quotes_verified, "magnitude": ins.magnitude,
            "confidence": ins.confidence, "is_official": bool(ins.is_official) and not doc_row.get("curated"),
            "is_rumor": ins.is_rumor,
            "published_at": ins.published_at, "novelty_entity": ins.novelty_entity,
            "novelty_global": ins.novelty_global, "artifacts": _names(ins.artifacts),
            "first_seen_artifacts": _names(ins.first_seen_artifacts),
            "people": [(p, people.get(p, p)) for p in ins.person_ids or []], "model": ins.model,
            "prompt_version": ins.prompt_version, "doc": doc_row}


def _exposure_rows(s: Session, dev: Development, policy: mat.Policy) -> list[Row]:
    """Every exposure link with how the alert rules treat it (`basis`): 'event' counts as exposure,
    'lab_relation' is the lab's curated relation (context only), 'below_threshold' is weaker than the
    policy's exposure_min_confidence."""
    pairs = s.execute(select(ExposureLink, Ticker).outerjoin(Ticker, Ticker.symbol == ExposureLink.ticker)
                      .where(ExposureLink.development_id == dev.id)).all()
    counted = {id(x) for x in mat.event_specific_links(mat.DevContext(dev=dev, exposures=[e for e, _ in pairs]),
                                                       policy)}
    rows = []
    for e, t in pairs:
        basis = "event" if id(e) in counted else "lab_relation" if mat.is_lab_level(e) else "below_threshold"
        rows.append({"ticker": e.ticker, "company": t.company if t else None,
                     "bit_weight": t.bit_weight if t else None, "theme": e.theme, "direction": e.direction,
                     "rationale": e.rationale, "method": e.method, "confidence": e.confidence,
                     "basis": basis, "counts": basis == "event"})
    rows.sort(key=lambda r: (not r["counts"], -(r["bit_weight"] or 0.0), -(r["confidence"] or 0.0)))
    return rows


def development_detail(s: Session, dev_id: int) -> Row | None:
    d = s.get(Development, dev_id)
    if d is None:
        return None
    card = build_cards(s, [dev_id])[0]
    rows = s.execute(select(Insight, Document, Source.kind).join(Document, Document.id == Insight.document_id)
                     .outerjoin(Source, Source.id == Document.source_id)
                     .where(Insight.development_id == dev_id)
                     .order_by(Insight.published_at.asc().nulls_last(), Insight.id)).all()
    pids = set(d.person_ids or []) | {p for ins, _, _ in rows for p in ins.person_ids or []}
    people = dict(s.execute(select(Person.id, Person.name).where(Person.id.in_(pids))).all()) if pids else {}
    docs: dict[int, Row] = {}
    insights = []
    for ins, doc, kind in rows:
        docs.setdefault(doc.id, _doc_row(doc, kind))
        insights.append(_insight_row(ins, docs[doc.id], people))
    scores = s.scalars(select(Score).where(Score.development_id == dev_id)
                       .order_by(Score.audience, Score.created_at.desc(), Score.id.desc()))
    shown = latest_scores(s, [dev_id])
    shown_ids = {x.id for x in shown.values()}
    alerts = s.scalars(select(Alert).where(Alert.development_id == dev_id).order_by(Alert.created_at.desc()))
    labels = s.scalars(select(Label).where(Label.development_id == dev_id).order_by(Label.created_at.desc()))
    policy = materiality_policy()[0]
    return {**card,
            "insights": insights, "people": sorted(people.items(), key=lambda kv: kv[1]),
            "documents": sorted(docs.values(),
                                key=lambda r: (r["published_at"] is None, r["published_at"] or datetime.min)),
            "score_detail": {aud: {"p_material": x.p_material, "relevance": x.p_material,
                                   "model_version": x.model_version, "prior": x.model_version == PRIOR_VERSION,
                                   "scored_at": x.created_at, "contributions": contribution_rows(x),
                                   "features": sorted((x.features or {}).items())}
                             for (_, aud), x in shown.items()},
            "score_history": [{"audience": x.audience, "p_material": x.p_material, "relevance": x.p_material,
                               "model_version": x.model_version, "prior": x.model_version == PRIOR_VERSION,
                               "shown": x.id in shown_ids, "created_at": x.created_at} for x in scores],
            "exposure": _exposure_rows(s, d, policy), "exposure_min_confidence": policy.exposure_min_confidence,
            "alerts_full": [_alert_row(a, d.title, card["lab_name"], d.lab_id, policy) for a in alerts],
            "labels": [{"audience": x.audience, "label": x.label, "annotator": x.annotator,
                        "reason": x.reason, "rubric_version": x.rubric_version, "created_at": x.created_at}
                       for x in labels]}


# ------------------------------------------------------------------------------------ why flagged
_SUFFIXES = ("_min", "_max", "_floor")
_UNKNOWN = {"lens_action": "no lens yet", "p": "not scored", "magnitude": "not rated", "lab_tier": "no lab"}


def _fmt_fact(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list | tuple | set):
        return ", ".join(str(v) for v in value) or "none"
    return str(value)


def _fact_key(key: str) -> str:
    return next((key[: -len(sfx)] for sfx in _SUFFIXES if key.endswith(sfx)), key)


def _needs(key: str, expected: Any) -> str:
    if key.endswith(("_min", "_floor")):
        return f">= {expected}"
    if key.endswith("_max"):
        return f"<= {expected}"
    if isinstance(expected, list):
        return str(expected[0]) if len(expected) == 1 else "one of " + ", ".join(str(v) for v in expected)
    return _fmt_fact(expected)


def _cond_text(cond: dict[str, Any]) -> str:
    return " and ".join(k if k in ("any", "all", "not") else f"{_fact_key(k)} {_needs(k, v)}"
                        for k, v in cond.items())


def first_failing(cond: dict[str, Any], facts: dict[str, Any]) -> str | None:
    """Why a rule did not fire: its first condition, in policy order, that does not hold (e.g.
    'magnitude 3 (needs >= 4)'); None when `cond` holds. Every condition is tested with
    materiality.matches itself, so the reason can never disagree with the decision."""
    for key, expected in cond.items():
        if mat.matches({key: expected}, facts):
            continue
        if key == "any":
            return "none of: " + " | ".join(first_failing(c, facts) or "holds" for c in expected)
        if key == "all":
            return next((r for c in expected if (r := first_failing(c, facts))), "a nested condition fails")
        if key == "not":  # the excluded condition holds: say which facts made it hold
            held = "; ".join(mat.condition_reasons(expected, facts)) or _cond_text(expected)
            return f"excluded: {held} (the rule requires not {_cond_text(expected)})"
        fact = _fact_key(key)
        value = facts.get(fact)
        text = f"{fact} {_UNKNOWN.get(fact, 'unknown') if value is None else _fmt_fact(value)} " \
               f"(needs {_needs(key, expected)})"
        if fact == "held_exposure" and facts.get("exposed_tickers"):
            text += f"; event-specific tickers {_fmt_fact(facts['exposed_tickers'])} are not BIT holdings"
        elif fact == "corroborated":
            text += f"; source classes: {_fmt_fact(facts.get('source_classes'))}"
        return text
    return None


def _why(ctx: mat.DevContext, audience: str, policy: mat.Policy) -> Row:
    """One audience's 'why flagged'. The keys severity, rule_severity, unconfirmed, rules (each with the
    facts it matched) and facts are materiality.explain()'s, unchanged; added for display: fired rules
    with their severity, guards, every rule that did not fire with its first failing condition and how
    many of its conditions hold, and the score the rules read with its top contributions."""
    ev = mat.evaluate(ctx, audience, policy)
    facts = ev.facts
    rules = [{"id": r, "description": policy.describe(r), "matched": mat.rule_reasons(r, facts, policy)}
             for r in ev.rules]
    by_id = {r.id: r for r in policy.rules.get(audience, ())}
    score = ctx.scores.get(audience)
    return {
        "severity": ev.severity, "rule_severity": ev.rule_severity, "unconfirmed": ev.unconfirmed,
        "rules": rules, "facts": facts,
        "fired": [{**r, "severity": by_id[r["id"]].severity} for r in rules if r["id"] in by_id],
        "guards": [r for r in rules if r["id"] not in by_id],
        "not_fired": [{"id": rule.id, "severity": rule.severity, "description": rule.description,
                       "failed": first_failing(rule.when, facts),
                       "holds": sum(mat.matches({k: v}, facts) for k, v in rule.when.items()),
                       "conditions": len(rule.when)}
                      for rule in by_id.values() if rule.id not in ev.fired],
        "score": None if score is None else {
            "p_material": score.p_material, "model_version": score.model_version,
            "prior": score.model_version == PRIOR_VERSION, "top": contribution_rows(score)[:3]},
    }


def explain_many(s: Session, dev_ids, policy: mat.Policy | None = None) -> dict[int, dict[str, Row]]:
    """materiality.explain() for many developments with one batched context load (the alerts page
    explains every alert it lists), each audience extended as in _why. When the policy cannot be
    loaded, every entry is {'error': ...} instead."""
    ids = sorted({int(i) for i in dev_ids if i is not None})
    err = None
    if policy is None:
        policy, err = materiality_policy()
    if err:
        return {i: {aud: {"error": err} for aud in AUDIENCES} for i in ids}
    return {i: {aud: _why(ctx, aud, policy) for aud in AUDIENCES}
            for i, ctx in mat.load_contexts(s, ids).items()}


# ------------------------------------------------------------------------------------ alerts / reports
def normalize_rules(rules, policy: mat.Policy | None = None) -> list[Row]:
    """Rules recorded on an alert when it was raised, with the policy's description of each."""
    out = []
    for r in rules or []:
        if isinstance(r, dict):
            name = r.get("rule") or r.get("name") or r.get("id") or "rule"
            detail = r.get("detail") or r.get("reason") or r.get("why") or ", ".join(
                f"{k}={v}" for k, v in r.items() if k not in {"rule", "name", "id"})
        else:
            name, detail = r, ""
        out.append({"name": str(name), "detail": str(detail),
                    "description": policy.describe(str(name)) if policy is not None else ""})
    return out


def delivery_view(delivered: Any) -> Row:
    """Alert.delivered mixes channel receipts with bookkeeping (guarded_severity, mode, log, slack
    retries). Only in_app and slack are channels; mode (live push vs historical backfill) and the
    pre-cap severity are shown on their own."""
    d = delivered if isinstance(delivered, dict) else {}
    channels = []
    if d.get("in_app"):
        channels.append({"name": "in_app", "label": "in-app", "state": "sent", "at": str(d["in_app"]), "error": None})
    slack_error = str(d["slack_error"]) if d.get("slack_error") else None
    if d.get("slack"):
        channels.append({"name": "slack", "label": "Slack", "state": "sent", "at": str(d["slack"]), "error": None})
    elif d.get("slack_pending") or slack_error:
        channels.append({"name": "slack", "label": "Slack", "state": "pending" if d.get("slack_pending") else "failed",
                         "at": None, "error": slack_error})
    mode = d.get("mode") if d.get("mode") in ("live", "historical") else None
    guarded = d.get("guarded_severity") if d.get("guarded_severity") in ("P1", "P2", "P3") else None
    return {"channels": channels, "mode": mode, "guarded_severity": guarded}


def _alert_row(a: Alert, dev_title: str | None, lab_name: str | None, lab_id: str | None,
               policy: mat.Policy | None = None) -> Row:
    return {"id": a.id, "development_id": a.development_id, "development_title": dev_title, "lab_id": lab_id,
            "lab_name": lab_name, "audience": a.audience, "severity": a.severity, "title": a.title,
            "body_md": a.body_md, "rules": normalize_rules(a.rules, policy), "status": a.status,
            "delivered": a.delivered or {}, "delivery": delivery_view(a.delivered), "created_at": a.created_at,
            "acked_at": a.acked_at, "acked_by": a.acked_by}


def _signal_fields(own: Lens | None, exposure: list[Row], lab: list[Row], e: Exposure, score: Score | None) -> Row:
    """Routing fields for BIT's Signals layer. `tickers` is the development's exposure exactly as the alert
    rules count it (basis 'lens': committed by the investment lens, with its direction; 'event': an
    event-specific exposure link); held_tickers is the rules' held_tickers fact; the lab's curated
    relations are listed apart (lab_relations) and never route an alert."""
    bases = frozenset(t["basis"] for t in exposure)
    basis = {frozenset({"lens"}): "lens", frozenset({"event"}): "exposure_map"}.get(
        bases, "lens+exposure_map" if bases else None)
    return {"relevance": score.p_material if score else None,
            "relevance_model_version": score.model_version if score else None,
            "action": own.action if own else None, "horizon": own.horizon if own else None,
            "tickers": [{"symbol": t["symbol"], "direction": t["direction"], "bit_weight": t["bit_weight"],
                         "basis": t["basis"]} for t in exposure],
            "held_tickers": list(e.held), "lab_relations": [t["symbol"] for t in lab], "ticker_basis": basis}


def alert_signals(s: Session, rows: list[Row]) -> list[Row]:
    """Add routing fields (relevance, lens action / horizon, exposure as the alert rules count it) and
    'why flagged' for the alert's audience, evaluated now (batched, in place)."""
    dev_ids = sorted({r["development_id"] for r in rows if r.get("development_id") is not None})
    if not dev_ids:
        return rows
    policy, err = materiality_policy()
    scores = latest_scores(s, dev_ids)
    ctxs = mat.load_contexts(s, dev_ids)
    tickers = {t.symbol: t for t in s.scalars(select(Ticker))}
    for r in rows:
        dev, ctx = r.get("development_id"), ctxs.get(r.get("development_id"))
        if ctx is None:  # the development is gone: nothing to route on
            r.update(_signal_fields(None, [], [], Exposure(), None), why_flagged=None)
            continue
        inv = ctx.lenses.get("investment")
        links = sorted(ctx.exposures, key=lambda x: (-(x.confidence or 0.0), x.id or 0))
        e = exposure_of(ctx.dev, links, inv, ctx.holdings, policy)
        exposure, lab = ticker_chips(e, inv, links, tickers)
        r.update(_signal_fields(ctx.lenses.get(r["audience"]), exposure, lab, e, scores.get((dev, r["audience"]))))
        r["why_flagged"] = {"error": err} if err else _why(ctx, r["audience"], policy)
        # the daily cap (applied when raised) or newer facts: show the recorded rules too
        now_ids = {x["id"] for x in r["why_flagged"].get("rules", [])}
        r["rules_changed"] = {x["name"] for x in r.get("rules") or []} != now_ids
    return rows


def alert_rows(s: Session, *, audience: str | None = None, severity: str | None = None,
               status: str | None = "open", since: datetime | None = None,
               severities: tuple[str, ...] | None = None, limit: int = 200) -> list[Row]:
    q = (select(Alert, Development.title, Development.lab_id)
         .outerjoin(Development, Development.id == Alert.development_id))
    if audience:
        q = q.where(Alert.audience == audience)
    if severity:
        q = q.where(Alert.severity == severity)
    if severities:
        q = q.where(Alert.severity.in_(severities))
    if status and status != "all":
        q = q.where(Alert.status == status)
    if since is not None:
        q = q.where(Alert.created_at >= since)
    newest = (Alert.created_at.desc(), Alert.id.desc())
    order = (Alert.severity, *newest) if severities else newest
    labs = lab_names(s)
    policy = materiality_policy()[0]
    return [_alert_row(a, title, labs.get(lab_id, lab_id) if lab_id else None, lab_id, policy)
            for a, title, lab_id in s.execute(q.order_by(*order).limit(limit))]


def ack_alert(s: Session, alert_id: int, status: str = "acked", by: str | None = None) -> Alert | None:
    """Set the status; who closed it and when is kept (cleared again on reopen)."""
    a = s.get(Alert, alert_id)
    if a is not None and a.status != status:
        a.status = status
        if status == "open":
            a.acked_at, a.acked_by = None, None
        else:
            a.acked_at, a.acked_by = utcnow(), by[:64] if by else None
        log.info("alert %s -> %s by %s", alert_id, status, by or "anonymous")
    return a


def report_rows(s: Session, audience: str | None = None, limit: int = 100) -> list[Row]:
    q = select(Report).order_by(Report.period_start.desc(), Report.audience).limit(limit)
    if audience:
        q = q.where(Report.audience == audience)
    return [{"id": r.id, "audience": r.audience, "title": r.title, "period_start": r.period_start,
             "period_end": r.period_end, "created_at": r.created_at, "model": r.model,
             "n_items": len(r.items or [])} for r in s.scalars(q)]


def report_detail(s: Session, report_id: int) -> Row | None:
    r = s.get(Report, report_id)
    if r is None:
        return None
    items = [i for i in r.items or [] if isinstance(i, dict) and str(i.get("development_id", "")).isdigit()]
    items.sort(key=lambda i: i.get("rank") or 0)
    titles = dev_titles(s, [int(i["development_id"]) for i in items])
    siblings = select(Report.id).where(Report.audience == r.audience).limit(1)
    start = Report.period_start
    prev_id = s.scalar(siblings.where(start < r.period_start).order_by(start.desc()))
    next_id = s.scalar(siblings.where(start > r.period_start).order_by(start))
    return {"id": r.id, "audience": r.audience, "title": r.title, "period_start": r.period_start,
            "period_end": r.period_end, "created_at": r.created_at, "model": r.model,
            "body_md": r.body_md or "", "prev_id": prev_id, "next_id": next_id,
            "items": [{"development_id": int(i["development_id"]), "rank": i.get("rank"),
                       "p_material": i.get("p_material"), "title": titles.get(int(i["development_id"]))}
                      for i in items]}


# ------------------------------------------------------------------------------------ register
def _exposure_badges(exposure, tickers: dict[str, Ticker]) -> list[Row]:
    out = []
    for e in exposure or []:
        if not isinstance(e, dict) or not e.get("ticker"):
            continue
        t = tickers.get(e["ticker"])
        out.append({"symbol": e["ticker"], "relation": e.get("relation"), "note": e.get("note"),
                    "company": t.company if t else None, "bit_weight": t.bit_weight if t else None,
                    "direction": "unclear", "why": e.get("relation")})
    return out


def labs_overview(s: Session, now: datetime) -> list[Row]:
    people = dict(s.execute(select(Person.lab_id, func.count()).group_by(Person.lab_id)).all())
    devs = {lab: (n, last) for lab, n, last in s.execute(
        select(Development.lab_id, func.count(), func.max(Development.first_seen_at))
        .where(_visible()).group_by(Development.lab_id))}
    recent = dict(s.execute(
        select(Development.lab_id, func.count())
        .where(_visible(), Development.first_seen_at >= now - timedelta(days=30))
        .group_by(Development.lab_id)).all())
    states: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lab_sources = list(s.scalars(select(Source).where(Source.lab_id.is_not(None))))
    imported = imported_ingest_times(s, lab_sources)
    for src in lab_sources:
        states[src.lab_id][source_state(src, now, imported.get(src.id))[0]] += 1
    tickers = {t.symbol: t for t in s.scalars(select(Ticker))}
    out = []
    for lab in s.scalars(select(Lab).order_by(Lab.tier, Lab.name)):
        n_devs, last = devs.get(lab.id, (0, None))
        out.append({"id": lab.id, "name": lab.name, "kind": lab.kind, "tier": lab.tier, "parent": lab.parent,
                    "description": lab.description, "people": people.get(lab.id, 0), "developments": n_devs,
                    "last_development": last, "developments_30d": recent.get(lab.id, 0),
                    "sources": dict(states.get(lab.id, {})),
                    "exposure": _exposure_badges(lab.public_exposure, tickers)})
    return out


def lab_detail(s: Session, lab_id: str, now: datetime) -> Row | None:
    lab = s.get(Lab, lab_id)
    if lab is None:
        return None
    tickers = {t.symbol: t for t in s.scalars(select(Ticker))}
    recent = list(s.scalars(select(Development.id).where(Development.lab_id == lab_id, _visible())
                            .order_by(Development.first_seen_at.desc()).limit(30)))
    identities = s.scalars(select(Identity).where(Identity.lab_id == lab_id, Identity.person_id.is_(None))
                           .order_by(Identity.platform, Identity.handle))
    people = s.scalars(select(Person).where(Person.lab_id == lab_id).order_by(Person.tier, Person.name))
    return {"lab": {"id": lab.id, "name": lab.name, "kind": lab.kind, "tier": lab.tier, "parent": lab.parent,
                    "homepage": lab.homepage, "description": lab.description, "meta": lab.meta or {}},
            "exposure": _exposure_badges(lab.public_exposure, tickers),
            "identities": [_identity_row(i) for i in identities],
            "sources": source_rows(s, now, lab_id=lab_id),
            "people": [_person_row(p, lab.name) for p in people],
            "documents": s.scalar(select(func.count(Document.id)).where(Document.lab_id == lab_id)) or 0,
            "developments": build_cards(s, recent)}


def _identity_row(i: Identity) -> Row:
    return {"platform": i.platform, "handle": i.handle, "url": i.url, "confidence": i.confidence,
            "method": i.method, "status": i.status, "evidence": i.evidence}


def _person_row(p: Person, lab_name: str | None, identities: int = 0, documents: int = 0) -> Row:
    return {"id": p.id, "name": p.name, "lab_id": p.lab_id, "lab_name": lab_name, "title": p.title,
            "tier": p.tier, "why_key": p.why_key, "inclusion_rule": p.inclusion_rule, "status": p.status,
            "contributor_score": p.contributor_score, "identities": identities, "documents": documents}


def people_rows(s: Session, lab: str | None = None, tier: str | None = None,
                q: str | None = None) -> list[Row]:
    stmt = select(Person).order_by(Person.tier, Person.contributor_score.desc().nulls_last(), Person.name)
    if lab:
        stmt = stmt.where(Person.lab_id == lab)
    if tier:
        stmt = stmt.where(Person.tier == tier)
    if q:
        stmt = stmt.where(or_(Person.name.ilike(f"%{q}%"), Person.title.ilike(f"%{q}%")))
    idents = dict(s.execute(select(Identity.person_id, func.count()).where(Identity.status == "confirmed")
                            .group_by(Identity.person_id)).all())
    docs = dict(s.execute(select(Document.person_id, func.count()).where(Document.person_id.is_not(None))
                          .group_by(Document.person_id)).all())
    labs = lab_names(s)
    return [_person_row(p, labs.get(p.lab_id), idents.get(p.id, 0), docs.get(p.id, 0))
            for p in s.scalars(stmt)]


def person_detail(s: Session, person_id: str, now: datetime) -> Row | None:
    p = s.get(Person, person_id)
    if p is None:
        return None
    labs = lab_names(s)
    affs = s.scalars(select(Affiliation).where(Affiliation.person_id == person_id)
                     .order_by(Affiliation.start_date.asc().nulls_first(), Affiliation.id))
    docs = s.scalars(select(Document).where(Document.person_id == person_id)
                     .order_by(Document.published_at.desc().nulls_last(), Document.id.desc()).limit(50))
    ins = s.execute(select(Insight, Document).join(Document, Document.id == Insight.document_id)
                    .where(or_(Document.person_id == person_id, _like_json(Insight.person_ids, person_id)))
                    .order_by(Insight.published_at.desc().nulls_last(), Insight.id.desc()).limit(50))
    dev_ids = list(s.scalars(select(Development.id)
                             .where(_visible(), _like_json(Development.person_ids, person_id))
                             .order_by(Development.first_seen_at.desc()).limit(20)))
    identities = s.scalars(select(Identity).where(Identity.person_id == person_id)
                           .order_by(Identity.status, Identity.platform))
    feats = p.contributor_features if isinstance(p.contributor_features, dict) else {}
    return {"person": _person_row(p, labs.get(p.lab_id)), "meta": p.meta or {},
            "features": sorted(feats.items()), "identities": [_identity_row(i) for i in identities],
            "affiliations": [{"org": labs.get(a.lab_id) or a.org_name or a.lab_id, "lab_id": a.lab_id,
                              "role": a.role, "start": a.start_date, "end": a.end_date,
                              "evidence": a.evidence, "source_url": a.source_url, "confidence": a.confidence}
                             for a in affs],
            "sources": source_rows(s, now, person_id=person_id),
            "documents": [_doc_row(d) for d in docs],
            "insights": [{"id": i.id, "title": i.title, "claim": i.claim, "event_type": i.event_type,
                          "published_at": i.published_at, "development_id": i.development_id,
                          "quotes_verified": i.quotes_verified, "doc": _doc_row(d)} for i, d in ins],
            "developments": build_cards(s, dev_ids)}


# ------------------------------------------------------------------------------------ health / status
def _never_ran(src: Source) -> bool:
    """No ingest bookkeeping at all on the row (a source the runner skipped or failed has some)."""
    return (src.last_run_at is None and src.last_success_at is None and not src.last_error
            and not src.consecutive_failures)


def imported_ingest_times(s: Session, sources: list[Source]) -> dict[str, datetime]:
    """Newest document fetch per source that never ran in this database. Its documents were ingested
    elsewhere (an imported snapshot, or bookkeeping reset by re-registering), so they, not 'never',
    say how fresh the source's data is. A source with any ingest job_run here (even a skip for a
    missing credential) has been tried by this database's runner and keeps its own state."""
    never = {src.id for src in sources if _never_ran(src)}
    if not never:
        return {}
    tried = {job.removeprefix("ingest:") for job in s.scalars(  # IN uses ix_job_run_job_started
        select(JobRun.job).where(JobRun.job.in_([f"ingest:{sid}" for sid in sorted(never)])).distinct())}
    never -= tried
    if not never:
        return {}
    rows = s.execute(select(Document.source_id, func.max(Document.fetched_at))
                     .where(Document.source_id.in_(sorted(never))).group_by(Document.source_id))
    return {sid: at for sid, at in rows if at is not None}


def source_state(src: Source, now: datetime, imported_at: datetime | None = None) -> tuple[str, str]:
    """ok | stale | failing | pending | disabled, plus a short human reason. `imported_at` (see
    imported_ingest_times) stands in for the poll times of a source that never ran here."""
    if not src.enabled:
        return "disabled", "disabled in config"
    fails = src.consecutive_failures or 0
    if fails >= FAILING_AFTER or (fails and src.last_success_at is None):
        return "failing", f"{fails} consecutive failure{'s' if fails != 1 else ''}"
    imported = imported_at if _never_ran(src) else None
    last_ok = src.last_success_at or imported
    if last_ok is None:
        return "pending", "never polled successfully"
    slo = timedelta(hours=src.freshness_slo_hours or 24 * 14)
    if now - last_ok > slo:
        what = "newest imported document is" if imported else "no successful poll for"
        return "stale", f"{what} {span(now - last_ok)}{' old' if imported else ''}"
    ref = src.last_new_item_at or imported or src.created_at
    if ref is not None and now - ref > slo:
        return "stale", f"no new item for {span(now - ref)} (SLO {src.freshness_slo_hours}h)"
    if imported:
        return "ok", f"no run recorded in this database; newest document fetched {span(now - imported)} ago"
    return "ok", "last poll failed once" if fails else ""


def _last_ingest_runs(s: Session) -> dict[str, JobRun]:
    j2 = aliased(JobRun)
    ids = select(func.max(j2.id)).where(j2.job.like("ingest:%")).group_by(j2.job)
    return {r.job: r for r in s.scalars(select(JobRun).where(JobRun.id.in_(ids)))}


def source_rows(s: Session, now: datetime, lab_id: str | None = None,
                person_id: str | None = None) -> list[Row]:
    q = select(Source).order_by(Source.lab_id.nulls_last(), Source.id)
    if lab_id:
        q = q.where(Source.lab_id == lab_id)
    if person_id:
        q = q.where(Source.person_id == person_id)
    runs = _last_ingest_runs(s)
    sources = list(s.scalars(q))
    imported = imported_ingest_times(s, sources)
    out = []
    for src in sources:
        at = imported.get(src.id) if _never_ran(src) else None
        state, why = source_state(src, now, at)
        run = runs.get(f"ingest:{src.id}")
        if run is not None and state == "pending" and run.error:
            why = f"last run {run.status}: {run.error}"  # e.g. skipped for a missing credential
        out.append({"id": src.id, "lab_id": src.lab_id, "person_id": src.person_id, "adapter": src.adapter,
                    "kind": src.kind, "source_class": src.source_class, "url": src.url,
                    "enabled": src.enabled, "cadence_minutes": src.cadence_minutes,
                    "slo_hours": src.freshness_slo_hours, "last_run_at": src.last_run_at,
                    "last_success_at": src.last_success_at, "last_new_item_at": src.last_new_item_at,
                    "imported_at": at, "failures": src.consecutive_failures or 0, "last_error": src.last_error,
                    "state": state, "reason": why, "last_run_status": run.status if run is not None else None})
    return out


def budget_summary(s: Session, now: datetime) -> Row:
    """Spend vs the hard caps the LLM gateway enforces (same UTC-midnight day boundary)."""
    st = get_settings()
    total_q = select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0))
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    x_q = (select(func.coalesce(func.sum(ApiUsage.cost_usd), 0.0))
           .where(func.lower(ApiUsage.provider).in_(X_PROVIDERS)))

    def line(spent, cap: float) -> Row:
        spent = float(spent or 0.0)
        return {"spent": round(spent, 4), "budget": cap, "pct": round(100 * spent / cap, 1) if cap else None}

    today = s.scalar(total_q.where(LLMCall.created_at >= midnight))
    return {"llm_total": line(s.scalar(total_q), st.llm_budget_total_usd),
            "llm_today": line(today, st.llm_budget_daily_usd),
            "x_total": line(s.scalar(x_q), st.x_budget_total_usd)}


def _health_reasons(counts: dict[str, int], ops: dict[str, int], over: list[str],
                    last_success: datetime | None, now: datetime, basis: str | None = "runs") -> list[str]:
    reasons = [f"{ops['crit']} critical ops event(s)"] if ops.get("crit") else []
    reasons += [f"budget exhausted: {', '.join(over)}"] if over else []
    reasons += [f"{counts[k]} {k} source(s)" for k in ("failing", "stale") if counts[k]]
    reasons += [f"{ops['warn']} warning ops event(s)"] if ops.get("warn") else []
    if last_success is None and sum(counts.values()) > counts["disabled"]:
        reasons.append("no successful ingest yet")
    elif last_success is not None and now - last_success > timedelta(hours=STALE_INGEST_HOURS):
        reasons.append(f"no ingest has run in this database; newest imported document is {span(now - last_success)} old"
                       if basis == "documents" else f"last successful ingest {span(now - last_success)} ago")
    return reasons


def health_summary(s: Session, now: datetime | None = None) -> Row:
    """ok | degraded | critical with the reasons, for the overview strip, /status and /api/health.

    `last_ingest_basis` says where `last_ingest_at` comes from: 'runs' (this database's ingest
    bookkeeping) or 'documents' (no source ever ran here, e.g. an imported snapshot: the newest
    document fetch, so the header shows how old the data is instead of 'never')."""
    now = now or utcnow()
    counts = {k: 0 for k in ("ok", "stale", "failing", "pending", "disabled")}
    sources = list(s.scalars(select(Source)))
    imported = imported_ingest_times(s, sources)
    for src in sources:
        counts[source_state(src, now, imported.get(src.id))[0]] += 1
    ops = {k: int(v) for k, v in s.execute(select(OpsEvent.severity, func.count())
                                           .where(OpsEvent.resolved_at.is_(None))
                                           .group_by(OpsEvent.severity)).all()}
    budgets = budget_summary(s, now)
    over = [k for k, b in budgets.items() if (b["pct"] or 0) >= 100]
    last_success = max((x.last_success_at for x in sources if x.last_success_at), default=None)
    basis = "runs" if last_success else None
    if last_success is None and imported:
        last_success, basis = max(imported.values()), "documents"
    reasons = _health_reasons(counts, ops, over, last_success, now, basis)
    status = "critical" if ops.get("crit") or over else ("degraded" if reasons else "ok")
    age = round((now - last_success).total_seconds() / 3600, 1) if last_success else None
    open_alerts = select(func.count(Alert.id)).where(Alert.status == "open")
    return {"status": status, "reasons": reasons, "sources": counts, "last_ingest_at": last_success,
            "last_ingest_basis": basis, "last_ingest_age_hours": age,
            "last_document_at": s.scalar(select(func.max(Document.fetched_at))),
            "open_ops_events": ops, "budgets": budgets,
            "counts": {"developments": s.scalar(select(func.count(Development.id))) or 0,
                       "documents": s.scalar(select(func.count(Document.id))) or 0,
                       "insights": s.scalar(select(func.count(Insight.id))) or 0,
                       "open_alerts": s.scalar(open_alerts) or 0}}


def _llm_costs(s: Session) -> list[Row]:
    err = func.sum(case((LLMCall.status.in_(LLM_ERROR_STATUSES), 1), else_=0))
    cached = func.sum(case((LLMCall.status == "cached", 1), else_=0))
    rows = s.execute(select(LLMCall.purpose, LLMCall.model, func.count(), cached, err,
                            func.sum(LLMCall.input_tokens), func.sum(LLMCall.output_tokens),
                            func.sum(LLMCall.cache_read_tokens), func.sum(LLMCall.cost_usd),
                            func.avg(LLMCall.latency_ms))
                     .group_by(LLMCall.purpose, LLMCall.model)
                     .order_by(func.sum(LLMCall.cost_usd).desc())).all()
    return [{"purpose": p, "model": m, "calls": int(n or 0), "cached": int(c or 0), "errors": int(e or 0),
             "input_tokens": int(i or 0), "output_tokens": int(o or 0), "cache_read_tokens": int(cr or 0),
             "usd": float(u or 0.0), "avg_latency_ms": float(lat or 0.0)}
            for p, m, n, c, e, i, o, cr, u, lat in rows]


def status_page(s: Session, now: datetime) -> Row:
    sev_rank = {"crit": 0, "warn": 1, "info": 2}
    ops = sorted(s.scalars(select(OpsEvent).where(OpsEvent.resolved_at.is_(None))),
                 key=lambda e: (sev_rank.get(e.severity, 3),
                                -(e.created_at.timestamp() if e.created_at else 0.0)))
    api = s.execute(select(ApiUsage.provider, ApiUsage.endpoint, func.count(), func.sum(ApiUsage.units),
                           func.sum(ApiUsage.cost_usd))
                    .group_by(ApiUsage.provider, ApiUsage.endpoint)
                    .order_by(func.sum(ApiUsage.cost_usd).desc())).all()
    runs = s.scalars(select(JobRun).order_by(JobRun.started_at.desc(), JobRun.id.desc()).limit(60))
    return {"health": health_summary(s, now), "sources": source_rows(s, now), "llm": _llm_costs(s),
            "ops": [{"kind": e.kind, "severity": e.severity, "subject": e.subject, "message": e.message,
                     "created_at": e.created_at} for e in ops],
            "api": [{"provider": p, "endpoint": ep, "calls": int(n or 0), "units": int(un or 0),
                     "usd": float(u or 0.0)} for p, ep, n, un, u in api],
            "runs": [{"job": r.job, "started_at": r.started_at, "finished_at": r.finished_at,
                      "status": r.status, "items_in": r.items_in, "items_new": r.items_new,
                      "items_failed": r.items_failed, "error": r.error,
                      "seconds": max(0.0, (r.finished_at - r.started_at).total_seconds())
                      if r.finished_at and r.started_at else None} for r in runs]}


# ------------------------------------------------------------------------------------ labelling
def _mixed_order(rows: list[tuple[int, float | None]]) -> list[int]:
    """Round-robin high / low / uncertain scores, each stratum in a stable pseudo-random order, so
    labels are not only confirmations of what the model already ranks first."""
    def key(i: int) -> str:
        return hashlib.sha1(str(i).encode()).hexdigest()

    hi = sorted((i for i, p in rows if p is not None and p >= 0.6), key=key)
    lo = sorted((i for i, p in rows if p is not None and p < 0.3), key=key)
    mid = sorted((i for i, p in rows if p is not None and 0.3 <= p < 0.6), key=key)
    unscored = sorted((i for i, p in rows if p is None), key=key)
    mixed = [i for trio in itertools.zip_longest(hi, lo, mid) for i in trio if i is not None]
    return mixed + unscored


def label_queue(s: Session, audience: str, annotator: str | None = None, pos: int = 0,
                include_others: bool = False) -> Row:
    """Developments without a human label for this audience (or, for agreement checks, without
    *my* label), in mixed-score order; `pos` skips ahead without writing anything."""
    done = select(Label.development_id).where(Label.audience == audience)
    if include_others and annotator:
        done = done.where(Label.annotator == annotator)
    else:
        done = done.where(Label.annotator.like("human:%"))
    rows = s.execute(_with_latest_score(select(Development.id, Score.p_material), audience, live_versions(s))
                     .where(_visible(), Development.id.not_in(done))).all()
    order = _mixed_order([(i, p) for i, p in rows])
    pos = pos % len(order) if order else 0  # skipping past the end wraps around
    mine: dict[int, int] = {}
    recent: list[Row] = []
    if annotator:
        mine = {int(k): int(v) for k, v in s.execute(
            select(Label.label, func.count())
            .where(Label.annotator == annotator, Label.audience == audience).group_by(Label.label)).all()}
        recent = [{"development_id": x.development_id, "title": t, "label": x.label, "audience": x.audience,
                   "created_at": x.created_at}
                  for x, t in s.execute(select(Label, Development.title)
                                        .join(Development, Development.id == Label.development_id)
                                        .where(Label.annotator == annotator)
                                        .order_by(Label.created_at.desc()).limit(8))]
    humans = select(func.count(Label.id)).where(Label.audience == audience, Label.annotator.like("human:%"))
    return {"item": development_detail(s, order[pos]) if order else None, "pos": pos, "remaining": len(order),
            "mine": mine, "human_labels": s.scalar(humans) or 0, "recent": recent}


def development_exists(s: Session, dev_id: int) -> bool:
    return s.get(Development, dev_id) is not None


def save_label(s: Session, dev_id: int, audience: str, value: int, annotator: str,
               reason: str | None = None) -> Label:
    """Upsert on (development, audience, annotator): relabelling replaces, never duplicates."""
    row = s.scalar(select(Label).where(Label.development_id == dev_id, Label.audience == audience,
                                       Label.annotator == annotator))
    if row is None:
        row = Label(development_id=dev_id, audience=audience, annotator=annotator)
        s.add(row)
    row.label, row.reason, row.rubric_version = value, reason or None, RUBRIC_VERSION
    row.created_at = utcnow()
    s.flush()
    return row
