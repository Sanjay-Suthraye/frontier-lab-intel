"""Rule-based exposure mapping: which public companies a development can move.

Deterministic and auditable on purpose: the investment lens builds on these links, and the closed
ticker list in config/exposure.yaml is what stops an LLM from inventing exposures. Three rules, each
leaving a rationale on the link:

  public_exposure  the lab's curated relations (config/labs.yaml): confidence 0.6, or 0.4 when the
                   register note says "verify"
  named company    a listed company or a brand it owns is named in the development text: 0.65
  theme keywords   a theme whose event types include the development's and whose keywords occur in
                   the text links that theme's tickers: 0.25 + 0.1 per distinct keyword, max 0.55

A ticker supported by two rules gets +0.1 (max 0.8). Each link records which kind of evidence it rests on
in `method`:

  rule  event-specific: the development's own text names the company or matches a theme (possibly also a
        lab relation, which corroborates it)
  lab   only the lab's curated relations support it. Such a link describes the lab, not the event (every
        OpenAI item would otherwise "touch" MSFT, NVDA, ORCL, AMD, AVGO and TSM), so it is shown and offered
        to the lens as context, but score features (and holding-exposure facts) count only 'rule' links.

At most MAX_LINKS links per development: 'rule' links before 'lab' links, each ranked by evidence
strength (confidence plus a bonus for event-specific evidence: a mention, a keyword theme, or a lab
relation whose theme this event type moves, so that, e.g., an agent launch keeps its
software-disruption links instead of only the lab's generic suppliers). BIT fund weight only breaks
ties: ranking holdings first let a weak lab-level link to a large holding crowd out a company the
development actually names. Lab links only fill the slots the development's own evidence leaves:
they are context, re-derivable from labs.yaml (the lens gets the lab's relations anyway), and ranking
them by strength alone let them (0.6, or 0.75 with the bonus) displace keyword-theme links (0.5-0.7),
which cost hold-out AUC in both audiences' models. On a refresh, a lab link the text now names is
upgraded in place, and a new event-specific link takes the place of the weakest lab link when the cap
is reached. Direction stays 'unclear': the audience lens decides it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from fli.config import CONFIG_DIR
from fli.db.models import Development, ExposureLink, Insight, Lab, Ticker
from fli.db.session import session_scope
from fli.timeutil import utcnow

log = logging.getLogger("fli.exposure")

MAX_LINKS = 8
PUBLIC_CONF, VERIFY_CONF, MENTION_CONF = 0.6, 0.4, 0.65
THEME_BASE, THEME_PER_HIT, THEME_MAX = 0.25, 0.10, 0.55
CORROBORATION_BONUS, MAX_CONF = 0.10, 0.80
EVENT_SPECIFIC_BONUS = 0.15  # ranking only; not stored
REFRESH_DAYS = 14
LAB_RULES = frozenset({"public_exposure"})  # rules that describe the lab rather than the development
SKIP_STATUSES = ("merged", "duplicate", "empty")  # structurally dead (cluster stage); "noise" etc. still scored

# relation in labs.yaml -> candidate theme ids, first one the ticker belongs to wins
RELATION_THEMES: dict[str, tuple[str, ...]] = {
    "investor_and_cloud_partner": ("hyperscaler_partners",),
    "investor_and_distribution": ("hyperscaler_partners",),
    "cloud_partner": ("hyperscaler_partners",),
    "compute_supplier": ("neocloud_capacity", "hyperscaler_partners"),
    "chip_supplier": ("accelerators", "hyperscaler_partners"),
    "chip_supplier_and_investor": ("accelerators",),
    "custom_silicon_partner": ("accelerators", "networking_optics"),
    "foundry_for_custom_silicon": ("foundry_equipment",),
    "server_supplier": ("accelerators",),
    "investor": ("lab_capital_markets",),
    "issuer": ("lab_capital_markets",),
    "parent": ("hyperscaler_partners", "ai_consumer_apps", "china_ai"),
    "competitor": ("china_ai", "ai_consumer_apps", "software_disruption"),
    "efficiency_shock_risk": ("china_ai", "accelerators"),
    "talent_drain": ("hyperscaler_partners",),
}
# brand / short names (case-sensitive, whole word) in addition to the company names in the register
ALIASES: dict[str, tuple[str, ...]] = {
    "AMZN": ("Amazon", "AWS", "Trainium"), "MSFT": ("Microsoft", "Azure"),
    "GOOGL": ("Google", "Alphabet", "TPU", "TPUs"), "META": ("Meta",), "NVDA": ("Nvidia", "NVIDIA"),
    "TSM": ("TSMC",), "MU": ("Micron",), "IREN": ("IREN", "Iris Energy"), "SIMO": ("Silicon Motion",),
    "AVGO": ("Broadcom",), "ORCL": ("Oracle",), "9984.T": ("SoftBank",), "BABA": ("Alibaba",),
    "000660.KS": ("SK hynix", "SK Hynix", "Hynix"), "005930.KS": ("Samsung",), "SMCI": ("Supermicro",),
    "DELL": ("Dell",), "MRVL": ("Marvell",), "ANET": ("Arista",), "TSLA": ("Tesla",),
}
AMBIGUOUS_NAMES = frozenset({"Coherent", "Cipher", "Lemonade", "Anthropic"})  # ordinary words / the lab itself
NO_MENTION = frozenset({"ANT"})  # not trading yet; reached only through the issuing lab's exposure


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    tickers: tuple[str, ...]
    event_types: frozenset[str]
    keywords: tuple[str, ...]


@dataclass
class Candidate:
    ticker: str
    theme: str | None
    confidence: float
    event_specific: bool
    rationale: list[str] = field(default_factory=list)
    rules: set[str] = field(default_factory=set)

    @property
    def method(self) -> str:
        """'lab' when only the lab's curated relations support the link, else 'rule' (event-specific)."""
        return "lab" if self.rules <= LAB_RULES else "rule"

    @property
    def strength(self) -> float:
        """Ranking key: confidence plus the event-specific bonus."""
        return self.confidence + (EVENT_SPECIFIC_BONUS if self.event_specific else 0.0)


@dataclass(frozen=True)
class TickerInfo:
    symbol: str
    company: str
    themes: tuple[str, ...]
    bit_weight: float


@lru_cache(maxsize=4)
def load_themes(path: str | None = None) -> tuple[Theme, ...]:
    cfg = yaml.safe_load(Path(path or CONFIG_DIR / "exposure.yaml").read_text(encoding="utf-8")) or {}
    return tuple(
        Theme(id=t["id"], name=t.get("name", t["id"]), tickers=tuple(str(x) for x in t.get("tickers", [])),
              event_types=frozenset(t.get("event_types", [])), keywords=tuple(str(k) for k in t.get("keywords", [])))
        for t in cfg.get("themes", [])
    )


@lru_cache(maxsize=1024)
def _keyword_re(keyword: str) -> re.Pattern:
    """Whole-word, case-insensitive (text is lowercased), optional plural, flexible space/hyphen."""
    body = r"[\s\-]+".join(re.escape(part) for part in keyword.lower().split())
    return re.compile(rf"(?<![a-z0-9]){body}(?:s|es)?(?![a-z0-9])")


@lru_cache(maxsize=1024)
def _name_re(name: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")


def mention_names(t: TickerInfo) -> tuple[str, ...]:
    if t.symbol in NO_MENTION:
        return ()
    names = {t.company, *ALIASES.get(t.symbol, ())} - AMBIGUOUS_NAMES
    return tuple(sorted(n for n in names if n and len(n) >= 3))


def theme_for_relation(relation: str, ticker_themes: Iterable[str]) -> str | None:
    candidates = RELATION_THEMES.get(relation, ())
    own = list(ticker_themes)
    return next((c for c in candidates if c in own), None) or (candidates[0] if candidates else None) or (
        own[0] if own else None)


def _flatten(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [s for v in value.values() for s in _flatten(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _flatten(v)]
    return []


def development_text(session: Session, dev: Development) -> str:
    """Title, summary, insight titles/claims, artifact and entity names: what the rules match on."""
    parts = [dev.title or "", dev.summary or "", *_flatten(dev.artifacts or [])]
    for ins in session.scalars(select(Insight).where(Insight.development_id == dev.id).order_by(Insight.id)):
        parts += [ins.title or "", ins.claim or "", *_flatten(ins.artifacts or []), *_flatten(ins.entities or {})]
    return "\n".join(p for p in parts if p)


def _moves(theme_by_id: Mapping[str, Theme], theme_id: str | None, event_type: str) -> bool:
    """True when the theme lists this event type, i.e. the link is specific to this kind of event."""
    theme = theme_by_id.get(theme_id or "")
    return theme is not None and event_type in theme.event_types


def _add(cands: dict[str, Candidate], ticker: str, theme: str | None, conf: float, specific: bool,
         rule: str, why: str) -> None:
    c = cands.get(ticker)
    if c is None:
        cands[ticker] = Candidate(ticker, theme, conf, specific, [why], {rule})
        return
    if rule not in c.rules:
        c.confidence = min(MAX_CONF, max(c.confidence, conf) + CORROBORATION_BONUS)
    else:
        c.confidence = max(c.confidence, conf)
    c.event_specific = c.event_specific or specific
    c.theme = c.theme or theme
    c.rules.add(rule)
    if why not in c.rationale:
        c.rationale.append(why)


def propose_links(event_type: str, lab: Lab | None, text: str, themes: Iterable[Theme],
                  tickers: Mapping[str, TickerInfo]) -> list[Candidate]:
    """All rule candidates for one development, ranked (uncapped). Pure: no database access."""
    themes = list(themes)
    theme_by_id = {t.id: t for t in themes}
    cands: dict[str, Candidate] = {}
    # (a) the lab's curated public exposure
    for entry in (lab.public_exposure or []) if lab is not None else []:
        sym = str(entry.get("ticker", ""))
        if sym not in tickers:
            continue
        relation, note = str(entry.get("relation", "")), str(entry.get("note") or "")
        theme = theme_for_relation(relation, tickers[sym].themes)
        specific = _moves(theme_by_id, theme, event_type)
        conf = VERIFY_CONF if "verify" in note.lower() else PUBLIC_CONF
        why = f"{lab.name} public exposure: {relation}" + (f" ({note})" if note else "")
        _add(cands, sym, theme, conf, specific, "public_exposure", why)
    # (b) named company or brand in the text
    for sym, t in tickers.items():
        hit = next((n for n in mention_names(t) if _name_re(n).search(text)), None)
        if hit:
            theme = next((th for th in t.themes if _moves(theme_by_id, th, event_type)),
                         t.themes[0] if t.themes else None)
            _add(cands, sym, theme, MENTION_CONF, True, "mention", f"named in development text: '{hit}'")
    # (c) theme keywords, only for themes this event type moves
    lowered = text.lower()
    for th in themes:
        if event_type not in th.event_types:
            continue
        hits = [kw for kw in th.keywords if _keyword_re(kw).search(lowered)]
        if not hits:
            continue
        conf = min(THEME_MAX, THEME_BASE + THEME_PER_HIT * len(hits))
        why = f"theme '{th.name}': {event_type} + keywords [{', '.join(hits[:4])}]"
        for sym in th.tickers:
            if sym in tickers:
                _add(cands, sym, th.id, conf, True, "theme", why)
    order = {sym: i for i, sym in enumerate(cands)}
    # the development's own evidence first, then strength; fund weight only breaks ties (see module docstring)
    return sorted(cands.values(), key=lambda c: (c.method != "rule", -round(c.strength, 6),
                                                 -tickers[c.ticker].bit_weight, order[c.ticker]))


def load_tickers(session: Session) -> dict[str, TickerInfo]:
    return {t.symbol: TickerInfo(t.symbol, t.company, tuple(t.themes or []), float(t.bit_weight or 0.0))
            for t in session.scalars(select(Ticker))}


def _pending(session: Session, limit: int | None) -> list[Development]:
    """Developments without links, or changed (recently) since their newest link: new insights may name
    new companies. The recency bound keeps re-checks finite, as the schema has no 'checked at' marker."""
    last = (select(ExposureLink.development_id.label("dev_id"), func.max(ExposureLink.created_at).label("at"))
            .group_by(ExposureLink.development_id).subquery())
    recent = utcnow() - timedelta(days=REFRESH_DAYS)
    q = (select(Development).outerjoin(last, last.c.dev_id == Development.id)
         .where(Development.status.not_in(SKIP_STATUSES))
         .where(or_(last.c.at.is_(None), and_(Development.updated_at > last.c.at, Development.updated_at >= recent)))
         .order_by(Development.first_seen_at.desc(), Development.id.desc()))
    return list(session.scalars(q.limit(limit) if limit else q))


def _rationale(c: Candidate) -> str:
    return "; ".join(c.rationale)[:1000]


def link_development(session: Session, dev: Development, themes: Iterable[Theme],
                     tickers: Mapping[str, TickerInfo], rule_counts: dict[str, int] | None = None
                     ) -> list[ExposureLink]:
    """Add missing rule links for one development; returns the new rows.

    Existing rows are left alone except this stage's own 'lab' links (rows other stages or people made,
    e.g. method 'llm', are never touched): one the development text now supports is upgraded in place
    to an event-specific 'rule' link, and when the MAX_LINKS cap is reached a stronger event-specific
    candidate takes the place of the weakest lab link."""
    existing = {lk.ticker: lk for lk in session.scalars(
        select(ExposureLink).where(ExposureLink.development_id == dev.id))}
    lab = session.get(Lab, dev.lab_id) if dev.lab_id else None
    ranked = propose_links(dev.event_type, lab, development_text(session, dev), themes, tickers)
    by_ticker = {c.ticker: c for c in ranked}

    def strength(lk: ExposureLink) -> float:  # 0 when the rules no longer propose it (e.g. labs.yaml changed)
        return by_ticker[lk.ticker].strength if lk.ticker in by_ticker else 0.0

    for sym, lk in existing.items():
        c = by_ticker.get(sym)
        if c is not None and lk.method == "lab" and c.method == "rule":
            lk.method, lk.rationale, lk.theme = "rule", _rationale(c), c.theme or lk.theme
            lk.confidence = round(max(float(lk.confidence or 0.0), c.confidence), 3)
    fresh = [c for c in ranked if c.ticker not in existing]
    chosen = fresh[:max(MAX_LINKS - len(existing), 0)]
    # cap reached: event-specific candidates take the place of the weakest lab links (re-derivable context)
    lab_links = sorted((lk for lk in existing.values() if lk.method == "lab"),
                       key=lambda lk: (strength(lk), tickers[lk.ticker].bit_weight if lk.ticker in tickers else 0.0))
    for c in fresh[len(chosen):]:
        if c.method == "rule" and lab_links and c.strength > strength(lab_links[0]):
            session.delete(lab_links.pop(0))
            chosen.append(c)
    new = [ExposureLink(development_id=dev.id, ticker=c.ticker, theme=c.theme, direction="unclear",
                        rationale=_rationale(c), method=c.method, confidence=round(c.confidence, 3))
           for c in chosen]
    session.add_all(new)
    if rule_counts is not None:
        for rule in (r for c in chosen for r in c.rules):
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    return new


def run_exposure(limit: int | None = None, **_: object) -> dict:
    """Pipeline stage: map pending developments to tickers. Idempotent (unique dev+ticker)."""
    themes = load_themes()
    created = linked = 0
    by_rule: dict[str, int] = {}
    by_method: dict[str, int] = {}
    with session_scope() as s:
        tickers = load_tickers(s)
        if not tickers:
            log.warning("ticker table is empty; run `fli register` first")
            return {"candidates": 0, "developments_linked": 0, "links_created": 0}
        pending = _pending(s, limit)
        for dev in pending:
            links = link_development(s, dev, themes, tickers, by_rule)
            created += len(links)
            linked += bool(links)
            for lk in links:
                by_method[lk.method] = by_method.get(lk.method, 0) + 1
    log.info("exposure: %d developments checked, %d linked, %d links %s", len(pending), linked, created, by_method)
    return {"candidates": len(pending), "developments_linked": linked, "links_created": created,
            "links_by_rule": by_rule, "links_by_method": by_method}
