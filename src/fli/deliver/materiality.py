"""Materiality policy: which developments are material, for whom, and how loudly.

The decision is deterministic and declarative (config/materiality.yaml) on purpose: an alert that
pages a PM must be explainable ("rule X fired because ..."), reproducible and table-testable. The
LLM's job ends at the lens (what it means, what to do); it never decides severity.

Flow: `load_contexts()` batch-loads everything a decision needs for many developments at once,
`facts_for()` flattens one development into a small dict of named facts, and `evaluate_facts()`
applies the audience's rules plus the global guards. Alerts and digests share these functions;
`condition_reasons()` turns a fired rule back into the facts that made it fire ("why flagged").

Holding exposure is event-specific on purpose: a lab's curated relations (exposure links with
method 'lab', e.g. "OpenAI -> MSFT") are true of every development at that lab, so they never make a
development "exposed to a BIT holding". Only links about this development (a named company, a
keyword theme, a relation this event type moves) at >= `exposure_min_confidence`, and lens tickers
the lens commits to (positive / negative / mixed, and the lens did not abstain), count.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, defer

from fli.config import CONFIG_DIR
from fli.db.models import (
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Lens,
    Person,
    Score,
    Source,
    Ticker,
)
from fli.taxonomy import AUDIENCES, EVENT_TYPES

log = logging.getLogger("fli.deliver.materiality")

POLICY_PATH = CONFIG_DIR / "materiality.yaml"
SEVERITY_RANK: dict[str | None, int] = {None: 0, "P3": 1, "P2": 2, "P1": 3}
INACTIVE_STATUSES = ("merged", "duplicate", "suppressed", "noise", "empty")  # "empty": emptied by re-clustering
CODE_HOSTS = ("github.com/", "huggingface.co/", "gitlab.com/")
CODE_SOURCE_KINDS = {"github_org", "hf_org"}
LAB_LEVEL_METHODS = frozenset({"lab"})  # the lab's curated relations: true of every development there
DIRECTIONAL = ("positive", "negative", "mixed")

FACTS: dict[str, str] = {
    "event_type": "development event type (fli.taxonomy.EVENT_TYPES)",
    "p": "latest p_material for the audience (None when unscored)",
    "verified": "at least one insight whose evidence quotes were verified against the source",
    "has_official": "reported by an official lab channel",
    "lab_tier": "register tier of the development's lab (1 = material by default)",
    "person_tiers": "register tiers (A/B/C) of the tracked people involved",
    "held_exposure": "a BIT Capital holding is exposed to this development (event-specific link or "
                     "directional lens ticker; lab-level relations do not count)",
    "held_tickers": "BIT holdings among the event-specific exposed tickers",
    "exposed_tickers": "event-specific exposure-link tickers (not lab-level, >= exposure_min_confidence)",
    "has_code": "a code or weights link (GitHub, Hugging Face) is attached",
    "source_classes": "distinct source classes (official, research, personal, press, ...)",
    "n_source_classes": "number of distinct source classes",
    "corroborated": "n_source_classes >= guards.min_corroborating_classes",
    "rumor_only": "every insight is flagged as a rumor / unconfirmed report",
    "personal_only": "only personal posts so far",
    "source_count": "distinct sources",
    "magnitude": "max extractor magnitude (1-5)",
    "novelty": "max entity-level novelty of the insights",
    "lens_action": "lens action (watch/review_position/... or adopt/trial/assess/hold)",
    "breaking_change": "an official insight announces a deprecation, retirement, shutdown, re-route, "
                       "breaking change or changed default (migration work, often with a deadline)",
}
# Deterministic, on official insights only: the extractor rates such changelog lines magnitude 2-3,
# so magnitude-gated rules alone would never push them.
BREAKING_CHANGE = re.compile(
    r"\b(deprecat\w*|retir(?:e|es|ed|ing|ement)|end[- ]of[- ]life|sunset\w*|shut(?:s|ting)? down|shutdown"
    r"|re-?rout\w*|breaking changes?(?! expected)|by default|no longer (?:available|supported)|will be removed)\b",
    re.IGNORECASE)
GUARDS: dict[str, str] = {
    "guard.unverified": "no insight with verified evidence quotes: not alerted",
    "guard.rumor_cap": "rumor-only and not corroborated by independent source classes: capped",
    "guard.unconfirmed": "personal posts or uncorroborated reports only: marked unconfirmed",
    "guard.daily_cap": "daily push budget (P1+P2) for this audience used up: this P2 is delivered "
                       "digest-only (P3); P1 is never capped",
}
_SUFFIXES = ("_min", "_max", "_floor")
_COMBINATORS = ("any", "all", "not")


class PolicyError(ValueError):
    """config/materiality.yaml is invalid (raised at load time, never mid-run)."""


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    when: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class Policy:
    rules: dict[str, tuple[Rule, ...]]
    version: int = 1
    daily_cap: int = 2  # P1+P2 pushes per audience per UTC day; only P2 overflow is demoted (to P3)
    live_horizon_hours: int = 48
    lookback_days: int = 14
    exposure_min_confidence: float = 0.0
    require_verified: bool = True
    rumor_cap: str = "P3"
    min_corroborating_classes: int = 2
    digest_top_n: int = 12
    digest_min_p: float = 0.3

    def rule(self, rule_id: str) -> Rule | None:
        return next((r for rules in self.rules.values() for r in rules if r.id == rule_id), None)

    def describe(self, rule_id: str) -> str:
        if rule_id in GUARDS:
            return GUARDS[rule_id]
        rule = self.rule(rule_id)
        return rule.description if rule is not None else ""


# --------------------------------------------------------------------------------- policy loading
def _validate_condition(cond: Any, where: str) -> None:
    if not isinstance(cond, dict) or not cond:
        raise PolicyError(f"{where}: condition must be a non-empty mapping")
    for key, value in cond.items():
        if key in ("any", "all"):
            if not isinstance(value, list) or not value:
                raise PolicyError(f"{where}: '{key}' needs a non-empty list")
            for sub in value:
                _validate_condition(sub, f"{where}.{key}")
            continue
        if key == "not":
            _validate_condition(value, f"{where}.not")
            continue
        fact = next((key[: -len(s)] for s in _SUFFIXES if key.endswith(s)), key)
        if fact not in FACTS:
            raise PolicyError(f"{where}: unknown fact '{key}' (known: {', '.join(sorted(FACTS))})")
        if fact == "event_type":
            bad = [v for v in (value if isinstance(value, list) else [value]) if v not in EVENT_TYPES]
            if bad:
                raise PolicyError(f"{where}: unknown event type(s) {bad}")


def parse_policy(raw: dict) -> Policy:
    """Validate a parsed YAML policy. Typos fail loudly here instead of silently never firing."""
    rules: dict[str, tuple[Rule, ...]] = {}
    seen: set[str] = set()
    for audience, specs in (raw.get("audiences") or {}).items():
        if audience not in AUDIENCES:
            raise PolicyError(f"unknown audience '{audience}'")
        parsed = []
        for spec in specs or []:
            rid, sev = spec.get("id"), spec.get("severity")
            if not rid or rid in seen:
                raise PolicyError(f"rule id missing or duplicated: {rid!r}")
            if sev not in ("P1", "P2", "P3"):
                raise PolicyError(f"{rid}: severity must be P1, P2 or P3")
            _validate_condition(spec.get("when"), rid)
            seen.add(rid)
            parsed.append(Rule(id=rid, severity=sev, when=dict(spec["when"]), description=spec.get("description", "")))
        rules[audience] = tuple(parsed)
    guards, digest = raw.get("guards") or {}, raw.get("digest") or {}
    if guards.get("rumor_cap", "P3") not in ("P1", "P2", "P3"):
        raise PolicyError("guards.rumor_cap must be P1, P2 or P3")
    return Policy(
        rules=rules, version=int(raw.get("version", 1)), daily_cap=int(raw.get("daily_cap", 2)),
        live_horizon_hours=int(raw.get("live_horizon_hours", 48)), lookback_days=int(raw.get("lookback_days", 14)),
        exposure_min_confidence=float(raw.get("exposure_min_confidence", 0.0)),
        require_verified=bool(guards.get("require_verified", True)), rumor_cap=guards.get("rumor_cap", "P3"),
        min_corroborating_classes=int(guards.get("min_corroborating_classes", 2)),
        digest_top_n=int(digest.get("top_n", 12)), digest_min_p=float(digest.get("min_p", 0.3)),
    )


@lru_cache(maxsize=8)
def _load_cached(path: str, _mtime: float) -> Policy:
    return parse_policy(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def load_policy(path: Path | None = None) -> Policy:
    """Load (and cache by mtime, so a running worker picks up edits) the materiality policy."""
    p = Path(path or POLICY_PATH)
    return _load_cached(str(p), p.stat().st_mtime)


# --------------------------------------------------------------------------------- context loading
@dataclass
class DevContext:
    """Everything a materiality decision, an alert body or a digest item needs about one development."""

    dev: Development
    lab: Lab | None = None
    people: list[Person] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    documents: dict[int, Document] = field(default_factory=dict)
    sources: dict[str, Source] = field(default_factory=dict)
    exposures: list[ExposureLink] = field(default_factory=list)
    scores: dict[str, Score] = field(default_factory=dict)
    lenses: dict[str, Lens] = field(default_factory=dict)
    holdings: dict[str, float] = field(default_factory=dict)  # BIT holdings: symbol -> weight %
    code_doc_ids: set[int] = field(default_factory=set)

    def evidence(self, k: int = 3) -> list[tuple[str, Document | None]]:
        """Top verified quotes with their documents: official first, then confidence, then age."""
        ranked = sorted(self.insights, key=lambda i: (not i.is_official, -(i.confidence or 0.0),
                                                      i.published_at or datetime.max, i.id or 0))
        out: list[tuple[str, Document | None]] = []
        seen: set[str] = set()
        for ins in ranked:
            for text in verified_quotes(ins):
                norm = " ".join(text.lower().split())
                if norm and norm not in seen:
                    seen.add(norm)
                    out.append((text, self.documents.get(ins.document_id)))
        return out[:k]

    def links(self, k: int = 3) -> list[Document]:
        """Distinct source documents for citations: official first, then earliest published."""
        docs = sorted(self.documents.values(), key=lambda d: (d.source_class != "official",
                                                              d.published_at or datetime.max, d.id))
        out, seen = [], set()
        for d in docs:
            if d.canonical_url not in seen:
                seen.add(d.canonical_url)
                out.append(d)
        return out[:k]


def _quote(q: Any) -> tuple[str, bool]:
    """Evidence quotes may be plain strings or {quote|text, verified|ok|found} dicts."""
    if isinstance(q, str):
        return q.strip(), True
    if isinstance(q, dict):
        text = str(q.get("quote") or q.get("text") or "").strip()
        flag = next((q[k] for k in ("verified", "ok", "found") if k in q), True)
        return text, flag is not False
    return "", False


def verified_quotes(ins: Insight) -> list[str]:
    """Quotes that individually passed verification (Insight.verification['quotes'] when the
    extractor recorded per-quote checks), else all quotes of a fully verified insight."""
    checks = (ins.verification or {}).get("quotes")
    if isinstance(checks, list) and checks:
        return [str(c.get("quote", "")).strip() for c in checks if isinstance(c, dict) and c.get("verified")]
    if not ins.quotes_verified:
        return []
    return [text for text, ok in map(_quote, ins.evidence_quotes or []) if ok and text]


def _chunks(values: Iterable[Any], n: int = 500) -> Iterator[list[Any]]:
    items = list(values)
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _in(session: Session, stmt_fn, ids: Iterable[Any]) -> list:
    return [row for chunk in _chunks(ids) for row in session.scalars(stmt_fn(chunk))]


def load_contexts(session: Session, dev_ids: Iterable[int]) -> dict[int, DevContext]:
    """Batch-load contexts (a fixed number of queries regardless of how many developments)."""
    ids = sorted(set(dev_ids))
    if not ids:
        return {}
    devs = _in(session, lambda c: select(Development).options(defer(Development.centroid))
               .where(Development.id.in_(c)), ids)
    insights = _in(session, lambda c: select(Insight).options(defer(Insight.embedding))
                   .where(Insight.development_id.in_(c)).order_by(Insight.id), ids)
    doc_ids = sorted({i.document_id for i in insights})
    docs = {d.id: d for d in _in(session, lambda c: select(Document).options(
        defer(Document.content_text), defer(Document.minhash)).where(Document.id.in_(c)), doc_ids)}
    code_docs = set(_in(session, lambda c: select(Document.id).where(
        Document.id.in_(c), or_(*[Document.content_text.contains(h) for h in CODE_HOSTS])), doc_ids))
    sources = {s.id: s for s in _in(session, lambda c: select(Source).where(Source.id.in_(c)),
                                    {d.source_id for d in docs.values()})}
    exposures = _in(session, lambda c: select(ExposureLink).where(ExposureLink.development_id.in_(c)), ids)
    scores = _in(session, lambda c: select(Score).where(Score.development_id.in_(c))
                 .order_by(Score.created_at, Score.id), ids)
    lenses = _in(session, lambda c: select(Lens).where(Lens.development_id.in_(c)), ids)
    person_ids = {p for d in devs for p in (d.person_ids or [])} | {p for i in insights for p in (i.person_ids or [])}
    people = {p.id: p for p in _in(session, lambda c: select(Person).where(Person.id.in_(c)),
                                   {str(p) for p in person_ids})}
    labs = {lab.id: lab for lab in _in(session, lambda c: select(Lab).where(Lab.id.in_(c)),
                                       {d.lab_id for d in devs if d.lab_id})}
    holdings = {t.symbol: float(t.bit_weight) for t in session.scalars(select(Ticker).where(Ticker.bit_weight.is_not(None)))}

    out = {d.id: DevContext(dev=d, lab=labs.get(d.lab_id), holdings=holdings) for d in devs}
    for ins in insights:
        ctx = out[ins.development_id]
        ctx.insights.append(ins)
        if (doc := docs.get(ins.document_id)) is not None:
            ctx.documents[doc.id] = doc
            if doc.source_id in sources:
                ctx.sources[doc.source_id] = sources[doc.source_id]
            if doc.id in code_docs:
                ctx.code_doc_ids.add(doc.id)
    for link in exposures:
        out[link.development_id].exposures.append(link)
    for sc in scores:  # ordered by created_at: the last one wins = latest score
        out[sc.development_id].scores[sc.audience] = sc
    for lens in lenses:
        out[lens.development_id].lenses[lens.audience] = lens
    for ctx in out.values():
        pids = {str(p) for p in (ctx.dev.person_ids or [])} | {str(p) for i in ctx.insights for p in (i.person_ids or [])}
        ctx.people = [people[p] for p in sorted(pids) if p in people]
    return out


# --------------------------------------------------------------------------------- facts + rules
def _has_code(ctx: DevContext) -> bool:
    if ctx.code_doc_ids:
        return True
    for doc in ctx.documents.values():
        src = ctx.sources.get(doc.source_id)
        if (src is not None and src.kind in CODE_SOURCE_KINDS) or any(h in (doc.url or "") for h in CODE_HOSTS):
            return True
        if any(isinstance(v, str) and any(h in v for h in CODE_HOSTS) for v in (doc.meta or {}).values()):
            return True
    for ins in ctx.insights:
        for art in ins.artifacts or []:
            if isinstance(art, dict) and (art.get("kind") == "repo" or any(h in str(art) for h in CODE_HOSTS)):
                return True
            if isinstance(art, str) and any(h in art for h in CODE_HOSTS):
                return True
    return False


def is_lab_level(link: ExposureLink) -> bool:
    """A lab's curated relation (config/labs.yaml public_exposure): says nothing about this event."""
    return (link.method or "rule") in LAB_LEVEL_METHODS


def event_specific_links(ctx: DevContext, policy: Policy) -> list[ExposureLink]:
    """Exposure links about this development (not lab-level) at >= policy.exposure_min_confidence."""
    return [x for x in ctx.exposures
            if not is_lab_level(x) and (x.confidence or 0.0) >= policy.exposure_min_confidence]


def directional_lens_tickers(lens: Lens | None) -> set[str]:
    """Tickers the lens commits to (positive / negative / mixed); none when it abstains (no_action)."""
    if lens is None or lens.action == "no_action":
        return set()
    return {str(t.get("symbol", "")).upper() for t in (lens.tickers or [])
            if isinstance(t, dict) and t.get("symbol") and t.get("direction") in DIRECTIONAL}


def facts_for(ctx: DevContext, audience: str, policy: Policy) -> dict[str, Any]:
    """Flatten one development into the named facts the rules can test (see FACTS)."""
    dev, score, lens = ctx.dev, ctx.scores.get(audience), ctx.lenses.get(audience)
    classes = sorted({c for c in (dev.source_classes or []) if c}
                     | {d.source_class for d in ctx.documents.values() if d.source_class})
    exposed = sorted({x.ticker for x in event_specific_links(ctx, policy)})
    lens_tickers = directional_lens_tickers(lens) if audience == "investment" else set()
    held = sorted((set(exposed) | lens_tickers) & set(ctx.holdings))
    rumor_only = bool(ctx.insights) and all(i.is_rumor for i in ctx.insights)
    magnitudes = [i.magnitude for i in ctx.insights if i.magnitude is not None]
    novelty = [i.novelty_entity for i in ctx.insights if i.novelty_entity is not None]
    return {
        "event_type": dev.event_type,
        "p": float(score.p_material) if score is not None else None,
        "verified": bool(dev.has_verified) or any(i.quotes_verified for i in ctx.insights),
        "has_official": bool(dev.has_official) or "official" in classes or any(i.is_official for i in ctx.insights),
        "lab_tier": ctx.lab.tier if ctx.lab is not None else None,
        "person_tiers": sorted({p.tier for p in ctx.people if p.tier}),
        "held_exposure": bool(held),
        "held_tickers": held,
        "exposed_tickers": exposed,
        "has_code": _has_code(ctx),
        "source_classes": classes,
        "n_source_classes": len(classes),
        "corroborated": len(classes) >= policy.min_corroborating_classes,
        "rumor_only": rumor_only,
        "personal_only": classes == ["personal"],
        "source_count": dev.source_count or len({d.source_id for d in ctx.documents.values()}),
        "magnitude": max(magnitudes) if magnitudes else None,
        "novelty": max(novelty) if novelty else None,
        "lens_action": lens.action if lens is not None else None,
        "breaking_change": any(i.is_official and BREAKING_CHANGE.search(f"{i.title or ''} {i.claim or ''}")
                               for i in ctx.insights),
    }


def matches(cond: dict[str, Any], facts: dict[str, Any]) -> bool:
    """True when every condition holds (see the header of config/materiality.yaml)."""
    for key, expected in cond.items():
        if key == "any":
            ok = any(matches(c, facts) for c in expected)
        elif key == "all":
            ok = all(matches(c, facts) for c in expected)
        elif key == "not":
            ok = not matches(expected, facts)
        elif key.endswith("_floor"):
            value = facts.get(key[: -len("_floor")])
            ok = value is None or value >= expected
        elif key.endswith(("_min", "_max")):
            value = facts.get(key[:-4])
            ok = value is not None and (value >= expected if key.endswith("_min") else value <= expected)
        else:
            value = facts.get(key)
            if isinstance(expected, list):
                ok = bool(set(value) & set(expected)) if isinstance(value, list | set | tuple) else value in expected
            else:
                ok = value == expected
        if not ok:
            return False
    return True


def _fmt(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list | tuple | set):
        return ", ".join(str(v) for v in value) or "none"
    return str(value)


def _fact_of(key: str) -> str:
    return next((key[: -len(s)] for s in _SUFFIXES if key.endswith(s)), key)


def _test(key: str, expected: Any) -> str:
    if key.endswith("_min"):
        return f">= {expected}"
    if key.endswith("_max"):
        return f"<= {expected}"
    if key.endswith("_floor"):
        return f"floor {expected}"
    return "/".join(str(v) for v in expected) if isinstance(expected, list) else _fmt(expected)


def condition_reasons(cond: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """The facts that satisfied `cond`, e.g. ['event_type funding_ipo', 'magnitude 5 (>= 4)']: what an
    alert shows under 'why flagged'. Call it for a condition that matches; for `any`, the first branch
    that holds is reported."""
    out: list[str] = []
    for key, expected in cond.items():
        if key == "any":
            branch = next((c for c in expected if matches(c, facts)), None)
            out += condition_reasons(branch, facts) if branch is not None else []
        elif key == "all":
            for sub in expected:
                out += condition_reasons(sub, facts)
        elif key == "not":
            for k, e in expected.items():
                if k in _COMBINATORS:
                    out.append(f"not {k}: {e}")
                    continue
                fact = _fact_of(k)
                value = facts.get(fact)
                shown = "no lens yet" if (fact == "lens_action" and value is None) else f"{fact} {_fmt(value)}"
                out.append(f"{shown} (not {_test(k, e)})")
        elif key.endswith(("_min", "_max", "_floor")):
            fact = _fact_of(key)
            value = facts.get(fact)
            out.append(f"{fact} unknown ({_test(key, expected)} passes)" if value is None
                       else f"{fact} {_fmt(value)} ({_test(key, expected)})")
        elif expected is True:
            out.append(f"held_exposure ({_fmt(facts.get('held_tickers'))})" if key == "held_exposure" else key)
        elif expected is False:
            out.append(f"not {key}")
        else:
            value = facts.get(key)
            if isinstance(expected, list) and isinstance(value, list | set | tuple):
                value = sorted(set(value) & set(expected), key=str)
            out.append(f"{key} {_fmt(value)}")
    return out


def rule_reasons(rule_id: str, facts: dict[str, Any], policy: Policy) -> list[str]:
    """Matched facts of one fired rule ([] for guards, unknown ids and rules that do not match)."""
    rule = policy.rule(rule_id)
    return condition_reasons(rule.when, facts) if rule is not None and matches(rule.when, facts) else []


@dataclass
class Evaluation:
    audience: str
    severity: str | None  # after guards, before the daily cap (alerts apply that)
    rule_severity: str | None  # strongest firing rule
    fired: list[str]
    guards: list[str]
    unconfirmed: bool
    facts: dict[str, Any]
    development_id: int | None = None

    @property
    def rules(self) -> list[str]:
        return self.fired + self.guards


def max_severity(severities: Iterable[str | None]) -> str | None:
    return max(severities, key=lambda s: SEVERITY_RANK.get(s, 0), default=None)


def evaluate_facts(facts: dict[str, Any], audience: str, policy: Policy,
                   development_id: int | None = None) -> Evaluation:
    """Apply the audience's rules, then the global guards. Pure function: the unit-test surface."""
    fired = [r for r in policy.rules.get(audience, ()) if matches(r.when, facts)]
    rule_sev = max_severity(r.severity for r in fired)
    severity, guards = rule_sev, []
    if severity and policy.require_verified and not facts.get("verified"):
        severity = None
        guards.append("guard.unverified")
    rumor = bool(facts.get("rumor_only")) and facts.get("n_source_classes", 0) < policy.min_corroborating_classes
    if severity and rumor and SEVERITY_RANK[severity] > SEVERITY_RANK[policy.rumor_cap]:
        severity = policy.rumor_cap
        guards.append("guard.rumor_cap")
    unconfirmed = bool(facts.get("personal_only")) or rumor
    if severity and unconfirmed:
        guards.append("guard.unconfirmed")
    return Evaluation(audience=audience, severity=severity, rule_severity=rule_sev, fired=[r.id for r in fired],
                      guards=guards, unconfirmed=unconfirmed, facts=facts, development_id=development_id)


def evaluate(ctx: DevContext, audience: str, policy: Policy | None = None) -> Evaluation:
    policy = policy or load_policy()
    return evaluate_facts(facts_for(ctx, audience, policy), audience, policy, development_id=ctx.dev.id)


def explain(session: Session, development_id: int, policy: Policy | None = None) -> dict[str, Any]:
    """Per-audience decision with fired rules, the facts each rule matched on, and all facts
    (web 'why flagged', eval suites)."""
    policy = policy or load_policy()
    ctx = load_contexts(session, [development_id]).get(development_id)
    if ctx is None:
        return {}
    out = {}
    for audience in AUDIENCES:
        ev = evaluate(ctx, audience, policy)
        out[audience] = {"severity": ev.severity, "rule_severity": ev.rule_severity, "unconfirmed": ev.unconfirmed,
                         "rules": [{"id": r, "description": policy.describe(r),
                                    "matched": rule_reasons(r, ev.facts, policy)} for r in ev.rules],
                         "facts": ev.facts}
    return out
