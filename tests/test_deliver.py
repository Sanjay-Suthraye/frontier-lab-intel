"""Materiality rules, alert caps / dedup / escalation, and digest citation handling.

No network and no LLM: Slack is mocked with respx, the lens module and the LLM gateway are faked.
"""

from __future__ import annotations

import itertools
import sys
import types
from datetime import datetime, timedelta

import httpx
import pytest
import respx
from sqlalchemy import select

from fli import config as fli_config
from fli.db.models import (
    Alert,
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Lens,
    Person,
    Report,
    Score,
    Source,
    Ticker,
)
from fli.deliver import materiality as mat

WEBHOOK = "https://hooks.slack.test/services/T000/B000/XXXX"
NOW = datetime(2026, 11, 4, 12, 0, 0)
_seq = itertools.count(1)


@pytest.fixture(autouse=True)
def _no_outbound(monkeypatch):
    """Never pick up real webhooks / ping keys from .env inside tests."""
    for var in ("SLACK_WEBHOOK_URL", "HEALTHCHECKS_PING_BASE"):
        monkeypatch.setenv(var, "")
    monkeypatch.delenv("FLI_PUBLIC_URL", raising=False)
    fli_config.get_settings.cache_clear()
    yield
    fli_config.get_settings.cache_clear()


class LensCalls(list):
    """Recorded ensure_lens calls for missing lenses (refresh checks go to `refreshed`); `actions`
    sets the verdict the fake lens writes per audience."""

    def __init__(self):
        super().__init__()
        self.actions = {"investment": "watch", "ai_team": "assess"}
        self.refreshed: list[tuple[list[int], str]] = []


@pytest.fixture()
def fake_lens(monkeypatch):
    """Stand-in for fli.deliver.lens (owned by another module): records calls, writes Lens rows."""
    calls = LensCalls()

    def ensure_lens(development_ids, audience, refresh=False):
        from fli.db.session import session_scope

        if refresh:  # the real one regenerates lenses older than the evidence; the fake only records
            calls.refreshed.append((list(development_ids), audience))
            return 0
        calls.append((list(development_ids), audience))
        with session_scope() as s:
            for dev_id in development_ids:
                if s.scalar(select(Lens).where(Lens.development_id == dev_id, Lens.audience == audience)) is None:
                    s.add(Lens(development_id=dev_id, audience=audience, headline=f"Lens headline {dev_id}",
                               so_what="Generated so-what.", what_to_do="Generated action.", tickers=[],
                               action=calls.actions[audience], model="fake", prompt_version="fake@v1"))

    module = types.ModuleType("fli.deliver.lens")
    module.ensure_lens = ensure_lens
    monkeypatch.setitem(sys.modules, "fli.deliver.lens", module)
    return calls


def seed_world(s) -> None:
    s.add_all([
        Lab(id="openai", name="OpenAI", kind="frontier_lab", tier=1),
        Lab(id="mistral", name="Mistral AI", kind="open_weight_lab", tier=2),
        Lab(id="tml", name="Thinking Machines Lab", kind="alumni_startup", tier=3),
        Ticker(symbol="NVDA", company="NVIDIA", bit_weight=8.0),
        Ticker(symbol="MSFT", company="Microsoft", bit_weight=4.4),
        Ticker(symbol="ORCL", company="Oracle"),
        Ticker(symbol="BABA", company="Alibaba"),
    ])
    s.flush()
    s.add_all([
        Person(id="ada", name="Ada Lovelace", lab_id="openai", tier="A"),
        Person(id="bob", name="Bob Builder", lab_id="openai", tier="B"),
        Source(id="openai:news", lab_id="openai", adapter="rss", kind="official_blog", source_class="official"),
        Source(id="mistral:news", lab_id="mistral", adapter="rss", kind="official_blog", source_class="official"),
        Source(id="x:ada", person_id="ada", adapter="x_timeline", kind="x_account", source_class="personal"),
        Source(id="press:wire", adapter="rss", kind="press", source_class="press"),
        Source(id="arxiv:labs", adapter="arxiv", kind="arxiv", source_class="research"),
    ])
    s.flush()


def add_dev(s, *, title="Development", event_type="model_release", lab_id="openai", person_ids=(),
            sources=("openai:news",), verified=True, rumor=False, p_inv=None, p_ai=None, first_seen=None,
            tickers=(), content="", headline=None, magnitude=None, lab_tickers=(), inv_action="review_position",
            ai_action="trial", lens_tickers=None, artifacts=()) -> Development:
    """One development with a document + insight per source. `tickers` are event-specific exposure links
    (confidence 0.7), `lab_tickers` lab-level ones (method 'lab'); `headline` adds a lens per audience."""
    first_seen = first_seen or NOW - timedelta(hours=3)
    dev = Development(title=title, event_type=event_type, lab_id=lab_id, person_ids=list(person_ids),
                      first_seen_at=first_seen, last_seen_at=first_seen, summary=f"{title} summary.")
    s.add(dev)
    s.flush()
    classes = set()
    for src_id in sources:
        src = s.get(Source, src_id)
        classes.add(src.source_class)
        n = next(_seq)
        doc = Document(source_id=src_id, natural_key=f"k{n}", url=f"https://{src_id.split(':')[0]}.example/p/{n}",
                       canonical_url=f"https://{src_id.split(':')[0]}.example/p/{n}", title=f"{title} ({src_id})",
                       published_at=first_seen, content_text=content, content_hash=f"h{n}", lab_id=lab_id,
                       source_class=src.source_class, status="extracted")
        s.add(doc)
        s.flush()
        s.add(Insight(document_id=doc.id, development_id=dev.id, lab_id=lab_id, person_ids=list(person_ids),
                      event_type=event_type, title=title, claim=f"{title} claim.",
                      evidence_quotes=[f"verbatim quote {n} about {title}"], quotes_verified=verified,
                      is_official=src.source_class == "official", is_rumor=rumor, published_at=first_seen,
                      confidence=0.8, magnitude=magnitude, artifacts=list(artifacts), prompt_version="t", model="t"))
    dev.source_classes, dev.has_official = sorted(classes), "official" in classes
    dev.has_verified, dev.source_count, dev.insight_count = verified, len(sources), len(sources)
    for audience, p in (("investment", p_inv), ("ai_team", p_ai)):
        if p is not None:
            s.add(Score(development_id=dev.id, audience=audience, p_material=p, model_version="test-v1",
                        contributions={"lab_tier": 0.9, "official": 0.4, "novelty": -0.1}))
    for sym in tickers:
        s.add(ExposureLink(development_id=dev.id, ticker=sym, direction="positive", confidence=0.7))
    for sym in lab_tickers:
        s.add(ExposureLink(development_id=dev.id, ticker=sym, direction="unclear", confidence=0.6, method="lab",
                           rationale="OpenAI public exposure: investor"))
    if headline:
        lens_tickers = ([{"symbol": "NVDA", "direction": "positive", "why": "demand"}] if lens_tickers is None
                        else lens_tickers)
        for audience, action in (("investment", inv_action), ("ai_team", ai_action)):
            s.add(Lens(development_id=dev.id, audience=audience, headline=headline, so_what="It matters because X.",
                       what_to_do="Do Y.", tickers=lens_tickers, action=action, model="t", prompt_version="t"))
    s.flush()
    return dev


# ------------------------------------------------------------------------------- rules (pure)
BASE_FACTS = dict(event_type="other", p=None, verified=True, has_official=False, lab_tier=None, person_tiers=[],
                  held_exposure=False, held_tickers=[], exposed_tickers=[], has_code=False,
                  source_classes=["official"], n_source_classes=1, corroborated=False, rumor_only=False,
                  personal_only=False, source_count=1, magnitude=None, novelty=None, lens_action=None,
                  breaking_change=False)
TIER1_RELEASE = dict(event_type="model_release", lab_tier=1, has_official=True)
HELD = dict(held_exposure=True, held_tickers=["NVDA"])

RULE_CASES = [
    # (audience, facts, expected severity, expected rule/guard ids, unconfirmed)
    # -- investment: talent
    ("investment", dict(event_type="talent_move", person_tiers=["A"], lab_tier=3), "P1", {"inv.p1.key_talent"}, False),
    ("investment", dict(event_type="talent_move", lab_tier=1, magnitude=4), "P1", {"inv.p1.key_talent"}, False),
    ("investment", dict(event_type="talent_move", lab_tier=1, magnitude=3), None, set(), False),  # e.g. a comms hire
    ("investment", dict(event_type="new_venture", lab_tier=3, person_tiers=["B"]), "P2",
     {"inv.p2.key_talent_tier_b"}, False),
    ("investment", dict(event_type="talent_move", person_tiers=["A"], verified=False), None, set(), False),
    ("investment", dict(event_type="talent_move", person_tiers=["A"], lens_action="watch"), "P2",
     {"inv.p2.key_talent_tier_a"}, False),  # watch vetoes the P1 page, not the P2 push
    ("investment", dict(event_type="talent_move", person_tiers=["A"], lens_action="update_thesis"), "P1",
     {"inv.p1.key_talent"}, False),
    ("investment", dict(event_type="new_venture", person_tiers=["B"], p=0.3), None, set(), False),  # p floor
    # -- investment: capital markets (no p gate on the P1: a scorer miss must not silence an S-1)
    ("investment", dict(event_type="funding_ipo", lab_tier=1, magnitude=5, p=0.04), "P1",
     {"inv.p1.tier1_capital_markets"}, False),
    ("investment", dict(event_type="funding_ipo", lab_tier=1, magnitude=3), "P2", {"inv.p2.capital_markets"}, False),
    ("investment", dict(event_type="funding_ipo", lab_tier=1, magnitude=5, lens_action="watch"), "P2",
     {"inv.p2.capital_markets"}, False),  # the lens vetoes P1, not the push
    ("investment", dict(event_type="funding_ipo", lab_tier=1, magnitude=5, lens_action="no_action"), None, set(),
     False),
    ("investment", dict(event_type="funding_ipo", lab_tier=3, magnitude=5), None, set(), False),
    # -- investment: infrastructure (held_exposure is event-specific, see test_held_exposure_is_event_specific)
    ("investment", dict(event_type="compute_deal", magnitude=4, **HELD), "P1", {"inv.p1.major_infra_deal"}, False),
    ("investment", dict(event_type="compute_deal", magnitude=3, p=0.95, **HELD), "P2",
     {"inv.p2.infra_deal_on_holding"}, False),
    ("investment", dict(event_type="compute_deal", magnitude=5, p=0.95), None, set(), False),  # no held exposure
    # -- investment: releases
    ("investment", dict(TIER1_RELEASE, magnitude=5, p=0.9), "P1",
     {"inv.p1.flagship_release", "inv.p2.major_release"}, False),
    ("investment", dict(TIER1_RELEASE, magnitude=5), "P1", {"inv.p1.flagship_release"}, False),  # unscored: floor
    ("investment", dict(TIER1_RELEASE, magnitude=5, p=0.9, lens_action="watch"), "P2", {"inv.p2.major_release"},
     False),
    ("investment", dict(TIER1_RELEASE, magnitude=5, p=0.3), None, set(), False),
    ("investment", dict(TIER1_RELEASE, magnitude=4, p=0.8), None, set(), False),
    ("investment", dict(TIER1_RELEASE, magnitude=4, p=0.9, lens_action="no_action"), None, set(), False),
    ("investment", dict(TIER1_RELEASE, magnitude=5, p=0.9, verified=False), None,
     {"inv.p1.flagship_release", "guard.unverified"}, False),  # rule fires, global guard suppresses
    ("investment", dict(event_type="open_weight_release", lab_tier=2, corroborated=True, magnitude=4, p=0.95), "P2",
     {"inv.p2.challenger_release"}, False),
    ("investment", dict(event_type="partnership", lab_tier=1, magnitude=3, p=0.9, **HELD), "P2",
     {"inv.p2.strategic_tier1"}, False),
    ("investment", dict(event_type="pricing_change", lab_tier=1, magnitude=3, p=0.9, **HELD), None, set(), False),
    # -- investment: guards
    ("investment", dict(event_type="talent_move", person_tiers=["A"], rumor_only=True, source_classes=["press"]),
     "P3", {"inv.p1.key_talent", "guard.rumor_cap", "guard.unconfirmed"}, True),
    ("investment", dict(event_type="talent_move", person_tiers=["A"], rumor_only=True, n_source_classes=2,
                        source_classes=["press", "personal"]), "P1", {"inv.p1.key_talent"}, False),
    ("investment", dict(event_type="talent_move", person_tiers=["A"], personal_only=True,
                        source_classes=["personal"]), "P1", {"inv.p1.key_talent", "guard.unconfirmed"}, True),
    # -- ai_team
    ("ai_team", dict(event_type="open_weight_release", has_official=True, magnitude=4, p=0.8), "P1",
     {"ai.p1.major_release"}, False),
    ("ai_team", dict(event_type="open_weight_release", has_official=True, magnitude=4, p=0.8, lens_action="hold"),
     None, set(), False),
    ("ai_team", dict(event_type="open_weight_release", magnitude=3, p=0.85, has_code=True), "P2",
     {"ai.p2.release"}, False),
    ("ai_team", dict(event_type="open_weight_release", magnitude=3, p=0.85), None, set(), False),  # no code, 1 class
    ("ai_team", dict(event_type="model_release", lab_tier=1, magnitude=3, p=0.85, source_classes=["market_data"]),
     "P2", {"ai.p2.tier1_listing"}, False),
    ("ai_team", dict(event_type="product_launch", magnitude=4, p=0.8), "P2", {"ai.p2.platform_change"}, False),
    ("ai_team", dict(event_type="research_paper", magnitude=4, p=0.85), "P2", {"ai.p2.major_research"}, False),
    ("ai_team", dict(event_type="research_paper", magnitude=3, p=0.95, has_code=True), None, set(), False),
    ("ai_team", dict(event_type="product_launch", breaking_change=True, has_official=True, magnitude=2, p=0.55),
     "P2", {"ai.p2.breaking_change"}, False),
    ("ai_team", dict(event_type="product_launch", breaking_change=True, has_official=True, p=0.4), None, set(), False),
    ("ai_team", dict(event_type="product_launch", breaking_change=True, has_official=True, p=0.9,
                     lens_action="hold"), None, set(), False),
    ("ai_team", dict(event_type="product_launch", breaking_change=True, p=0.9, source_classes=["press"]), None,
     set(), False),  # deprecations are taken from official changelogs only
    ("ai_team", dict(TIER1_RELEASE, magnitude=5), None, set(), False),  # unscored
]


def test_every_push_rule_carries_the_lens_veto():
    """The lens can veto, never promote: investment P1 not no_action/watch, P2 not no_action, ai_team not hold."""
    policy = mat.load_policy()
    veto = {("investment", "P1"): ["no_action", "watch"], ("investment", "P2"): ["no_action"],
            ("ai_team", "P1"): ["hold"], ("ai_team", "P2"): ["hold"]}
    for audience, rules in policy.rules.items():
        for rule in rules:
            assert rule.severity in ("P1", "P2"), rule.id
            assert rule.when.get("not") == {"lens_action": veto[(audience, rule.severity)]}, rule.id
            assert "lens_action" not in rule.when, rule.id  # a lens verdict never raises severity by itself
    assert policy.daily_cap == 2 and policy.exposure_min_confidence == 0.6 and policy.version == 2


@pytest.mark.parametrize("audience,overrides,severity,rules,unconfirmed", RULE_CASES)
def test_rule_table(audience, overrides, severity, rules, unconfirmed):
    policy = mat.load_policy()
    facts = {**BASE_FACTS, **overrides}
    if "n_source_classes" not in overrides:
        facts["n_source_classes"] = len(facts["source_classes"])
    ev = mat.evaluate_facts(facts, audience, policy)
    assert ev.severity == severity
    assert rules <= set(ev.rules), ev.rules
    assert ev.unconfirmed is unconfirmed


def test_policy_validation_catches_typos():
    good = {"audiences": {"investment": [{"id": "r1", "severity": "P1", "when": {"p_min": 0.5}}]}}
    assert mat.parse_policy(good).rules["investment"][0].id == "r1"
    for bad in (
        {"audiences": {"investment": [{"id": "r1", "severity": "P1", "when": {"p_minimum": 0.5}}]}},
        {"audiences": {"investment": [{"id": "r1", "severity": "P1", "when": {"event_type": ["model_drop"]}}]}},
        {"audiences": {"investment": [{"id": "r1", "severity": "P0", "when": {"p_min": 0.5}}]}},
        {"audiences": {"pm_team": [{"id": "r1", "severity": "P1", "when": {"p_min": 0.5}}]}},
        {"audiences": {"investment": [{"id": "r1", "severity": "P1", "when": {"p_min": 0.5}},
                                      {"id": "r1", "severity": "P2", "when": {"p_min": 0.4}}]}},
    ):
        with pytest.raises(mat.PolicyError):
            mat.parse_policy(bad)


def test_facts_from_database(db):
    with db() as s:
        seed_world(s)
        dev = add_dev(s, event_type="research_paper", sources=("arxiv:labs", "x:ada"), person_ids=["ada"],
                      p_ai=0.8, tickers=("NVDA", "ORCL"), content="Code: https://github.com/openai/thing")
        s.commit()
        ctx = mat.load_contexts(s, [dev.id])[dev.id]
        facts = mat.facts_for(ctx, "ai_team", mat.load_policy())
    assert facts["has_code"] is True
    assert facts["person_tiers"] == ["A"]
    assert facts["held_tickers"] == ["NVDA"] and facts["exposed_tickers"] == ["NVDA", "ORCL"]
    assert facts["source_classes"] == ["personal", "research"] and facts["corroborated"] is True
    assert facts["p"] == 0.8 and facts["lab_tier"] == 1
    assert facts["breaking_change"] is False


def test_held_exposure_is_event_specific(db):
    """Lab-level relations, weak links and non-directional or abstaining lens tickers never make a
    development 'exposed to a BIT holding'."""
    with db() as s:
        seed_world(s)
        s.add(Ticker(symbol="AMZN", company="Amazon", bit_weight=9.6))
        s.add(Ticker(symbol="SIMO", company="Silicon Motion", bit_weight=3.9))
        s.flush()
        dev = add_dev(s, title="OpenAI ships a model", tickers=("ORCL",), lab_tickers=("MSFT",), headline="Model",
                      inv_action="watch", lens_tickers=[{"symbol": "AMZN", "direction": "positive", "why": "AWS"},
                                                        {"symbol": "NVDA", "direction": "unclear", "why": "maybe"}])
        s.add(ExposureLink(development_id=dev.id, ticker="SIMO", direction="unclear", confidence=0.45,
                           rationale="theme 'Memory': model_release + keywords [context window]"))
        s.commit()
        policy = mat.load_policy()
        ctx = mat.load_contexts(s, [dev.id])[dev.id]
        facts = mat.facts_for(ctx, "investment", policy)
        assert facts["exposed_tickers"] == ["ORCL"]  # MSFT is lab-level, SIMO below 0.6
        assert facts["held_tickers"] == ["AMZN"] and facts["held_exposure"] is True  # NVDA: no direction
        assert mat.facts_for(ctx, "ai_team", policy)["held_tickers"] == []  # lens tickers are investment-only
        ctx.lenses["investment"].action = "no_action"  # an abstaining lens contributes no tickers
        facts = mat.facts_for(ctx, "investment", policy)
        assert facts["held_tickers"] == [] and facts["held_exposure"] is False


def test_breaking_change_fact_reads_official_insights_only(db):
    with db() as s:
        seed_world(s)
        official = add_dev(s, title="Gemini 2.5 is deprecated and will be removed on Dec 1", event_type="product_launch")
        press = add_dev(s, title="Lab deprecates its old model", event_type="product_launch", sources=("press:wire",))
        preview = add_dev(s, title="Harness in preview, breaking changes expected", event_type="technique")
        s.commit()
        policy = mat.load_policy()
        ctxs = mat.load_contexts(s, [official.id, press.id, preview.id])
        assert mat.facts_for(ctxs[official.id], "ai_team", policy)["breaking_change"] is True
        assert mat.facts_for(ctxs[press.id], "ai_team", policy)["breaking_change"] is False
        assert mat.facts_for(ctxs[preview.id], "ai_team", policy)["breaking_change"] is False


def test_condition_reasons_name_the_facts_that_fired():
    policy = mat.load_policy()
    facts = {**BASE_FACTS, "event_type": "talent_move", "person_tiers": ["A", "C"], "lab_tier": 3, "p": 0.61}
    assert mat.rule_reasons("inv.p1.key_talent", facts, policy) == [
        "event_type talent_move", "verified", "person_tiers A", "no lens yet (not no_action/watch)"]
    facts = {**BASE_FACTS, **TIER1_RELEASE, "magnitude": 5, "p": 0.912, "lens_action": "update_thesis",
             "held_exposure": True, "held_tickers": ["NVDA", "TSM"]}
    assert mat.rule_reasons("inv.p1.flagship_release", facts, policy) == [
        "event_type model_release", "lab_tier 1", "has_official", "magnitude 5 (>= 5)", "p 0.91 (floor 0.5)",
        "lens_action update_thesis (not no_action/watch)"]
    assert mat.condition_reasons({"held_exposure": True, "p_floor": 0.5}, {**facts, "p": None}) == [
        "held_exposure (NVDA, TSM)", "p unknown (floor 0.5 passes)"]
    assert mat.rule_reasons("guard.daily_cap", facts, policy) == []
    assert mat.rule_reasons("ai.p2.release", facts, policy) == []  # does not match: nothing to explain


def test_explain_reports_matched_facts(db):
    with db() as s:
        seed_world(s)
        dev = add_dev(s, title="OpenAI files S-1", event_type="funding_ipo", magnitude=5, p_inv=0.97)
        s.commit()
        out = mat.explain(s, dev.id)
    inv = out["investment"]
    assert inv["severity"] == "P1" and inv["rules"][0]["id"] == "inv.p1.tier1_capital_markets"
    assert "magnitude 5 (>= 4)" in inv["rules"][0]["matched"]
    assert out["ai_team"]["severity"] is None


# ------------------------------------------------------------------------------- alerts
def _alerts(db, audience=None):
    with db() as s:
        q = select(Alert).order_by(Alert.id)
        if audience:
            q = q.where(Alert.audience == audience)
        return list(s.scalars(q))


def test_alert_body_cites_evidence_and_explains(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    with db() as s:
        seed_world(s)
        dev = add_dev(s, title="OpenAI ships GPT-6", sources=("openai:news", "press:wire"), p_inv=0.8, p_ai=0.8,
                      tickers=("NVDA", "MSFT"), headline="GPT-6 resets the frontier", magnitude=5)
        s.commit()
        dev_id = dev.id
    out = run_alerts(now=NOW)
    assert out["created"] == 2 and not fake_lens  # lenses existed: no lens generated ...
    assert sorted(fake_lens.refreshed) == [([dev_id], "ai_team"), ([dev_id], "investment")]  # ... only refresh-checked
    inv = _alerts(db, "investment")[0]
    assert inv.severity == "P1" and inv.dedup_key == f"investment:{dev_id}"
    assert inv.rules == ["inv.p1.flagship_release"]  # p 0.80 is below the P2 major-release bar (0.85)
    body = inv.body_md
    for needle in ("GPT-6 resets the frontier", "**So what:**", "**What to do:**", "NVDA (positive, BIT 8.0%): demand",
                   "verbatim quote", "https://openai.example/p/", "official, 2026-11-04", "Why flagged",
                   "p_material 0.80", "lab_tier (+0.90)",
                   "Matched: event_type model_release; lab_tier 1; has_official; magnitude 5 (>= 5); p 0.80 (floor 0.5); "
                   "lens_action review_position (not no_action/watch)"):
        assert needle in body, needle
    tickers = next(line for line in body.splitlines() if line.startswith("- **Tickers:**"))
    assert "MSFT" not in tickers  # a lens exists: its tickers only, no padding from exposure links
    assert inv.delivered["mode"] == "live" and inv.delivered["in_app"]
    assert inv.created_at == NOW
    ai = _alerts(db, "ai_team")[0]
    assert ai.severity == "P1" and ai.rules == ["ai.p1.major_release", "ai.p2.tier1_release", "ai.p2.release"]  # 2 source classes


def test_alert_tickers_fall_back_to_exposure_links_only_without_a_lens(db):
    from fli.deliver.alerts import run_alerts

    with db() as s:
        seed_world(s)
        add_dev(s, title="OpenAI signs compute deal", event_type="compute_deal", p_inv=0.95, magnitude=4,
                tickers=("NVDA", "ORCL"), lab_tickers=("MSFT",))
        s.commit()
    run_alerts(now=NOW, ensure_lenses=False)
    inv = _alerts(db, "investment")[0]
    assert inv.severity == "P1" and inv.rules == ["inv.p1.major_infra_deal", "inv.p2.infra_deal_on_holding"]
    assert ("- **Tickers:** NVDA (positive, BIT 8.0%); ORCL (positive); MSFT (lab-level link, BIT 4.4%)"
            in inv.body_md)
    assert "held_exposure (NVDA)" in inv.body_md and "no lens yet (not no_action/watch)" in inv.body_md
    assert "check the thesis for NVDA." in inv.body_md  # default action from the event-specific holding


def test_lens_verdict_applies_after_the_missing_lens_is_generated(db, fake_lens):
    """No lens at planning time passes the veto; the re-plan after ensure_lens applies the verdict."""
    from fli.deliver.alerts import run_alerts

    with db() as s:
        seed_world(s)
        watched = add_dev(s, title="Flagship A", p_inv=0.9, magnitude=5).id
        dismissed = add_dev(s, title="Flagship B", p_inv=0.9, magnitude=5, lab_id="openai").id
        s.commit()
    fake_lens.actions["investment"] = "watch"
    with db() as s:  # B gets an abstaining lens up front; A's lens is generated during the run ("watch")
        s.add(Lens(development_id=dismissed, audience="investment", headline="Immaterial", so_what="x",
                   what_to_do="y", tickers=[], action="no_action", model="t", prompt_version="t"))
        s.commit()
    run_alerts(now=NOW)
    alerts = {a.development_id: a for a in _alerts(db, "investment")}
    assert set(alerts) == {watched}  # no_action vetoes every investment rule
    assert alerts[watched].severity == "P2" and alerts[watched].rules == ["inv.p2.major_release"]
    assert fake_lens[0] == ([watched], "investment")


def test_daily_cap_never_demotes_p1_and_caps_p2_overflow(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    with db() as s:
        seed_world(s)
        p1 = add_dev(s, title="Flagship", p_inv=0.88, magnitude=5, headline="Landmark", inv_action="update_thesis").id
        p2 = [add_dev(s, title=f"Release {i}", p_inv=p, magnitude=4).id for i, p in enumerate((0.86, 0.95, 0.9))]
        s.commit()
    run_alerts(now=NOW)
    alerts = {a.development_id: a for a in _alerts(db, "investment")}
    assert alerts[p1].severity == "P1" and alerts[p2[1]].severity == "P2"  # budget of 2: P1 + the best P2
    capped = [alerts[p2[0]], alerts[p2[2]]]
    assert all(a.severity == "P3" and "guard.daily_cap" in a.rules for a in capped)
    assert all(a.title.startswith("Lens headline") for a in capped)
    assert len(fake_lens) == 1  # one batched ensure_lens call for the missing investment lenses
    # a landmark arriving after the budget is spent still pushes; a later P2 does not
    with db() as s:
        late_p1 = add_dev(s, title="Second flagship", p_inv=0.9, magnitude=5, headline="Landmark 2",
                          inv_action="update_thesis").id
        late_p2 = add_dev(s, title="Late release", p_inv=0.99, magnitude=4).id
        s.commit()
    run_alerts(now=NOW + timedelta(hours=1))
    alerts = {a.development_id: a for a in _alerts(db, "investment")}
    assert alerts[late_p1].severity == "P1" and "guard.daily_cap" not in alerts[late_p1].rules
    assert alerts[late_p2].severity == "P3" and "guard.daily_cap" in alerts[late_p2].rules
    assert "P1 is never capped" in mat.GUARDS["guard.daily_cap"]
    # re-running is a no-op: capped alerts are not re-pushed later the same day or the next
    again = run_alerts(now=NOW + timedelta(hours=2))
    assert again.get("created", 0) == 0 and again.get("escalated", 0) == 0
    assert len(_alerts(db, "investment")) == 6


def test_cap_contention_prefers_the_lens_verdict(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    with db() as s:
        seed_world(s)
        ids = [add_dev(s, title=f"Release {i}", p_inv=p, magnitude=4, headline=f"H{i}", inv_action=action).id
               for i, (p, action) in enumerate(((0.99, "watch"), (0.97, "watch"), (0.86, "update_thesis")))]
        s.commit()
    run_alerts(now=NOW)
    sev = {a.development_id: a.severity for a in _alerts(db, "investment")}
    assert sev == {ids[2]: "P2", ids[0]: "P2", ids[1]: "P3"}  # thesis update first, then by p


def test_dedup_and_realert_on_escalation(db, fake_lens, monkeypatch):
    from fli.deliver.alerts import run_alerts

    monkeypatch.setenv("SLACK_WEBHOOK_URL", WEBHOOK)
    fli_config.get_settings.cache_clear()
    with db() as s:
        seed_world(s)
        dev = add_dev(s, title="Ada joins rival lab", event_type="talent_move", lab_id="tml", person_ids=["ada"],
                      sources=("press:wire",), rumor=True, p_inv=0.7)
        s.commit()
        dev_id = dev.id
    fake_lens.actions["investment"] = "update_thesis"  # the lens backs the move (a 'watch' lens would veto P1)
    with respx.mock(assert_all_called=False) as router:
        route = router.post(WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        first = run_alerts(now=NOW)
        alert = _alerts(db, "investment")[0]
        assert first["created"] == 1 and alert.severity == "P3"
        assert {"guard.rumor_cap", "guard.unconfirmed"} <= set(alert.rules)
        assert alert.title.startswith("Unconfirmed: ")
        assert route.call_count == 0  # P3 is digest-only

        assert run_alerts(now=NOW + timedelta(minutes=30)).get("unchanged") == 1
        assert len(_alerts(db)) == 1

        with db() as s:  # the lab confirms: an official, non-rumor source joins the development
            src = s.get(Source, "openai:news")
            doc = Document(source_id=src.id, natural_key="confirm", url="https://openai.example/ada",
                           canonical_url="https://openai.example/ada", title="Welcome Ada", content_hash="hc",
                           published_at=NOW, source_class="official", status="extracted")
            s.add(doc)
            s.flush()
            s.add(Insight(document_id=doc.id, development_id=dev_id, event_type="talent_move", title="t",
                          claim="c", evidence_quotes=["Ada joins us"], quotes_verified=True, is_official=True,
                          published_at=NOW, prompt_version="t", model="t"))
            d = s.get(Development, dev_id)
            d.source_classes, d.last_seen_at = ["official", "press"], NOW
            s.commit()
        second = run_alerts(now=NOW + timedelta(hours=1))
        assert second["escalated"] == 1 and second["slack"]["sent"] == 1
        assert route.call_count == 1
        payload = route.calls[0].request.content.decode()
        assert "[P1] Investment" in payload
    alert = _alerts(db, "investment")[0]
    assert alert.severity == "P1" and not alert.title.startswith("Unconfirmed")
    assert [e["severity"] for e in alert.delivered["log"]] == ["P3", "P1"]
    assert alert.delivered["slack_pending"] is False and alert.delivered["log"][-1]["channels"] == ["in_app", "slack"]
    assert len(_alerts(db)) == 1  # still one row per audience + development


def test_historical_alerts_are_dated_and_not_pushed(db, fake_lens, monkeypatch):
    from fli.deliver.alerts import run_alerts

    monkeypatch.setenv("SLACK_WEBHOOK_URL", WEBHOOK)
    fli_config.get_settings.cache_clear()
    old = NOW - timedelta(days=40)
    with db() as s:
        seed_world(s)
        dev = add_dev(s, title="Old release", p_inv=0.9, first_seen=old, magnitude=5)
        s.flush()
        dev.updated_at = old  # untouched (and not re-scored) for 40 days: outside the 14-day lookback
        for score in s.scalars(select(Score).where(Score.development_id == dev.id)):
            score.created_at = old
        s.commit()
    with respx.mock(assert_all_called=False) as router:
        route = router.post(WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        assert run_alerts(now=NOW)["evaluated"] == 0
        out = run_alerts(now=NOW, backfill=True)
        assert out["created"] == 1 and route.call_count == 0
    alert = _alerts(db, "investment")[0]
    assert alert.created_at == old and alert.delivered["mode"] == "historical"
    assert "slack_pending" not in alert.delivered


def test_unverified_developments_never_alert(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    with db() as s:
        seed_world(s)
        add_dev(s, title="Unverified claim", verified=False, p_inv=0.95, p_ai=0.95, magnitude=5)
        s.commit()
    assert run_alerts(now=NOW).get("created", 0) == 0
    assert _alerts(db) == []


def test_slack_failure_is_recorded_and_retried(db, fake_lens, monkeypatch):
    from fli.deliver.alerts import run_alerts

    monkeypatch.setenv("SLACK_WEBHOOK_URL", WEBHOOK)
    fli_config.get_settings.cache_clear()
    with db() as s:
        seed_world(s)
        add_dev(s, title="Release", p_inv=0.9, magnitude=5)
        s.commit()
    with respx.mock(assert_all_called=False) as router:
        route = router.post(WEBHOOK).mock(side_effect=[httpx.Response(500, text="boom"), httpx.Response(200, text="ok")])
        assert run_alerts(now=NOW)["slack"] == {"sent": 0, "failed": 1}
        alert = _alerts(db, "investment")[0]
        assert alert.delivered["slack_pending"] is True and "HTTP 500" in alert.delivered["slack_error"]
        assert WEBHOOK not in str(alert.delivered)
        assert run_alerts(now=NOW + timedelta(minutes=30))["slack"] == {"sent": 1, "failed": 0}
        assert route.call_count == 2


# ------------------------------------------------------------------------------- follow-ups
LAUNCH = NOW - timedelta(days=5)


def add_insight(s, dev_id, *, title, event_type="product_launch", magnitude=3, published, source="openai:news",
                artifacts=(), first_seen=(), rumor=False, verified=True, created=None) -> Insight:
    """One more document + insight joining a development; aggregates updated as clustering would."""
    dev, src = s.get(Development, dev_id), s.get(Source, source)
    n = next(_seq)
    doc = Document(source_id=source, natural_key=f"k{n}", url=f"https://{source.split(':')[0]}.example/p/{n}",
                   canonical_url=f"https://{source.split(':')[0]}.example/p/{n}", title=title, published_at=published,
                   content_text="", content_hash=f"h{n}", lab_id=dev.lab_id, source_class=src.source_class,
                   status="extracted")
    s.add(doc)
    s.flush()
    ins = Insight(document_id=doc.id, development_id=dev_id, lab_id=dev.lab_id, event_type=event_type, title=title,
                  claim=f"{title}, the post says.", evidence_quotes=[f"verbatim quote {n} about {title}"],
                  quotes_verified=verified, is_official=src.source_class == "official", is_rumor=rumor,
                  published_at=published, confidence=0.8, magnitude=magnitude, artifacts=list(artifacts),
                  first_seen_artifacts=list(first_seen), prompt_version="t", model="t",
                  **({"created_at": created} if created else {}))
    s.add(ins)
    dev.last_seen_at = max(dev.last_seen_at, published)
    dev.source_classes = sorted({*(dev.source_classes or []), src.source_class})
    dev.has_official = bool(dev.has_official) or src.source_class == "official"
    dev.insight_count = (dev.insight_count or 0) + 1
    s.flush()
    return ins


def astra_launch(db) -> int:
    """GPT-6 Astra launch, alerted P1 to both audiences right away (lenses exist)."""
    from fli.deliver.alerts import run_alerts

    with db() as s:
        seed_world(s)
        s.add(Source(id="openrouter:models", adapter="openrouter", kind="model_index", source_class="market_data"))
        dev_id = add_dev(s, title="OpenAI releases GPT-6 Astra", p_inv=0.8, p_ai=0.8, magnitude=5, first_seen=LAUNCH,
                         headline="GPT-6 Astra resets the frontier", inv_action="update_thesis",
                         artifacts=["GPT-6 Astra"]).id
        s.commit()
    out = run_alerts(now=LAUNCH + timedelta(minutes=30))
    assert out["created"] == 2 and out["followups"] == 0
    return dev_id


def test_new_sub_event_in_an_alerted_development_gets_a_follow_up(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    dev_id = astra_launch(db)
    parents = {a.audience: a for a in _alerts(db)}
    assert parents["investment"].delivered["covers"] == [i.id for i in _insights(db, dev_id)]
    with db() as s:  # the launch burst and later re-reports of the same launch are not news ...
        add_insight(s, dev_id, title="OpenAI introduces GPT-6 Astra, a new flagship generation",
                    event_type="model_release", magnitude=5, published=LAUNCH + timedelta(hours=3),
                    artifacts=["GPT-6 Astra"], first_seen=["GPT-6 Astra Max"])  # within 12h: the burst
        add_insight(s, dev_id, title="GPT-6 Astra Pro listed on OpenRouter", event_type="model_release",
                    magnitude=4, published=LAUNCH + timedelta(days=1), source="openrouter:models",
                    artifacts=["GPT-6 Astra Pro"], first_seen=["GPT-6 Astra Pro"])  # a listing
        add_insight(s, dev_id, title="OpenAI launches GPT-6 Astra for business", event_type="model_release",
                    magnitude=4, published=LAUNCH + timedelta(days=3), artifacts=["GPT-6 Astra"])  # nothing new
        s.commit()
    quiet = run_alerts(now=LAUNCH + timedelta(days=3, hours=1))
    assert quiet["followups"] == 0 and len(_alerts(db)) == 2
    with db() as s:  # ... a new product built on it is
        sub = add_insight(s, dev_id, title="OpenAI launches ChatGPT for Financial Services with GPT-6 Astra",
                          published=NOW - timedelta(hours=5), artifacts=["ChatGPT for Financial Services", "GPT-6 Astra"],
                          first_seen=["ChatGPT for Financial Services"]).id
        s.commit()
    out = run_alerts(now=NOW)
    assert out["followups"] == 2 and out["created"] == 0 and out["by_severity"] == {"investment:P2": 1, "ai_team:P2": 1}
    follow = {a.audience: a for a in _alerts(db) if a.dedup_key.count(":") == 2}
    inv = follow["investment"]
    assert inv.dedup_key == f"investment:{dev_id}:{sub}" and inv.severity == "P2"  # the sub-event is not a landmark
    assert inv.title == "Follow-up: OpenAI launches ChatGPT for Financial Services with GPT-6 Astra"
    assert inv.rules[0] == "followup.sub_event" and "inv.p1.flagship_release" in inv.rules
    assert inv.created_at == NOW and inv.delivered["mode"] == "live" and inv.status == "open"
    assert inv.delivered["followup_of"] == parents["investment"].id and inv.delivered["insight_id"] == sub
    assert inv.delivered["covers"] == [i.id for i in _insights(db, dev_id)]
    body = inv.body_md
    for needle in (f"New since P1 alert #{parents['investment'].id} of {LAUNCH:%Y-%m-%d}: GPT-6 Astra resets the frontier",
                   "- **What is new:** OpenAI launches ChatGPT for Financial Services with GPT-6 Astra, the post says.",
                   "- **So what:** It matters because X.", "- **Tickers:** NVDA (positive, BIT 8.0%): demand",
                   "verbatim quote", "follow-up 1 of at most 2; development rules: P1",
                   "Matched: new event type product_launch; first mention of ChatGPT for Financial Services; "
                   "magnitude 3",
                   "`inv.p1.flagship_release`"):
        assert needle in body, needle
    evidence = body.split("**Evidence**", 1)[1]
    assert evidence.index("about OpenAI launches ChatGPT") < evidence.index("about OpenAI releases GPT-6 Astra")
    assert all((a.title, a.severity) == (parents[a.audience].title, "P1") for a in _alerts(db) if a.id in
               {p.id for p in parents.values()})  # the original alerts are untouched
    assert ([dev_id], "investment") in fake_lens.refreshed  # the lens is refreshed before the push if stale
    again = run_alerts(now=NOW + timedelta(hours=1))
    assert again["followups"] == 0 and len(_alerts(db)) == 4  # idempotent: the follow-up covers everything


def _insights(db, dev_id) -> list[Insight]:
    with db() as s:
        return list(s.scalars(select(Insight).where(Insight.development_id == dev_id).order_by(Insight.id)))


def test_follow_ups_are_capped_per_development_and_by_the_daily_budget(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    dev_id = astra_launch(db)
    day = NOW - timedelta(days=3)
    with db() as s:  # two other investment pushes already used that day's budget
        for k, sev in enumerate(("P1", "P2")):
            s.add(Alert(development_id=dev_id, audience="investment", severity=sev, title="other", body_md="b",
                        rules=[], dedup_key=f"investment:other{k}", created_at=day - timedelta(hours=k + 1)))
        add_insight(s, dev_id, title="OpenAI cuts GPT-6 Astra cached input price", event_type="pricing_change",
                    magnitude=3, published=day - timedelta(hours=1))
        s.commit()
    run_alerts(now=day)
    capped = next(a for a in _alerts(db, "investment") if a.dedup_key.count(":") == 2)
    assert capped.severity == "P3" and capped.rules[-1] == "guard.daily_cap"  # P2 overflow: digest only
    assert capped.delivered["guarded_severity"] == "P2"
    assert next(a for a in _alerts(db, "ai_team") if a.dedup_key.count(":") == 2).severity == "P2"
    with db() as s:  # a landmark sub-event of a P1 development is pushed at P1, budget or not
        landmark = add_insight(s, dev_id, title="OpenAI files S-1 for a Nasdaq listing", event_type="funding_ipo",
                               magnitude=5, published=day + timedelta(days=1), first_seen=["OpenAI S-1"]).id
        s.commit()
    run_alerts(now=day + timedelta(days=1, hours=1))
    p1 = next(a for a in _alerts(db, "investment") if a.dedup_key.endswith(f":{landmark}"))
    assert p1.severity == "P1" and "guard.daily_cap" not in p1.rules
    with db() as s:  # a third sub-event: the development already has two follow-ups per audience
        add_insight(s, dev_id, title="OpenAI partners with Oracle on Astra capacity", event_type="partnership",
                    magnitude=4, published=NOW - timedelta(hours=2), first_seen=["Stargate Astra"])
        s.commit()
    assert run_alerts(now=NOW)["followups"] == 0
    assert sum(a.dedup_key.count(":") == 2 for a in _alerts(db)) == 4  # 2 per audience


def test_follow_up_needs_a_qualifying_development_and_an_unsuppressed_alert(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    dev_id = astra_launch(db)
    with db() as s:
        s.scalar(select(Lens).where(Lens.development_id == dev_id, Lens.audience == "investment")).action = "no_action"
        s.scalar(select(Alert).where(Alert.dedup_key == f"ai_team:{dev_id}")).status = "suppressed"
        add_insight(s, dev_id, title="OpenAI launches ChatGPT for Financial Services", published=NOW - timedelta(hours=2),
                    first_seen=["ChatGPT for Financial Services"])
        add_insight(s, dev_id, title="Rumor: OpenAI plans Astra phone", event_type="strategy_shift", magnitude=4,
                    published=NOW - timedelta(hours=2), source="press:wire", rumor=True, first_seen=["Astra phone"])
        s.commit()
    out = run_alerts(now=NOW)  # investment lens now abstains (the veto applies); ai_team alert was suppressed
    assert out["followups"] == 0 and len(_alerts(db)) == 2


def test_ai_team_follows_up_on_an_official_deprecation_and_old_alerts_count_what_they_saw(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    dev_id = astra_launch(db)
    rendered = LAUNCH + timedelta(minutes=30)
    with db() as s:  # an alert from before coverage was recorded: it covers what existed when rendered
        for a in s.scalars(select(Alert)):
            a.delivered = {k: v for k, v in a.delivered.items() if k != "covers"} | {"in_app": rendered.isoformat()}
        add_insight(s, dev_id, title="OpenAI Responses API usage docs", event_type="technique", magnitude=3,
                    published=NOW - timedelta(days=4), created=rendered - timedelta(minutes=5),
                    first_seen=["Responses cookbook"])  # extracted before the render: covered
        add_insight(s, dev_id, title="gpt-5.5 is deprecated and will be removed on 2026-12-01", magnitude=2,
                    published=NOW - timedelta(hours=3), artifacts=["gpt-5.5"], first_seen=["gpt-5.5-legacy"])
        s.commit()
    out = run_alerts(now=NOW)
    assert out["followups"] == 1  # magnitude 2: material for the AI team as a breaking change only
    (follow,) = [a for a in _alerts(db) if a.dedup_key.count(":") == 2]
    assert follow.audience == "ai_team" and follow.severity == "P2"
    assert "official deprecation / breaking change" in follow.body_md
    assert "- **What is new:** gpt-5.5 is deprecated" in follow.body_md and "Also new" not in follow.body_md


def test_follow_up_reasons_need_something_new_and_material():
    from fli.deliver.alerts import followup_reasons

    def ins(i, **kw) -> Insight:
        base = dict(id=i, event_type="model_release", title="t", claim="c", evidence_quotes=["a verified quote"],
                    quotes_verified=True, magnitude=3, is_official=True, artifacts=[], first_seen_artifacts=[])
        return Insight(**{**base, **kw})

    covered = [ins(1, artifacts=["Timestral-5M", "GPT-6 Astra", "SWE-bench Verified"])]
    assert followup_reasons(ins(2, first_seen_artifacts=["GPT-6 Astra Pro"]), covered, "investment") == [
        "first mention of GPT-6 Astra Pro", "magnitude 3"]  # a new variant of a named family is new
    assert followup_reasons(ins(3, first_seen_artifacts=["Timestral"]), covered, "ai_team") == []  # family already named
    assert followup_reasons(ins(4, first_seen_artifacts=["ChatGPT"]), covered, "ai_team") == []  # umbrella brand
    assert followup_reasons(ins(5, magnitude=5), covered, "ai_team") == []  # a re-report, however big
    assert followup_reasons(ins(6, event_type="pricing_change"), covered, "ai_team") == [
        "new event type pricing_change", "magnitude 3"]
    assert followup_reasons(ins(7, event_type="other"), covered, "ai_team") == []  # the catch-all is no new type
    assert followup_reasons(ins(8, event_type="pricing_change", is_rumor=True), covered, "ai_team") == []
    assert followup_reasons(ins(9, event_type="pricing_change", quotes_verified=False), covered, "ai_team") == []
    assert followup_reasons(ins(10, event_type="pricing_change", magnitude=2), covered, "ai_team") == []
    sunset = ins(11, event_type="product_launch", magnitude=2, title="GPT-5 will be retired on Jan 5")
    assert followup_reasons(sunset, covered, "ai_team")[-1] == "official deprecation / breaking change"
    assert followup_reasons(sunset, covered, "investment") == []


def test_live_follow_up_is_pushed_to_slack(db, fake_lens, monkeypatch):
    from fli.deliver.alerts import run_alerts

    dev_id = astra_launch(db)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", WEBHOOK)
    fli_config.get_settings.cache_clear()
    with db() as s:
        add_insight(s, dev_id, title="OpenAI launches ChatGPT for Financial Services", published=NOW - timedelta(hours=2),
                    first_seen=["ChatGPT for Financial Services"])
        s.commit()
    with respx.mock(assert_all_called=False) as router:
        route = router.post(WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        out = run_alerts(now=NOW)
    assert out["followups"] == 2 and out["slack"] == {"sent": 2, "failed": 0}
    payloads = [c.request.content.decode() for c in route.calls]
    assert any("[P2] Investment: Follow-up: OpenAI launches ChatGPT for Financial Services" in p for p in payloads)


def test_historical_alerts_take_the_refreshed_lens_and_pushed_ones_stay_as_sent(db, fake_lens):
    from fli.deliver.alerts import run_alerts

    old = NOW - timedelta(days=40)
    with db() as s:
        seed_world(s)
        backfilled = add_dev(s, title="Old release", p_inv=0.9, first_seen=old, magnitude=5,
                             headline="Rumor-stage headline", inv_action="update_thesis").id
        s.commit()
    run_alerts(now=NOW, backfill=True)
    with db() as s:
        pushed = add_dev(s, title="New release", p_inv=0.9, magnitude=5, headline="Live headline",
                         inv_action="update_thesis").id
        s.commit()
    run_alerts(now=NOW)
    before = {a.development_id: a for a in _alerts(db, "investment")}
    assert before[backfilled].delivered["mode"] == "historical" and before[pushed].delivered["mode"] == "live"
    with db() as s:  # lens.run_lens regenerated both lenses after the lab confirmed
        for lens in s.scalars(select(Lens).where(Lens.audience == "investment")):
            lens.headline, lens.so_what = "Confirmed-stage headline", "Now backed by the lab's own post."
            lens.checks = {"refresh": {"count": 1, "at": (NOW + timedelta(minutes=10)).isoformat(timespec="seconds")}}
        s.commit()
    out = run_alerts(now=NOW + timedelta(minutes=30))
    assert out["rerendered"] == 1
    after = {a.development_id: a for a in _alerts(db, "investment")}
    hist = after[backfilled]
    assert hist.title == "Confirmed-stage headline" and "Now backed by the lab's own post." in hist.body_md
    assert (hist.severity, hist.rules, hist.created_at) == (before[backfilled].severity, before[backfilled].rules, old)
    assert after[pushed].title == "Live headline"  # what was pushed stays on record
    assert "rerendered" not in run_alerts(now=NOW + timedelta(hours=1))  # idempotent


# ------------------------------------------------------------------------------- digests
WEEK = datetime(2026, 10, 26)  # Monday; the last full week before NOW
IN_WEEK = datetime(2026, 10, 28, 10, 0)


def fake_gateway(monkeypatch, outputs: dict | None = None, fail: bool = False) -> list[dict]:
    """Replace the digest's LLM gateway: records calls, returns canned structured output."""
    from fli.deliver import digest
    from fli.llm.client import LLMResult

    calls: list[dict] = []

    class FakeGateway:
        def __init__(self, settings=None):
            if fail:
                raise RuntimeError("ANTHROPIC_API_KEY is not set; LLM stages cannot run")

        async def structured(self, **kw):
            calls.append(kw)
            return LLMResult((outputs or {})[kw["ref"]["audience"]], "ok", 0.05, kw["model"], "digest@v1+test")

    monkeypatch.setattr(digest, "LLMGateway", FakeGateway)
    return calls


def _investment_week(s) -> dict[str, int]:
    seed_world(s)
    return {
        "a": add_dev(s, title="OpenAI ships GPT-6", p_inv=0.8, first_seen=IN_WEEK, tickers=("NVDA",),
                     headline="GPT-6 resets the frontier").id,
        "b": add_dev(s, title="OpenAI signs compute deal", event_type="compute_deal", p_inv=0.6,
                     first_seen=IN_WEEK, tickers=("MSFT",)).id,
        "u": add_dev(s, title="Ada hints at a move", event_type="talent_move", sources=("x:ada",),
                     person_ids=["ada"], p_inv=0.5, first_seen=IN_WEEK).id,
        "d": add_dev(s, title="Minor OpenAI update", p_inv=0.4, first_seen=IN_WEEK).id,
        "low": add_dev(s, title="Low relevance", p_inv=0.2, first_seen=IN_WEEK).id,
        "unverified": add_dev(s, title="Unverified rumour", verified=False, p_inv=0.9, first_seen=IN_WEEK).id,
        "other_week": add_dev(s, title="Last month", p_inv=0.9, first_seen=datetime(2026, 10, 1)).id,
    }


INVESTMENT_OUTPUT = {
    "headline": "GPT-6 is the story of the week [1][42].",
    "top_of_mind": [
        {"items": [1], "title": "GPT-6 ships", "text": "OpenAI shipped GPT-6 [1, 99], see [details](https://evil.example/x).",
         "tickers": ["NVDA", "ZZZZ"], "action": "review_position"},
        {"items": [3], "title": "Ada may move", "text": "Ada might join a rival [3].", "tickers": [], "action": "watch"},
        {"items": [99], "title": "Invented", "text": "Something the model made up [99].", "tickers": [],
         "action": "watch"},
    ],
    "by_theme": [{"items": [2], "title": "Hyperscalers", "text": "The compute deal favours MSFT."}],
    "watchlist": [],
}


def test_digest_maps_citations_to_sources_and_drops_invalid(db, fake_lens, monkeypatch):
    from fli.deliver.digest import run_digest

    calls = fake_gateway(monkeypatch, {"investment": INVESTMENT_OUTPUT})
    with db() as s:
        ids = _investment_week(s)
        s.commit()
    out = run_digest(audience="investment", week="2026-10-28", now=NOW)
    assert out["reports"][0]["status"] == "created" and len(calls) == 1
    call = calls[0]
    assert call["effort"] == "high" and call["purpose"] == "digest"  # medium under-generated (effort sweep)
    assert call["model"] == fli_config.get_settings().model_digest
    assert "[1] OpenAI ships GPT-6" in call["user"] and "UNCONFIRMED" in call["user"]
    assert "Low relevance" not in call["user"] and "Unverified rumour" not in call["user"]
    with db() as s:
        rep = s.scalar(select(Report).where(Report.audience == "investment"))
    body = rep.body_md
    assert rep.period_start == WEEK and rep.period_end == WEEK + timedelta(days=7)
    assert rep.model == fli_config.get_settings().model_digest
    assert [i["development_id"] for i in rep.items] == [ids["a"], ids["b"], ids["u"], ids["d"]]
    assert [i["rank"] for i in rep.items] == [1, 2, 3, 4] and rep.items[0]["p_material"] == 0.8
    assert "OpenAI shipped GPT-6 [1], see details." in body
    assert "[99]" not in body and "[42]" not in body and "Invented" not in body and "evil.example" not in body
    assert "_Tickers: NVDA; Suggested action: review position_" in body and "ZZZZ" not in body
    assert "The compute deal favours MSFT. [2]" in body  # declared item appended as citation
    assert body.index("## Watchlist (unconfirmed)") < body.index("Ada might join a rival [3]")
    labs = body_section(body, "Quiet labs")  # no documents that week: not monitored, not "quiet"
    assert "Not monitored this week (no documents collected, so silence is not evidence): Mistral AI (tier 2); " \
           "Thinking Machines Lab (alumni startups)." in labs
    assert "Every tracked lab with documents this week had at least one material development." in labs
    sources = body.split("## Sources", 1)[1]
    assert "- [1] OpenAI ships GPT-6 (OpenAI, 2026-10-28): [" in sources and "https://openai.example/p/" in sources
    assert "- [3] Ada hints at a move (unconfirmed)" in sources
    # the model left item 4 out: it is listed deterministically, never silently dropped
    also = body_section(body, "Also this week").strip()
    assert also == f"- **Lens headline {ids['d']}** (watch): Generated so-what. Generated action. [4]"
    assert "- [4] Minor OpenAI update (OpenAI, 2026-10-28)" in sources
    assert [i["cited"] for i in rep.items] == [True, True, True, True]
    assert [i["also_this_week"] for i in rep.items] == [False, False, False, True]
    holdings = body_section(body, "Holdings check")
    assert "- **NVDA** (8.0%): positive. Top item: GPT-6 resets the frontier [1]." in holdings
    assert f"- **MSFT** (4.4%): direction not assessed. Top item: Lens headline {ids['b']} [2]." in holdings


def test_digest_nothing_material_skips_the_model(db, monkeypatch):
    from fli.deliver.digest import run_digest

    calls = fake_gateway(monkeypatch, fail=True)
    with db() as s:
        seed_world(s)
        add_dev(s, title="Only noise", p_inv=0.1, p_ai=0.1, first_seen=IN_WEEK)
        s.commit()
    out = run_digest(audience="all", week=WEEK.date(), now=NOW)
    assert {r["status"] for r in out["reports"]} == {"created"} and not calls
    with db() as s:
        reports = list(s.scalars(select(Report).order_by(Report.audience)))
    assert len(reports) == 2
    for rep in reports:
        assert "Nothing material this week." in rep.body_md and rep.model is None and rep.items == []
    assert "## Quiet labs" in reports[1].body_md  # investment lists the quiet labs


def test_digest_falls_back_without_model_then_regenerates(db, fake_lens, monkeypatch):
    from fli.deliver.digest import run_digest

    fake_gateway(monkeypatch, fail=True)
    with db() as s:
        _investment_week(s)
        s.commit()
    first = run_digest(audience="investment", now=NOW)
    assert first["reports"][0]["model"] == "fallback" and "model unavailable" in first["reports"][0]["note"]
    with db() as s:
        rep = s.scalar(select(Report))
    assert "GPT-6 resets the frontier" in rep.body_md and "[1]" in rep.body_md and "## Sources" in rep.body_md
    top = body_section(rep.body_md, "Top of mind")
    assert top.count("### ") == 3 and "[3]" not in top  # the three confirmed items; unconfirmed kept out
    assert "[3]" in body_section(rep.body_md, "Watchlist (unconfirmed)")
    calls = fake_gateway(monkeypatch, {"investment": INVESTMENT_OUTPUT})
    second = run_digest(audience="investment", now=NOW)
    assert second["reports"][0]["status"] == "updated" and len(calls) == 1
    third = run_digest(audience="investment", now=NOW)
    assert third["reports"][0]["status"] == "exists" and len(calls) == 1
    with db() as s:
        assert s.query(Report).count() == 1


def body_section(body: str, heading: str) -> str:
    return body.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]


def test_digest_ai_team_radar_rules(db, fake_lens, monkeypatch):
    from fli.deliver.digest import run_digest

    output = {"headline": "", "papers": [], "tools_techniques": [
        {"items": [2], "title": "Speculative decoding recipe", "text": "Cuts latency [2]."}],
        "radar": [{"items": [1], "ring": "trial", "title": "Open model", "text": "Try it on extraction [1]."},
                  {"items": [3], "ring": "adopt", "title": "Rumoured model", "text": "Said to be great [3]."},
                  {"items": [2], "ring": "hold", "title": "Empty entry", "text": " [2]"}]}  # seen live: empty text
    calls = fake_gateway(monkeypatch, {"ai_team": output})
    with db() as s:
        seed_world(s)
        add_dev(s, title="Mistral open weights", event_type="open_weight_release", lab_id="mistral",
                sources=("mistral:news",), p_ai=0.9, first_seen=IN_WEEK)
        add_dev(s, title="Decoding trick", event_type="technique", p_ai=0.7, first_seen=IN_WEEK)
        add_dev(s, title="Rumoured model", sources=("press:wire",), rumor=True, p_ai=0.6, first_seen=IN_WEEK)
        s.commit()
    run_digest(audience="ai_team", now=NOW)
    assert len(calls) == 1 and "artifacts" not in calls[0]["user"].split("Items:")[0]
    with db() as s:
        body = s.scalar(select(Report)).body_md
    radar = body_section(body, "Radar")
    assert "**Trial**" in radar and "Try it on extraction [1]" in radar
    assert "**Adopt**" not in radar and radar.index("**Assess**") < radar.index("Said to be great [3]")
    assert "- **Empty entry**: Generated so-what. Generated action. [2]" in radar  # backfilled from the lens
    assert "Nothing material this week." in body_section(body, "Papers worth reading")
    assert "Cuts latency [2]" in body_section(body, "Tools and techniques")


def test_digest_backfill_covers_every_missing_week(db, monkeypatch):
    from fli.deliver.digest import periods, run_digest

    fake_gateway(monkeypatch, fail=True)
    weeks = periods(None, True, fli_config.get_settings(), NOW)
    assert weeks[0] == datetime(2026, 7, 27) and weeks[-1] == WEEK and len(weeks) == 14
    assert periods("2026-11-04", False, fli_config.get_settings(), NOW) == [datetime(2026, 11, 2)]
    out = run_digest(backfill=True, now=NOW)
    assert out["weeks"] == 14 and len(out["reports"]) == 28
    again = run_digest(backfill=True, now=NOW)
    assert {r["status"] for r in again["reports"]} == {"exists"}
    with db() as s:
        assert s.query(Report).count() == 28


def test_mid_week_preview_is_provisional(db, fake_lens, monkeypatch):
    from fli.deliver.digest import run_digest

    calls = fake_gateway(monkeypatch, {"investment": INVESTMENT_OUTPUT})
    with db() as s:
        _investment_week(s)
        s.commit()
    preview = run_digest(audience="investment", week="2026-10-28", now=datetime(2026, 10, 30, 9))
    assert preview["reports"][0]["status"] == "created"
    final = run_digest(audience="investment", now=NOW)  # Monday run: the previewed week is regenerated
    assert final["reports"][0]["status"] == "updated" and len(calls) == 2
    assert run_digest(audience="investment", now=NOW)["reports"][0]["status"] == "exists"
