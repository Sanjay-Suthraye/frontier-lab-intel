"""Digest fixes from the review: output validation + uncached retry, 'Also this week', cited-only
'reported', alert-aware selection, the investment 'Holdings check', and quiet vs not-monitored labs.

No network and no LLM: the gateway and the lens module are faked (as in test_deliver.py).
"""

from __future__ import annotations

import dataclasses
import sys
import types
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from test_deliver import IN_WEEK, NOW, WEEK, add_dev, body_section, seed_world

from fli import config as fli_config
from fli.db.models import Alert, Lab, Lens, Report, Ticker
from fli.deliver import digest
from fli.deliver import materiality as mat
from fli.llm.client import BudgetExceeded, LLMResult

DEGENERATE = {"headline": "Opus fast mode is the story of the week",  # seen live (W44): one empty entry
              "radar": [{"title": "Claude Opus 5.5 fast mode", "text": "", "items": [], "ring": "trial"}],
              "papers": [], "tools_techniques": []}
GOOD_AI = {"headline": "",
           "radar": [{"items": [1], "ring": "trial", "title": "Open model", "text": "Try it on extraction [1]."},
                     {"items": [2], "ring": "assess", "title": "Decoding", "text": "Read the decoding trick [2]."}],
           "papers": [], "tools_techniques": []}


@pytest.fixture(autouse=True)
def _no_outbound(monkeypatch):
    for var in ("SLACK_WEBHOOK_URL", "HEALTHCHECKS_PING_BASE"):
        monkeypatch.setenv(var, "")
    fli_config.get_settings.cache_clear()
    yield
    fli_config.get_settings.cache_clear()


@pytest.fixture()
def lens_stub(monkeypatch):
    """fli.deliver.lens stand-in: writes a 'watch' / 'assess' lens for developments that lack one."""
    def ensure_lens(development_ids, audience):
        from fli.db.session import session_scope

        with session_scope() as s:
            for dev_id in development_ids:
                if s.scalar(select(Lens).where(Lens.development_id == dev_id, Lens.audience == audience)) is None:
                    s.add(Lens(development_id=dev_id, audience=audience, headline=f"Lens headline {dev_id}",
                               so_what="Generated so-what.", what_to_do="Generated action.", tickers=[],
                               action="watch" if audience == "investment" else "assess", model="fake",
                               prompt_version="fake@v1"))

    module = types.ModuleType("fli.deliver.lens")
    module.ensure_lens = ensure_lens
    monkeypatch.setitem(sys.modules, "fli.deliver.lens", module)


def scripted_gateway(monkeypatch, script: dict[str, list]) -> list[dict]:
    """A gateway that answers each call from `script[audience]` in order: a dict (fresh output),
    (dict, 'cached') (a response-cache hit, which the real gateway returns without validating), or an
    exception. Like LLMGateway.structured, a fresh output the caller's validate() rejects comes back
    as status 'invalid' with no data (and would not be cached)."""
    calls: list[dict] = []

    class Gateway:
        def __init__(self, settings=None):
            pass

        async def structured(self, **kw):
            calls.append(kw)
            item = script[kw["ref"]["audience"]].pop(0)
            if isinstance(item, BaseException):
                raise item
            data, status = item if isinstance(item, tuple) else (item, "ok")
            if status == "ok" and kw.get("validate") is not None and not kw["validate"](data):
                return LLMResult(None, "invalid", 0.05, kw["model"], "digest@test",
                                 "rejected by caller validation (not cached)")
            return LLMResult(data, status, 0.05 if status == "ok" else 0.0, kw["model"], "digest@test")

    monkeypatch.setattr(digest, "LLMGateway", Gateway)
    return calls


def _ai_week(s) -> list[int]:
    seed_world(s)
    return [add_dev(s, title="Mistral open weights", event_type="open_weight_release", lab_id="mistral",
                    sources=("mistral:news",), p_ai=0.9, first_seen=IN_WEEK).id,
            add_dev(s, title="Decoding trick", event_type="technique", p_ai=0.7, first_seen=IN_WEEK).id]


def _report(db, audience: str) -> Report:
    with db() as s:
        return s.scalar(select(Report).where(Report.audience == audience))


# ------------------------------------------------------------------------------- validation + retry
def test_degenerate_output_is_rejected_and_retried_on_the_retry_model(db, lens_stub, monkeypatch):
    calls = scripted_gateway(monkeypatch, {"ai_team": [DEGENERATE, GOOD_AI]})
    with db() as s:
        _ai_week(s)
        s.commit()
    out = digest.run_digest(audience="ai_team", now=NOW)
    rep = _report(db, "ai_team")
    assert len(calls) == 2 and calls[0].get("use_cache", True) is True and calls[1]["use_cache"] is False
    assert calls[0]["validate"](DEGENERATE) is False and calls[0]["validate"](GOOD_AI) is True
    settings = fli_config.get_settings()
    assert calls[0]["model"] == settings.model_digest and calls[1]["model"] == settings.model_digest_retry
    assert settings.model_digest_retry != settings.model_digest  # a resend to the same model repeats the failure
    assert rep.model == settings.model_digest_retry  # the retry's output is published, under its own model
    assert "Try it on extraction [1]" in rep.body_md and "## Also this week" not in rep.body_md
    note = out["reports"][0]["note"]
    assert "no usable entry" in note and f"retry on {settings.model_digest_retry} succeeded" in note
    assert out["cost_usd"] == pytest.approx(0.10)  # both attempts are paid for


def test_two_unusable_outputs_fall_back_to_the_lens_renderer(db, lens_stub, monkeypatch):
    calls = scripted_gateway(monkeypatch, {"ai_team": [DEGENERATE, DEGENERATE]})
    with db() as s:
        _ai_week(s)
        s.commit()
    out = digest.run_digest(audience="ai_team", now=NOW)
    rep = _report(db, "ai_team")
    assert len(calls) == 2 and rep.model == "fallback"
    assert "Generated so-what. Generated action. [1]" in rep.body_md and "[2]" in rep.body_md
    assert out["reports"][0]["note"].count("rejected: no usable entry") == 2  # first try and retry
    assert all(i["cited"] for i in rep.items)


def test_low_coverage_output_is_rejected(db, lens_stub, monkeypatch):
    """Seen live (ai_team 2026-08-17): a model digest citing 2 of 12 selected items."""
    thin = {"headline": "", "top_of_mind": [{"items": [1], "title": "GPT-6", "text": "OpenAI shipped GPT-6 [1].",
                                             "tickers": [], "action": "watch"}], "by_theme": [], "watchlist": []}
    fuller = {**thin, "by_theme": [{"items": [2, 4], "title": "Compute", "text": "Deal and update [2][4]."}]}
    calls = scripted_gateway(monkeypatch, {"investment": [thin, fuller]})
    with db() as s:
        seed_world(s)
        for i, p in enumerate((0.8, 0.7, 0.6, 0.5)):
            add_dev(s, title=f"OpenAI item {i}", p_inv=p, first_seen=IN_WEEK)
        s.commit()
    out = digest.run_digest(audience="investment", now=NOW)
    assert len(calls) == 2 and "cites 1 of 4 selected items (minimum 60%)" in out["reports"][0]["note"]
    rep = _report(db, "investment")
    assert rep.model != "fallback" and [i["also_this_week"] for i in rep.items] == [False, False, True, False]
    assert "- **Lens headline" in body_section(rep.body_md, "Also this week") and "[3]" in rep.body_md


def test_cached_unusable_output_is_revalidated_and_refetched(db, lens_stub, monkeypatch):
    """A bad output cached before validation existed comes back as a cache hit: re-checked, refetched."""
    calls = scripted_gateway(monkeypatch, {"ai_team": [(DEGENERATE, "cached"), GOOD_AI]})
    with db() as s:
        _ai_week(s)
        s.commit()
    digest.run_digest(audience="ai_team", now=NOW)
    assert len(calls) == 2 and calls[1]["use_cache"] is False
    assert _report(db, "ai_team").model == fli_config.get_settings().model_digest_retry


def test_budget_stop_is_not_retried(db, lens_stub, monkeypatch):
    calls = scripted_gateway(monkeypatch, {"ai_team": [BudgetExceeded("LLM daily budget reached")]})
    with db() as s:
        _ai_week(s)
        s.commit()
    out = digest.run_digest(audience="ai_team", now=NOW)
    assert len(calls) == 1 and _report(db, "ai_team").model == "fallback"
    assert "BudgetExceeded" in out["reports"][0]["note"]


def test_schema_writes_entries_before_the_headline():
    for schema in digest.SCHEMAS.values():
        assert list(schema["properties"])[-1] == "headline"
    assert digest.SECTIONS == {"investment": ["top_of_mind", "by_theme", "watchlist"],
                               "ai_team": ["radar", "papers", "tools_techniques"]}


# ------------------------------------------------------------------------------- reported = cited only
def test_items_a_report_never_cited_are_not_treated_as_reported(db, lens_stub, monkeypatch):
    scripted_gateway(monkeypatch, {"investment": []})
    monkeypatch.setattr(digest, "LLMGateway", None)  # model unavailable: deterministic render
    prev = WEEK - timedelta(days=7)
    with db() as s:
        seed_world(s)
        dropped = add_dev(s, title="Dropped last week", p_inv=0.9, first_seen=prev + timedelta(days=2))
        shown = add_dev(s, title="Shown last week", p_inv=0.9, first_seen=prev + timedelta(days=2))
        for dev in (dropped, shown):
            dev.last_seen_at = IN_WEEK  # new evidence this week
        s.add(Report(audience="investment", period_start=prev, period_end=WEEK, title="t", body_md="b",
                     model="claude-opus-5-5", created_at=WEEK + timedelta(hours=6),
                     items=[{"development_id": dropped.id, "rank": 1, "cited": False},
                            {"development_id": shown.id, "rank": 2, "cited": True}]))
        s.commit()
        dropped_id, shown_id = dropped.id, shown.id
    digest.run_digest(audience="investment", now=NOW)
    with db() as s:
        rep = s.scalar(select(Report).where(Report.audience == "investment", Report.period_start == WEEK))
    assert [i["development_id"] for i in rep.items] == [dropped_id]
    assert shown_id not in {i["development_id"] for i in rep.items}
    assert digest._shown(Report(items=[{"development_id": 7}, {"development_id": 8, "cited": False}])) == {7}


# ------------------------------------------------------------------------------- selection
def _alert(s, dev_id: int, audience: str, severity: str, status: str = "open") -> None:
    s.add(Alert(development_id=dev_id, audience=audience, severity=severity, title="t", body_md="b", rules=[],
                dedup_key=f"{audience}:{dev_id}", status=status, created_at=IN_WEEK, delivered={}))


def test_selection_always_includes_alerted_and_drops_dismissed(db):
    policy = dataclasses.replace(mat.load_policy(), digest_top_n=3)
    with db() as s:
        seed_world(s)
        p1 = add_dev(s, title="P1 alerted", p_inv=0.5, first_seen=IN_WEEK).id
        p2_low = add_dev(s, title="P2 alerted, dismissed by the lens", p_inv=0.2, first_seen=IN_WEEK,
                         headline="x", inv_action="no_action").id
        dismissed = add_dev(s, title="High p, dismissed", p_inv=0.95, first_seen=IN_WEEK, headline="x",
                            inv_action="no_action").id
        watched = add_dev(s, title="Watch", p_inv=0.6, first_seen=IN_WEEK, headline="x", inv_action="watch").id
        p3 = add_dev(s, title="Digest-only P3", p_inv=0.25, first_seen=IN_WEEK).id
        suppressed = add_dev(s, title="Suppressed alert", p_inv=0.9, first_seen=IN_WEEK, headline="x",
                             inv_action="no_action").id
        last_month = add_dev(s, title="Alerted last month", p_inv=0.9, first_seen=datetime(2026, 10, 1)).id
        _alert(s, p1, "investment", "P1")
        _alert(s, p2_low, "investment", "P2")
        _alert(s, p3, "investment", "P3")
        _alert(s, suppressed, "investment", "P1", status="suppressed")
        s.add(Alert(development_id=last_month, audience="investment", severity="P1", title="t", body_md="b",
                    rules=[], dedup_key=f"investment:{last_month}", created_at=datetime(2026, 10, 1), delivered={}))
        s.commit()
        assert digest.select_items(s, "investment", WEEK, policy, set()) == [p1, p2_low, p3]  # P3 first in the fill
        roomy = dataclasses.replace(policy, digest_top_n=12)
        picked = digest.select_items(s, "investment", WEEK, roomy, set())
        assert picked == [p1, p2_low, p3, watched] and dismissed not in picked and suppressed not in picked
        tight = dataclasses.replace(policy, digest_top_n=1)  # alerted items are never cut by top_n
        assert digest.select_items(s, "investment", WEEK, tight, set()) == [p1, p2_low]
        assert digest.select_items(s, "investment", WEEK, roomy, {p1}) == [p2_low, p3, watched]  # reported


def test_ai_selection_drops_hold_unless_alerted(db):
    policy = mat.load_policy()
    with db() as s:
        seed_world(s)
        hold = add_dev(s, title="Hold", p_ai=0.9, first_seen=IN_WEEK, headline="x", ai_action="hold").id
        hold_alerted = add_dev(s, title="Hold but pushed", p_ai=0.6, first_seen=IN_WEEK, headline="x",
                               ai_action="hold").id
        trial = add_dev(s, title="Trial", p_ai=0.5, first_seen=IN_WEEK, headline="x", ai_action="trial").id
        _alert(s, hold_alerted, "ai_team", "P2")
        s.commit()
        assert digest.select_items(s, "ai_team", WEEK, policy, set()) == [hold_alerted, trial]
        assert hold not in digest.select_items(s, "ai_team", WEEK, policy, set())


def test_payload_marks_alerts_and_lab_monitoring(db):
    with db() as s:
        seed_world(s)
        dev = add_dev(s, title="P1 alerted", p_inv=0.5, first_seen=IN_WEEK).id
        add_dev(s, title="Mistral minor", lab_id="mistral", sources=("mistral:news",), p_inv=0.1, first_seen=IN_WEEK)
        _alert(s, dev, "investment", "P1")
        s.commit()
        policy = mat.load_policy()
        plan = digest.build_plan(s, "investment", WEEK, digest.select_items(s, "investment", WEEK, policy, set()),
                                 policy)
    payload = digest.render_payload(plan)
    assert "| alert: P1 (pushed)" in payload
    assert "Quiet labs (documents collected, nothing material): Mistral AI (tier 2)" in payload
    assert "Not monitored this week (no documents collected, so their silence is not evidence): " \
           "Thinking Machines Lab (alumni startups)" in payload


# ------------------------------------------------------------------------------- holdings check + labs
def test_holdings_check_and_quiet_vs_unmonitored_labs(db, lens_stub, monkeypatch):
    monkeypatch.setattr(digest, "LLMGateway", None)  # deterministic render
    with db() as s:
        seed_world(s)
        s.add_all([Ticker(symbol="AMZN", company="Amazon", bit_weight=9.6),
                   Ticker(symbol="MU", company="Micron Technology", bit_weight=6.5),
                   Ticker(symbol="SIMO", company="Silicon Motion", bit_weight=3.9)])
        s.get(Lab, "openai").public_exposure = [{"ticker": "SIMO", "relation": "supplier"}]
        s.flush()
        add_dev(s, title="OpenAI ships GPT-6", p_inv=0.8, first_seen=IN_WEEK, tickers=("NVDA",),
                headline="GPT-6 resets the frontier")
        add_dev(s, title="OpenAI chip news", event_type="custom_silicon", p_inv=0.7, first_seen=IN_WEEK,
                headline="Custom chip", inv_action="update_thesis",
                lens_tickers=[{"symbol": "NVDA", "direction": "negative", "why": "substitution"},
                              {"symbol": "MSFT", "direction": "positive", "why": "capacity"},
                              {"symbol": "AMZN", "direction": "unclear", "why": "maybe"}])
        add_dev(s, title="Mistral minor note", lab_id="mistral", sources=("mistral:news",), p_inv=0.1,
                first_seen=IN_WEEK, content="Runs on Amazon infrastructure.")
        s.commit()
    digest.run_digest(audience="investment", now=NOW)
    body = _report(db, "investment").body_md
    lines = body_section(body, "Holdings check").strip().splitlines()
    assert lines == [
        "- **AMZN** (9.6%): no new development.",  # named in a document, but no item moved it
        "- **NVDA** (8.0%): mixed. Top item: GPT-6 resets the frontier [1]; also [2].",
        "- **MU** (6.5%): no source coverage (no document collected this week names it or comes from a lab linked "
        "to it).",
        "- **MSFT** (4.4%): positive. Top item: Custom chip [2].",
        "- **SIMO** (3.9%): no new development.",  # OpenAI documents: a lab whose curated exposure lists it
    ]
    labs = body_section(body, "Quiet labs")
    assert "No material developments this week from: Mistral AI (tier 2)." in labs
    assert "Not monitored this week (no documents collected, so silence is not evidence): Thinking Machines Lab " \
           "(alumni startups)." in labs
    assert body.index("## Holdings check") < body.index("## Quiet labs") < body.index("## Sources")


def test_quiet_week_still_reports_holdings_and_lab_coverage(db, monkeypatch):
    monkeypatch.setattr(digest, "LLMGateway", None)
    with db() as s:
        seed_world(s)
        s.commit()
    digest.run_digest(audience="investment", week=WEEK.date(), now=NOW)
    body = _report(db, "investment").body_md
    assert "Nothing material this week." in body
    assert "- **NVDA** (8.0%): no source coverage" in body
    assert "Not monitored this week (no documents collected, so silence is not evidence): OpenAI (tier 1); " \
           "Mistral AI (tier 2); Thinking Machines Lab (alumni startups)." in body
