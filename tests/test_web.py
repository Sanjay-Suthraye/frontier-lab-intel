"""Web UI + JSON API on a seeded synthetic database (no network, no LLM)."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import update

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
from fli.timeutil import utcnow

PAGES = ["/", "/feed/investment", "/feed/ai-team", "/feed/investment?week=all&bit=1&min_p=0.5",
         "/feed/ai-team?week=2026-08-03&lab=anthropic&event_type=model_release", "/alerts",
         "/alerts?status=all", "/reports", "/reports?audience=ai-team", "/register/labs", "/register/people",
         "/register/people?tier=A&q=jane", "/label", "/label?audience=ai-team", "/status", "/api"]
MARKDOWN = ("# Week in frontier labs\n\nSee [the release](https://example.com/claude-x) and "
            "[bad](javascript:alert(1)).\n\n<script>alert('xss')</script>\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n")
LOCAL = ("127.0.0.1", 50000)  # a laptop running `fli serve`
REMOTE = ("203.0.113.7", 41000)  # anyone else (TEST-NET-3)
LIVE = {"investment": "logreg-v1", "ai_team": "logreg-v1"}  # the seeded "deployed" model version
REPO = Path(__file__).resolve().parents[1]
AUDIENCES = ("investment", "ai_team")
# A fixed materiality policy for the 'why flagged' tests: config/materiality.yaml is tuned on its own.
TEST_POLICY = {
    "version": 9, "exposure_min_confidence": 0.6,
    "guards": {"require_verified": True, "rumor_cap": "P3", "min_corroborating_classes": 2},
    "audiences": {
        "investment": [
            {"id": "t.inv.p1.flagship", "severity": "P1", "description": "Tier-1 flagship on an official channel",
             "when": {"event_type": ["model_release"], "lab_tier": [1], "has_official": True, "magnitude_min": 5,
                      "not": {"lens_action": ["no_action", "watch"]}}},
            {"id": "t.inv.p2.capital", "severity": "P2", "description": "Funding at a tier-1/2 lab",
             "when": {"event_type": ["funding_ipo"], "lab_tier": [1, 2]}},
            {"id": "t.inv.p2.holding", "severity": "P2", "description": "Major development exposing a holding",
             "when": {"held_exposure": True, "magnitude_min": 4, "p_min": 0.95}},
        ],
        "ai_team": [
            {"id": "t.ai.p2.release", "severity": "P2", "description": "Verified release",
             "when": {"event_type": ["model_release"], "verified": True, "not": {"lens_action": ["hold"]}}},
        ],
    },
}


# ------------------------------------------------------------------------------------ seed
def _doc(src: Source, key: str, title: str, published: datetime, **kw) -> Document:
    return Document(source_id=src.id, natural_key=key, url=f"https://example.com/{key}",
                    canonical_url=f"https://example.com/{key}", title=title, published_at=published,
                    content_text="body", content_hash=key, source_class=src.source_class,
                    status="extracted", **kw)


def _insight(doc: Document, dev: Development, title: str, quotes: list, verified: bool, **kw) -> Insight:
    return Insight(document_id=doc.id, development_id=dev.id, lab_id=dev.lab_id, event_type=dev.event_type,
                   title=title, claim=f"{title} (claim)", evidence_quotes=quotes, quotes_verified=verified,
                   published_at=doc.published_at, prompt_version="extract-v1", model="claude-haiku-5-5", **kw)


def _seed_register(s, now: datetime) -> dict[str, Source]:
    """Two labs, two people with identities and history, and one source in each health state."""
    s.add_all([Ticker(symbol="AMZN", company="Amazon", bit_weight=9.6, bit_funds=["BIT GTL"]),
               Ticker(symbol="NVDA", company="NVIDIA", bit_weight=8.0),
               Ticker(symbol="MSFT", company="Microsoft", bit_weight=4.4),
               Ticker(symbol="GOOGL", company="Alphabet"),
               Lab(id="anthropic", name="Anthropic", kind="frontier_lab", tier=1, description="Claude.",
                   public_exposure=[{"ticker": "AMZN", "relation": "investor_and_cloud_partner"}]),
               Lab(id="mistral", name="Mistral AI", kind="open_weight_lab", tier=2)])
    s.flush()
    s.add_all([Person(id="jane-doe", name="Jane Doe", lab_id="anthropic", tier="A",
                      why_key="Leads pretraining", inclusion_rule="seed_founder", contributor_score=0.82,
                      contributor_features={"papers_90d": 4, "citations_velocity": 12.5}),
               Person(id="bob-roe", name="Bob Roe", lab_id="mistral", tier="B")])
    s.flush()
    s.add_all([Affiliation(person_id="jane-doe", org_name="OldCo", role="Researcher",
                           start_date=datetime(2018, 1, 1), end_date=datetime(2020, 12, 31)),
               Affiliation(person_id="jane-doe", lab_id="anthropic", role="Research lead",
                           start_date=datetime(2021, 1, 1), evidence="team page"),
               Identity(person_id="jane-doe", platform="x", handle="janedoe", evidence="curated seed"),
               Identity(person_id="jane-doe", platform="github", handle="jdoe", method="fuzzy",
                        status="candidate", confidence=0.6, evidence="same display name"),
               Identity(lab_id="anthropic", platform="x", handle="anthropicai")])
    sources = {
        "ok": Source(id="anthropic:news", lab_id="anthropic", adapter="rss", kind="official_blog",
                     source_class="official", url="https://www.anthropic.com/news", last_run_at=now,
                     last_success_at=now - timedelta(hours=1), last_new_item_at=now - timedelta(days=1)),
        "stale": Source(id="mistral:news", lab_id="mistral", adapter="rss", kind="official_blog",
                        source_class="official", created_at=now - timedelta(days=60),
                        last_success_at=now - timedelta(days=30), last_new_item_at=now - timedelta(days=30)),
        "failing": Source(id="x:janedoe", person_id="jane-doe", adapter="x_timeline", kind="x_account",
                          source_class="personal", consecutive_failures=3, last_error="HTTPStatusError: 503"),
        # never ran here but has a document (imported): judged by that document, so "ok"
        "imported": Source(id="press:reuters", adapter="rss", kind="press", source_class="press"),
        "pending": Source(id="sec:edgar", adapter="edgar_fts", kind="sec", source_class="filing"),
    }
    s.add_all(sources.values())
    s.flush()
    return sources


def _seed_developments(s, now: datetime, src: dict[str, Source]) -> tuple[Development, ...]:
    """An official release (+ press), a personal-only rumour and an old low-value item."""
    day = timedelta(days=1)
    d1 = _doc(src["ok"], "claude-x", "Introducing Claude X", now - day)
    d2 = _doc(src["failing"], "tweet-1", "Hearing Mistral is raising", now - 2 * day, person_id="bob-roe")
    d3 = _doc(src["imported"], "reuters-1", "Anthropic launches Claude X", now - day, person_id="jane-doe")
    d4 = _doc(src["ok"], "docs-fix", "Docs typo fix", now - 20 * day)
    dev1 = Development(title="Anthropic releases Claude X", event_type="model_release", lab_id="anthropic",
                       person_ids=["jane-doe"], artifacts=["claude-x"], first_seen_at=now - day,
                       last_seen_at=now, insight_count=2, source_count=2,
                       source_classes=["official", "press"], has_official=True, has_verified=True,
                       summary="New frontier model.")
    dev2 = Development(title="Mistral reportedly raising", event_type="funding_ipo", lab_id="mistral",
                       first_seen_at=now - 2 * day, last_seen_at=now - 2 * day, insight_count=1,
                       source_count=1, source_classes=["personal"])
    dev3 = Development(title="Minor docs update", event_type="other", lab_id="anthropic",
                       first_seen_at=now - 20 * day, last_seen_at=now - 20 * day, insight_count=1,
                       source_count=1, source_classes=["official"], has_official=True)
    s.add_all([d1, d2, d3, d4, dev1, dev2, dev3])
    s.flush()
    s.add_all([
        _insight(d1, dev1, "Claude X released", [{"quote": "We are releasing Claude X", "verified": True}],
                 True, is_official=True, magnitude=5, person_ids=["jane-doe"]),
        _insight(d3, dev1, "Press confirms launch", ["a quote that is not in the text"], False),
        _insight(d2, dev2, "Mistral raise rumour", ["raising at a big valuation"], True, is_rumor=True,
                 person_ids=["bob-roe"]),
        _insight(d4, dev3, "Docs fixed", [], False, is_official=True),
    ])
    return dev1, dev2, dev3


def _seed_analysis(s, dev1: Development, dev2: Development, dev3: Development) -> None:
    """Scores (incl. an older model version that must lose), lenses and exposure links."""
    def score(dev, audience, p, version="logreg-v1", **kw):
        return Score(development_id=dev.id, audience=audience, p_material=p, model_version=version, **kw)

    s.add_all([
        score(dev1, "investment", 0.30, "heuristic-v0"),  # older row first: the newest (higher id) wins
        score(dev1, "investment", 0.91, features={"tier1_lab": 1, "official_source": 1, "novelty": 0.8},
              contributions={"official_source": 0.8, "tier1_lab": 0.5, "novelty_penalty": -0.1}),
        score(dev1, "ai_team", 0.75),
        score(dev2, "investment", 0.55, contributions={"funding": {"contribution": 0.4, "feature_value": 1}}),
        score(dev2, "ai_team", 0.2), score(dev3, "investment", 0.12), score(dev3, "ai_team", 0.05),
        Lens(development_id=dev1.id, audience="investment", headline="Trainium demand tailwind for AMZN",
             so_what="Claude X is trained on AWS.", what_to_do="Review AMZN position sizing.",
             tickers=[{"symbol": "AMZN", "direction": "positive", "why": "AWS / Trainium"}],
             action="review_position", horizon="weeks", model="claude-sonnet-5-5", prompt_version="lens-v1"),
        Lens(development_id=dev1.id, audience="ai_team", headline="Trial Claude X for extraction",
             so_what="Better long-context recall.", what_to_do="Run our eval suite on it.",
             action="trial", model="claude-sonnet-5-5", prompt_version="lens-v1"),
        # event-specific (counts as exposure), the lab's curated relation (context only), and a link
        # weaker than the policy's exposure_min_confidence (0.6; counts nowhere)
        ExposureLink(development_id=dev1.id, ticker="NVDA", theme="accelerators", direction="positive",
                     rationale="training compute", method="rule", confidence=0.7),
        ExposureLink(development_id=dev1.id, ticker="MSFT", theme="hyperscaler_partners",
                     rationale="lab relation: cloud partner", method="lab", confidence=0.6),
        ExposureLink(development_id=dev2.id, ticker="GOOGL", theme="hyperscaler_partners", rationale="rival",
                     method="rule", confidence=0.5),
    ])


def _seed_delivery_and_ops(s, now: datetime, dev1: Development, dev2: Development,
                           dev3: Development) -> dict:
    a1 = Alert(development_id=dev1.id, audience="investment", severity="P1", title="Claude X: AMZN tailwind",
               body_md="**Review** [the post](https://example.com/claude-x)", dedup_key="a1",
               rules=["tier1_official_release", {"rule": "bit_holding_exposure", "detail": "AMZN 9.6%"}],
               delivered={"slack": "2026-11-03T10:00:00Z"})
    a2 = Alert(development_id=dev2.id, audience="investment", severity="P2", title="Mistral raise",
               body_md="x", dedup_key="a2", status="acked")
    a3 = Alert(development_id=dev1.id, audience="ai_team", severity="P3", title="Trial Claude X", body_md="y",
               dedup_key="a3")
    r1 = Report(audience="investment", period_start=now - timedelta(days=9),
                period_end=now - timedelta(days=2), title="Investment digest", body_md=MARKDOWN,
                model="claude-opus-5-5",
                items=[{"development_id": dev1.id, "rank": 1, "p_material": 0.91}])
    s.add_all([a1, a2, a3, r1,
               Label(development_id=dev3.id, audience="investment", label=0, annotator="human:carol",
                     rubric_version="r1"),
               JobRun(job="ingest:anthropic:news", status="ok", items_in=3, items_new=1, finished_at=now),
               JobRun(job="stage:triage", status="error", error="RuntimeError: boom", finished_at=now),
               LLMCall(purpose="triage", model="claude-haiku-5-5", prompt_version="t1", input_tokens=1200,
                       output_tokens=80, cost_usd=0.02, status="ok", latency_ms=800),
               LLMCall(purpose="lens", model="claude-sonnet-5-5", prompt_version="l1", status="cached"),
               LLMCall(purpose="extract", model="claude-haiku-5-5", prompt_version="e1", status="error"),
               ApiUsage(provider="x", endpoint="users/:id/tweets", units=100, cost_usd=0.5),
               OpsEvent(kind="source_failing", severity="crit", subject="x:janedoe", message="3 failures",
                        dedup_key="ops1"),
               OpsEvent(kind="source_stale", severity="warn", subject="old", message="resolved",
                        dedup_key="ops2", resolved_at=now)])
    s.flush()
    return {"a1": a1.id, "a2": a2.id, "r1": r1.id}


def seed(factory) -> dict:
    now = utcnow()
    with factory() as s:
        sources = _seed_register(s, now)
        dev1, dev2, dev3 = _seed_developments(s, now, sources)
        _seed_analysis(s, dev1, dev2, dev3)
        ids = _seed_delivery_and_ops(s, now, dev1, dev2, dev3)
        s.commit()
        return {"dev1": dev1.id, "dev2": dev2.id, "dev3": dev3.id, **ids}


@pytest.fixture()
def live(monkeypatch):
    """Pin the live scoring versions (the real ones come from data/models/*.json)."""
    from fli.web import queries

    versions = dict(LIVE)
    monkeypatch.setattr(queries, "live_versions", lambda *_a, **_k: dict(versions))
    return versions


@pytest.fixture()
def app_(db, monkeypatch, live):
    monkeypatch.delenv("FLI_BASIC_AUTH", raising=False)
    monkeypatch.delenv("FLI_AUTH_ALL", raising=False)
    from fli.web.app import app

    return app


@pytest.fixture()
def client(app_):
    """A local client: without credentials configured, only loopback may write."""
    return TestClient(app_, client=LOCAL)


@pytest.fixture()
def remote(app_):
    return TestClient(app_, client=REMOTE)


@pytest.fixture()
def ids(db):
    return seed(db)


@pytest.fixture()
def policy(tmp_path, monkeypatch):
    """Point the materiality policy (alerts and web alike) at TEST_POLICY."""
    from fli.deliver import materiality

    path = tmp_path / "materiality.yaml"
    path.write_text(yaml.safe_dump(TEST_POLICY, sort_keys=False))
    monkeypatch.setattr(materiality, "POLICY_PATH", path)
    return materiality.load_policy(path)


def _basic(user: str, password: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


# ------------------------------------------------------------------------------------ pages
def test_every_page_renders_on_empty_db(client):
    for url in PAGES:
        r = client.get(url)
        assert r.status_code == 200, url
        assert "Frontier Lab Intelligence" in r.text


def test_every_page_renders_on_seeded_db(client, ids):
    details = [f"/developments/{ids['dev1']}", f"/developments/{ids['dev2']}", f"/reports/{ids['r1']}",
               "/register/labs/anthropic", "/register/labs/mistral", "/register/people/jane-doe",
               "/register/people/bob-roe"]
    for url in PAGES + details:
        r = client.get(url)
        assert r.status_code == 200, (url, r.text[:300])
    assert "Trainium demand tailwind" in client.get("/").text


def test_missing_objects_are_404(client, ids):
    for url in ["/developments/999999", "/register/labs/nope", "/register/people/nope", "/reports/999",
                "/feed/nope", "/developments/not-a-number", "/no/such/page"]:
        r = client.get(url, headers={"accept": "text/html"})
        assert r.status_code == 404, url
        assert "text/html" in r.headers["content-type"]
    r = client.get("/api/reports/999")
    assert r.status_code == 404 and r.json()["detail"] == "Report 999 not found"


# ------------------------------------------------------------------------------------ feeds
def test_feed_ranks_by_latest_score_and_filters(client, ids):
    page = client.get("/feed/investment").text
    assert page.index("Anthropic releases Claude X") < page.index("Mistral reportedly raising")
    # the live model version's row wins over the other version; the score is called relevance
    assert "relevance = 0.910" in page and "relevance = 0.300" not in page and "p(material)" not in page
    assert "Minor docs update" not in page  # outside the default 7-day window
    # relevance 0.12 is under the investment default (0.35); "any" shows it
    assert "Minor docs update" not in client.get("/feed/investment?week=all").text
    assert "Minor docs update" in client.get("/feed/investment?week=all&min_p=0").text
    strong = client.get("/feed/investment?min_p=0.7").text
    assert "Claude X" in strong and "Mistral reportedly" not in strong
    bit = client.get("/feed/investment?bit=1&week=all").text
    assert "Claude X" in bit and "Mistral reportedly" not in bit and "Minor docs" not in bit
    # dev2's only link (GOOGL, confidence 0.5) is under the policy's exposure_min_confidence: the alert
    # rules ignore it, so the ticker filter does too
    assert "Mistral reportedly" not in client.get("/feed/investment?ticker=GOOGL").text
    assert "Claude X" in client.get("/feed/investment?ticker=nvda").text  # event-specific link, any case
    assert "Claude X" not in client.get("/feed/investment?ticker=MSFT").text  # lab relation only
    lab = client.get("/feed/investment?lab=mistral").text
    assert "Mistral reportedly" in lab and "Claude X" not in lab
    assert "Claude X" not in client.get("/feed/investment?event_type=funding_ipo").text


def test_feed_text_search(client, ids):
    page = client.get("/feed/investment", params={"q": "mistral", "week": "all"}).text
    assert "Mistral reportedly raising" in page and "Anthropic releases Claude X" not in page


def test_feed_card_badges(client, ids):
    page = client.get("/feed/investment").text
    assert page.count('class="badge unconf"') == 1  # the personal-only rumour, not the official release
    assert "AMZN<b>9.6%</b>" in page and "NVDA<b>8.0%</b>" in page  # BIT-weight badges
    assert "Review AMZN position sizing." in page and "review position" in page
    ai = client.get("/feed/ai-team").text
    assert "Trial Claude X for extraction" in ai and "claude-x" in ai


def _dev(s, now: datetime, title: str, p: float | None, action: str | None = None, audience="investment",
         alert: tuple[str, str] | None = None, **kw) -> int:
    """One extra development seen an hour ago, with a live score, a lens and optionally an alert."""
    d = Development(title=title, event_type="model_release", lab_id="anthropic", first_seen_at=now - timedelta(hours=1),
                    last_seen_at=now, insight_count=1, source_count=1, source_classes=["official"],
                    has_official=True, **kw)
    s.add(d)
    s.flush()
    if p is not None:
        s.add(Score(development_id=d.id, audience=audience, p_material=p, model_version=LIVE[audience]))
    if action:
        s.add(Lens(development_id=d.id, audience=audience, headline=f"{title} lens", so_what="s", what_to_do="w",
                   action=action, model="m", prompt_version="v"))
    if alert:
        s.add(Alert(development_id=d.id, audience=audience, severity=alert[0], status=alert[1], title=title,
                    body_md="b", dedup_key=f"{audience}:{d.id}"))
    s.flush()
    return d.id


def _order(page: str, titles: list[str]) -> list[str]:
    found = [(page.find(f">{t}</a>"), t) for t in titles]
    return [t for pos, t in sorted(found) if pos >= 0]


def test_feed_triage_order_and_dismissed_toggle(client, db, ids):
    now = utcnow()
    with db() as s:
        _dev(s, now, "Watch item", 0.99, "watch")
        _dev(s, now, "Thesis item", 0.40, "update_thesis")
        _dev(s, now, "Pushed no lens", 0.50, None, alert=("P2", "open"))
        _dev(s, now, "Dismissed item", 0.95, "no_action")
        _dev(s, now, "Suppressed P1", 0.36, "review_position", alert=("P1", "suppressed"))
        _dev(s, now, "Below default", 0.30, "update_thesis")
        s.commit()
    titles = ["Anthropic releases Claude X", "Mistral reportedly raising", "Pushed no lens", "Thesis item",
              "Suppressed P1", "Watch item", "Below default", "Dismissed item"]
    page = client.get("/feed/investment").text
    # pushed (P1/P2, acked counts, suppressed does not) -> lens action rank -> relevance
    assert _order(page, titles) == ["Anthropic releases Claude X", "Mistral reportedly raising", "Pushed no lens",
                                    "Thesis item", "Suppressed P1", "Watch item"]
    assert "data-dismissed-note" in page and "1 development the lens marked <b>no action</b> is hidden" in page
    assert '<option value="0.35" selected>35 (default)</option>' in page
    every = client.get("/feed/investment?dismissed=1&min_p=0").text
    # rank 4 (thesis/position) by relevance, then watch, then what the lens dismissed
    assert _order(every, titles)[-3:] == ["Below default", "Watch item", "Dismissed item"]
    assert "data-dismissed-note" not in every and 'card dev dismissed' in every
    assert 'name="dismissed" value="1" checked' in every
    # the overview's top list uses the same defaults
    top = client.get("/").text
    assert "Dismissed item lens" not in top and "Watch item lens" in top
    assert "1 dismissed by the lens not shown" in top


def test_ai_feed_hides_hold_without_a_default_floor(client, db, ids):
    now = utcnow()
    with db() as s:
        _dev(s, now, "Hold item", 0.97, "hold", audience="ai_team")
        _dev(s, now, "Assess item", 0.20, "assess", audience="ai_team")
        _dev(s, now, "Adopt item", 0.60, "adopt", audience="ai_team")
        s.commit()
    page = client.get("/feed/ai-team").text
    assert "Hold item" not in page and "Assess item" in page  # no relevance floor for the AI team
    assert _order(page, ["Adopt item", "Assess item"]) == ["Adopt item", "Assess item"]
    assert "the lens marked <b>hold</b> is hidden" in page
    assert "Hold item" in client.get("/feed/ai-team?dismissed=1").text


def test_scores_read_the_live_version_then_the_newest(client, db, ids, live):
    from fli.web import queries as q

    with db() as s:  # an old row of another version with the highest id: max(id) would pick it
        s.add(Score(development_id=ids["dev1"], audience="investment", p_material=0.05, model_version="old-v",
                    created_at=utcnow() - timedelta(days=10)))
        s.commit()
    assert "relevance = 0.910" in client.get("/feed/investment").text
    live["investment"] = "heuristic-v0"  # the live model wins even over a newer row
    assert "relevance = 0.300" in client.get(f"/developments/{ids['dev1']}").text
    live.clear()  # live version unknown: newest by created_at, not max(id)
    with db() as s:
        s.execute(update(Score).where(Score.model_version == "heuristic-v0")
                  .values(created_at=utcnow() - timedelta(days=1)))
        s.commit()
        assert {k: v.p_material for k, v in q.latest_scores(s, [ids["dev1"]]).items()} == {
            (ids["dev1"], "investment"): 0.91, (ids["dev1"], "ai_team"): 0.75}
    live.update(investment="never-scored-v9")  # the live model has not scored dev1 yet: newest row
    data = client.get("/api/developments", params={"audience": "investment"}).json()
    assert data["items"][0]["relevance"] == 0.91 and data["items"][0]["scores"]["investment"]["live"] is False


def test_live_versions_read_the_model_files_as_scoring_decides(db, tmp_path):
    """The web reads only the 'version' field of data/models/scoring_<audience>.json, yet picks the same
    live version as fli.analyze.scoring.current_versions() in every state a deployment goes through."""
    from dataclasses import replace

    import fli.analyze.scoring as scoring
    from fli.web import queries as q

    assert (q.PRIOR_VERSION, q.MODELS_DIR) == (scoring.PRIOR_VERSION, scoring.MODELS_DIR)
    models = tmp_path / "models"
    models.mkdir()

    def deploy(aud: str, version: str) -> Path:
        return scoring.save_model(replace(scoring.prior_model(aud), version=version, validated=True), models)

    def same(s) -> dict:
        mine = q.live_versions(s, models_dir=models)
        assert mine == scoring.current_versions(s, models_dir=models)
        return mine

    with db() as s:
        dev = Development(title="d", event_type="other", first_seen_at=utcnow(), last_seen_at=utcnow())
        s.add(dev)
        s.commit()  # (commits: scoring records an ops event for an unreadable file in its own session)
        assert same(s) == {"investment": "prior-v0", "ai_team": "prior-v0"}  # nothing deployed or scored
        s.add(Score(development_id=dev.id, audience="investment", p_material=0.4, model_version="prior-v0"))
        s.commit()
        path = deploy("investment", "lr-new")  # deployed, not re-scored yet: pages keep the newest rows
        assert q.deployed_version("investment", models) == "lr-new"
        assert same(s)["investment"] == "prior-v0"
        s.add(Score(development_id=dev.id, audience="investment", p_material=0.7, model_version="lr-new"))
        s.commit()
        assert same(s)["investment"] == "lr-new"
        path.write_text('{"audience": "investment", "version": "lr-new", "coeffi')  # half-written
        os.utime(path, ns=(1, 1))  # a new mtime: the cached version is not reused
        assert q.deployed_version("investment", models) is None  # unreadable, never raises
        assert same(s)["investment"] == "lr-new"  # the newest row, as scoring falls back to
        path.write_text(json.dumps({**scoring.prior_model("ai_team").to_dict(), "version": "lr-x"}))
        os.utime(path, ns=(2, 2))
        assert q.deployed_version("investment", models) is None  # another audience's model
        assert same(s)["investment"] == "lr-new"
        assert q.live_versions(models_dir=models) == {"ai_team": "prior-v0"}  # no session: files only


def test_live_version_is_cached_per_file_version(tmp_path):
    from fli.web import queries as q

    path = tmp_path / "scoring_investment.json"
    path.write_text(json.dumps({"audience": "investment", "version": "lr-1"}))
    os.utime(path, ns=(10, 10))
    q._model_file_version.cache_clear()
    assert q.deployed_version("investment", tmp_path) == "lr-1"
    assert q.deployed_version("investment", tmp_path) == "lr-1"
    assert q._model_file_version.cache_info().hits == 1  # the second call did not re-read the file
    path.write_text(json.dumps({"audience": "investment", "version": "lr-2"}))  # `fli train` redeploys
    os.utime(path, ns=(20, 20))
    assert q.deployed_version("investment", tmp_path) == "lr-2"
    assert q.deployed_version("ai_team", tmp_path) == q.PRIOR_VERSION  # no file: the prior


def test_web_process_never_imports_scikit_learn(tmp_path):
    """Startup and page views read model versions from the JSON files; the scoring module (scikit-learn,
    scipy: ~0.45 s and ~85 MB) stays out of the web process."""
    script = (
        "import sys\n"
        "from fastapi.testclient import TestClient\n"
        "from fli.web.app import app\n"
        "with TestClient(app, client=('127.0.0.1', 5000)) as c:\n"
        "    codes = [c.get(u).status_code for u in ('/', '/feed/investment?bit=1', '/alerts', '/api/alerts',\n"
        "                                          '/developments/1', '/status')]\n"
        "heavy = sorted(m for m in ('sklearn', 'scipy', 'fli.analyze.scoring') if m in sys.modules)\n"
        "print(codes, heavy)\n")
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'web.sqlite'}", "FLI_BASIC_AUTH": "",
           "FLI_AUTH_ALL": "0", "RAW_DIR": str(tmp_path / "raw")}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=REPO,
                         timeout=120, check=True).stdout.strip()
    assert out == "[200, 200, 200, 200, 404, 200] []", out


def test_unvalidated_prior_is_badged(client, db, ids, live):
    with db() as s:
        s.add(Score(development_id=ids["dev2"], audience="investment", p_material=0.66, model_version="prior-v0"))
        s.commit()
    live["investment"] = "prior-v0"
    page = client.get("/feed/investment").text
    assert "unvalidated prior (prior-v0" in page  # page note: the live model is the prior
    assert page.count('class="badge prior"') == 1  # dev2 is scored by the prior; dev1 has no prior row
    dev = client.get(f"/developments/{ids['dev2']}").text
    assert 'class="badge prior"' in dev and "relevance = 0.660" in dev
    assert 'class="badge prior"' not in client.get(f"/developments/{ids['dev3']}").text


def test_lens_warnings_shapes():
    from fli.web.queries import lens_warnings

    assert lens_warnings(None) == [] and lens_warnings({}) == [] and lens_warnings({"unsupported_numbers": []}) == []
    w = lens_warnings({"unsupported_numbers": ["80%", "$2"]})
    assert len(w) == 1 and "80%, $2" in w[0]
    assert "80%" in lens_warnings({"unsupported_numbers": {"headline": ["80%"], "so_what": []}})[0]
    assert "7" in lens_warnings({"unsupported_numbers": 7})[0]
    w = lens_warnings({"headline_corrected": True, "original_headline": "OpenAI cuts prices 80%"})
    assert len(w) == 1 and "rewritten" in w[0] and "OpenAI cuts prices 80%" in w[0]
    assert "rewritten" in lens_warnings({"headline": {"corrected": True}})[0]
    assert lens_warnings({"headline_corrected": False, "unsupported_numbers": None}) == []


def test_development_page_warns_about_lens_checks(client, db, ids):
    with db() as s:
        lens = s.query(Lens).filter_by(development_id=ids["dev1"], audience="investment").one()
        lens.checks = {"unsupported_numbers": ["80%"], "headline_corrected": True,
                       "original_headline": "Claude X cuts prices 80%"}
        s.commit()
    page = client.get(f"/developments/{ids['dev1']}").text
    assert page.count('class="note small lens-warn"') == 2
    assert "Numbers not found in the evidence the lens was given: 80%" in page
    assert "Claude X cuts prices 80%" in page
    assert "lens-warn" not in client.get(f"/developments/{ids['dev2']}").text


def test_lens_tickers_accept_either_key_name():
    from fli.web.queries import lens_tickers

    raw = [{"ticker": "MU", "rationale": "HBM"}, {"symbol": "TSM", "direction": "positive"}, "NVDA",
           {"x": 1}, 7]
    got = [(t["symbol"], t["why"]) for t in lens_tickers(raw)]
    assert got == [("MU", "HBM"), ("TSM", None), ("NVDA", None)]


# ------------------------------------------------------------------------------------ development
def test_development_page_shows_evidence_and_why(client, ids):
    page = client.get(f"/developments/{ids['dev1']}").text
    assert "We are releasing Claude X" in page
    assert "&#10003; verified" in page and "&#10007; unverified" in page
    assert "official source" in page and "+0.800" in page and "-0.100" in page  # contributions table
    assert "tier1_official_release" in page and "bit_holding_exposure" in page  # fired rules
    assert "training compute" in page and "accelerators" in page  # exposure link
    assert "Trial Claude X for extraction" in page and "Trainium demand tailwind" in page  # both lenses
    assert "https://example.com/reuters-1" in page  # every source listed
    rumour = client.get(f"/developments/{ids['dev2']}").text
    assert "unconfirmed" in rumour and "+0.400" in rumour  # nested contribution shape


def test_untrusted_urls_and_redirects_are_neutralised(client, db, ids):
    from fli.web.app import _safe_next

    with db() as s:
        dev = s.get(Development, ids["dev2"])
        doc = Document(source_id="press:reuters", natural_key="evil", url="javascript:alert(1)",
                       canonical_url="javascript:alert(1)", title="Evil link", content_hash="evil",
                       source_class="press")
        s.add(doc)
        s.flush()
        s.add(_insight(doc, dev, "Evil insight", [], False))
        s.commit()
    page = client.get(f"/developments/{ids['dev2']}").text
    assert "Evil link" in page and 'href="javascript:' not in page
    for bad in ["//evil.com", "/\t/evil.com", "https://evil.com", "/\\evil.com", "", None]:
        assert _safe_next(bad, "/fallback") == "/fallback", bad
    assert _safe_next("/label?audience=investment&pos=2", "/x") == "/label?audience=investment&pos=2"
    form = {"development_id": ids["dev1"], "audience": "investment", "label": "1", "annotator": "al",
            "next": "//evil.com"}
    r = client.post("/api/labels", data=form, follow_redirects=False)
    assert r.headers["location"] == f"/developments/{ids['dev1']}"


def test_report_markdown_keeps_links_and_escapes_html(client, ids):
    page = client.get(f"/reports/{ids['r1']}").text
    link = '<a href="https://example.com/claude-x" rel="noopener noreferrer" target="_blank">the release</a>'
    assert link in page
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "javascript:" not in page and "<table>" in page


# ------------------------------------------------------------------------------------ labels / alerts
def test_label_post_form_upserts_and_sets_cookie(client, db, ids):
    form = {"development_id": ids["dev1"], "audience": "investment", "label": "2", "annotator": "Alice",
            "next": "/label?audience=investment"}
    r = client.post("/api/labels", data=form, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/label?audience=investment"
    assert client.cookies.get("fli_annotator") == "alice"
    relabel = {**form, "label": "1", "reason": "context only"}
    assert client.post("/api/labels", data=relabel, follow_redirects=False).status_code == 303
    with db() as s:
        rows = s.query(Label).filter_by(development_id=ids["dev1"], audience="investment").all()
        assert [(x.annotator, x.label, x.rubric_version, x.reason) for x in rows] == [
            ("human:alice", 1, "r1", "context only")]
    assert "human:alice" in client.get(f"/developments/{ids['dev1']}").text


def test_label_post_json_and_validation(client, db, ids):
    body = {"development_id": ids["dev2"], "audience": "ai-team", "label": 0, "annotator": "bob"}
    ok = client.post("/api/labels", json=body)
    assert ok.status_code == 200 and ok.json()["label"]["annotator"] == "human:bob"
    assert ok.json()["label"]["audience"] == "ai_team"
    bad = [({"label": 3}, 400), ({"audience": "pm"}, 400), ({"development_id": 999999}, 404),
           ({"annotator": "<script>"}, 400), ({"label": "x"}, 400), ({"label": 1.5}, 400),
           ({"label": True}, 400)]
    for patch, code in bad:
        attempt = {**body, "audience": "investment", "label": 1, **patch}
        assert client.post("/api/labels", json=attempt).status_code == code, patch
    assert client.post("/api/labels", data={"development_id": "1"}, follow_redirects=False,
                       headers={"accept": "text/html"}).status_code == 400
    with db() as s:
        assert s.query(Label).filter(Label.annotator.like("human:%")).count() == 2  # carol + bob


def test_label_queue_excludes_human_labelled(client, db, ids):
    from fli.web import queries as q

    with db() as s:
        queue = q.label_queue(s, "investment", "human:alice")
        assert queue["remaining"] == 2 and queue["item"]["id"] in (ids["dev1"], ids["dev2"])
        assert q.label_queue(s, "investment", pos=2)["pos"] == 0  # skipping past the end wraps around
        assert q.label_queue(s, "ai_team")["remaining"] == 3
        q.save_label(s, ids["dev1"], "investment", 2, "human:alice")
        s.commit()
        assert q.label_queue(s, "investment", "human:alice")["remaining"] == 1
        # agreement mode: items others labelled come back, mine stay out
        assert q.label_queue(s, "investment", "human:alice", include_others=True)["remaining"] == 2
    client.cookies.set("fli_annotator", "alice")
    page = client.get("/label").text
    assert "data-label-form" in page and "Mistral reportedly raising" in page


def test_mixed_order_interleaves_high_low_and_uncertain():
    from fli.web.queries import _mixed_order

    rows = [(1, 0.95), (2, 0.9), (3, 0.1), (4, 0.05), (5, 0.45), (6, None)]
    order = _mixed_order(rows)
    assert sorted(order) == [1, 2, 3, 4, 5, 6] and order[-1] == 6
    assert set(order[:3]) & {1, 2} and set(order[:3]) & {3, 4}
    assert order == _mixed_order(list(reversed(rows)))  # stable, independent of input order


def test_ack_alert_form_and_json(client, db, ids):
    r = client.post(f"/api/alerts/{ids['a1']}/ack", data={"next": "/alerts"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.post(f"/api/alerts/{ids['a2']}/ack", json={"status": "open"}).json()["status"] == "open"
    assert client.post(f"/api/alerts/{ids['a2']}/ack", json={"status": "bogus"}).status_code == 400
    assert client.post("/api/alerts/99999/ack", json={}).status_code == 404
    with db() as s:
        assert s.get(Alert, ids["a1"]).status == "acked" and s.get(Alert, ids["a2"]).status == "open"
    assert "Claude X: AMZN tailwind" not in client.get("/alerts").text  # default view: open only


# ------------------------------------------------------------------------------------ auth
def test_basic_auth_protects_writes(client, ids, monkeypatch):
    monkeypatch.setenv("FLI_BASIC_AUTH", "pm:s3cret")
    body = {"development_id": ids["dev1"], "audience": "investment", "label": 1, "annotator": "pm"}
    r = client.post("/api/labels", json=body)
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
    assert client.post("/api/labels", json=body, headers=_basic("pm", "wrong")).status_code == 401
    assert client.post("/api/labels", json=body, headers=_basic("pm", "s3cret")).status_code == 200
    assert client.post(f"/api/alerts/{ids['a1']}/ack", json={}).status_code == 401
    assert client.get("/").status_code == 200  # reads stay open unless FLI_AUTH_ALL=1
    # the authenticated user becomes the annotator when no name is given
    client.cookies.clear()
    anon = {k: v for k, v in body.items() if k != "annotator"}
    r = client.post("/api/labels", json=anon, headers=_basic("pm", "s3cret"))
    assert r.json()["label"]["annotator"] == "human:pm"


def test_authenticated_login_is_the_annotator_whatever_the_body_says(client, db, ids, monkeypatch):
    monkeypatch.setenv("FLI_BASIC_AUTH", "PM Desk:s3cret")
    client.cookies.set("fli_annotator", "carol")
    spoof = {"development_id": ids["dev1"], "audience": "investment", "label": 2, "annotator": "carol"}
    r = client.post("/api/labels", json=spoof, headers=_basic("PM Desk", "s3cret"))
    assert r.status_code == 200 and r.json()["label"]["annotator"] == "human:pm desk"
    form = {**spoof, "label": "0", "annotator": "mallory", "next": "/label"}
    assert client.post("/api/labels", data=form, headers=_basic("PM Desk", "s3cret"),
                       follow_redirects=False).status_code == 303
    with db() as s:
        rows = s.query(Label).filter(Label.development_id == ids["dev1"]).all()
        assert [(x.annotator, x.label) for x in rows] == [("human:pm desk", 0)]  # upserted, never carol/mallory
    # the forms show the login instead of a free-text name field; a stale cookie name is never shown
    page = client.get(f"/developments/{ids['dev1']}", headers=_basic("PM Desk", "s3cret")).text
    assert 'name="annotator"' not in page and "as human:pm desk" in page
    queue = client.get("/label", headers=_basic("PM Desk", "s3cret")).text
    assert 'name="annotator"' not in queue and "human:pm desk" in queue
    client.cookies.set("fli_annotator", "carol")
    anon_queue = client.get("/label").text  # not challenged yet: the login is unknown, not "carol"
    assert "Labelling as <b>your login</b>" in anon_queue and "carol" not in anon_queue


def test_login_annotator_maps_any_login_to_a_valid_name():
    from fli.web.app import clean_annotator, login_annotator

    assert login_annotator("pm") == "pm" and login_annotator("Anna.B@bit") == "anna.b@bit"
    assert login_annotator("ops+desk/1") == "ops-desk-1" and clean_annotator(login_annotator("ops+desk/1"))
    assert login_annotator("x" * 80) == "x" * 40 and login_annotator("+++") == "-" and login_annotator("") == "user"


def test_writes_fail_closed_for_remote_clients_without_credentials(remote, db, ids):
    body = {"development_id": ids["dev1"], "audience": "investment", "label": 2, "annotator": "eve"}
    r = remote.post("/api/labels", json=body)
    assert r.status_code == 403 and "FLI_BASIC_AUTH" in r.json()["detail"]
    assert remote.post(f"/api/alerts/{ids['a1']}/ack", json={}).status_code == 403
    form = remote.post("/api/labels", data={**body, "label": "2"}, headers={"accept": "text/html"})
    assert form.status_code == 403 and "text/html" in form.headers["content-type"]
    assert "403 · Not allowed" in form.text and "FLI_BASIC_AUTH" in form.text
    assert remote.get("/feed/investment").status_code == 200  # reads stay open without FLI_AUTH_ALL
    with db() as s:
        assert s.query(Label).filter(Label.annotator == "human:eve").count() == 0
        assert s.get(Alert, ids["a1"]).status == "open"


def test_proxied_and_spoofed_local_requests_count_as_remote(app_, ids):
    body = {"development_id": ids["dev1"], "audience": "investment", "label": 1, "annotator": "al"}
    local = TestClient(app_, client=LOCAL)
    # a reverse proxy on the same host connects from 127.0.0.1 but adds forwarding headers
    for hdr in ({"x-forwarded-for": "198.51.100.4"}, {"x-real-ip": "198.51.100.4"},
                {"forwarded": "for=198.51.100.4"}, {"x-forwarded-host": "fli.example.com"}):
        assert local.post("/api/labels", json=body, headers=hdr).status_code == 403, hdr
    for peer in (("::1", 1), ("::ffff:127.0.0.1", 1), ("127.0.0.5", 1)):
        assert TestClient(app_, client=peer).post("/api/labels", json=body).status_code == 200, peer
    for peer in (("testclient", 1), ("localhost", 1), ("", 0), ("10.0.0.8", 1), ("172.17.0.1", 1)):
        assert TestClient(app_, client=peer).post("/api/labels", json=body).status_code == 403, peer


def test_is_local_client():
    from fli.web.auth import is_local_client

    assert is_local_client(("127.0.0.1", 1), ["host"]) and is_local_client(["::1", 1], [])
    assert not is_local_client(None, []) and not is_local_client(("127.0.0.1", 1), ["X-Forwarded-For"])
    assert not is_local_client(("[::1]", 1), ["x-forwarded-proto"]) and is_local_client(("[::1]", 1), [])
    assert not is_local_client(("192.168.65.1", 1), [])  # Docker Desktop's NAT: remote on purpose


def test_remote_writes_with_credentials_are_recorded_under_the_login(remote, db, ids, monkeypatch):
    monkeypatch.setenv("FLI_BASIC_AUTH", "pm:s3cret")
    body = {"development_id": ids["dev2"], "audience": "investment", "label": 1}
    assert remote.post("/api/labels", json=body).status_code == 401
    ok = remote.post("/api/labels", json=body, headers=_basic("pm", "s3cret"))
    assert ok.status_code == 200 and ok.json()["label"]["annotator"] == "human:pm"
    ack = remote.post(f"/api/alerts/{ids['a1']}/ack", json={}, headers=_basic("pm", "s3cret")).json()
    assert ack["status"] == "acked" and ack["acked_by"] == "human:pm"


def test_ack_records_who_and_reopen_clears_it(client, db, ids):
    client.cookies.set("fli_annotator", "alice")
    assert client.post(f"/api/alerts/{ids['a1']}/ack", json={}).json()["acked_by"] == "human:alice"
    with db() as s:
        a = s.get(Alert, ids["a1"])
        assert a.acked_by == "human:alice" and a.acked_at is not None
    page = client.get("/alerts?status=all").text
    assert 'title="by human:alice"' in page
    assert client.post(f"/api/alerts/{ids['a1']}/ack", json={"status": "open"}).json()["acked_by"] is None
    with db() as s:
        assert s.get(Alert, ids["a1"]).acked_at is None


def test_auth_all_protects_pages_but_not_probes(client, ids, monkeypatch):
    monkeypatch.setenv("FLI_BASIC_AUTH", "pm:s3cret")
    monkeypatch.setenv("FLI_AUTH_ALL", "1")
    for url in ["/", "/feed/investment", "/api/developments", "/api/health", "/api/openapi.json"]:
        assert client.get(url).status_code == 401, url
    assert client.get("/", headers=_basic("pm", "s3cret")).status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_auth_all_without_credentials_locks_out_remote_clients(client, remote, ids, monkeypatch):
    monkeypatch.setenv("FLI_AUTH_ALL", "1")
    monkeypatch.setenv("FLI_BASIC_AUTH", "")  # protection asked for, but nothing to check against
    r = remote.get("/feed/investment")
    assert r.status_code == 403 and "FLI_BASIC_AUTH" in r.text
    assert remote.get("/api/developments").status_code == 403
    assert remote.get("/healthz").status_code == 200 and remote.get("/static/style.css").status_code == 200
    assert client.get("/").status_code == 200  # a local operator can still look


def test_cross_origin_writes_are_rejected(client, ids):
    body = {"development_id": ids["dev1"], "audience": "investment", "label": 1, "annotator": "eve"}
    evil = client.post("/api/labels", json=body, headers={"origin": "https://evil.example"})
    assert evil.status_code == 403
    assert client.post("/api/labels", json=body, headers={"origin": "http://testserver"}).status_code == 200


# ------------------------------------------------------------------------------------ JSON API
def test_api_developments(client, ids):
    data = client.get("/api/developments", params={"audience": "investment", "min_p": 0.5}).json()
    assert [i["id"] for i in data["items"]] == [ids["dev1"], ids["dev2"]]
    top = data["items"][0]
    assert top["p_material"] == 0.91 and top["url"].endswith(f"/developments/{ids['dev1']}")
    assert top["relevance"] == 0.91 and top["scores"]["investment"]["live"] is True
    assert top["scores"]["investment"]["prior"] is False and top["lenses"]["investment"]["warnings"] == []
    assert top["lenses"]["investment"]["action"] == "review_position"
    assert [t["symbol"] for t in top["tickers"]][:2] == ["AMZN", "NVDA"]
    urls = {x["url"] for x in top["sources"]}
    assert urls == {"https://example.com/claude-x", "https://example.com/reuters-1"}
    assert top["first_seen_at"].endswith("Z") and data["items"][1]["unconfirmed"] is True
    both = client.get("/api/developments", params={"since": "2026-01-01"}).json()
    assert {i["id"] for i in both["items"]} == {ids["dev1"], ids["dev2"], ids["dev3"]}
    for params in ({"since": "garbage"}, {"audience": "pm"}, {"min_p": 2}):
        assert client.get("/api/developments", params=params).status_code == 422, params
    one = client.get(f"/api/developments/{ids['dev1']}").json()
    assert one["insights"][0]["quotes"][0] == {"text": "We are releasing Claude X", "verified": True}


def test_updated_since_catches_rescored_developments(client, db, ids, live):
    old = utcnow() - timedelta(days=30)
    with db() as s:
        for model, col in ((Development, Development.updated_at), (Score, Score.created_at),
                           (Lens, Lens.created_at), (Alert, Alert.created_at)):
            s.execute(update(model).values({col: old}))
        s.add(Score(development_id=ids["dev2"], audience="investment", p_material=0.6, model_version="v2"))
        s.commit()
    live["investment"] = "v2"  # a new model went live and re-scored dev2
    since = (utcnow() - timedelta(days=1)).isoformat()
    data = client.get("/api/developments", params={"audience": "investment", "updated_since": since}).json()
    assert [i["id"] for i in data["items"]] == [ids["dev2"]] and data["items"][0]["p_material"] == 0.6
    assert data["items"][0]["relevance"] == 0.6


def test_alert_chips_show_channels_and_mode_not_bookkeeping(client, db, ids):
    from fli.web.queries import delivery_view

    with db() as s:
        s.get(Alert, ids["a2"]).status = "open"
        s.get(Alert, ids["a2"]).delivered = {
            "in_app": "2026-11-04T13:47:56", "guarded_severity": "P1", "mode": "historical",
            "log": [{"at": "2026-11-04T13:47:56", "severity": "P2", "guarded": "P1", "mode": "historical"}]}
        s.commit()
    page = client.get("/alerts").text
    for junk in ("sent: guarded_severity", "sent: mode", "sent: log", "sent: in_app", ">log<"):
        assert junk not in page, junk
    assert ">in-app</span>" in page and ">historical</span>" in page and ">rules said P1</span>" in page
    assert ">Slack</span>" in page  # a1 was delivered to Slack (legacy shape: just a timestamp)
    dev = client.get(f"/developments/{ids['dev2']}").text
    assert ">in-app</span>" in dev and "guarded_severity" not in dev
    pending = delivery_view({"in_app": "t", "slack_pending": True, "slack_attempts": 1, "mode": "live"})
    assert [(c["label"], c["state"]) for c in pending["channels"]] == [("in-app", "sent"), ("Slack", "pending")]
    failed = delivery_view({"in_app": "t", "slack_pending": False, "slack_error": "HTTP 500"})
    assert failed["channels"][1]["state"] == "failed" and failed["channels"][1]["error"] == "HTTP 500"
    assert delivery_view(None) == {"channels": [], "mode": None, "guarded_severity": None}
    assert delivery_view({"mode": "weird", "guarded_severity": "P9"})["mode"] is None


def test_api_alerts_routing_fields(client, db, ids):
    """Routing tickers are the exposure the materiality rules count: the lens's committed tickers plus
    event-specific links (held_tickers == the rules' held_tickers fact); lab relations ride apart."""
    items = {a["id"]: a for a in client.get("/api/alerts").json()["items"]}
    inv = items[ids["a1"]]
    assert inv["tickers"] == [{"symbol": "AMZN", "direction": "positive", "bit_weight": 9.6, "basis": "lens"},
                              {"symbol": "NVDA", "direction": "positive", "bit_weight": 8.0, "basis": "event"}]
    assert inv["held_tickers"] == ["AMZN", "NVDA"] and inv["ticker_basis"] == "lens+exposure_map"
    assert inv["lab_relations"] == ["MSFT"]  # the lab's curated relation: context, never routing
    assert (inv["action"], inv["horizon"], inv["relevance"]) == ("review_position", "weeks", 0.91)
    assert inv["relevance_model_version"] == "logreg-v1" and inv["delivery"]["channels"][0]["name"] == "slack"
    ai = next(a for a in items.values() if a["audience"] == "ai_team")
    assert (ai["action"], ai["relevance"]) == ("trial", 0.75) and ai["held_tickers"] == ["AMZN", "NVDA"]
    # no lens and only a link under exposure_min_confidence: nothing to route on
    noisy = items[ids["a2"]]
    assert noisy["tickers"] == [] and noisy["held_tickers"] == [] and noisy["lab_relations"] == []
    assert noisy["ticker_basis"] is None and noisy["action"] is None
    with db() as s:  # a lens that abstains commits to nothing; the event-specific NVDA link still counts
        s.query(Lens).filter_by(development_id=ids["dev1"], audience="investment").one().action = "no_action"
        s.commit()
    again = {a["id"]: a for a in client.get("/api/alerts").json()["items"]}[ids["a1"]]
    assert again["held_tickers"] == ["NVDA"] and [t["symbol"] for t in again["tickers"]] == ["NVDA"]
    assert again["ticker_basis"] == "exposure_map"
    page = client.get("/alerts?status=all").text
    assert "act-no_action" in page


def test_health_uses_imported_documents_when_no_source_ran(client, db, monkeypatch):
    from fli.web import queries as q

    now = utcnow()
    with db() as s:  # an imported snapshot: documents, sources with no ingest bookkeeping at all
        s.add_all([Source(id="a:news", adapter="rss", kind="official_blog", source_class="official"),
                   Source(id="b:news", adapter="rss", kind="official_blog", source_class="official",
                          freshness_slo_hours=48),
                   Source(id="c:none", adapter="rss", kind="press", source_class="press"),
                   Source(id="d:skipped", adapter="edgar_fts", kind="sec", source_class="filing")])
        s.flush()
        for sid, age in (("a:news", 5), ("a:news", 30), ("b:news", 100), ("d:skipped", 3)):
            s.add(Document(source_id=sid, natural_key=f"{sid}-{age}", url=f"https://example.com/{sid}/{age}",
                           canonical_url=f"https://example.com/{sid}/{age}", title="t", content_hash=f"{sid}{age}",
                           source_class="official", fetched_at=now - timedelta(hours=age)))
        # the runner tried d:skipped in this database (missing credential): its own state wins
        s.add(JobRun(job="ingest:d:skipped", status="skipped", error="missing SEC_USER_AGENT", finished_at=now))
        s.commit()
        h = q.health_summary(s, now)
        # no more "no successful ingest yet": only the genuinely stale imported source is a reason
        assert h["reasons"] == ["1 stale source(s)"] and h["last_ingest_basis"] == "documents"
        assert h["last_ingest_at"] == now - timedelta(hours=5)
        assert h["sources"] == {"ok": 1, "stale": 1, "failing": 0, "pending": 2, "disabled": 0}
        strip = client.get("/").text.split("Last ingest")[1][:400]
        assert "5 h ago" in strip and "· imported" in strip and "never" not in strip
        status = client.get("/status").text
        assert "No ingest has run in this database (imported data)" in status and "· imported" in status
        rows = {r["id"]: r for r in q.source_rows(s, now)}
        assert rows["b:news"]["state"] == "stale" and "newest imported document is 4d 4h old" in rows["b:news"]["reason"]
        assert rows["a:news"]["imported_at"] == now - timedelta(hours=5) and rows["c:none"]["state"] == "pending"
        assert rows["d:skipped"]["state"] == "pending" and rows["d:skipped"]["imported_at"] is None
        assert "missing SEC_USER_AGENT" in rows["d:skipped"]["reason"]
        later = q.health_summary(s, now + timedelta(days=2))  # nothing ingested since: say so
        assert later["status"] == "degraded"
        assert "no ingest has run in this database; newest imported document is 2d 5h old" in later["reasons"]
        s.add(Source(id="e:ran", adapter="rss", kind="official_blog", source_class="official", last_run_at=now,
                     last_success_at=now - timedelta(hours=1), last_new_item_at=now - timedelta(hours=1)))
        s.commit()
        h = q.health_summary(s, now)  # once anything ran here, last ingest is the real one
        assert h["last_ingest_basis"] == "runs" and h["last_ingest_at"] == now - timedelta(hours=1)
    assert "· imported" in client.get("/status").text  # never-run sources' rows still show the import


def test_api_alerts_reports_health(client, ids):
    alerts = client.get("/api/alerts", params={"audience": "investment"}).json()["items"]
    assert {a["id"] for a in alerts} == {ids["a1"], ids["a2"]}
    open_p1 = client.get("/api/alerts", params={"status": "open", "severity": "P1"}).json()["items"]
    assert [a["id"] for a in open_p1] == [ids["a1"]]
    assert open_p1[0]["rules"][1]["name"] == "bit_holding_exposure"
    rep = client.get(f"/api/reports/{ids['r1']}").json()
    assert rep["items"][0]["development_id"] == ids["dev1"] and "Week in frontier labs" in rep["body_md"]
    assert '<a href="https://example.com/claude-x"' in rep["body_html"] and "<script>" not in rep["body_html"]
    assert client.get("/api/reports").json()["count"] == 1
    h = client.get("/api/health").json()
    # press:reuters never ran here but has a fresh document: ok; sec:edgar has nothing: pending
    assert h["sources"] == {"ok": 2, "stale": 1, "failing": 1, "pending": 1, "disabled": 0}
    assert h["last_ingest_basis"] == "runs"
    assert h["status"] == "critical" and h["open_ops_events"] == {"crit": 1}  # resolved event not counted
    assert h["budgets"]["llm_total"]["spent"] == 0.02 and h["budgets"]["x_total"]["spent"] == 0.5
    assert {"1 critical ops event(s)", "1 failing source(s)", "1 stale source(s)"} <= set(h["reasons"])
    assert client.get("/healthz").json() == {"ok": True}


# ------------------------------------------------------------------------------------ register / status
def test_register_pages(client, ids):
    labs = client.get("/register/labs").text
    assert "Anthropic" in labs and "AMZN<b>9.6%</b>" in labs
    jane = client.get("/register/people/jane-doe").text
    assert "Leads pretraining" in jane and "seed_founder" in jane and "same display name" in jane
    assert "OldCo" in jane and "citations velocity" in jane and "Anthropic launches Claude X" in jane
    assert "Anthropic releases Claude X" in jane  # development via person_ids
    lab = client.get("/register/labs/anthropic").text
    assert "anthropicai" in lab and "Jane Doe" in lab and "anthropic:news" in lab


def test_status_page_tables(client, ids):
    page = client.get("/status").text
    assert "HTTPStatusError: 503" in page and "stage:triage" in page and "users/:id/tweets" in page
    assert "claude-haiku-5-5" in page and "3 failures" in page and "1 failing source(s)" in page
    assert ">resolved<" not in page  # resolved ops events are not listed


# ------------------------------------------------------------------------------------ why flagged
def _why_section(page: str, n: int = 0) -> str:
    """The n-th audience card's 'why flagged' panel (investment first) up to its score table."""
    return page.split("<h3>Why flagged</h3>")[n + 1].split("Score contributions")[0]


def test_development_page_explains_fired_and_missed_rules(client, ids, policy):
    inv = _why_section(client.get(f"/developments/{ids['dev1']}").text)
    assert "Rules give" in inv and "sev-P1" in inv and "t.inv.p1.flagship" in inv
    for fact in ("event_type model_release", "lab_tier 1", "has_official", "magnitude 5 (&gt;= 5)",
                 "lens_action review_position (not no_action/watch)"):  # what made the rule fire
        assert fact in inv, fact
    assert "2 rules did not fire" in inv
    assert "event_type model_release (needs funding_ipo)" in inv  # t.inv.p2.capital: its first failing test
    assert "p 0.91 (needs &gt;= 0.95)" in inv and "2 of 3 conditions hold" in inv  # t.inv.p2.holding
    assert "Facts the rules read" in inv and "held_tickers" in inv
    ai = _why_section(client.get(f"/developments/{ids['dev1']}").text, 1)
    assert "t.ai.p2.release" in ai and "lens_action trial (not hold)" in ai
    rumour = _why_section(client.get(f"/developments/{ids['dev2']}").text)
    assert "t.inv.p2.capital" in rumour and "rules said P2; a guard capped it" in rumour
    assert "guard.rumor_cap" in rumour and "guard.unconfirmed" in rumour and 'class="badge unconf"' in rumour
    assert "No materiality rule fires: no alert" in _why_section(client.get(f"/developments/{ids['dev3']}").text)
    why = client.get(f"/api/developments/{ids['dev1']}").json()["why_flagged"]["investment"]
    assert why["severity"] == "P1" and [r["id"] for r in why["rules"]] == ["t.inv.p1.flagship"]
    assert {r["id"]: r["failed"] for r in why["not_fired"]} == {
        "t.inv.p2.capital": "event_type model_release (needs funding_ipo)",
        "t.inv.p2.holding": "p 0.91 (needs >= 0.95)"}
    assert why["facts"]["held_tickers"] == ["AMZN", "NVDA"] and why["score"]["p_material"] == 0.91


def test_why_flagged_is_materiality_explain(db, ids, policy):
    """The panel is materiality.explain() itself (plus display keys), batched or not."""
    from fli.deliver import materiality
    from fli.web import queries as q

    with db() as s:
        for dev in (ids["dev1"], ids["dev2"], ids["dev3"]):
            mine, theirs = q.explain_many(s, [dev])[dev], materiality.explain(s, dev)
            assert set(mine) == set(theirs) == set(AUDIENCES)
            for aud, entry in theirs.items():
                assert {k: mine[aud][k] for k in entry} == entry, (dev, aud)
        batch = q.explain_many(s, [ids["dev1"], ids["dev2"], ids["dev3"]])
        assert batch[ids["dev2"]] == q.explain_many(s, [ids["dev2"]])[ids["dev2"]]


def test_first_failing_condition_shapes():
    from fli.web.queries import first_failing

    f = {"event_type": "model_release", "magnitude": 3, "p": 0.4, "verified": False, "lab_tier": 3,
         "person_tiers": [], "held_exposure": False, "exposed_tickers": ["GOOGL"], "lens_action": "watch",
         "corroborated": False, "source_classes": ["official"]}
    assert first_failing({"magnitude_min": 4}, f) == "magnitude 3 (needs >= 4)"
    assert first_failing({"magnitude_min": 4}, {**f, "magnitude": None}) == "magnitude not rated (needs >= 4)"
    assert first_failing({"p_floor": 0.5}, f) == "p 0.40 (needs >= 0.5)"
    assert first_failing({"p_floor": 0.5}, {**f, "p": None}) is None  # an unknown value passes a floor
    assert first_failing({"p_max": 0.3}, f) == "p 0.40 (needs <= 0.3)"
    assert first_failing({"event_type": ["funding_ipo"]}, f) == "event_type model_release (needs funding_ipo)"
    assert first_failing({"lab_tier": [1, 2]}, f) == "lab_tier 3 (needs one of 1, 2)"
    assert first_failing({"verified": True}, f) == "verified no (needs yes)"
    assert first_failing({"held_exposure": True}, f) == (
        "held_exposure no (needs yes); event-specific tickers GOOGL are not BIT holdings")
    assert first_failing({"corroborated": True}, f) == "corroborated no (needs yes); source classes: official"
    # policy order, passing conditions skipped
    assert first_failing({"event_type": ["funding_ipo"], "magnitude_min": 4}, f).startswith("event_type")
    assert first_failing({"verified": False, "magnitude_min": 4}, f) == "magnitude 3 (needs >= 4)"
    assert first_failing({"any": [{"person_tiers": ["A"]}, {"magnitude_min": 2, "lab_tier": [1, 2]}]}, f) == (
        "none of: person_tiers none (needs A) | lab_tier 3 (needs one of 1, 2)")
    assert first_failing({"all": [{"lab_tier": [3]}, {"magnitude_min": 4}]}, f) == "magnitude 3 (needs >= 4)"
    assert first_failing({"not": {"lens_action": ["no_action", "watch"]}}, f) == (
        "excluded: lens_action watch (the rule requires not lens_action one of no_action, watch)")
    assert first_failing({"not": {"lens_action": ["no_action"]}}, {**f, "lens_action": None}) is None
    assert first_failing({"magnitude_min": 1, "lab_tier": [3]}, f) is None


def test_first_failing_agrees_with_the_rule_engine(db, ids):
    """For every rule of the live policy and thousands of fact combinations, a reason is given exactly
    when the rule does not match (the panel can never contradict an alert)."""
    import random

    from fli.deliver import materiality
    from fli.taxonomy import EVENT_TYPES
    from fli.web.queries import first_failing

    pol = materiality.load_policy()
    with db() as s:
        ctxs = materiality.load_contexts(s, [ids["dev1"], ids["dev2"], ids["dev3"]])
        base = [materiality.facts_for(c, aud, pol) for c in ctxs.values() for aud in AUDIENCES]
    choices = {"event_type": list(EVENT_TYPES), "lab_tier": [1, 2, 3, None], "magnitude": [None, 1, 2, 3, 4, 5],
               "p": [None, 0.2, 0.5, 0.86, 0.95], "person_tiers": [[], ["A"], ["B", "C"]],
               "lens_action": [None, "no_action", "watch", "update_thesis", "hold", "adopt", "trial"],
               "held_exposure": [True, False], "verified": [True, False], "has_official": [True, False],
               "corroborated": [True, False], "breaking_change": [True, False], "has_code": [True, False],
               "n_source_classes": [1, 2, 3], "novelty": [None, 0.1, 0.9], "rumor_only": [True, False]}
    rules = [r for audience_rules in pol.rules.values() for r in audience_rules]
    rng = random.Random(7)
    for _ in range(2000):
        facts = {**rng.choice(base), **{k: rng.choice(v) for k, v in choices.items() if rng.random() < 0.7}}
        for rule in rules:
            assert (first_failing(rule.when, facts) is None) == materiality.matches(rule.when, facts), rule.id


def test_alerts_page_explains_each_alert(client, db, ids, policy):
    page = client.get("/alerts?status=all").text
    a1 = page[page.index("Claude X: AMZN tailwind"):]
    a1 = a1[: a1.index("</article>")]
    assert "Why flagged" in a1 and "t.inv.p1.flagship" in a1 and "magnitude 5 (&gt;= 5)" in a1
    assert "did not fire" in a1 and "event_type model_release (needs funding_ipo)" in a1
    assert "top contributions: official source" in a1 and "+0.800" in a1  # the score the rules read
    assert "recorded when raised:" in a1 and "tier1_official_release" in a1  # differs from today's rules
    items = {a["id"]: a for a in client.get("/api/alerts").json()["items"]}
    why = items[ids["a1"]]["why_flagged"]
    assert why["severity"] == "P1" and why["rules"][0]["matched"][:2] == ["event_type model_release", "lab_tier 1"]
    assert why["score"]["model_version"] == "logreg-v1" and why["score"]["top"][0]["feature"] == "official_source"
    assert [g["id"] for g in items[ids["a2"]]["why_flagged"]["guards"]] == ["guard.rumor_cap", "guard.unconfirmed"]
    with db() as s:  # recorded rules == today's: no second list
        s.get(Alert, ids["a1"]).rules = ["t.inv.p1.flagship"]
        s.commit()
    page = client.get("/alerts?status=all").text
    a1 = page[page.index("Claude X: AMZN tailwind"):]
    assert "recorded when raised:" not in a1[: a1.index("</article>")]


def test_pages_degrade_when_the_materiality_policy_is_broken(client, ids, tmp_path, monkeypatch):
    from fli.deliver import materiality

    bad = tmp_path / "broken.yaml"
    bad.write_text("audiences:\n  investment:\n    - {id: x, severity: P9, when: {magnitude_min: 1}}\n")
    monkeypatch.setattr(materiality, "POLICY_PATH", bad)
    page = client.get(f"/developments/{ids['dev1']}")
    assert page.status_code == 200 and "Materiality policy unavailable (PolicyError" in page.text
    for url in ("/alerts?status=all", "/feed/investment?bit=1&week=all", "/"):
        assert client.get(url).status_code == 200, url
    assert "PolicyError" in client.get("/api/alerts").json()["items"][0]["why_flagged"]["error"]


# ------------------------------------------------------------------------------------ exposure / curated
def test_holding_exposure_has_one_definition_everywhere(client, db, ids, policy):
    """The BIT filter, the ticker filter, the holdings tiles, the chips and the API count exactly what the
    materiality rules count (the held_tickers fact); lab relations are only context."""
    from fli.deliver import materiality
    from fli.web import queries as q

    now = utcnow()
    with db() as s:
        # the lens abstained: its directional NVDA is no exposure; AMZN only through the lab's relation
        quiet = _dev(s, now, "Docs refresh", 0.6, "no_action")
        s.query(Lens).filter_by(development_id=quiet).one().tickers = [
            {"symbol": "NVDA", "direction": "positive", "why": "stretch"}]
        s.add(ExposureLink(development_id=quiet, ticker="AMZN", method="lab", confidence=0.6, rationale="investor"))
        # a lens ticker without a direction, and an event-specific link to a non-holding
        vague = _dev(s, now, "Partnership note", 0.7, "watch")
        s.query(Lens).filter_by(development_id=vague).one().tickers = [{"symbol": "AMZN", "direction": "unclear"}]
        s.add(ExposureLink(development_id=vague, ticker="GOOGL", method="rule", confidence=0.8, rationale="partner"))
        s.commit()
        every_id = [ids["dev1"], ids["dev2"], ids["dev3"], quiet, vague]
        ctxs = materiality.load_contexts(s, every_id)
        held = {i: materiality.facts_for(ctxs[i], "investment", policy)["held_tickers"] for i in every_id}
        cards = {c["id"]: c for c in q.build_cards(s, every_id, "investment")}
        radar = {t["symbol"]: t for t in q.holdings_radar(s, now - timedelta(days=30), None)}
    assert held == {ids["dev1"]: ["AMZN", "NVDA"], ids["dev2"]: [], ids["dev3"]: [], quiet: [], vague: []}
    for i in every_id:
        assert cards[i]["held_tickers"] == held[i]
        assert sorted(t["symbol"] for t in cards[i]["tickers"] if t["bit_weight"] is not None) == held[i]
    assert [t["symbol"] for t in cards[vague]["tickers"]] == ["GOOGL"]  # counted, just not a holding
    assert cards[quiet]["tickers"] == [] and [t["symbol"] for t in cards[quiet]["lab_relations"]] == ["AMZN"]
    assert [t["symbol"] for t in cards[ids["dev1"]]["lab_relations"]] == ["MSFT"]

    def listed(url: str) -> set[int]:
        page = client.get(url).text
        return {i for i in every_id if f'href="/developments/{i}"' in page}

    everything = "/feed/investment?week=all&min_p=0&dismissed=1"
    assert listed(everything + "&bit=1") == {i for i in every_id if held[i]} == {ids["dev1"]}
    assert listed(everything + "&ticker=NVDA") == {ids["dev1"]}  # not the abstaining lens's NVDA
    assert listed(everything + "&ticker=AMZN") == {ids["dev1"]}  # not the lab-level or direction-less AMZN
    assert listed(everything + "&ticker=MSFT") == set()  # a lab relation only
    assert listed(everything + "&ticker=GOOGL") == {vague}  # 0.8 counts; dev2's 0.5 link does not
    assert (radar["AMZN"]["count"], radar["AMZN"]["lab_only"]) == (1, 1)
    assert (radar["NVDA"]["count"], radar["NVDA"]["lab_only"]) == (1, 0)
    assert (radar["MSFT"]["count"], radar["MSFT"]["lab_only"]) == (0, 1)
    # a tile links to exactly the developments it counts
    assert 'href="/feed/investment?ticker=NVDA&amp;week=7d&amp;min_p=0&amp;dismissed=1"' in client.get("/").text
    assert "1 development:" in client.get("/feed/investment?ticker=NVDA&week=7d&min_p=0&dismissed=1").text
    feed = client.get("/feed/investment").text
    assert "AMZN<b>9.6%</b>" in feed and "NVDA<b>8.0%</b>" in feed and "MSFT<b>4.4%</b>" not in feed
    assert "lab relation · MSFT" in feed
    dev1 = client.get(f"/developments/{ids['dev1']}").text
    assert "lab relation (context)" in dev1 and "committed tickers): <b>AMZN, NVDA</b>" in dev1
    assert "not counted (below 0.60)" in client.get(f"/developments/{ids['dev2']}").text
    assert "not counted as exposure: the lens abstained (no action)" in client.get(f"/developments/{quiet}").text
    assert "not counted as exposure: no committed direction" in client.get(f"/developments/{vague}").text
    api = {i["id"]: i for i in client.get("/api/developments", params={"since": "2026-01-01"}).json()["items"]}
    assert api[ids["dev1"]]["held_tickers"] == ["AMZN", "NVDA"]
    assert [t["symbol"] for t in api[ids["dev1"]]["lab_relations"]] == ["MSFT"]


def test_curated_register_entries_link_their_source_and_are_never_official(client, db, ids):
    now = utcnow()
    cited = "https://techcrunch.example/jane-joins-mistral"
    with db() as s:
        src = Source(id="register:affiliations", adapter="register_diff", kind="register", source_class="curated")
        s.add(src)
        s.flush()
        doc = _doc(src, "reg-jane", "Jane Doe joins Mistral AI as chief scientist", now - timedelta(hours=3),
                   person_id="jane-doe", meta={"source_url": cited})
        bare = _doc(src, "reg-bob", "Bob Roe leaves Mistral AI", now - timedelta(hours=2))  # cites its own URL
        dev = Development(title="Jane Doe joins Mistral", event_type="talent_move", lab_id="mistral",
                          person_ids=["jane-doe"], first_seen_at=now - timedelta(hours=3), last_seen_at=now,
                          insight_count=2, source_count=1, source_classes=["curated"], has_verified=True)
        s.add_all([doc, bare, dev])
        s.flush()
        s.add_all([  # is_official=True simulates an upstream slip: the page must still not say "official"
            _insight(doc, dev, "Jane Doe joins Mistral", [{"quote": "chief scientist", "verified": True}], True,
                     is_official=True),
            _insight(bare, dev, "Bob Roe leaves", [], False),
            Score(development_id=dev.id, audience="investment", p_material=0.8, model_version=LIVE["investment"])])
        s.commit()
        dev_id = dev.id
    badge = f'class="badge curated" href="{cited}"'
    page = client.get(f"/developments/{dev_id}").text
    assert badge in page and ">curated register entry</a>" in page
    assert 'class="badge curated" href="https://example.com/reg-bob"' in page  # no cited URL: its own
    assert ">official</span>" not in page and 'class="badge unconf"' not in page  # curated confirms, never official
    feed = client.get("/feed/investment").text
    assert badge in feed[feed.index("Jane Doe joins Mistral"):][:4000]  # the card's class badge links too
    assert badge in client.get("/register/people/jane-doe").text
    item = next(i for i in client.get("/api/developments", params={"audience": "investment"}).json()["items"]
                if i["id"] == dev_id)
    assert item["unconfirmed"] is False and item["has_official"] is False
    assert item["curated_sources"] == [{"title": "Jane Doe joins Mistral AI as chief scientist", "cited_url": cited},
                                       {"title": "Bob Roe leaves Mistral AI", "cited_url": "https://example.com/reg-bob"}]
    sources = {x["url"]: x for x in item["sources"]}
    assert sources["https://example.com/reg-jane"]["curated"] is True
    assert sources["https://example.com/reg-jane"]["cited_url"] == cited
    detail = client.get(f"/api/developments/{dev_id}").json()
    assert [i["is_official"] for i in detail["insights"]] == [False, False]
    assert detail["documents"][0]["cited_url"] == cited
    official = client.get(f"/api/developments/{ids['dev1']}").json()
    assert official["insights"][0]["is_official"] is True and official["documents"][0]["curated"] is False


def test_cited_url_reads_what_the_register_collector_writes():
    from fli.web.queries import cited_url

    assert cited_url("https://a.example", {"source_url": " https://b.example ", "sources": ["x"]}) == "https://b.example"
    assert cited_url("https://a.example", {"sources": ["", "https://c.example"]}) == "https://c.example"
    assert cited_url("https://a.example", {"source_urls": ["https://d.example"]}) == "https://d.example"
    assert cited_url("https://a.example", None) == "https://a.example" and cited_url(None, {}) is None


# ------------------------------------------------------------------------------------ deployment
def _deploy_check(tmp_path: Path, env_text: str | None) -> subprocess.CompletedProcess:
    """Run deploy.sh's .env check only (FLI_DEPLOY_CHECK_ONLY=1) in a scratch tree, with a scratch
    HOME and key path and fake aws/curl/ssh that fail, so nothing real can ever be touched."""
    root = tmp_path / "repo"
    (root / "deploy" / "aws").mkdir(parents=True)
    shutil.copy(REPO / "deploy" / "aws" / "deploy.sh", root / "deploy" / "aws" / "deploy.sh")
    if env_text is not None:
        (root / ".env").write_text(env_text)
    fake = tmp_path / "bin"
    fake.mkdir()
    for tool in ("aws", "curl", "ssh", "scp", "rsync"):
        (fake / tool).write_text("#!/bin/sh\necho \"fake $0 called\" >&2\nexit 97\n")
        (fake / tool).chmod(0o755)
    env = {"PATH": f"{fake}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(tmp_path / "home"),
           "FLI_KEY_FILE": str(tmp_path / "home" / "key.pem"), "FLI_DEPLOY_CHECK_ONLY": "1"}
    return subprocess.run(["bash", str(root / "deploy" / "aws" / "deploy.sh")], env=env, capture_output=True,
                          text=True, timeout=30)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(("env_text", "ok"), [
    (None, False),
    ("ANTHROPIC_API_KEY=x\n", False),
    ("FLI_BASIC_AUTH=\n", False),
    ("FLI_BASIC_AUTH=pm:\n", False),
    ("FLI_BASIC_AUTH=:a-long-password\n", False),
    ("FLI_BASIC_AUTH=pm:short\n", False),
    ("# FLI_BASIC_AUTH=pm:a-long-password\n", False),
    ("FLI_BASIC_AUTH=pm:a-long-password\nFLI_BASIC_AUTH=\n", False),  # the last assignment wins
    ("FLI_BASIC_AUTH=pm:a-long-password   # team login\n", True),
    ("FLI_BASIC_AUTH=\"pm:quoted pass #1x\"\n", True),
    ("export FLI_BASIC_AUTH='pm:single-quoted'\n", True),
])
def test_deploy_refuses_without_basic_auth(tmp_path, env_text, ok):
    r = _deploy_check(tmp_path, env_text)
    assert "fake" not in r.stderr, r.stderr  # never reaches AWS, ssh or curl
    if ok:
        assert r.returncode == 0 and "auth check passed" in r.stdout, r.stderr
    else:
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        assert ("FLI_BASIC_AUTH" if env_text is not None else "missing") in r.stderr, r.stderr


def test_deploy_scripts_harden_the_public_instance():
    overlay = yaml.safe_load((REPO / "deploy" / "aws" / "docker-compose.aws.yml").read_text())
    web_env = overlay["services"]["web"]["environment"]
    assert web_env["FLI_AUTH_ALL"] == "1"
    assert web_env["FLI_BASIC_AUTH"].startswith("${FLI_BASIC_AUTH:?")  # compose refuses to start without it
    script = (REPO / "deploy" / "aws" / "deploy.sh").read_text()
    assert script.index("FLI_BASIC_AUTH") < script.index("create-key-pair")  # checked before anything exists
    assert "FLI_PORT=127.0.0.1:8000" in script and "/healthz" in script and "-K -" in script
    example = (REPO / ".env.example").read_text()
    assert "FLI_BASIC_AUTH=" in example and "FLI_AUTH_ALL=1" in example
    assert "\nFLI_AUTH_ALL=\n" not in example  # an empty bool would break Settings()
