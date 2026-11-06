"""Market (OpenRouter), press (GDELT) and filings (EDGAR full-text search) collectors."""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import respx
import tenacity

from fli import config as fli_config
from fli.db.models import Source
from fli.ingest.collectors import filings, market, press
from fli.ingest.records import CollectorUnavailable, SourceInfo, Window

FIX = Path(__file__).parent / "fixtures" / "market"
WINDOW = Window(since=datetime(2026, 8, 1), until=datetime(2026, 11, 4, 12))


def load(name: str):
    return json.loads((FIX / name).read_text())


def src(adapter: str, config: dict, lab_id: str | None = None, sid: str = "t:src") -> SourceInfo:
    return SourceInfo(id=sid, adapter=adapter, kind="x", source_class="press", url=None, config=config, lab_id=lab_id)


# ------------------------------------------------------------------------------- OpenRouter
def _openrouter(router: respx.Router) -> respx.Route:
    return router.get(market.OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(200, json=load("openrouter_models.json")))


def test_openrouter_selects_new_base_models_from_tracked_labs(settings):
    with respx.mock() as router:
        _openrouter(router)
        recs = market.openrouter_models(src("openrouter_models", {}), WINDOW, settings)
    by_key = {r.natural_key: r for r in recs}
    # variants (':batch'), floating aliases ('~openai/...'), untracked providers and pre-window models are out
    assert {k: r.lab_id for k, r in by_key.items()} == {
        "or:anthropic/claude-haiku-5.5": "anthropic", "or:x-ai/grok-4.7": "xai",
        "or:meta/muse-spark-1.3": "meta-ai", "or:reflection/beam-1.0-501b-a23b": "reflection-ai",
        "or:deepseek/deepseek-v4.1-flash-1017": "deepseek"}
    haiku = by_key["or:anthropic/claude-haiku-5.5"]
    assert haiku.url == "https://openrouter.ai/anthropic/claude-haiku-5.5"
    assert haiku.published_at.date().isoformat() == "2026-10-07"
    m = haiku.meta
    assert (m["prompt_usd_per_mtok"], m["completion_usd_per_mtok"]) == (0.1, 0.5)
    assert m["variants"]["batch"] == {"prompt": 0.05, "completion": 0.25, "cache_read": 0.005}
    assert m["long_context_pricing"]["min_prompt_tokens"] == 100000 and m["long_context_pricing"]["prompt"] == 0.5
    assert m["context_length"] == 1000000
    assert "$0.10 input, $0.50 output" in haiku.content_text and "Batch variant" in haiku.content_text
    assert [r.published_at for r in recs] == sorted((r.published_at for r in recs), reverse=True)


def test_openrouter_extra_providers_and_full_window_refresh(settings):
    narrow = Window(since=datetime(2026, 11, 1), until=WINDOW.until)  # incremental run
    with respx.mock() as router:
        _openrouter(router)
        recs = market.openrouter_models(src("openrouter_models", {"extra_providers": ["z-ai"]}), narrow, settings)
    keys = {r.natural_key: r.lab_id for r in recs}
    assert keys["or:z-ai/glm-5.3-flashx-legal"] is None  # untracked open-weight leader, no lab attribution
    assert "or:anthropic/claude-haiku-5.5" in keys  # pricing refresh covers the whole tracked window


def test_openrouter_time_of_day_and_long_context_overrides(settings):
    with respx.mock() as router:
        _openrouter(router)
        recs = market.openrouter_models(src("openrouter_models", {"extra_providers": ["minimax"]}), WINDOW, settings)
    by_key = {r.natural_key: r for r in recs}
    deepseek = by_key["or:deepseek/deepseek-v4.1-flash-1017"]  # real off-peak windows
    assert deepseek.meta["long_context_pricing"] is None and len(deepseek.meta["time_of_day_pricing"]) == 6
    assert "Time-of-day pricing (6 UTC windows): input $0.15 to $0.30 per million tokens." in deepseek.content_text
    minimax = by_key["or:minimax/minimax-m3.1-flash"]
    assert minimax.lab_id is None and len(minimax.meta["time_of_day_pricing"]) == 2


def test_openrouter_skips_malformed_entries(settings):
    payload = load("openrouter_models.json")
    haiku = next(i for i, m in enumerate(payload["data"]) if m["id"] == "anthropic/claude-haiku-5.5")
    grok = next(i for i, m in enumerate(payload["data"]) if m["id"] == "x-ai/grok-4.7")
    payload["data"][haiku]["created"] = "not-a-timestamp"  # rejected while selecting
    payload["data"][grok]["context_length"] = "unknown"  # fails while rendering
    with respx.mock() as router:
        router.get(market.OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(200, json=payload))
        recs = market.openrouter_models(src("openrouter_models", {}), WINDOW, settings)
    assert {r.natural_key for r in recs} == {"or:meta/muse-spark-1.3", "or:reflection/beam-1.0-501b-a23b",
                                             "or:deepseek/deepseek-v4.1-flash-1017"}


def test_per_mtok_conversion():
    assert market.per_mtok("0.000003") == 3.0
    assert market.per_mtok("-1") is None and market.per_mtok(None) is None


# ------------------------------------------------------------------------------- GDELT
def test_parse_gdelt_real_payloads():
    arts = press.parse_gdelt((FIX / "gdelt_anthropic.json").read_text())
    assert len(arts) == 7 and {a["language"] for a in arts} == {"English"}
    assert press.parse_gdelt("") == [] and press.parse_gdelt("{}") == []
    with pytest.raises(press.GdeltThrottled):
        press.parse_gdelt((FIX / "gdelt_throttled.txt").read_text())
    with pytest.raises(press.GdeltQueryError, match="illegal character"):
        press.parse_gdelt((FIX / "gdelt_query_error.txt").read_text())


def test_title_cleanup_and_syndication_key():
    raw = "Stock Market Today Live : Gift Nifty points to gap - up opening ; crude oil , US election in focus"
    assert press.clean_title(raw) == "Stock Market Today Live: Gift Nifty points to gap-up opening; crude oil, US election in focus"
    assert press.clean_title("Anthropic ' s compute bill ( estimated )") == "Anthropic's compute bill (estimated)"
    assert press.title_key("Anthropic files for IPO - Reuters") == press.title_key("Anthropic files for IPO | Yahoo Finance")
    assert press.detect_lab("Anthropic warns TSMC disruptions could hit AI chip supply") == "anthropic"
    assert press.detect_lab("OpenAI and Anthropic race for compute") is None
    assert press.detect_lab("SpaceXAI raises again") == "xai"


def test_articles_filtering_collapses_syndicated_copies():
    arts = load("gdelt_anthropic.json")["articles"]
    first = arts[0]
    extra = [
        {**first, "url": "https://syndicated.example.com/copy", "domain": "syndicated.example.com"},  # same title+day
        {**first, "url": "https://es.example.com/a", "title": "Anthropic solicita su salida a bolsa", "language": "Spanish"},
        {**first, "url": "https://spam.example.net/b", "title": "Unrelated headline", "domain": "spam.example.net"},
    ]
    cfg = {"deny_domains": ["example.net"]}
    seen: set = set()
    recs = press.articles_to_records(arts + extra, query={"q": "q", "lab": "anthropic"}, seen=seen, cfg=cfg,
                                     since=datetime(2026, 10, 27), until=datetime(2026, 11, 4), limit=50)
    urls = [r.url for r in recs]
    assert len(recs) == 7 and "https://syndicated.example.com/copy" not in urls
    assert all("example" not in u for u in urls)
    r0 = recs[0]
    assert r0.content_text == r0.title and r0.source_class == "press" and r0.lab_id == "anthropic"
    assert r0.natural_key == "https://thedailytimes.com/ap/business/anthropic-files-for-potentially-the-biggest-ipo-ever-as-ceo-warns-of-risk-to-humanity/article_5dfd4b80-6ce8-5fb9-89d3-5c05dce075a7.html"
    assert r0.meta["domain"] == "thedailytimes.com" and r0.published_at == datetime(2026, 10, 27, 5, 15)


def test_gdelt_news_chunks_window_and_tags_queries(settings, monkeypatch):
    monkeypatch.setattr(press, "GDELT_MIN_INTERVAL_S", 0.0)
    source = src("gdelt_news", {"queries": [{"q": '"Anthropic" (IPO OR funding)', "lab": "anthropic"}], "chunk_days": 7})
    window = Window(since=datetime(2026, 10, 21), until=datetime(2026, 11, 4))
    with respx.mock() as router:
        route = router.get(press.GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, text=(FIX / "gdelt_anthropic.json").read_text()))
        recs = press.gdelt_news(source, window, settings)
    assert route.call_count == 2  # two 7-day chunks
    params = route.calls[0].request.url.params
    assert params["query"] == '"Anthropic" (IPO OR funding) sourcelang:english'
    assert params["mode"] == "artlist" and params["format"] == "json" and params["sort"] == "hybridrel"
    assert (params["startdatetime"], params["enddatetime"]) == ("20261028000000", "20261104000000")
    assert len(recs) == 7 and all(r.lab_id == "anthropic" for r in recs)


def test_gdelt_persistent_throttle_skips_whole_run(settings, monkeypatch):
    monkeypatch.setattr(press, "GDELT_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(press.gdelt_articles.retry, "wait", tenacity.wait_none())
    with respx.mock() as router:
        route = router.get(press.GDELT_DOC_URL).mock(
            return_value=httpx.Response(429, text=(FIX / "gdelt_throttled.txt").read_text()))
        with pytest.raises(CollectorUnavailable, match="GDELT throttled"):
            press.gdelt_news(src("gdelt_news", {"queries": ["Anthropic"]}), WINDOW, settings)
    assert route.call_count == 3  # retried with backoff before giving up


def test_gdelt_query_preflight_rejects_unquoted_dashes(settings):
    press.validate_query('"Anthropic" (IPO OR "S-1")')
    with respx.mock(assert_all_called=False) as router:
        route = router.get(press.GDELT_DOC_URL)
        with pytest.raises(press.GdeltQueryError, match="quoted"):
            press.gdelt_news(src("gdelt_news", {"queries": ['"Anthropic" (IPO OR S-1)']}), WINDOW, settings)
    assert not route.called


def test_gdelt_chunks_are_newest_first_and_cover_window():
    spans = press.chunks(datetime(2026, 10, 1), datetime(2026, 10, 31), 14)
    assert spans[0][1] == datetime(2026, 10, 31) and spans[-1][0] == datetime(2026, 10, 1)
    assert all(a[0] == b[1] for a, b in itertools.pairwise(spans))


# ------------------------------------------------------------------------------- EDGAR full-text search
def test_edgar_hits_collapse_to_one_record_per_filing():
    recs = filings.hits_to_records(load("edgar_fts_anthropic.json")["hits"]["hits"],
                                   query={"q": '"Anthropic"', "lab": "anthropic"})
    by_key = {r.natural_key: r for r in recs}
    assert set(by_key) == {"sec:0009999999-26-000123", "sec:0008888888-26-000201"}
    s1 = by_key["sec:0009999999-26-000123"]
    assert s1.url == "https://www.sec.gov/Archives/edgar/data/9999999/000999999926000123/fixture-s1.htm"
    assert s1.author == "FIXTURE ISSUER PBC (FXT)" and s1.meta["form"] == "S-1"
    assert s1.meta["matched_files"] == ["fixture-s1.htm", "fixture-ex10-1.htm"]
    assert s1.published_at == datetime(2026, 10, 26) and s1.lab_id == "anthropic" and s1.source_class == "filing"
    eightk = by_key["sec:0008888888-26-000201"]
    assert "1.01 Entry into a Material Definitive Agreement" in eightk.content_text
    assert eightk.title == "FIXTURE CLOUD CORP (FXC) 8-K (2026-10-28): Current report"


def test_edgar_fts_paginates_and_sends_user_agent(settings, monkeypatch):
    monkeypatch.setattr(filings, "REQUEST_GAP_S", 0.0)
    payload = load("edgar_fts_anthropic.json")
    hits = payload["hits"]["hits"]
    page1 = {**payload, "hits": {**payload["hits"], "hits": hits[:2]}}
    page2 = {**payload, "hits": {**payload["hits"], "hits": hits[2:]}}
    ua_settings = settings.model_copy(update={"sec_user_agent": "FLI Research test@example.com"})
    with respx.mock() as router:
        route = router.get(filings.EFTS_URL).mock(side_effect=[httpx.Response(200, json=page1),
                                                               httpx.Response(200, json=page2)])
        recs = filings.edgar_fts(src("edgar_fts", {"queries": [{"q": '"Anthropic"', "lab": "anthropic"}]}),
                                 WINDOW, ua_settings)
    assert route.call_count == 2 and len(recs) == 2
    first, second = (c.request for c in route.calls)
    assert first.headers["user-agent"] == "FLI Research test@example.com"
    assert first.url.params["forms"] == "8-K,S-1,10-Q,10-K" and first.url.params["dateRange"] == "custom"
    assert (first.url.params["startdt"], first.url.params["enddt"]) == ("2026-08-01", "2026-11-04")
    assert "from" not in first.url.params and second.url.params["from"] == "2"


def test_edgar_is_skipped_without_user_agent(db, monkeypatch):
    from fli.ingest.runner import run_ingest

    monkeypatch.setenv("SEC_USER_AGENT", "")
    fli_config.get_settings.cache_clear()
    with db() as s:
        s.add(Source(id="sec:edgar-fts", adapter="edgar_fts", kind="sec", source_class="filing",
                     config={"queries": ['"Anthropic"']}))
        s.commit()
    out = run_ingest(source_ids=["sec:edgar-fts"])
    assert out["sec:edgar-fts"] == {"status": "skipped", "reason": "missing settings: sec_user_agent"}
