"""X (metered, budget-guarded) and Bluesky collectors, against recorded or schema-built fixtures."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import requests
import respx
import tweepy

from fli.db.models import ApiUsage, Identity, Lab, Source
from fli.ingest.collectors import social
from fli.ingest.records import COLLECTORS, CollectorUnavailable, SourceInfo, Window
from fli.ingest.runner import persist_records

FIX = Path(__file__).parent / "fixtures" / "social"


def load(name: str):
    return json.loads((FIX / name).read_text())


class FakeX:
    """Stands in for tweepy.Client(return_type=dict); records the parameters of every call."""

    def __init__(self, pages: list[dict] | None = None, lookup: dict | None = None, error: Exception | None = None):
        self.pages = list(pages or [])
        self.lookup = lookup or load("x_user_lookup.json")
        self.error = error
        self.user_calls: list[dict] = []
        self.timeline_calls: list[dict] = []

    def get_user(self, **kw):
        self.user_calls.append(kw)
        if self.error:
            raise self.error
        return self.lookup

    def get_users_tweets(self, user_id, **kw):
        self.timeline_calls.append({"user_id": user_id, **kw})
        if self.error:
            raise self.error
        return self.pages.pop(0) if self.pages else load("x_timeline_empty.json")


def http_error(cls, status: int, body: dict):
    resp = requests.Response()
    resp.status_code, resp.reason = status, "error"
    resp._content = json.dumps(body).encode()
    resp.headers["content-type"] = "application/json"
    return cls(resp)


@pytest.fixture()
def x_source(db):
    with db() as s:
        s.add(Lab(id="anthropic", name="Anthropic", kind="frontier_lab", tier=1))
        s.flush()  # identity.lab_id has no ORM relationship, so insert order is not inferred
        s.add(Source(id="anthropic:x", lab_id="anthropic", adapter="x_timeline", kind="x_account",
                     source_class="official", config={"handle": "AnthropicAI"}))
        s.add(Identity(platform="x", handle="anthropicai", lab_id="anthropic"))
        s.commit()
    return SourceInfo(id="anthropic:x", adapter="x_timeline", kind="x_account", source_class="official",
                      url=None, config={"handle": "AnthropicAI"}, lab_id="anthropic")


WINDOW = Window(since=datetime(2026, 8, 1), until=datetime(2026, 11, 4, 12))


def ledger(db) -> list[tuple[str, int, float]]:
    with db() as s:
        return [(u.endpoint, u.units, round(u.cost_usd, 6)) for u in s.query(ApiUsage).order_by(ApiUsage.id)]


def test_collectors_registered():
    assert COLLECTORS["x_timeline"].requires == ("x_bearer_token",)
    assert "bluesky_feed" in COLLECTORS


def test_x_post_to_record_uses_note_tweet_and_expands_links():
    posts = load("x_timeline_page1.json")["data"]
    first, long_post, quote, media = (social.x_post_to_record(p, "AnthropicAI") for p in posts)
    assert first.url == "https://x.com/AnthropicAI/status/1985112233445566778"
    assert "https://www.anthropic.com/news/fixture-announcement" in first.content_text
    assert "t.co" not in first.content_text
    assert "(Fixture announcement page)" in first.content_text  # link title from entities
    assert first.meta["metrics"]["like_count"] == 6120 and first.author == "AnthropicAI"
    assert first.published_at == datetime(2026, 10, 28, 16, 0, 5)
    assert long_post.meta["long_post"] and "final sentence that a truncated read would lose" in long_post.content_text
    assert long_post.meta["urls"] == ["https://www.anthropic.com/research/fixture-paper"]
    assert quote.meta["is_quote"] and quote.meta["quoted_id"] == "1983000000000000000"
    assert media.meta["urls"] == [] and "pic.x.com" not in media.content_text and "t.co" not in media.content_text


def test_x_timeline_caches_user_id_and_meters_usage(db, settings, monkeypatch, x_source):
    fake = FakeX(pages=[load("x_timeline_page1.json"), load("x_timeline_page2.json")])
    monkeypatch.setattr(social, "_x_client", lambda _s: fake)
    records = social.x_timeline(x_source, WINDOW, settings)
    assert len(records) == 5
    assert len(fake.user_calls) == 1 and fake.user_calls[0]["username"] == "anthropicai"
    first = fake.timeline_calls[0]
    assert first["user_id"] == "1353836358901501952"
    assert first["exclude"] == ["retweets", "replies"] and "note_tweet" in first["tweet_fields"]
    assert first["start_time"] == WINDOW.since and "since_id" not in first and first["max_results"] == 60
    assert fake.timeline_calls[1]["pagination_token"] == "7140dibdnow9c7btw4b3v1i0k"
    assert ledger(db) == [(social.X_LOOKUP_ENDPOINT, 1, 0.01), (social.X_TIMELINE_ENDPOINT, 4, 0.02),
                          (social.X_TIMELINE_ENDPOINT, 1, 0.005)]  # settled to posts returned, not page size
    with db() as s:
        ident = s.query(Identity).filter_by(platform="x", handle="anthropicai").one()
        assert ident.meta["x_user_id"] == "1353836358901501952" and ident.meta["x_followers"] == 1100000
        persist_records(s, s.get(Source, "anthropic:x"), records)
        s.commit()

    # second run: cached id (no lookup), since_id = newest stored post, small page
    fake2 = FakeX()
    monkeypatch.setattr(social, "_x_client", lambda _s: fake2)
    assert social.x_timeline(x_source, WINDOW, settings) == []
    assert fake2.user_calls == []
    call = fake2.timeline_calls[0]
    assert call["since_id"] == "1985112233445566778" and "start_time" not in call and call["max_results"] == 20
    assert ledger(db)[-1] == (social.X_TIMELINE_ENDPOINT, 0, 0.0)


def test_x_budget_guard_refuses_before_spending(db, settings, monkeypatch, x_source):
    with db() as s:
        s.add(ApiUsage(provider="x", endpoint="earlier runs", units=1, cost_usd=9.70))
        s.commit()
    fake = FakeX(pages=[load("x_timeline_page1.json")])
    monkeypatch.setattr(social, "_x_client", lambda _s: fake)
    with pytest.raises(CollectorUnavailable, match="X budget guard"):
        social.x_timeline(x_source, WINDOW, settings)  # lookup ($0.01) fits, a 60-post page ($0.30) does not
    assert len(fake.user_calls) == 1 and fake.timeline_calls == []
    assert [e for e, _, _ in ledger(db)] == ["earlier runs", social.X_LOOKUP_ENDPOINT]

    tight = settings.model_copy(update={"x_budget_total_usd": 9.70})
    fake2 = FakeX()
    monkeypatch.setattr(social, "_x_client", lambda _s: fake2)
    with db() as s:  # forget the cached id so the next run needs a lookup
        ident = s.query(Identity).filter_by(handle="anthropicai").one()
        ident.meta = {}
        s.commit()
    with pytest.raises(CollectorUnavailable):
        social.x_timeline(x_source, WINDOW, tight)
    assert fake2.user_calls == []  # refused before the request was sent


def test_x_reservations_hold_under_concurrency(db):
    book = social.XLedger(budget_usd=0.35)
    outcomes: list[str] = []

    def worker():
        try:
            book.reserve(social.X_TIMELINE_ENDPOINT, 60, social.X_COST_PER_POST)  # $0.30 each
            outcomes.append("ok")
        except CollectorUnavailable:
            outcomes.append("refused")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["ok", "refused", "refused", "refused"]
    with db() as s:
        assert social.x_spent_usd(s) == pytest.approx(0.30)


@pytest.mark.parametrize("error, reason", [
    (lambda: http_error(tweepy.errors.Forbidden, 403, load("x_error_403.json")), "403 forbidden"),
    (lambda: http_error(tweepy.errors.Unauthorized, 401, {"title": "Unauthorized", "detail": "Unauthorized"}),
     "401 unauthorized"),
    (lambda: http_error(tweepy.errors.HTTPException, 402, {"title": "CreditsDepleted", "detail": "no credits"}),
     "402 payment required"),
    (lambda: http_error(tweepy.errors.TooManyRequests, 429, {"title": "Too Many Requests", "detail": "slow"}),
     "429 rate limited"),
])
def test_x_access_errors_skip_and_are_not_billed(db, settings, monkeypatch, x_source, error, reason):
    fake = FakeX(error=error())
    monkeypatch.setattr(social, "_x_client", lambda _s: fake)
    with pytest.raises(CollectorUnavailable, match=reason):
        social.x_timeline(x_source, WINDOW, settings)
    assert ledger(db) == []  # rejected requests release their reservation


def test_x_403_reason_carries_api_detail(db, settings, monkeypatch, x_source):
    monkeypatch.setattr(social, "_x_client",
                        lambda _s: FakeX(error=http_error(tweepy.errors.Forbidden, 403, load("x_error_403.json"))))
    with pytest.raises(CollectorUnavailable, match="attached to a Project"):
        social.x_timeline(x_source, WINDOW, settings)


def test_x_unknown_outcome_keeps_worst_case(db, settings, monkeypatch, x_source):
    monkeypatch.setattr(social, "_x_client", lambda _s: FakeX(error=requests.Timeout("read timed out")))
    with pytest.raises(requests.Timeout):
        social.x_timeline(x_source, WINDOW, settings)
    assert ledger(db) == [(social.X_LOOKUP_ENDPOINT, 1, 0.01)]  # may have been served: stays booked


def test_x_client_has_timeout_and_hides_token_logs(settings):
    import logging

    client = social._x_client(settings.model_copy(update={"x_bearer_token": "test-token"}))
    assert client.session.request.keywords["timeout"] == settings.http_timeout_s
    assert not logging.getLogger("tweepy.client").isEnabledFor(logging.DEBUG)


# ------------------------------------------------------------------------------- Bluesky
BSKY_SOURCE = SourceInfo(id="x:test-bsky", adapter="bluesky_feed", kind="bluesky_account", source_class="personal",
                         url=None, config={"handle": "yann-lecun.bsky.social"})


def _bsky_router(router: respx.Router) -> respx.Route:
    pages = {None: load("bsky_feed_page1.json"), "2024-11-28T19:54:33.361Z": load("bsky_feed_page2.json")}
    return router.get(social.BSKY_FEED_URL).mock(
        side_effect=lambda request: httpx.Response(200, json=pages[request.url.params.get("cursor")]))


def test_bluesky_feed_paginates_expands_facets_and_skips_reposts(settings):
    with respx.mock(assert_all_called=True) as router:
        route = _bsky_router(router)
        recs = social.bluesky_feed(BSKY_SOURCE, Window(since=datetime(2024, 11, 1), until=datetime(2024, 12, 31)),
                                   settings)
    assert route.call_count == 2
    assert route.calls[0].request.url.params["filter"] == "posts_no_replies"
    assert len(recs) == 4  # five items, one repost skipped
    assert all(r.author == "yann-lecun.bsky.social" for r in recs)
    first = recs[0]
    assert first.url.startswith("https://bsky.app/profile/yann-lecun.bsky.social/post/")
    assert first.natural_key.startswith("at://") and first.meta["platform"] == "bluesky"
    assert "Links:" in first.content_text and first.meta["urls"][0].startswith("http")
    assert first.meta["metrics"]["likeCount"] is not None
    assert any(r.meta["is_quote"] for r in recs)


def test_bluesky_stops_paging_once_older_than_window(settings):
    with respx.mock() as router:
        route = _bsky_router(router)
        recs = social.bluesky_feed(BSKY_SOURCE, Window(since=datetime(2024, 12, 1), until=datetime(2024, 12, 31)),
                                   settings)
    assert route.call_count == 1
    assert [r.published_at.date().isoformat() for r in recs] == ["2024-12-15"]


def test_apply_link_facets_uses_utf8_byte_offsets():
    text = "café → example.com/sh..."
    start = len("café → ".encode())
    facets = [{"index": {"byteStart": start, "byteEnd": len(text.encode())},
               "features": [{"$type": "app.bsky.richtext.facet#link", "uri": "https://example.com/short/full-path"}]}]
    out, urls = social._apply_link_facets(text, facets)
    assert out == "café → https://example.com/short/full-path"
    assert urls == ["https://example.com/short/full-path"]
