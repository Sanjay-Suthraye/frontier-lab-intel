"""Official-channel collectors (rss, sitemap, docs_changelog) against trimmed real responses.

Fixtures in tests/fixtures/feeds/ were cut from live responses on 2026-11-04 (alignment.openai.com,
anthropic.com, platform.claude.com, developers.openai.com, platform.kimi.ai, docs.x.ai,
api-docs.deepseek.com, periodic.com). No network: every request goes through respx.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from fli.ingest.collectors import feeds
from fli.ingest.records import CollectorUnavailable, SourceInfo, Window

FIX = Path(__file__).parent / "fixtures" / "feeds"
NOW = datetime(2026, 11, 4, 12, 0)
WINDOW = Window(since=datetime(2026, 8, 1), until=NOW)
XML = {"content-type": "application/xml"}
HTML = {"content-type": "text/html; charset=utf-8"}
MD = {"content-type": "text/markdown; charset=utf-8"}

ALIGN = "https://alignment.openai.com"
ANTHROPIC = "https://www.anthropic.com"


def _fx(name: str) -> bytes:
    return (FIX / name).read_bytes()


def _src(adapter: str, url: str, **config) -> SourceInfo:
    return SourceInfo(id=f"test:{adapter}", adapter=adapter, kind="official_blog", source_class="official",
                      url=url, config={"delay_s": 0, **config})


@pytest.fixture()
def fixed_now(monkeypatch):
    """Changelog headings without a year are resolved relative to 'now'."""
    monkeypatch.setattr(feeds, "utcnow", lambda: NOW)
    return NOW


# --------------------------------------------------------------------------------- date labels
def test_date_labels_cover_real_heading_styles():
    label = feeds._date_label
    assert label("Oct 30") == feeds._Label(None, 10, 30)  # OpenAI
    assert label("November 3, 2026 \ue09a") == feeds._Label(2026, 11, 3)  # Claude (icon glyph)
    assert label("Date: 2026-10-17 \u200b") == feeds._Label(2026, 10, 17)  # DeepSeek
    assert label('November 2, 2026  {% id="11-02-2026" %}') == feeds._Label(2026, 11, 2)  # Gemini
    assert label("May 10th, 2024") == feeds._Label(2024, 5, 10)
    assert label("October, 2026") == feeds._Label(2026, 10, None)
    assert label("November") == feeds._Label(None, 11, None)  # xAI month heading
    assert label("2025") == feeds._Label(2025, None, None)
    for text in ("Grok 4.7", "Release notes", "Feb 30, 2026", "Introducing Gemini 3 (May 2026)"):
        assert label(text) is None


def test_missing_year_comes_from_context_then_recency():
    assert feeds._resolve(feeds._Label(None, 10, 30), 2024, NOW) == date(2024, 10, 30)
    assert feeds._resolve(feeds._Label(None, 11, 3), None, NOW) == date(2026, 11, 3)
    assert feeds._resolve(feeds._Label(None, 12, 15), None, NOW) == date(2025, 12, 15)  # never the future


# ------------------------------------------------------------------------------------------- rss
def _alignment_routes(mock: respx.MockRouter, robots: str | None = None) -> dict[str, respx.Route]:
    if robots is None:
        mock.get(f"{ALIGN}/robots.txt").respond(404)
    else:
        mock.get(f"{ALIGN}/robots.txt").respond(200, text=robots)
    mock.get(f"{ALIGN}/rss.xml").respond(200, content=_fx("alignment_rss.xml"), headers=XML)
    return {
        "article": mock.get(f"{ALIGN}/how-we-detect-frontier-model-misalignment/").respond(
            200, content=_fx("alignment_article.html"), headers=HTML),
        "missing": mock.get(f"{ALIGN}/eval-production-behavior-gap/").respond(404),
    }


def test_rss_window_full_text_cache_and_stable_keys(settings):
    src = _src("rss", f"{ALIGN}/rss.xml", fetch_full_text=True)
    with respx.mock(assert_all_called=False) as mock:
        routes = _alignment_routes(mock)
        first = feeds.collect_rss(src, WINDOW, settings)
        second = feeds.collect_rss(src, WINDOW, settings)

    # newest first; the 2025-12-01 post and the commented-out <item> template are not records
    assert [r.url for r in first] == [f"{ALIGN}/eval-production-behavior-gap/",
                                      f"{ALIGN}/how-we-detect-frontier-model-misalignment/"]
    assert [r.natural_key for r in first] == [r.natural_key for r in second]
    assert first[0].published_at == datetime(2026, 11, 3, 8, 0)  # "00:00 -0800" stored as naive UTC
    full = first[1]
    assert full.meta["full_text"] is True and full.meta["summary"]
    assert len(full.content_text) > 2000 and "misalign" in full.content_text.lower()
    assert full.author and "Marcus Williams" in full.author
    assert first[0].meta["full_text"] is False and first[0].content_text  # page 404 -> summary kept
    assert routes["article"].call_count == 1  # second run served from the page cache
    assert routes["missing"].call_count == 2  # failures are not cached


def test_rss_respects_robots_for_article_pages_and_feed(settings):
    src = _src("rss", f"{ALIGN}/rss.xml", fetch_full_text=True)
    with respx.mock(assert_all_called=False) as mock:
        routes = _alignment_routes(mock, robots="User-agent: *\nDisallow: /how-we-detect\n")
        recs = feeds.collect_rss(src, WINDOW, settings)
    assert routes["article"].call_count == 0
    assert recs[1].meta["full_text"] is False and len(recs[1].content_text) < 300

    for robots in ("User-agent: *\nDisallow: /\n", "User-agent: *\nContent-Signal: search=yes, ai-input=no\n"):
        with respx.mock(assert_all_called=False) as mock:
            routes = _alignment_routes(mock, robots=robots)
            with pytest.raises(CollectorUnavailable):
                feeds.collect_rss(src, WINDOW, settings)
            assert not mock.calls.last.request.url.path.endswith("rss.xml")


def test_rss_max_items_caps_records_and_page_budget(settings):
    src = _src("rss", f"{ALIGN}/rss.xml", fetch_full_text=True)
    with respx.mock(assert_all_called=False) as mock:
        routes = _alignment_routes(mock)
        recs = feeds.collect_rss(src, Window(since=WINDOW.since, until=NOW, max_items=1), settings)
    assert len(recs) == 1 and routes["article"].call_count == 0


# --------------------------------------------------------------------------------------- sitemap
ANTHROPIC_PAGES = {
    "research/provable-inference": "anthropic_provable_inference.html",
    "news/a-new-initiative-for-developing-third-party-model-evaluations": "anthropic_old_post.html",
    "news/expanding-partnership-gsa": "anthropic_gsa.html",
}


def _anthropic_routes(mock: respx.MockRouter) -> dict[str, respx.Route]:
    index = (b'<?xml version="1.0" encoding="UTF-8"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/'
             b'sitemap/0.9"><sitemap><loc>https://www.anthropic.com/sitemap-0.xml</loc></sitemap></sitemapindex>')
    mock.get(f"{ANTHROPIC}/robots.txt").respond(200, text="User-agent: *\nDisallow: /news/expanding-partnership-gsa\n")
    mock.get(f"{ANTHROPIC}/sitemap.xml").respond(200, content=index, headers=XML)
    mock.get(f"{ANTHROPIC}/sitemap-0.xml").respond(200, content=_fx("anthropic_sitemap.xml"), headers=XML)
    routes = {path: mock.get(f"{ANTHROPIC}/{path}").respond(200, content=_fx(name), headers=HTML)
              for path, name in ANTHROPIC_PAGES.items()}
    routes["failing"] = mock.get(f"{ANTHROPIC}/news/ai-for-science-expands-foundations-research-institutions").respond(503)
    routes["other"] = mock.get(url__regex=r"https://www\.anthropic\.com/(claude-corps|news/announcements).*").respond(200)
    return routes


def test_sitemap_follows_index_dates_from_page_and_respects_robots(settings, monkeypatch):
    monkeypatch.setattr(feeds, "RETRY_WAIT_S", 0)
    src = _src("sitemap", f"{ANTHROPIC}/sitemap.xml", include=[r"^https://www\.anthropic\.com/(news|research)/[^/]+$"],
               exclude=[r"/news/(announcements|features)$"])
    with respx.mock(assert_all_called=False) as mock:
        routes = _anthropic_routes(mock)
        recs = feeds.collect_sitemap(src, WINDOW, settings)
        again = feeds.collect_sitemap(src, WINDOW, settings)

    assert [r.url for r in recs] == [f"{ANTHROPIC}/research/provable-inference"]
    rec = recs[0]
    assert rec.published_at == datetime(2026, 11, 2)  # article:published_time, not lastmod (11-03)
    assert rec.meta["date_source"] == "meta" and rec.meta["lastmod"].startswith("2026-11-03")
    assert rec.natural_key == "https://anthropic.com/research/provable-inference"  # canonical, stable
    assert rec.title == "Progress on provable inference for large language models"
    assert len(rec.content_text) > 1000
    assert [r.natural_key for r in again] == [rec.natural_key]
    # a 2024 post whose lastmod was bumped by a CMS edit: fetched once, dropped, then cached
    assert routes["news/a-new-initiative-for-developing-third-party-model-evaluations"].call_count == 1
    assert routes["news/expanding-partnership-gsa"].call_count == 0  # robots.txt disallow
    assert routes["other"].call_count == 0  # not included / excluded / lastmod before the window
    assert routes["failing"].call_count == 6  # 3 attempts per run, never cached


def test_sitemap_lastmod_prefilter_skips_unchanged_pages(settings, monkeypatch):
    monkeypatch.setattr(feeds, "RETRY_WAIT_S", 0)
    src = _src("sitemap", f"{ANTHROPIC}/sitemap.xml", include=[r"/(news|research)/[^/]+$"])
    with respx.mock(assert_all_called=False) as mock:
        routes = _anthropic_routes(mock)
        recs = feeds.collect_sitemap(src, Window(since=datetime(2026, 10, 20), until=NOW), settings)
    assert [r.url for r in recs] == [f"{ANTHROPIC}/research/provable-inference"]
    assert routes["news/a-new-initiative-for-developing-third-party-model-evaluations"].call_count == 0


def test_sitemap_url_dates_and_page_budget(settings):
    base = "https://api-docs.deepseek.com"
    src = _src("sitemap", f"{base}/sitemap.xml", include=[r"/news/news\d{6}$"], max_fetch=1,
               url_date_regex=r"news(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{base}/robots.txt").respond(200, text="User-agent: *\nAllow: /\n")
        mock.get(f"{base}/sitemap.xml").respond(200, content=_fx("deepseek_sitemap.xml"), headers=XML)
        newest = mock.get(f"{base}/news/news261017").respond(200, content=_fx("deepseek_news261017.html"), headers=HTML)
        older = mock.get(url__regex=r".*/news/news2(608|509).*").respond(200, content=b"<html></html>", headers=HTML)
        recs = feeds.collect_sitemap(src, Window(since=WINDOW.since, until=NOW), settings)
    assert len(recs) == 1 and recs[0].published_at == datetime(2026, 10, 17)
    assert recs[0].meta["date_source"] == "url" and recs[0].title.startswith("DeepSeek-V4.1-Flash-1017")
    assert newest.call_count == 1 and older.call_count == 0  # 2025 URL pre-filtered, Aug URL over budget


def test_sitemap_index_pages_discover_posts_missing_from_sitemap(settings):
    base = "https://periodic.com"
    sitemap = (b'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
               b'<url><loc>https://periodic.com/</loc></url><url><loc>https://periodic.com/news</loc></url></urlset>')
    src = _src("sitemap", f"{base}/sitemap.xml", include=[r"^https://periodic\.com/news/[^/]+$"],
               index_pages=[f"{base}/news"], lastmod="ignore")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{base}/robots.txt").respond(200, text="User-agent: *\nAllow: /\n")
        mock.get(f"{base}/sitemap.xml").respond(200, content=sitemap, headers=XML)
        mock.get(f"{base}/news").respond(200, content=_fx("periodic_news_index.html"), headers=HTML)
        mock.get(f"{base}/news/building-labs-that-learn").respond(200, content=_fx("periodic_article.html"), headers=HTML)
        mock.get(url__regex=r"https://periodic\.com/news/.+").respond(404)
        recs = feeds.collect_sitemap(src, WINDOW, settings)
    assert [r.url for r in recs] == [f"{base}/news/building-labs-that-learn"]
    assert recs[0].published_at == datetime(2026, 9, 15) and recs[0].meta["date_source"] == "byline"


# -------------------------------------------------------------------------------- docs_changelog
def _changelog(settings, url: str, fixture: str, headers: dict, window: Window = WINDOW, **config):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__regex=r"https://[^/]+/robots\.txt").respond(404)
        mock.get(url).respond(200, content=_fx(fixture), headers=headers)
        return feeds.collect_docs_changelog(_src("docs_changelog", url, **config), window, settings)


def test_changelog_markdown_sections_window_and_keys(settings, fixed_now):
    url = "https://platform.claude.com/docs/en/release-notes/overview.md"
    recs = _changelog(settings, url, "claude_release_notes.md", MD, title="Claude release notes")
    page = "https://platform.claude.com/docs/en/release-notes/overview"
    assert [r.published_at.date() for r in recs] == [date(2026, 11, 3), date(2026, 11, 2), date(2026, 10, 30),
                                                     date(2026, 8, 1)]  # the May 2024 section is outside
    assert recs[0].natural_key == f"{page}#2026-11-03" and recs[0].url == f"{page}#november-3-2026"
    assert recs[0].title == "Claude release notes – 2026-11-03"
    assert "watermark" in recs[0].content_text and "](" not in recs[0].content_text  # markdown stripped
    again = _changelog(settings, url, "claude_release_notes.md", MD, title="Claude release notes")
    assert [r.natural_key for r in again] == [r.natural_key for r in recs]


def test_changelog_day_headings_take_year_from_month_headings(settings, fixed_now):
    url = "https://developers.openai.com/api/docs/changelog.md"
    recs = _changelog(settings, url, "openai_changelog.md", MD, window=Window(datetime(2025, 1, 1), NOW))
    assert [r.published_at.date() for r in recs] == [date(2026, 10, 30), date(2026, 10, 28), date(2026, 8, 4),
                                                     date(2025, 10, 30)]
    assert recs[0].content_text.startswith("Feature · Model: gpt-6-rosalind")
    assert "Released GPT-6 Rosalind, bringing" in recs[0].content_text  # link text kept inline


def test_changelog_month_only_page_gives_one_record_per_entry(settings, fixed_now):
    url = "https://platform.kimi.ai/docs/platform-changelog.md"
    recs = _changelog(settings, url, "kimi_changelog.md", MD, title="Kimi changelog")
    page = "https://platform.kimi.ai/docs/platform-changelog"
    assert [r.natural_key for r in recs] == [f"{page}#2026-10/batch-api-now-supports-kimi-k3-1-pro-and-kimi-k3",
                                             f"{page}#2026-09/web-search-api", f"{page}#2026-09/billing-method-change"]
    assert recs[0].published_at == datetime(2026, 10, 1) and recs[0].meta["date_precision"] == "month"
    assert recs[1].title == "Kimi changelog – September 2026: 🔍 Web Search API"


def test_changelog_html_date_badges_and_year_headings(settings, fixed_now):
    url = "https://docs.x.ai/developers/release-notes"
    recs = _changelog(settings, url, "xai_release_notes.html", HTML, window=Window(datetime(2024, 1, 1), NOW),
                      date_selector="div")
    days = [r.published_at.date() for r in recs]
    assert days == [date(2026, 11, 2), date(2026, 10, 27), date(2026, 10, 14), date(2026, 10, 2),
                    date(2025, 12, 16), date(2024, 11, 19), date(2024, 11, 4)]
    oct14 = recs[2]
    assert oct14.title.endswith("Grok Voice 3.0 is available; Grok Voice 2.0 deprecation")  # same day merged
    assert recs[-1].title.endswith("API Public Beta")


# ----------------------------------------------------------------------------------- end to end
def test_runner_persists_changelog_idempotently(db, fixed_now):
    from fli.db.models import Document, Source
    from fli.ingest.runner import run_ingest

    url = "https://platform.claude.com/docs/en/release-notes/overview.md"
    with db() as s:
        s.add(Source(id="anthropic:release-notes", adapter="docs_changelog", kind="docs_changelog",
                     source_class="official", url=url, config={"title": "Claude release notes"}))
        s.commit()
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://platform.claude.com/robots.txt").respond(404)
        mock.get(url).respond(200, content=_fx("claude_release_notes.md"), headers=MD)
        first = run_ingest(source_ids=["anthropic:release-notes"], since=datetime(2026, 8, 1))
        second = run_ingest(source_ids=["anthropic:release-notes"], since=datetime(2026, 8, 1))
    assert first["anthropic:release-notes"]["new"] == 4
    assert second["anthropic:release-notes"] == {"status": "ok", "fetched": 4, "new": 0, "duplicates": 0}
    with db() as s:
        assert s.query(Document).count() == 4


def test_unreachable_robots_means_unavailable_not_crash(settings, monkeypatch):
    monkeypatch.setattr(feeds, "RETRY_WAIT_S", 0)
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{ALIGN}/robots.txt").mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(CollectorUnavailable, match="robots.txt unreachable"):
            feeds.collect_rss(_src("rss", f"{ALIGN}/rss.xml"), WINDOW, settings)


def test_rss_paging_stops_past_last_page_or_when_page_param_is_ignored(settings):
    src = _src("rss", f"{ALIGN}/rss.xml", max_pages=5)
    window = Window(since=datetime(2025, 1, 1), until=NOW)  # every entry is in window: keep paging
    with respx.mock(assert_all_called=False) as mock:
        page2 = mock.get(f"{ALIGN}/rss.xml", params={"paged": "2"}).respond(404)  # WordPress past the end
        _alignment_routes(mock)
        recs = feeds.collect_rss(src, window, settings)
    assert len(recs) == 3 and page2.call_count == 1
    with respx.mock(assert_all_called=False) as mock:  # a server that ignores ?paged=N
        _alignment_routes(mock)
        recs = feeds.collect_rss(src, window, settings)
        assert len(recs) == 3 and len(mock.calls) == 3  # robots, page 1, page 2 (repeat) -> stop


def test_html_text_keeps_inline_markup_inline_and_drops_comments():
    html = "<div><!-- build\nnote --><p>Released <a href='x'>GPT-6\nRosalind</a>, bringing</p><pre>a\nb</pre></div>"
    assert feeds._html_to_text(html) == "Released GPT-6 Rosalind, bringing\n\na\nb"
