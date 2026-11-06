"""Official lab channels: RSS/Atom feeds, sitemaps and single-page changelogs.

Collectors (the `adapter` value in config/sources_feeds.yaml):

  rss             RSS/Atom feed. Summary-only entries can be completed with the article's main text
                  (`fetch_full_text`); WordPress feeds can be paged back for a backfill (`max_pages`).
  sitemap         sitemap.xml (nested indexes followed, optional `index_pages`) for official sites
                  without a feed. lastmod only pre-filters: the published date comes from the page.
  docs_changelog  one release-notes page (markdown preferred, HTML fallback) split at headings that
                  are dates; one record per date, or per entry on pages dated by month only.

Politeness lives in `_Fetcher`: a descriptive User-Agent, robots.txt (including a
`Content-Signal: ai-input=no` opt-out) checked before every fetch, a per-run page budget
(`max_fetch`; `backfill_max_fetch` when the window is longer than 14 days) and a delay between page
fetches. Extracted articles are cached per source under `<raw_dir>/_cache/feeds/`, so the runner's
2-day overlap and CMS lastmod churn do not trigger re-downloads; deleting the cache is always safe.

Dates, URLs and authors come from feed or page metadata, never from an LLM. Datetimes are naive UTC.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import re
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Self
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import feedparser
import httpx
import markdown
import trafilatura
from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree

from fli.config import Settings
from fli.ingest.records import CollectorUnavailable, IngestRecord, SourceInfo, Window, collector
from fli.textutil import canonical_url, clean_text, slugify
from fli.timeutil import to_naive_utc, utcnow

log = logging.getLogger("fli.feeds")

BACKFILL_SPAN = timedelta(days=14)  # a window longer than this is a backfill (bigger page budget)
FUTURE_SLACK = timedelta(days=2)  # feeds in far-east timezones post "tomorrow"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
RETRY_WAIT_S = 2.0
CACHE_VERSION = 1  # bump when extraction changes so cached articles are re-extracted
CACHE_MAX = 3000
HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
HEADING_SELECTOR = "h1, h2, h3, h4, h5, h6"


# ------------------------------------------------------------------------------------- HTTP
class _Fetcher:
    """One run's HTTP access: shared client, robots.txt per host, page budget, politeness delay."""

    def __init__(self, settings: Settings, cfg: dict[str, Any], window: Window) -> None:
        self.ua = settings.user_agent
        self.client = httpx.Client(headers={"User-Agent": self.ua}, timeout=settings.http_timeout_s,
                                   follow_redirects=True)
        budget = int(cfg.get("max_fetch", 25))
        if window.until - window.since > BACKFILL_SPAN:
            budget = int(cfg.get("backfill_max_fetch", budget))
        self.budget = min(budget, window.max_items) if window.max_items else budget
        self.delay = float(cfg.get("delay_s", 1.0))
        self.pages = 0
        self._robots: dict[str, tuple[RobotFileParser, float, str]] = {}
        self._last = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.close()

    def get(self, url: str) -> httpx.Response:
        """GET with two retries on transient failures (network errors, 429, 5xx)."""
        for attempt in range(3):
            try:
                resp = self.client.get(url)
                if resp.status_code not in RETRY_STATUS or attempt == 2:
                    return resp
            except httpx.TransportError:
                if attempt == 2:
                    raise
            time.sleep(RETRY_WAIT_S * 2**attempt)
        raise AssertionError("unreachable")

    def _host(self, url: str) -> tuple[RobotFileParser, float, str]:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            self._robots[base] = self._load_robots(base)
        return self._robots[base]

    def _load_robots(self, base: str) -> tuple[RobotFileParser, float, str]:
        """Parse robots.txt like urllib does (401/403/5xx: deny, other 4xx: allow) plus Content-Signal."""
        parser = RobotFileParser(base + "/robots.txt")
        try:
            resp = self.get(base + "/robots.txt")
        except httpx.HTTPError as exc:
            parser.disallow_all = True
            return parser, self.delay, f"robots.txt unreachable ({exc.__class__.__name__})"
        if resp.status_code in (401, 403) or resp.status_code >= 500:
            parser.disallow_all = True
            return parser, self.delay, f"robots.txt answered HTTP {resp.status_code}"
        if resp.status_code >= 400:
            parser.allow_all = True
            return parser, self.delay, ""
        parser.parse(resp.text.splitlines())
        if re.search(r"(?im)^\s*content-signal:[^\n]*\bai-input\s*=\s*no\b", resp.text):
            parser.disallow_all = True
            return parser, self.delay, "Content-Signal ai-input=no"
        return parser, max(self.delay, float(parser.crawl_delay(self.ua) or 0)), "robots.txt disallows"

    def allowed(self, url: str) -> bool:
        return self._host(url)[0].can_fetch(self.ua, url)

    def document(self, url: str) -> httpx.Response:
        """Fetch the source's own feed/sitemap/page; refusal by the site is 'unavailable', not failure."""
        if not self.allowed(url):
            raise CollectorUnavailable(f"{self._host(url)[2]}: {url}")
        resp = self.get(url)
        resp.raise_for_status()
        return resp

    def page(self, url: str) -> str | None:
        """Fetch one article page within budget and robots rules; None when skipped or failed."""
        if self.budget <= 0:
            return None
        if not self.allowed(url):
            log.info("%s: %s", self._host(url)[2], url)
            return None
        wait = self._host(url)[1] - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self.budget -= 1
        self.pages += 1
        try:
            resp = self.get(url)
        except httpx.HTTPError as exc:
            log.warning("fetch failed %s: %s", url, exc)
            return None
        finally:
            self._last = time.monotonic()
        if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "text/html"):
            log.info("fetch %s -> HTTP %s %s", url, resp.status_code, resp.headers.get("content-type"))
            return None
        return resp.text


class _PageCache:
    """Extracted articles of one source, keyed by URL and a version (sitemap lastmod / entry date)."""

    def __init__(self, settings: Settings, source_id: str) -> None:
        self.path = Path(settings.raw_dir) / "_cache" / "feeds" / f"{slugify(source_id)}.json.gz"
        self.dirty = False
        try:
            with gzip.open(self.path, "rt", encoding="utf-8") as fh:
                self.items: dict[str, dict[str, Any]] = json.load(fh)
        except (OSError, EOFError, ValueError):
            self.items = {}

    def get(self, url: str, version: str) -> dict[str, Any] | None:
        hit = self.items.get(url)
        return hit if hit and hit.get("v") == version and hit.get("cv") == CACHE_VERSION else None

    def put(self, url: str, version: str, value: dict[str, Any]) -> dict[str, Any]:
        self.items[url] = {**value, "v": version, "cv": CACHE_VERSION, "at": utcnow().isoformat()}
        self.dirty = True
        return self.items[url]

    def save(self) -> None:
        if not self.dirty:
            return
        keep = sorted(self.items.items(), key=lambda kv: kv[1].get("at", ""), reverse=True)[:CACHE_MAX]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(dict(keep), fh)
            os.replace(tmp, self.path)
        except OSError as exc:  # the cache only saves requests
            log.warning("page cache not saved (%s): %s", self.path, exc)


# ------------------------------------------------------------------------------------ dates
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = (r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
        r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_LABELS = [re.compile(p, re.IGNORECASE) for p in (
    r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})",
    _MON + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{4}))?",
    r"(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r",?(?:\s+(?P<y>\d{4}))?",
    _MON + r",?\s+(?P<y>\d{4})",
    _MON,
    r"(?P<y>(?:19|20)\d{2})",
)]
_LABEL_NOISE = re.compile(r"\{%.*?%\}|[*_#\[\]()|`]")
_LABEL_PREFIX = re.compile(r"^(?:date|released?|updated?|published|week of)\s*[:\-–]?\s*", re.IGNORECASE)
_URL_DATES = (re.compile(r"/(?P<y>20\d\d)/(?P<m>\d\d?)/(?P<d>\d\d?)(?:/|$)"),
              re.compile(r"(?P<y>20\d\d)-(?P<m>\d\d)-(?P<d>\d\d)"))


@dataclass(frozen=True)
class _Label:
    """A heading that is only a date; month/day are None for "2026" / "October 2026" headings."""

    year: int | None
    month: int | None
    day: int | None


def _date_label(text: str) -> _Label | None:
    """Parse "Oct 30", "November 3, 2026", "Date: 2026-10-17", "October, 2026", "2025" (else None)."""
    t = _LABEL_NOISE.sub(" ", text)
    t = "".join(" " if unicodedata.category(ch)[0] in "CS" else ch for ch in t)  # icons, emoji, ZWSP
    t = _LABEL_PREFIX.sub("", " ".join(t.split())).strip(" .,:;-–—")
    if not t or len(t) > 32:
        return None
    for rx in _LABELS:
        if (m := rx.fullmatch(t)) is None:
            continue
        g = m.groupdict()
        month = int(g["m"]) if g.get("m") else (_MONTHS[g["mon"][:3].lower()] if g.get("mon") else None)
        label = _Label(int(g["y"]) if g.get("y") else None, month, int(g["d"]) if g.get("d") else None)
        try:
            date(label.year or 2000, label.month or 1, label.day or 1)
        except ValueError:
            return None
        return label
    return None


def _resolve(label: _Label, ctx_year: int | None, now: datetime) -> date | None:
    """Calendar date of a label. A missing year comes from the enclosing month/year heading or,
    failing that, is the most recent year that does not put the date in the future."""
    month, day = label.month or 1, label.day or 1
    year = label.year or ctx_year
    if year is None:
        year = now.year if (month, day) <= (now.month, now.day + 2) else now.year - 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _url_date(url: str, pattern: str | None = None) -> datetime | None:
    """Date encoded in a URL: a configured regex with y/m/d groups, /YYYY/MM/DD/ or YYYY-MM-DD."""
    for rx in ([re.compile(pattern)] if pattern else []) + list(_URL_DATES):
        if m := rx.search(url):
            y, mo, d = (int(m.group(k)) for k in ("y", "m", "d"))
            try:
                return datetime(y + 2000 if y < 100 else y, mo, d)
            except ValueError:
                continue
    return None


# ------------------------------------------------------------------------------------- text
_BLOCKS = ("p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "pre",
           "blockquote", "section", "article", "header", "footer", "aside", "dt", "dd", "figcaption",
           "br", "hr")


def _raw_text(node: Tag) -> str:
    """Text with newlines at block boundaries and inline markup kept inline, so an LLM quote of a
    sentence containing a link or code span still matches the stored text verbatim."""
    for junk in node.find_all(["script", "style", "template", "svg"]):
        junk.decompose()
    for s in node.find_all(string=lambda s: "\n" in s):  # HTML whitespace: soft line wraps are spaces
        if type(s) is NavigableString and s.find_parent("pre") is None:  # not comments/CDATA
            s.replace_with(s.replace("\n", " "))
    for tag in node.find_all(_BLOCKS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for cell in node.find_all(["td", "th"]):
        cell.insert_after(" ")
    return node.get_text("")


def _html_to_text(html: str | None) -> str:
    if not html:
        return ""
    if "<" not in html:
        return clean_text(html)
    soup = BeautifulSoup(html, "lxml")
    return clean_text(_raw_text(soup.body or soup))


def _plain(value: str | None) -> str | None:
    text = _html_to_text(value).replace("\n", " ").strip()
    return text or None


def _patterns(values: list[str] | str | None) -> list[re.Pattern[str]]:
    return [re.compile(v) for v in ([values] if isinstance(values, str) else values or [])]


def _wanted(url: str, include: list[re.Pattern[str]], exclude: list[re.Pattern[str]]) -> bool:
    return (not include or any(r.search(url) for r in include)) and not any(r.search(url) for r in exclude)


def _key(value: str) -> str:
    """Natural keys must fit the 512-char column; very long ids are hashed (still stable)."""
    return value if len(value) <= 512 else "sha1:" + hashlib.sha1(value.encode()).hexdigest()


# ---------------------------------------------------------------------------------- articles
_META_DATES = ("article:published_time", "og:published_time", "datepublished", "citation_publication_date",
               "citation_date", "dc.date.issued", "dcterms.issued", "parsely-pub-date", "publish-date",
               "publishdate", "pubdate", "date")


def _meta_date(soup: BeautifulSoup) -> datetime | None:
    """Publication date from meta tags or JSON-LD (never the modified date)."""
    found: dict[str, str] = {}
    for m in soup.find_all("meta"):
        key = str(m.get("property") or m.get("name") or m.get("itemprop") or "").lower()
        if key in _META_DATES and m.get("content"):
            found.setdefault(key, str(m["content"]))
    for key in _META_DATES:
        if key in found and (dt := to_naive_utc(found[key])):
            return dt
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            if dt := _jsonld_date(json.loads(script.string or "")):
                return dt
        except ValueError:
            continue
    return None


def _jsonld_date(node: Any) -> datetime | None:
    if isinstance(node, list):
        return next((d for d in map(_jsonld_date, node) if d), None)
    if isinstance(node, dict):
        if isinstance(node.get("datePublished"), str):
            return to_naive_utc(node["datePublished"])
        return next((d for d in map(_jsonld_date, node.values()) if d), None)
    return None


def _byline_date(soup: BeautifulSoup) -> datetime | None:
    """A short string near the top of the article that is only a full date ("October 5, 2026"),
    else the article's first <time datetime>."""
    root = soup.find("article") or soup.find("main") or soup.body
    for i, s in enumerate(root.stripped_strings if root else []):
        if i > 80:
            break
        if len(s) <= 32 and (lab := _date_label(s)) and lab.day and lab.year:
            return datetime(lab.year, lab.month or 1, lab.day)
    stamp = root.find("time", datetime=True) if root else None
    return to_naive_utc(str(stamp["datetime"])) if stamp else None


def _page_title(soup: BeautifulSoup) -> str | None:
    """The article's h1 (no " | Site" suffix), else og:title, else <title>."""
    h1 = (soup.find("article") or soup.find("main") or soup).find("h1") or soup.find("h1")
    og = soup.find("meta", property="og:title")
    for cand in (h1.get_text(" ", strip=True) if h1 else None, og.get("content") if og else None,
                 soup.title.string if soup.title else None):
        if isinstance(cand, str) and 3 <= len(cand.strip()) <= 300:
            return " ".join(cand.split())
    return None


def _extract(html: str, url: str, url_date_pattern: str | None = None) -> dict[str, Any]:
    """Title, main text, author and publication date (with its provenance) of an article page."""
    soup = BeautifulSoup(html, "lxml")
    doc = trafilatura.bare_extraction(html, url=url, with_metadata=True, include_comments=False,
                                      include_tables=True, favor_recall=True, as_dict=True) or {}
    # trafilatura's own date guess comes last: it was wrong on 3 of 8 sampled lab pages
    found = (("meta", _meta_date(soup)), ("url", _url_date(url, url_date_pattern)),
             ("byline", _byline_date(soup)), ("trafilatura", to_naive_utc(doc.get("date"))))
    how, published = next(((k, v) for k, v in found if v), (None, None))
    return {"title": _page_title(soup) or doc.get("title"), "text": clean_text(doc.get("text") or ""),
            "author": doc.get("author") or None, "date": published.isoformat() if published else None,
            "date_source": how}


def _article(http: _Fetcher, cache: _PageCache, url: str, version: str, floor: datetime | None = None,
             url_date_pattern: str | None = None) -> dict[str, Any] | None:
    """Cached extraction of one page. With `floor` (sitemap discovery, where the page is the only
    date authority), pages dated before the tracked window are cached without their text."""
    if (hit := cache.get(url, version)) is not None:
        return hit
    if (html := http.page(url)) is None:
        return None
    art = _extract(html, url, url_date_pattern)
    if floor and art["date"] and datetime.fromisoformat(art["date"]) < floor:
        art["text"] = ""
    return cache.put(url, version, art)


def _floor(settings: Settings) -> datetime:
    return datetime.combine(settings.backfill_start, datetime.min.time())


# --------------------------------------------------------------------------------------- rss
def _entry_date(entry: Any) -> datetime | None:
    t = entry.get("published_parsed") or entry.get("updated_parsed")  # feedparser normalises to UTC
    return datetime(*t[:6]) if t else None


def _entry_url(entry: Any) -> str:
    url = entry.get("link") or entry.get("id") or ""
    return url if url.startswith("http") else ""


def _entry_text(entry: Any) -> str:
    """Longest of content:encoded / summary, as plain text."""
    best = max(entry.get("content") or [], key=lambda c: len(c.get("value", "")), default=None)
    if best is not None and len(best.get("value", "")) >= len(entry.get("summary", "")):
        value, kind = best.get("value", ""), best.get("type", "text/html")
    else:
        value, kind = entry.get("summary", ""), entry.get("summary_detail", {}).get("type", "text/html")
    return clean_text(value) if kind == "text/plain" else _html_to_text(value)


def _feed_entries(http: _Fetcher, url: str, cfg: dict[str, Any], since: datetime) -> tuple[list, str | None]:
    """Entries of the feed; with max_pages > 1, older pages (WordPress ?paged=N) until `since`."""
    entries: list[Any] = []
    seen: set[str] = set()
    title = None
    for page in range(1, int(cfg.get("max_pages", 1)) + 1):
        sep = "&" if "?" in url else "?"
        page_url = url if page == 1 else f"{url}{sep}{cfg.get('page_param', 'paged')}={page}"
        try:
            resp = http.document(page_url)
        except httpx.HTTPStatusError as exc:
            if page > 1 and exc.response.status_code == 404:
                break  # past the last WordPress page
            raise
        parsed = feedparser.parse(io.BytesIO(resp.content),
                                  response_headers={"content-type": resp.headers.get("content-type", "")})
        if not parsed.entries and page == 1 and parsed.bozo:
            raise ValueError(f"unparseable feed {url}: {parsed.get('bozo_exception')}")
        title = title or parsed.feed.get("title")
        new = [e for e in parsed.entries if (e.get("id") or e.get("link")) not in seen]
        seen.update(e.get("id") or e.get("link") for e in new)
        entries.extend(new)
        dates = [d for d in map(_entry_date, new) if d]
        if not new or not dates or min(dates) < since:
            break
    return entries, title


def _entry_record(entry: Any, published: datetime, feed_title: str | None, cfg: dict[str, Any],
                  http: _Fetcher, cache: _PageCache) -> IngestRecord:
    """The feed is the authority for date, URL and title; the article page only adds text."""
    url = _entry_url(entry)
    text = _entry_text(entry)
    tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
    meta: dict[str, Any] = {"feed": feed_title, "tags": tags, "full_text": False}
    author = entry.get("author") or None
    if cfg.get("fetch_full_text") and len(text) < int(cfg.get("min_chars", 600)):
        version = entry.get("updated") or entry.get("published") or ""
        art = _article(http, cache, url, version)
        if art and len(art["text"]) > len(text):
            meta.update(full_text=True, summary=text[:1000])
            text, author = art["text"], author or art.get("author")
    return IngestRecord(natural_key=_key(entry.get("id") or url), url=url, title=_plain(entry.get("title")),
                        content_text=text, published_at=published, author=author, meta=meta)


@collector("rss", description="RSS/Atom feed of an official channel, optionally with full article text")
def collect_rss(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    """config: fetch_full_text, min_chars (600), max_fetch (25), backfill_max_fetch, max_pages (1),
    page_param ("paged"), include / exclude (URL regexes), delay_s (1.0)."""
    cfg = source.config
    include, exclude = _patterns(cfg.get("include")), _patterns(cfg.get("exclude"))
    with _Fetcher(settings, cfg, window) as http:
        entries, feed_title = _feed_entries(http, source.url or "", cfg, window.since)
        dated = sorted(((d, e) for e in entries if (d := _entry_date(e)) and _entry_url(e)
                        and window.since <= d <= window.until + FUTURE_SLACK
                        and _wanted(_entry_url(e), include, exclude)), key=lambda de: de[0], reverse=True)
        cache = _PageCache(settings, source.id)
        records = [_entry_record(e, d, feed_title, cfg, http, cache) for d, e in dated[: window.max_items or None]]
        cache.save()
    log.info("%s: %d feed entries, %d in window, %d pages fetched", source.id, len(entries), len(records),
             http.pages)
    return records


# ----------------------------------------------------------------------------------- sitemap
def _xml(body: bytes) -> etree._Element:
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(body, parser=parser)
    if root is None:
        raise ValueError("empty or unparseable sitemap")
    return root


def _sitemap_urls(http: _Fetcher, url: str, since: datetime | None, cfg: dict[str, Any],
                  depth: int = 0) -> Iterator[tuple[str, str]]:
    """(loc, raw lastmod) for every page; sitemap indexes are followed up to two levels deep,
    skipping children whose own (trusted) lastmod predates the window."""
    root = _xml(http.document(url).content)
    if etree.QName(root).localname == "sitemapindex":
        sub = _patterns(cfg.get("sitemap_include"))
        for sm in root.findall("{*}sitemap")[:50]:
            loc, lastmod = (sm.findtext("{*}loc") or "").strip(), to_naive_utc(sm.findtext("{*}lastmod"))
            if loc and depth < 2 and _wanted(loc, sub, []) and not (since and lastmod and lastmod < since):
                yield from _sitemap_urls(http, loc, since, cfg, depth + 1)
        return
    for node in root.findall("{*}url"):
        if loc := (node.findtext("{*}loc") or "").strip():
            yield loc, (node.findtext("{*}lastmod") or "").strip()


def _index_links(http: _Fetcher, page_url: str) -> list[str]:
    """Same-host links of an official index page (for sites whose sitemap omits their posts)."""
    html = http.page(page_url)
    if html is None:
        return []
    host = urlsplit(page_url).netloc
    links = (urljoin(page_url, a["href"]).split("#")[0] for a in BeautifulSoup(html, "lxml").find_all("a", href=True))
    return list(dict.fromkeys(u for u in links if urlsplit(u).netloc == host))


@collector("sitemap", description="Official site without a feed: sitemap.xml discovery + article extraction")
def collect_sitemap(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    """config: include / exclude (URL regexes), sitemaps (default: the source url), sitemap_include,
    index_pages, lastmod ("trust" | "ignore" when a CMS rewrites it on every build), url_date_regex
    (named groups y, m, d), max_fetch (25), backfill_max_fetch, delay_s (1.0)."""
    cfg = source.config
    include, exclude = _patterns(cfg.get("include")), _patterns(cfg.get("exclude"))
    trust = cfg.get("lastmod", "trust") != "ignore"
    pattern = cfg.get("url_date_regex")
    with _Fetcher(settings, cfg, window) as http:
        found: dict[str, str] = {}
        for sm in cfg.get("sitemaps") or [source.url]:
            for loc, lastmod in _sitemap_urls(http, sm, window.since if trust else None, cfg):
                found.setdefault(loc, lastmod)
        for page in cfg.get("index_pages") or []:
            for loc in _index_links(http, page):
                found.setdefault(loc, "")
        candidates = []
        for loc, raw_lastmod in found.items():
            lastmod = to_naive_utc(raw_lastmod) if trust else None
            hint = _url_date(loc, pattern) or lastmod  # lastmod >= publication, so it can only exclude
            if _wanted(loc, include, exclude) and not (hint and hint < window.since - timedelta(days=1)):
                candidates.append((hint or datetime.min, loc, raw_lastmod if trust else "", lastmod))
        candidates.sort(key=lambda c: c[0], reverse=True)  # most recently changed first
        cache = _PageCache(settings, source.id)
        records = []
        for _, loc, version, lastmod in candidates:
            art = _article(http, cache, loc, version, _floor(settings), pattern)
            if art is None:
                continue
            published = to_naive_utc(art["date"]) or lastmod
            if published is None or not art["text"] or not window.since <= published <= window.until + FUTURE_SLACK:
                continue
            records.append(IngestRecord(
                natural_key=_key(canonical_url(loc)), url=loc, title=art["title"], content_text=art["text"],
                published_at=published, author=art.get("author"),
                meta={"lastmod": lastmod.isoformat() if lastmod else None,
                      "date_source": art["date_source"] if art["date"] else "lastmod"}))
        cache.save()
    records.sort(key=lambda r: r.published_at or datetime.min, reverse=True)
    log.info("%s: %d urls, %d candidates, %d pages fetched, %d in window", source.id, len(found),
             len(candidates), http.pages, len(records))
    return records[: window.max_items or None]


# ---------------------------------------------------------------------------- docs_changelog
_SENTINEL = re.compile(r"\x00([DH])(\d+)\x00")


def _sections(root: Tag, date_selector: str | None, now: datetime) -> list[dict[str, Any]]:
    """Dated sections of a changelog page, in page order.

    Headings, bold-only paragraphs and elements matching `date_selector` (e.g. "div" date badges)
    whose whole text is a date become markers; they are swapped for sentinels so one pass over the
    text splits the page. Month/year markers supply the year to later markers without one. Other
    headings stay inside the section as titled blocks (entries on month-only pages) unless they are
    at or above the marker's own heading level, which ends the section.
    """
    labels: list[tuple[_Label, str, int]] = []  # label, anchor, heading level (0 = not a heading)
    heads: list[tuple[str, int]] = []
    selector = HEADING_SELECTOR + (f", {date_selector}" if date_selector else "")
    bold = [s.parent for s in root.select("p > strong") if s.parent.get_text(strip=True) == s.get_text(strip=True)]
    for el in reversed(root.select(selector) + bold):  # innermost first, so nested matches resolve once
        text = clean_text(el.get_text(" ", strip=True))
        if el.parent is None or not text or "\x00" in text:
            continue
        level = int(el.name[1]) if el.name in HEADINGS else 0
        if len(text) <= 64 and (label := _date_label(text)):
            anchor = el.get("id") or (m.group(1) if (m := re.search(r'id="([^"]+)"', text)) else slugify(text))
            labels.append((label, str(anchor), level))
            el.replace_with(f"\n\x00D{len(labels) - 1}\x00\n")
        elif level:
            heads.append((text, level))
            el.replace_with(f"\n\x00H{len(heads) - 1}\x00\n")
    parts = _SENTINEL.split(_raw_text(root))
    sections: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    ctx_year: int | None = None
    for i in range(1, len(parts), 3):
        kind, idx, after = parts[i], int(parts[i + 1]), parts[i + 2]
        if kind == "H":
            text, level = heads[idx]
            if cur is not None and cur["level"] and level <= cur["level"]:
                cur = None  # a sibling/parent heading that is not a date ends the dated section
            elif cur is not None:
                cur["blocks"].append((text, after))
            continue
        label, anchor, level = labels[idx]
        if label.day is None and label.year:
            ctx_year = label.year
        day = _resolve(label, ctx_year, now) if label.month else None
        cur = None if day is None else {"day": day, "precision": "day" if label.day else "month",
                                        "anchor": anchor, "level": level, "blocks": [(None, after)]}
        if cur is not None:
            sections.append(cur)
    return sections


def _section_text(blocks: list[tuple[str | None, str]]) -> str:
    return clean_text("\n\n".join(f"{h}\n{t}" if h else t for h, t in blocks))


def _changelog_records(sections: list[dict[str, Any]], link: str, page_title: str,
                       window: Window) -> list[IngestRecord]:
    """One record per date (sections sharing a date are merged, e.g. duplicated date badges or one
    list per product); month-only pages give one record per entry heading."""
    groups: dict[str, dict[str, Any]] = {}
    has_days = any(s["precision"] == "day" for s in sections)
    lo, hi = window.since.date(), (window.until + FUTURE_SLACK).date()
    for s in sections:
        day: date = s["day"]
        if s["precision"] == "day":
            items = [(day.isoformat(), s["anchor"], day.isoformat(), s["blocks"])] if lo <= day <= hi else []
        else:
            month_end = (day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if has_days or month_end < lo or day > hi:
                continue  # on day-level pages month headings only carry the year
            entries = [(h, t) for h, t in s["blocks"] if h]
            items = [(f"{day:%Y-%m}/{slugify(h)[:96]}", slugify(h), f"{day:%B %Y}", [(h, t)]) for h, t in entries]
            items = items or [(f"{day:%Y-%m}", s["anchor"], f"{day:%B %Y}", s["blocks"])]
        for frag, anchor, label, blocks in items:
            group = groups.setdefault(frag, {"anchor": anchor, "label": label, "section": s, "blocks": []})
            group["blocks"].extend(blocks)
    records = []
    for frag, g in groups.items():
        if not (text := _section_text(g["blocks"])):
            continue
        heads = "; ".join(h for h, _ in g["blocks"] if h)
        day, precision = g["section"]["day"], g["section"]["precision"]
        records.append(IngestRecord(
            natural_key=_key(f"{link}#{frag}"), url=f"{link}#{g['anchor']}",
            title=f"{page_title} – {g['label']}{': ' + heads if heads else ''}"[:300], content_text=text,
            published_at=datetime(day.year, day.month, day.day),
            meta={"page": link, "section_date": day.isoformat(), "date_precision": precision}))
    return sorted(records, key=lambda r: r.published_at or datetime.min, reverse=True)


@collector("docs_changelog", description="Single release-notes page split into dated sections")
def collect_docs_changelog(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    """config: title (page name used in record titles), link_url (human page cited; default: the url
    without a .md suffix), format ("markdown" | "html"; default from content type / .md suffix),
    container (CSS, html only; default "main"), date_selector (CSS for date markers that are not
    headings, e.g. "div" date badges)."""
    cfg, url = source.config, source.url or ""
    link = cfg.get("link_url") or url.removesuffix(".md")
    with _Fetcher(settings, cfg, window) as http:
        resp = http.document(url)
    ctype = resp.headers.get("content-type", "")
    fmt = cfg.get("format") or ("markdown" if url.endswith(".md") or "markdown" in ctype
                                or ctype.startswith("text/plain") else "html")
    if fmt == "markdown":
        soup = BeautifulSoup(markdown.markdown(resp.text, extensions=["tables", "fenced_code"]), "lxml")
        root: Tag = soup.body or soup
    else:
        soup = BeautifulSoup(resp.text, "lxml")
        root = soup.select_one(cfg.get("container", "main")) or soup.body or soup
    h1 = root.find("h1")
    page_title = cfg.get("title") or (h1.get_text(" ", strip=True) if h1 else None) or link
    sections = _sections(root, cfg.get("date_selector"), utcnow())
    records = _changelog_records(sections, link, page_title, window)
    log.info("%s: %d dated sections, %d in window", source.id, len(sections), len(records))
    return records[: window.max_items or None]
