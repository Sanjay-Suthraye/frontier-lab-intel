"""Press collector: GDELT DOC 2.0 article lists (headlines only).

Press is a secondary source: it surfaces deals, hires and departures that labs do not announce
themselves. Only headlines are stored (no article bodies are fetched); headline triage downstream
decides what is worth extracting. GDELT asks for at most one request every 5 seconds, so requests
are serialised process-wide, whatever thread the runner uses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fli.config import Settings
from fli.ingest.records import CollectorUnavailable, IngestRecord, SourceInfo, Window, collector
from fli.textutil import canonical_url
from fli.timeutil import to_naive_utc

log = logging.getLogger("fli.ingest.press")

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_INTERVAL_S = 5.0
GDELT_LOOKBACK_DAYS = 89  # the DOC API only searches a rolling ~3-month window
GDELT_TIMEOUT_S = 90.0  # responses regularly take 15-40 s
_GDELT_LOCK = threading.Lock()
_last_request_end = [0.0]

# title aliases used when a query spans several labs (exactly one match -> attributed)
LAB_ALIASES: dict[str, tuple[str, ...]] = {
    "openai": ("openai", "chatgpt", "sam altman"),
    "anthropic": ("anthropic", "dario amodei"),
    "google-deepmind": ("deepmind", "demis hassabis"),
    "meta-ai": ("meta superintelligence", "meta ai", "superintelligence labs"),
    "xai": ("xai", "spacexai", "grok"),
    "mistral": ("mistral ai", "mistral"),
    "deepseek": ("deepseek",),
    "qwen": ("qwen",),
    "moonshot": ("moonshot ai", "kimi"),
    "thinking-machines": ("thinking machines", "mira murati"),
    "ssi": ("safe superintelligence", "ilya sutskever"),
    "reflection-ai": ("reflection ai",),
    "periodic-labs": ("periodic labs",),
    "discovery-loop": ("discovery loop",),
}
_ALIAS_RX = {lab: re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b") for lab, names in LAB_ALIASES.items()}


class GdeltQueryError(ValueError):
    """GDELT answered with a plain-text error (bad syntax, too short/long): a config problem."""


class GdeltThrottled(RuntimeError):
    """GDELT asked us to slow down; retried with backoff."""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError | GdeltThrottled)


def parse_gdelt(text: str) -> list[dict]:
    """Article list from a DOC API response (errors arrive as plain text with HTTP 200)."""
    body = (text or "").strip()
    if not body:
        return []
    if not body.startswith("{"):
        if "limit requests" in body.lower():
            raise GdeltThrottled(body[:200])
        raise GdeltQueryError(f"GDELT rejected the query: {body[:200]}")
    return json.loads(body, strict=False).get("articles") or []  # titles may hold raw control chars


@retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=5, min=5, max=60), reraise=True)
def gdelt_articles(client: httpx.Client, query: str, start: datetime, end: datetime, max_records: int = 250) -> list[dict]:
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": max_records,
              "sort": "hybridrel", "startdatetime": f"{start:%Y%m%d%H%M%S}", "enddatetime": f"{end:%Y%m%d%H%M%S}"}
    with _GDELT_LOCK:  # one request at a time, >= 5 s after the previous one finished
        wait = _last_request_end[0] + GDELT_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.get(GDELT_DOC_URL, params=params)
        finally:
            _last_request_end[0] = time.monotonic()
    resp.raise_for_status()
    return parse_gdelt(resp.text)


def clean_title(title: str | None) -> str:
    """Undo GDELT tokeniser spacing ("Live : Gift", "gap - up", "Anthropic ' s")."""
    t = " ".join((title or "").split())
    t = re.sub(r"\s+([,.;:?!%)\]])", r"\1", t)
    t = re.sub(r"([(\[$])\s+", r"\1", t)
    t = re.sub(r"\s*'\s*s\b", "'s", t)
    return re.sub(r"(?<=\w) - (?=[a-z])", "-", t)


def title_key(title: str) -> str:
    """Normalised headline for collapsing syndicated copies (outlet suffix dropped)."""
    t = clean_title(title).lower()
    t = re.sub(r"\s+[-|–—]\s+[^-|–—]{2,60}$", "", t)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def detect_lab(title: str) -> str | None:
    low = clean_title(title).lower()
    hits = [lab for lab, rx in _ALIAS_RX.items() if rx.search(low)]
    return hits[0] if len(hits) == 1 else None


def _domain_in(domain: str, patterns: list[str]) -> bool:
    d = domain.lower()
    return any(d == p.lower() or d.endswith("." + p.lower()) for p in patterns)


def _natural_key(url: str) -> str:
    canon = canonical_url(url)
    return canon if len(canon) <= 512 else "sha1:" + hashlib.sha1(canon.encode()).hexdigest()


def chunks(since: datetime, until: datetime, days: int) -> list[tuple[datetime, datetime]]:
    """Newest-first [start, end] spans covering the window."""
    out, end = [], until
    while end > since:
        start = max(since, end - timedelta(days=days))
        out.append((start, end))
        end = start
    return out


def articles_to_records(articles: list[dict], *, query: dict, seen: set[tuple[str, str]], cfg: dict,
                        since: datetime, until: datetime, limit: int) -> list[IngestRecord]:
    """Filter (English, domains, window), collapse syndicated copies by title+day, cap.
    seendate can trail the requested end by an hour or two, hence the tolerance."""
    out: list[IngestRecord] = []
    for a in articles:
        if len(out) >= limit:
            break
        title, url, domain = clean_title(a.get("title")), a.get("url") or "", (a.get("domain") or "").lower()
        published = to_naive_utc(a.get("seendate"))
        if not title or not url or published is None or not since <= published <= until + timedelta(hours=6):
            continue
        if cfg.get("english_only", True) and (a.get("language") or "English") != "English":
            continue
        if cfg.get("allow_domains") and not _domain_in(domain, cfg["allow_domains"]):
            continue
        if _domain_in(domain, cfg.get("deny_domains") or []):
            continue
        key = (title_key(title), f"{published:%Y-%m-%d}")
        if key in seen:
            continue
        seen.add(key)
        out.append(IngestRecord(
            natural_key=_natural_key(url), url=url, title=title, content_text=title,
            published_at=published, lab_id=query.get("lab") or detect_lab(title), source_class="press",
            meta={"domain": domain, "language": a.get("language"), "sourcecountry": a.get("sourcecountry"),
                  "query": query["q"], "seendate": a.get("seendate")},
        ))
    return out


def validate_query(query: str) -> None:
    """Catch GDELT's commonest rejection locally: a dash outside quotes ("S-1" must be quoted)."""
    unquoted = re.sub(r'"[^"]*"', " ", query)
    if re.search(r"\w-\w", unquoted):
        raise GdeltQueryError(f"dashed keyword must be quoted for GDELT: {query[:120]}")


def _queries(source: SourceInfo) -> list[dict]:
    raw: list[Any] = source.config.get("queries") or ([source.config["query"]] if source.config.get("query") else [])
    out = []
    for q in raw:
        spec = {"q": q} if isinstance(q, str) else dict(q)
        spec.setdefault("lab", source.lab_id)
        if "sourcelang:" not in spec["q"].lower() and source.config.get("english_only", True):
            spec["q"] = f"{spec['q']} sourcelang:english"
        validate_query(spec["q"])
        out.append(spec)
    if not out:
        raise ValueError(f"{source.id}: config.queries is required")
    return out


@collector("gdelt_news", description="GDELT DOC 2.0 headlines for lab event queries (press)")
def gdelt_news(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    cfg = source.config
    since = max(window.since, window.until - timedelta(days=GDELT_LOOKBACK_DAYS))
    per_query = int(cfg.get("max_per_query") or 150)
    per_chunk = int(cfg.get("max_per_chunk") or 40)
    if window.max_items:
        per_query = min(per_query, window.max_items)
    records: list[IngestRecord] = []
    seen: set[tuple[str, str]] = set()
    with httpx.Client(timeout=GDELT_TIMEOUT_S, headers={"User-Agent": settings.user_agent}) as client:
        for query in _queries(source):
            got: list[IngestRecord] = []
            for start, end in chunks(since, window.until, int(cfg.get("chunk_days") or 14)):
                if len(got) >= per_query:
                    break
                try:
                    arts = gdelt_articles(client, query["q"], start, end)
                except (GdeltThrottled, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code != 429:
                        raise
                    # all-or-nothing: a partial run would advance last_success and lose backfill chunks
                    raise CollectorUnavailable(f"GDELT throttled (HTTP 429) after retries: {exc}"[:300]) from exc
                got += articles_to_records(arts, query=query, seen=seen, cfg=cfg, since=start, until=end,
                                           limit=min(per_chunk, per_query - len(got)))
            log.info("gdelt %s: %d headlines for %s", source.id, len(got), query["q"][:80])
            records += got
    unique = {r.natural_key: r for r in records}  # the same URL can match two queries
    return list(unique.values())
