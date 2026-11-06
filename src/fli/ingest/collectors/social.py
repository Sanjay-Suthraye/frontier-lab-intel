"""Social collectors: X (metered, budget-guarded) and Bluesky (free public AppView).

X is pay-per-use ($0.005 per post read, $0.010 per user lookup), so this module treats it like
money: every request first *reserves* its worst-case cost in the `api_usage` ledger (refused with
`CollectorUnavailable` when spent + worst case would exceed X_BUDGET_TOTAL_USD) and the reservation
is settled to the actual number of resources returned afterwards. Reserving before the call keeps
the cap hard even when several X sources run in parallel threads. Handles are resolved to numeric
user ids once and cached on the Identity row, and incremental runs read with `since_id` so posts
already stored are never paid for twice.
"""

from __future__ import annotations

import functools
import logging
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import httpx
import tweepy
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fli.config import Settings
from fli.db.models import ApiUsage, Document, Identity
from fli.db.session import session_scope
from fli.ingest.records import CollectorUnavailable, IngestRecord, SourceInfo, Window, collector
from fli.timeutil import to_naive_utc, utcnow

log = logging.getLogger("fli.ingest.social")
# tweepy logs full request headers - including the bearer token - at DEBUG level.
logging.getLogger("tweepy").setLevel(logging.WARNING)

X_PROVIDER = "x"
X_COST_PER_POST = 0.005
X_COST_PER_USER_LOOKUP = 0.010
X_TIMELINE_ENDPOINT = "GET /2/users/:id/tweets"
X_LOOKUP_ENDPOINT = "GET /2/users/by/username/:username"
X_TWEET_FIELDS = ["created_at", "public_metrics", "entities", "referenced_tweets", "note_tweet"]
X_PAGE_MIN, X_PAGE_MAX = 5, 100  # API limits for max_results on the user timeline
DEFAULT_MAX_POSTS = 60
X_ERROR_REASONS = {
    401: "unauthorized: bearer token invalid or revoked",
    402: "payment required: pay-per-use credits exhausted",
    403: "forbidden: this app/tier has no access to the endpoint",
    429: "rate limited",
}
_X_LOCK = threading.Lock()  # serialises check+reserve across the runner's worker threads

BSKY_FEED_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
_MEDIA_PATH = re.compile(r"/(?:photo|video)/\d+/?$")


# ------------------------------------------------------------------------------- X budget ledger
def x_spent_usd(session) -> float:
    """Total metered X spend recorded in the ledger (reservations included)."""
    total = session.scalar(
        select(func.coalesce(func.sum(ApiUsage.cost_usd), 0.0)).where(ApiUsage.provider == X_PROVIDER)
    )
    return float(total or 0.0)


class XLedger:
    """Hard budget for X: reserve worst case before a request, settle to actual after it."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = float(budget_usd)

    def reserve(self, endpoint: str, units: int, unit_cost: float) -> int:
        worst = round(units * unit_cost, 6)
        with _X_LOCK, session_scope() as s:
            spent = x_spent_usd(s)
            if spent + worst > self.budget_usd + 1e-9:
                raise CollectorUnavailable(
                    f"X budget guard: spent ${spent:.3f} + worst case ${worst:.3f} for {endpoint} "
                    f"would exceed X_BUDGET_TOTAL_USD=${self.budget_usd:.2f}"
                )
            row = ApiUsage(provider=X_PROVIDER, endpoint=endpoint, units=units, cost_usd=worst)
            s.add(row)
            s.flush()
            return row.id

    def settle(self, row_id: int, units: int | None, unit_cost: float) -> None:
        """units=None: the API rejected the request (not billed), so drop the reservation."""
        with session_scope() as s:
            row = s.get(ApiUsage, row_id)
            if row is None:
                return
            if units is None:
                s.delete(row)
            else:
                row.units, row.cost_usd = units, round(units * unit_cost, 6)


def _x_error(exc: tweepy.errors.HTTPException) -> Exception:
    """Map quota/credential failures to CollectorUnavailable; other HTTP errors stay failures."""
    status = getattr(exc.response, "status_code", None)
    if status not in X_ERROR_REASONS:
        return exc
    reason = f"X API {status} {X_ERROR_REASONS[status]}"
    reset = getattr(exc, "reset_time", None)
    if status == 429 and reset:
        reason += f" until {datetime.fromtimestamp(int(reset), UTC):%Y-%m-%d %H:%M} UTC"
    if exc.api_messages:
        reason += f" ({'; '.join(map(str, exc.api_messages))[:200]})"
    return CollectorUnavailable(reason)


def metered_call(ledger: XLedger, endpoint: str, worst_units: int, unit_cost: float,
                 call: Callable[[], dict], billed: Callable[[dict], int]) -> dict:
    """Run one X request under the ledger. Unknown outcomes (timeouts) keep the worst case."""
    row_id = ledger.reserve(endpoint, worst_units, unit_cost)
    try:
        resp = call()
    except tweepy.errors.HTTPException as exc:
        ledger.settle(row_id, None, unit_cost)
        raise _x_error(exc) from exc
    ledger.settle(row_id, billed(resp), unit_cost)
    return resp


# ------------------------------------------------------------------------------- X API access
def _x_client(settings: Settings) -> tweepy.Client:
    client = tweepy.Client(bearer_token=settings.x_bearer_token, return_type=dict, wait_on_rate_limit=False)
    client.user_agent = settings.user_agent
    # tweepy sets no timeout on its requests session; a hung read would pin a worker thread forever
    client.session.request = functools.partial(client.session.request, timeout=settings.http_timeout_s)
    return client


def resolve_x_user_id(client: Any, ledger: XLedger, handle: str, source: SourceInfo) -> str:
    """Numeric user id for a handle; looked up once ever, then cached in Identity.meta."""
    norm = handle.lstrip("@").lower()
    with session_scope() as s:
        ident = s.scalar(select(Identity).where(Identity.platform == "x", Identity.handle == norm))
        cached = (ident.meta or {}).get("x_user_id") if ident is not None else None
    if cached:
        return str(cached)
    resp = metered_call(
        ledger, X_LOOKUP_ENDPOINT, 1, X_COST_PER_USER_LOOKUP,
        lambda: client.get_user(username=norm, user_fields=["public_metrics", "verified_type", "created_at"]),
        billed=lambda r: 1 if r.get("data") else 0,
    )
    user = resp.get("data")
    if not user:
        raise LookupError(f"X user @{handle} not found: {resp.get('errors')}")
    _cache_user(norm, user, source)
    return str(user["id"])


def _cache_user(norm: str, user: dict, source: SourceInfo) -> None:
    meta = {"x_user_id": str(user["id"]), "x_username": user.get("username"), "x_name": user.get("name"),
            "x_followers": (user.get("public_metrics") or {}).get("followers_count"),
            "x_verified_type": user.get("verified_type"), "x_looked_up_at": utcnow().isoformat()}
    try:
        with session_scope() as s:
            ident = s.scalar(select(Identity).where(Identity.platform == "x", Identity.handle == norm))
            if ident is None:  # source declared without a register identity: record it as curated seed
                ident = Identity(platform="x", handle=norm, lab_id=source.lab_id, person_id=source.person_id,
                                 url=f"https://x.com/{user.get('username') or norm}", method="seed",
                                 status="confirmed", evidence=f"x_timeline source {source.id}")
                s.add(ident)
            ident.meta = {**(ident.meta or {}), **meta}
    except IntegrityError:  # a concurrent run cached it first
        log.debug("identity x:%s cached concurrently", norm)


def latest_post_id(source_id: str) -> str | None:
    """Highest stored post id for the source: the since_id that makes incremental reads free of repeats."""
    with session_scope() as s:
        keys = s.scalars(select(Document.natural_key).where(Document.source_id == source_id)).all()
    ids = [int(k) for k in keys if k.isdigit()]
    return str(max(ids)) if ids else None


def fetch_x_posts(client: Any, ledger: XLedger, user_id: str, *, start_time: datetime,
                  since_id: str | None, cap: int, page_size: int) -> list[dict]:
    """Original posts (no retweets/replies), newest first, at most ~cap (API minimum page is 5)."""
    posts: list[dict] = []
    token: str | None = None
    while len(posts) < cap:
        page = max(X_PAGE_MIN, min(X_PAGE_MAX, page_size, cap - len(posts)))
        params: dict[str, Any] = {"exclude": ["retweets", "replies"], "tweet_fields": X_TWEET_FIELDS,
                                  "max_results": page, "pagination_token": token}
        if since_id:
            params["since_id"] = since_id
        else:
            params["start_time"] = start_time
        resp = metered_call(ledger, X_TIMELINE_ENDPOINT, page, X_COST_PER_POST,
                            lambda p=params: client.get_users_tweets(user_id, **p),
                            billed=lambda r: len(r.get("data") or []))
        posts.extend(resp.get("data") or [])
        token = (resp.get("meta") or {}).get("next_token")
        if not token:
            break
    return posts


# ------------------------------------------------------------------------------- X records
def _snippet(text: str, limit: int = 110) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _x_links(entities: dict | None) -> list[dict]:
    out = []
    for e in (entities or {}).get("urls") or []:
        expanded = e.get("unwound_url") or e.get("expanded_url") or e.get("url")
        if not expanded:
            continue
        media = bool(e.get("media_key")) or bool(_MEDIA_PATH.search(urlsplit(expanded).path))
        out.append({"tco": e.get("url"), "url": expanded, "title": e.get("title"), "media": media})
    return out


def x_post_to_record(post: dict, handle: str) -> IngestRecord:
    """Full text (note_tweet for long posts) with t.co links expanded, plus a Links list."""
    note = post.get("note_tweet") or {}
    text = note.get("text") or post.get("text") or ""
    links = _x_links(note.get("entities")) + _x_links(post.get("entities"))
    urls: list[dict] = []
    for link in links:
        if link["tco"]:
            text = text.replace(link["tco"], "" if link["media"] else link["url"])
        if not link["media"] and link["url"] not in {u["url"] for u in urls}:
            urls.append(link)
    text = text.strip()
    body = text
    if urls:
        body += "\n\nLinks:\n" + "\n".join(
            f"- {u['url']}" + (f" ({u['title']})" if u.get("title") else "") for u in urls)
    quoted = next((r.get("id") for r in post.get("referenced_tweets") or [] if r.get("type") == "quoted"), None)
    pid = str(post["id"])
    return IngestRecord(
        natural_key=pid,
        url=f"https://x.com/{handle}/status/{pid}",
        title=f"@{handle}: {_snippet(text)}",
        content_text=body,
        published_at=to_naive_utc(post.get("created_at")),
        author=handle,
        meta={"platform": "x", "post_id": pid, "metrics": post.get("public_metrics") or {},
              "urls": [u["url"] for u in urls], "is_quote": quoted is not None, "quoted_id": quoted,
              "long_post": bool(note.get("text"))},
    )


@collector("x_timeline", requires=("x_bearer_token",),
           description="X API v2 user timeline: original posts only, metered and budget-guarded")
def x_timeline(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    handle = str(source.config.get("handle") or "").lstrip("@")
    if not handle:
        raise ValueError(f"{source.id}: config.handle is required")
    cap = int(source.config.get("max_posts") or DEFAULT_MAX_POSTS)
    if window.max_items:
        cap = min(cap, window.max_items)
    ledger = XLedger(settings.x_budget_total_usd)
    client = _x_client(settings)
    user_id = resolve_x_user_id(client, ledger, handle, source)
    since_id = None if source.config.get("reread") else latest_post_id(source.id)
    # incremental reads are usually a handful of posts: a small page keeps the reserved worst case low
    page_size = int(source.config.get("page_size") or (20 if since_id else X_PAGE_MAX))
    posts = fetch_x_posts(client, ledger, user_id, start_time=window.since, since_id=since_id,
                          cap=cap, page_size=page_size)
    records = [x_post_to_record(p, handle) for p in posts if p.get("id")]
    log.info("x %s: %d posts (since_id=%s)", handle, len(records), since_id)
    # posts after since_id were paid for: keep them even if older than this run's window
    floor = None if since_id else window.since
    return [r for r in records
            if r.published_at is None or ((floor is None or r.published_at >= floor) and r.published_at <= window.until)]


# ------------------------------------------------------------------------------- Bluesky
def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


@retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _get_json(client: httpx.Client, url: str, params: dict) -> Any:
    resp = client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def _apply_link_facets(text: str, facets: list | None) -> tuple[str, list[str]]:
    """Bluesky shortens displayed links; facets carry the full URI at UTF-8 byte offsets."""
    raw = text.encode("utf-8")
    spans: list[tuple[int, int, str]] = []
    for facet in facets or []:
        idx = facet.get("index") or {}
        for feat in facet.get("features") or []:
            if feat.get("$type") == "app.bsky.richtext.facet#link" and feat.get("uri"):
                spans.append((int(idx.get("byteStart", -1)), int(idx.get("byteEnd", -1)), feat["uri"]))
    for start, end, uri in sorted(spans, reverse=True):
        if 0 <= start < end <= len(raw):
            raw = raw[:start] + uri.encode("utf-8") + raw[end:]
    return raw.decode("utf-8", errors="replace"), [uri for _, _, uri in sorted(spans)]


def bsky_item_to_record(item: dict) -> IngestRecord | None:
    """One feed item -> record; reposts are skipped (like retweets on X)."""
    if str((item.get("reason") or {}).get("$type", "")).endswith("reasonRepost"):
        return None
    post = item.get("post") or {}
    rec, uri = post.get("record") or {}, post.get("uri") or ""
    author = (post.get("author") or {}).get("handle") or ""
    if not uri or not author:
        return None
    text, urls = _apply_link_facets(rec.get("text") or "", rec.get("facets"))
    embed = rec.get("embed") or {}
    external = embed.get("external") or (embed.get("media") or {}).get("external") or {}
    links = [{"url": u, "title": None} for u in urls]
    if external.get("uri") and external["uri"] not in urls:
        links.append({"url": external["uri"], "title": external.get("title")})
    quoted = None
    if str(embed.get("$type", "")).startswith("app.bsky.embed.record"):
        inner = embed.get("record") or {}
        quoted = inner.get("uri") or (inner.get("record") or {}).get("uri")
    body = text.strip()
    if links:
        body += "\n\nLinks:\n" + "\n".join(
            f"- {lk['url']}" + (f" ({lk['title']})" if lk.get("title") else "") for lk in links)
    published = to_naive_utc(rec.get("createdAt"))
    indexed = to_naive_utc(post.get("indexedAt"))
    if published and indexed and published > indexed + timedelta(days=1):
        published = indexed  # createdAt is client-asserted; never trust a future date
    return IngestRecord(
        natural_key=uri,
        url=f"https://bsky.app/profile/{author}/post/{uri.rsplit('/', 1)[-1]}",
        title=f"@{author}: {_snippet(text)}",
        content_text=body,
        published_at=published,
        author=author,
        meta={"platform": "bluesky", "uri": uri, "cid": post.get("cid"), "urls": [lk["url"] for lk in links],
              "is_quote": quoted is not None, "quoted_uri": quoted, "langs": rec.get("langs") or [],
              "metrics": {k: post.get(k) for k in ("likeCount", "repostCount", "replyCount", "quoteCount")}},
    )


@collector("bluesky_feed", description="Bluesky public AppView author feed (posts, no replies/reposts)")
def bluesky_feed(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    handle = str(source.config.get("handle") or "").lstrip("@")
    if not handle:
        raise ValueError(f"{source.id}: config.handle is required")
    cap = int(source.config.get("max_posts") or 100)
    if window.max_items:
        cap = min(cap, window.max_items)
    max_pages = int(source.config.get("max_pages") or 10)
    out: list[IngestRecord] = []
    cursor: str | None = None
    with httpx.Client(timeout=settings.http_timeout_s, headers={"User-Agent": settings.user_agent}) as client:
        for _ in range(max_pages):
            params = {"actor": handle, "filter": "posts_no_replies", "limit": 100}
            if cursor:
                params["cursor"] = cursor
            data = _get_json(client, BSKY_FEED_URL, params)
            feed = data.get("feed") or []
            records = [r for r in (bsky_item_to_record(it) for it in feed) if r is not None]
            out.extend(r for r in records if r.published_at is None or window.since <= r.published_at <= window.until)
            oldest = min((r.published_at for r in records if r.published_at), default=None)
            cursor = data.get("cursor")
            if not cursor or not feed or len(out) >= cap or (oldest is not None and oldest < window.since):
                break
    log.info("bluesky %s: %d posts", handle, len(out[:cap]))
    return out[:cap]
