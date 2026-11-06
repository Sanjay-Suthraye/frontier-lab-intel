"""Research collectors: Hugging Face daily papers and arXiv author feeds.

Both emit one record per paper keyed by the version-less arXiv id, so the runner's canonical-URL
dedup (https://arxiv.org/abs/<id>) links the same paper across the two sources. Lab and person
attribution comes from the register at run time (people.yaml evolves independently of this code),
never from an LLM, and every match records *how* it was made (`meta.lab_match.via`).
The HTTP helpers here (`retrying`, `hf_client`) are shared with the code collectors.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import arxiv
import httpx
from sqlalchemy import select
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from fli.config import Settings
from fli.db.models import Identity, Lab, Person
from fli.db.session import session_scope
from fli.ingest.records import CollectorUnavailable, IngestRecord, SourceInfo, Window, collector
from fli.timeutil import to_naive_utc

log = logging.getLogger("fli.ingest.research")

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAGE_SIZE = 100  # API maximum: limit=200 is rejected with HTTP 400
HF_MAX_PAGES = 5  # busiest observed day had 109 papers
DEFAULT_MIN_UPVOTES = 150  # ~top 1% of daily papers (Aug-Oct 2026 sample: p97=101, p99=145)
DEFAULT_CATEGORIES = ("cs.", "stat.", "eess.")
OLD_STREAK_STOP = 5  # consecutive pre-window results before we stop paging an arXiv query
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?$")
_TIER_RANK = {"A": 0, "B": 1, "C": 2}
_FOLD = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "æ": "ae", "ß": "ss", "đ": "d", "ı": "i"})

OrgIndex = tuple[dict[str, tuple[str, str]], dict[str, str]]  # (handle -> (lab, via), lab-name key -> lab)


class TransientError(Exception):
    """Retryable HTTP condition (5xx, 429, network)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def retrying() -> Retrying:
    """Retry policy for transient HTTP failures: 3 attempts, exponential backoff up to 8 s."""
    return Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=8),
        retry=retry_if_exception_type(TransientError),
        reraise=True,
    )


def hf_client(settings: Settings) -> httpx.Client:
    headers = {"User-Agent": settings.user_agent}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"
    return httpx.Client(headers=headers, timeout=settings.http_timeout_s, follow_redirects=True)


# ------------------------------------------------------------------------------ name matching
def name_tokens(name: str) -> list[str]:
    """ASCII-folded lowercase tokens; "Last, First" is reordered to "First Last"."""
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first} {last}"
    folded = unicodedata.normalize("NFKD", name.translate(_FOLD)).encode("ascii", "ignore").decode()
    return [t for t in re.split(r"[^a-z'\-]+", folded.lower()) if t.strip("'-")]


@dataclass(frozen=True)
class TrackedName:
    """One searchable name of a tracked person."""

    person_id: str
    lab_id: str | None
    tier: str
    first: str
    last: str
    query: str  # what we send to arXiv: full name without single-letter initials

    @classmethod
    def from_name(cls, raw: str, person_id: str, lab_id: str | None, tier: str | None) -> TrackedName | None:
        toks = name_tokens(raw)
        if len(toks) < 2:
            return None
        return cls(person_id, lab_id, tier or "C", toks[0], toks[-1], " ".join(t for t in toks if len(t) > 1))

    def matches(self, author: str, allow_initial: bool = False) -> bool:
        toks = name_tokens(author)
        if len(toks) < 2 or toks[-1] != self.last:
            return False
        first = toks[0]
        return first == self.first or (allow_initial and len(first) == 1 and self.first.startswith(first))


class PeopleIndex:
    """Tracked names bucketed by last name, plus HF usernames of tracked people."""

    def __init__(self, names: Iterable[TrackedName], hf_users: dict[str, TrackedName] | None = None):
        self.by_last: dict[str, list[TrackedName]] = defaultdict(list)
        for n in names:
            self.by_last[n.last].append(n)
        self.hf_users = hf_users or {}

    def match(self, author: str, hf_user: str | None = None) -> TrackedName | None:
        """Exact first+last match (no initials: HF author lists are not pre-filtered by name)."""
        if hf_user and hf_user.lower() in self.hf_users:
            return self.hf_users[hf_user.lower()]
        toks = name_tokens(author)
        return next((c for c in self.by_last.get(toks[-1] if toks else "", []) if c.matches(author)), None)


def best_person(people: list[TrackedName]) -> TrackedName | None:
    return min(people, key=lambda p: (_TIER_RANK.get(p.tier, 3), p.person_id)) if people else None


def load_people_index() -> PeopleIndex:
    """Person names + confirmed arxiv_name identities + HF usernames, read from the register."""
    names: set[TrackedName] = set()
    hf_users: dict[str, TrackedName] = {}
    with session_scope() as s:
        people = {p.id: p for p in s.scalars(select(Person))}
        for p in people.values():
            if tn := TrackedName.from_name(p.name, p.id, p.lab_id, p.tier):
                names.add(tn)
        q = select(Identity).where(Identity.person_id.is_not(None), Identity.status == "confirmed")
        for ident in s.scalars(q.where(Identity.platform.in_(("arxiv_name", "hf")))):
            p = people.get(ident.person_id)
            if p is not None and ident.platform == "hf":
                if tn := TrackedName.from_name(p.name, p.id, p.lab_id, p.tier):
                    hf_users[ident.handle.lower()] = tn
            elif p is not None and (tn := TrackedName.from_name(ident.handle, p.id, p.lab_id, p.tier)):
                names.add(tn)
    return PeopleIndex(names, hf_users)


def _alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def load_org_index(aliases: dict[str, str]) -> OrgIndex:
    """HF org handle -> (lab_id, via). Handles come from register `hf` identities and curated
    aliases; lab *names* only match an org handle exactly and only when >= 6 chars, because short
    names collide with unrelated orgs (HF orgs "XAI" and "SSI" are not xAI or SSI). Display names
    are never used: org "GoogIe" (capital i) shows the fullname "Google"."""
    handles: dict[str, tuple[str, str]] = {}
    names: dict[str, str] = {}
    with session_scope() as s:
        for lab in s.scalars(select(Lab)):
            if len(key := _alnum(re.sub(r"\(.*?\)", "", lab.name))) >= 6:
                names[key] = lab.id
        for ident in s.scalars(
            select(Identity).where(Identity.platform == "hf", Identity.lab_id.is_not(None))
        ):
            handles[ident.handle.lower()] = (ident.lab_id, "hf_identity")
    for handle, lab_id in (aliases or {}).items():
        handles.setdefault(str(handle).lower(), (lab_id, "alias"))
    return handles, names


def match_org(org: dict | None, index: OrgIndex) -> dict | None:
    handle = str((org or {}).get("name") or "")
    if hit := index[0].get(handle.lower()):
        return {"lab_id": hit[0], "via": hit[1], "value": handle}
    if handle and (lab_id := index[1].get(_alnum(handle))):
        return {"lab_id": lab_id, "via": "lab_name", "value": handle}
    return None


def _author_line(names: list[str]) -> str | None:
    return (", ".join(names[:3]) + (" et al." if len(names) > 3 else "")) if names else None


def _cap(records: list[IngestRecord], max_items: int | None) -> list[IngestRecord]:
    records.sort(key=lambda r: r.published_at or datetime.min, reverse=True)
    return records[:max_items] if max_items else records


# ------------------------------------------------------------------------- hf_daily_papers
def get_json(client: httpx.Client, url: str, params: dict[str, Any]) -> Any:
    """GET with retries on 5xx/429/network errors; other HTTP errors raise immediately."""
    for attempt in retrying():
        with attempt:
            try:
                resp = client.get(url, params=params)
            except httpx.TransportError as exc:
                raise TransientError(str(exc)) from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                raise TransientError(f"HTTP {resp.status_code} for {url}", resp.status_code)
            resp.raise_for_status()
            return resp.json()
    return None  # pragma: no cover - retrying() always returns or raises


def fetch_daily(client: httpx.Client, day: date) -> list[dict]:
    """All papers featured on one day. The API always sends a `next` link, so stop on a short page."""
    items: list[dict] = []
    for page in range(HF_MAX_PAGES):
        params = {"date": day.isoformat(), "limit": HF_PAGE_SIZE, "p": page}
        batch = get_json(client, HF_DAILY_PAPERS_API, params) or []
        items.extend(batch)
        if len(batch) < HF_PAGE_SIZE:
            break
    return items


def _days(window: Window, settings: Settings, recheck_days: int, max_days: int) -> list[date]:
    """Window days, newest first. Re-reading the last `recheck_days` lets late upvotes count."""
    end = window.until.date()
    start = min(window.since.date(), end - timedelta(days=recheck_days))
    start = max(start, settings.backfill_start, end - timedelta(days=max_days))
    return [end - timedelta(days=i) for i in range((end - start).days + 1)]


def paper_record(
    item: dict, day: date, orgs: OrgIndex, people: PeopleIndex, cfg: dict
) -> IngestRecord | None:
    """Turn one daily-papers item into a record, or None when it is neither lab-related nor popular."""
    paper = item.get("paper") or {}
    pid = str(paper.get("id") or "").strip()
    title = " ".join(str(paper.get("title") or item.get("title") or "").split())
    if not pid or not title:
        return None
    authors = [a for a in paper.get("authors") or [] if not a.get("hidden") and a.get("name")]
    names = [str(a["name"]).strip() for a in authors]
    matched = (people.match(str(a["name"]), (a.get("user") or {}).get("user")) for a in authors)
    tracked = list(dict.fromkeys(t for t in matched if t))
    org = paper.get("organization") or item.get("organization") or {}
    lab, lead = match_org(org, orgs), best_person(tracked)
    if lab is None and lead is not None and lead.lab_id:
        lab = {"lab_id": lead.lab_id, "via": "tracked_author", "value": lead.person_id}
    upvotes = int(paper.get("upvotes") or 0)
    popular = upvotes >= int(cfg.get("min_upvotes_unmatched", DEFAULT_MIN_UPVOTES))
    selection = "lab" if lab else "tracked_author" if tracked else "upvotes" if popular else None
    if selection is None:
        return None
    m = _ARXIV_ID.search(pid)
    content = f"{title}\n\n{str(paper.get('summary') or item.get('summary') or '').strip()}"
    if paper.get("ai_summary") and cfg.get("include_ai_summary", True):
        content += (
            f"\n\nHugging Face AI summary (machine-generated, not the authors' words): {paper['ai_summary']}"
        )
    published = to_naive_utc(paper.get("publishedAt") or item.get("publishedAt"))
    meta = {
        "arxiv_id": m.group(1) if m else None,
        "authors": names[:60],
        "upvotes": upvotes,
        "num_comments": item.get("numComments"),
        "github_repo": paper.get("githubRepo"),
        "github_stars": paper.get("githubStars"),
        "project_page": paper.get("projectPage"),
        "organization": {"name": org.get("name"), "fullname": org.get("fullname")} if org else None,
        "hf_url": f"https://huggingface.co/papers/{pid}",
        "daily_date": day.isoformat(),
        "lab_match": lab,
        "tracked_people": [t.person_id for t in tracked],
        "selection": selection,
        "ai_keywords": (paper.get("ai_keywords") or [])[:12],
    }
    return IngestRecord(
        natural_key=m.group(1) if m else pid,
        url=f"https://arxiv.org/abs/{m.group(1)}" if m else meta["hf_url"],
        title=title,
        content_text=content,
        published_at=published or datetime(day.year, day.month, day.day),
        author=_author_line(names),
        lab_id=lab["lab_id"] if lab else None,
        person_id=lead.person_id if lead else None,
        meta=meta,
    )


@collector("hf_daily_papers", description="HF daily papers: tracked labs/people + top-upvoted papers")
def hf_daily_papers(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    cfg = source.config or {}
    orgs, people = load_org_index(cfg.get("org_aliases") or {}), load_people_index()
    days = _days(window, settings, int(cfg.get("recheck_days", 7)), int(cfg.get("max_days", 120)))
    found: dict[str, IngestRecord] = {}
    seen = 0
    with hf_client(settings) as client:
        for day in days:
            try:
                items = fetch_daily(client, day)
            except TransientError as exc:
                if exc.status == 429:
                    raise CollectorUnavailable(f"Hugging Face rate limit: {exc}") from exc
                raise
            seen += len(items)
            for item in items:
                if (rec := paper_record(item, day, orgs, people, cfg)) and rec.natural_key not in found:
                    found[rec.natural_key] = rec
    by: dict[str, int] = defaultdict(int)
    for rec in found.values():
        by[rec.meta["selection"]] += 1
    log.info("%s: %d days, %d papers seen, kept %d %s", source.id, len(days), seen, len(found), dict(by))
    return _cap(list(found.values()), window.max_items)


# ---------------------------------------------------------------------------- arxiv_authors
_ARXIV_LOCK = threading.Lock()  # one shared client => >= delay_seconds between requests per process
_arxiv_shared: arxiv.Client | None = None
Found = dict[str, tuple[arxiv.Result, list[TrackedName], list[str]]]  # arxiv id -> (result, people, names)


def _arxiv_client(delay: float, page_size: int) -> arxiv.Client:
    global _arxiv_shared
    if _arxiv_shared is None:
        _arxiv_shared = arxiv.Client(page_size=page_size, delay_seconds=delay, num_retries=3)
    _arxiv_shared.page_size, _arxiv_shared.delay_seconds = page_size, delay
    return _arxiv_shared


def arxiv_targets(cfg: dict) -> list[TrackedName]:
    """Confirmed arxiv_name identities joined to their person (read at run time)."""
    overrides = cfg.get("author_overrides") or {}
    targets: set[TrackedName] = set()
    with session_scope() as s:
        rows = s.execute(
            select(Identity.handle, Person.id, Person.lab_id, Person.tier)
            .join(Person, Person.id == Identity.person_id)
            .where(Identity.platform == "arxiv_name", Identity.status == "confirmed")
        ).all()
    for handle, person_id, lab_id, tier in rows:
        ov = overrides.get(person_id) or {}
        if ov.get("skip"):
            continue
        if tn := TrackedName.from_name(str(ov.get("query_name") or handle), person_id, lab_id, tier):
            targets.add(tn)
        else:
            log.warning("arxiv_name %r of %s is not a 'first last' name; skipped", handle, person_id)
    return sorted(targets, key=lambda t: (t.person_id, t.query))


def window_results(results: Iterator[arxiv.Result], since: datetime) -> Iterator[arxiv.Result]:
    """Yield in-window results; stop after a streak of older ones (sort order is not strictly monotonic)."""
    streak = 0
    for res in results:
        published = to_naive_utc(res.published)
        if published is not None and published < since:
            streak += 1
            if streak >= OLD_STREAK_STOP:
                return
            continue
        streak = 0
        yield res


def _scan_batch(
    client: arxiv.Client,
    batch: list[TrackedName],
    since: datetime,
    cfg: dict,
    counts: dict[str, int],
    found: Found,
) -> None:
    """One OR-query for a batch of names; attribute each in-window result to the tracked authors on it."""
    per_author = int(cfg.get("max_results_per_author", 25))
    prefixes = tuple(cfg.get("category_prefixes", DEFAULT_CATEGORIES))
    search = arxiv.Search(
        query=" OR ".join(f'au:"{t.query}"' for t in batch),
        max_results=per_author * len(batch),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    for res in window_results(client.results(search), since):
        hits = [(t, a.name) for t in batch for a in res.authors if t.matches(a.name, allow_initial=True)]
        people = [t for t in dict.fromkeys(t for t, _ in hits) if counts[t.person_id] < per_author]
        categories = res.categories or [res.primary_category]
        if not people or (prefixes and not any(c.startswith(prefixes) for c in categories)):
            continue  # namesake outside the query's intent, or an off-topic category
        for t in people:
            counts[t.person_id] += 1
        key = re.sub(r"v\d+$", "", res.get_short_id())
        prev_people, prev_names = found[key][1:] if key in found else ([], [])
        found[key] = (
            res,
            list(dict.fromkeys(prev_people + people)),
            list(dict.fromkeys(prev_names + [n for _, n in hits])),
        )


def arxiv_record(res: arxiv.Result, matched: list[TrackedName], authors_matched: list[str]) -> IngestRecord:
    short = res.get_short_id()
    arxiv_id = re.sub(r"v\d+$", "", short)
    names = [a.name for a in res.authors]
    lead = best_person(matched)
    content = f"{' '.join(res.title.split())}\n\n{res.summary.strip()}"
    if res.comment:
        content += f"\n\nComments: {res.comment.strip()}"
    meta = {
        "arxiv_id": arxiv_id,
        "version": short[len(arxiv_id) :] or None,
        "authors": names[:60],
        "primary_category": res.primary_category,
        "categories": res.categories,
        "comment": res.comment or None,
        "journal_ref": res.journal_ref or None,
        "doi": res.doi or None,
        "pdf_url": res.pdf_url,
        "updated_at": to_naive_utc(res.updated).isoformat() if res.updated else None,
        "tracked_people": [t.person_id for t in matched],
        "matched_authors": authors_matched,
    }
    return IngestRecord(
        natural_key=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        title=" ".join(res.title.split()),
        content_text=content,
        published_at=to_naive_utc(res.published),
        author=_author_line(names),
        person_id=lead.person_id if lead else None,
        lab_id=lead.lab_id if lead else None,
        meta=meta,
    )


@collector("arxiv_authors", description="arXiv API: recent papers by tracked people (arxiv_name identities)")
def arxiv_authors(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    cfg = source.config or {}
    targets = arxiv_targets(cfg)
    if not targets:
        log.info("%s: no confirmed arxiv_name identities in the register", source.id)
        return []
    size = max(1, int(cfg.get("authors_per_query", 8)))
    batches = [targets[i : i + size] for i in range(0, len(targets), size)]
    page_size = min(100, size * int(cfg.get("max_results_per_author", 25)))
    found: Found = {}
    counts: dict[str, int] = defaultdict(int)
    failed = 0
    with _ARXIV_LOCK:
        client = _arxiv_client(float(cfg.get("delay_seconds", 3.0)), page_size)
        for batch in batches:
            try:
                _scan_batch(client, batch, window.since, cfg, counts, found)
            except (arxiv.ArxivError, OSError) as exc:  # keep other batches; next run's overlap retries
                failed += 1
                log.warning("%s: arXiv query for %s failed: %s", source.id, [t.query for t in batch], exc)
    if failed and failed == len(batches):
        raise RuntimeError(f"all {failed} arXiv author queries failed")
    records = [arxiv_record(res, people, names) for res, people, names in found.values()]
    log.info(
        "%s: %d names, %d queries (%d failed), %d papers",
        source.id,
        len(targets),
        len(batches),
        failed,
        len(records),
    )
    return _cap(records, window.max_items)
