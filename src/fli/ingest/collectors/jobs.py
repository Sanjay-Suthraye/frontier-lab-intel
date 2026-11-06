"""Hiring signals from labs' public job boards (Greenhouse, Ashby, Lever posting APIs).

Allowed exception to the collector rule: this collector owns the `job_posting` snapshot table. Each
run upserts the board's current postings (first_seen_at / last_seen_at / removed_at) and then
returns ONE record per lab per completed ISO week ('hiring:<lab>:<YYYY-Www>') summarising new and
removed postings and notable shifts. Those records become `hiring_signal` documents downstream.

New postings are dated by the board's own publish date when it has one (Greenhouse
first_published, Ashby publishedAt, Lever createdAt), so the backfill can be reconstructed from
the first snapshot. Only completed weeks are emitted, because the runner never rewrites the text
of a document it has already stored.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fli.config import Settings
from fli.db.models import JobPosting, Lab
from fli.db.session import session_scope
from fli.ingest.records import IngestRecord, SourceInfo, Window, collector
from fli.timeutil import to_naive_utc, utcnow, week_start

log = logging.getLogger("fli.ingest.jobs")

API_URLS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{board}/departments",  # ~12x smaller than jobs?content=true
    "greenhouse_jobs": "https://boards-api.greenhouse.io/v1/boards/{board}/jobs",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{board}",
    "lever": "https://api.lever.co/v0/postings/{board}",
}
BOARD_URLS = {
    "greenhouse": "https://job-boards.greenhouse.io/{board}",
    "ashby": "https://jobs.ashbyhq.com/{board}",
    "lever": "https://jobs.lever.co/{board}",
}

CATEGORIES = ("hardware_chips", "robotics", "devices", "infrastructure_datacenter", "research",
              "applied_product", "go_to_market", "security", "policy", "other")
# signal categories for investment readers: compute build-out, custom silicon, new hardware lines, policy
SIGNAL_CATEGORIES = ("hardware_chips", "robotics", "devices", "infrastructure_datacenter", "policy")


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Precedence: specific title vocabulary, then the org unit (team/department), then weak device words,
# then broad vocabularies. Calibrated on the Anthropic, OpenAI, xAI, TML, Reflection and Periodic boards.
TITLE_STRONG: list[tuple[str, re.Pattern[str]]] = [
    ("hardware_chips", _rx(r"\b(asic|silicon|chips?|soc|rtl|verilog|physical design|design verification|dft|"
                           r"tape[- ]?out|accelerators?|serdes|analog|mixed[- ]signal|fpga|hardware architect\w*|"
                           r"kernels? engineer\w*|custom hardware)\b")),
    ("robotics", _rx(r"\b(robot\w*|humanoid|manipulation|embodied|actuat\w*|teleop\w*|locomotion|mechatronic\w*)\b")),
    ("infrastructure_datacenter", _rx(
        r"\b(data ?cent(?:er|re)s?|supercomput\w*|power systems?|energy|gas supply|substations?|construction|"
        r"site selection|site acquisition|capacity (?:planning|deployment)|critical facilities|"
        r"facilit(?:y|ies) (?:engineer|technician)\w*|mep|hvac|cooling|commissioning|battery storage|grid|"
        r"interconnection|electrical (?:engineer\w*|construction|maintenance)|controls engineer\w*)\b")),
    ("devices", _rx(r"\b(consumer devices?|devices?|wearables?|consumer hardware|industrial design\w*|firmware|"
                    r"product design engineer\w*)\b")),
    ("security", _rx(r"\b(security|threat|detection (?:and|&) response|red team\w*|cyber\w*|appsec|"
                     r"incident response|insider risk)\b")),
    ("policy", _rx(r"\b(policy|policies|government|public sector|regulat\w*|legislat\w*|geopolit\w*|"
                   r"national security|global affairs|public affairs|elections?)\b")),
]
ORG_STRONG: list[tuple[str, re.Pattern[str]]] = [
    ("hardware_chips", _rx(r"\b(hardware|silicon|chips?|asic|kernels)\b")),
    ("robotics", _rx(r"\brobotics?\b")),
    ("devices", _rx(r"\b(consumer devices?|devices?)\b")),
    ("infrastructure_datacenter", _rx(r"\b(data ?cent(?:er|re)s?|datacenter design|compute|supercomputing|"
                                      r"infrastructure|fleet\w*)\b")),
    ("security", _rx(r"\bsecurity\b")),
    ("policy", _rx(r"\b(public policy|policy|global affairs|for gov|government)\b")),
]
# device words that only mean "devices" on an engineering/manufacturing role
TITLE_WEAK = [("devices", _rx(r"\b(mechanical|electrical|optical|acoustics?|camera|display|battery|manufacturing|npi)\b"
                              r".*\b(engineer\w*|technician|scientist)\b"))]
BROAD_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("other", _rx(r"\b(finance|financial|accounting|accountant|tax|legal|counsel|paralegal|recruit\w*|"
                  r"talent (?:acquisition|partner|sourc\w*)|"
                  r"people|hr|payroll|workplace|executive assistant|executive business partner|it|procurement|"
                  r"treasury|audit|controller|g&a)\b")),
    ("research", _rx(r"\b(research\w*|scientist|pre[- ]?training|post[- ]?training|reinforcement learning|alignment|"
                     r"interpretability|evals?|evaluations?|reasoning|multimodal|model behavio\w*|"
                     r"member of technical staff|safety systems|safeguards)\b")),
    ("go_to_market", _rx(r"\b(sales|account (?:executive|manager|director)|partnerships?|business development|"
                         r"marketing|go[- ]to[- ]market|gtm|customer success|solutions? (?:architect|engineer)\w*|"
                         r"forward deployed|deployment strateg\w*|revenue|alliances|demand gen\w*|communications|"
                         r"brand|technical success|support|developer relations|devrel|commercial|user operations)\b")),
    ("infrastructure_datacenter", _rx(r"\b(infrastructure|infra|compute|clusters?|hpc|networking|network engineer\w*|"
                                      r"fleet|reliability|sre|scaling|capacity|hardware)\b")),
    ("applied_product", _rx(r"\b(software|engineer\w*|engineering|product|designer|design|developer|front[- ]?end|"
                            r"back[- ]?end|full[- ]?stack|ios|android|mobile|applied|api|platform|data scien\w*|"
                            r"machine learning|ml)\b")),
]
LATE_LISTING = timedelta(days=7)  # first seen this long after its publish date: count it when seen
SENIOR = _rx(r"\b(head of|director|vp|vice president|chief|founding|general manager|principal)\b")


def categorise(title: str, department: str | None = None, team: str | None = None) -> str:
    """Keyword rules over title and org unit; first matching (pass, rule) wins."""
    org = " ".join(x for x in (team, department) if x)
    passes = ((TITLE_STRONG, title or ""), (ORG_STRONG, org), (TITLE_WEAK, title or ""),
              (BROAD_RULES, title or ""), (BROAD_RULES, org))
    for rules, text in passes:
        for category, pattern in rules:
            if text and pattern.search(text):
                return category
    return "other"


# ------------------------------------------------------------------------------- board adapters
@dataclass(frozen=True)
class Posting:
    posting_id: str
    title: str
    department: str | None
    team: str | None
    location: str | None
    url: str | None
    published_at: datetime | None


def _clean(value: Any) -> str | None:
    text = " ".join(str(value).split()) if value else ""
    return text or None


def parse_greenhouse(data: dict) -> list[Posting]:
    """Accepts the /departments payload (default) or /jobs?content=true."""
    rows: list[tuple[dict, str | None]] = []
    if "departments" in data:
        # leaves first, so a job listed under a parent and a child keeps the specific department
        for dep in sorted(data["departments"], key=lambda d: bool(d.get("child_ids"))):
            name = None if dep.get("name") in (None, "No Department") else dep["name"]
            rows += [(job, name) for job in dep.get("jobs") or []]
    else:
        for job in data.get("jobs") or []:
            names = [d["name"] for d in job.get("departments") or [] if d.get("name")]
            rows.append((job, " / ".join(names) or None))
    out: dict[str, Posting] = {}
    for job, dept in rows:
        pid = str(job.get("id") or "")
        if pid and pid not in out:
            out[pid] = Posting(pid, _clean(job.get("title")) or "(untitled)", _clean(dept), None,
                               _clean((job.get("location") or {}).get("name")), job.get("absolute_url"),
                               to_naive_utc(job.get("first_published")))
    return list(out.values())


def parse_ashby(data: dict) -> list[Posting]:
    out = []
    for job in data.get("jobs") or []:
        if job.get("isListed") is False or not job.get("id"):
            continue
        locs = [job.get("location")] + [s.get("location") for s in job.get("secondaryLocations") or []]
        out.append(Posting(str(job["id"]), _clean(job.get("title")) or "(untitled)", _clean(job.get("department")),
                           _clean(job.get("team")), _clean("; ".join(x for x in locs if x)), job.get("jobUrl"),
                           to_naive_utc(job.get("publishedAt"))))
    return out


def parse_lever(data: list) -> list[Posting]:
    out = []
    for job in data or []:
        cats = job.get("categories") or {}
        created = job.get("createdAt")
        published = datetime.fromtimestamp(created / 1000, UTC).replace(tzinfo=None) if created else None
        locs = cats.get("allLocations") or [cats.get("location")]
        out.append(Posting(str(job["id"]), _clean(job.get("text")) or "(untitled)", _clean(cats.get("department")),
                           _clean(cats.get("team")), _clean("; ".join(x for x in locs if x)), job.get("hostedUrl"),
                           published))
    return out


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


@retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _get_json(client: httpx.Client, url: str, params: dict | None = None) -> Any:
    resp = client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def fetch_board(client: httpx.Client, ats: str, board: str, config: dict) -> list[Posting]:
    if ats == "greenhouse":
        if config.get("greenhouse_endpoint") == "jobs":
            return parse_greenhouse(_get_json(client, API_URLS["greenhouse_jobs"].format(board=board),
                                              {"content": "true"}))
        return parse_greenhouse(_get_json(client, API_URLS["greenhouse"].format(board=board)))
    if ats == "ashby":
        return parse_ashby(_get_json(client, API_URLS["ashby"].format(board=board)))
    if ats == "lever":
        return parse_lever(_get_json(client, API_URLS["lever"].format(board=board), {"mode": "json"}))
    raise ValueError(f"unknown ats '{ats}' (greenhouse | ashby | lever)")


# ------------------------------------------------------------------------------- snapshot table
def upsert_snapshot(session: Session, lab_id: str, board: str, postings: list[Posting], now: datetime,
                    token: str | None = None) -> dict:
    """Idempotent upsert of the current board state; postings that disappeared get removed_at.
    `board` is the ATS (the job_posting.board column); `token` tells two boards on one ATS apart."""
    existing = {r.posting_id: r for r in session.scalars(
        select(JobPosting).where(JobPosting.lab_id == lab_id, JobPosting.board == board))
        if token is None or (r.meta or {}).get("board_token", token) == token}
    first_snapshot = not existing
    active_before = sum(1 for r in existing.values() if r.removed_at is None)
    seen: set[str] = set()
    new = relisted = 0
    for p in postings:
        seen.add(p.posting_id)
        row = existing.get(p.posting_id)
        if row is None:
            row = JobPosting(lab_id=lab_id, board=board, posting_id=p.posting_id, first_seen_at=now,
                             meta={"baseline": True} if first_snapshot and p.published_at is None else {})
            if not first_snapshot and p.published_at is not None and p.published_at < now - LATE_LISTING:
                # re-opened or late-published role: its publish week was already summarised
                row.meta = {**row.meta, "late_listed": True}
            session.add(row)
            new += 1
        elif row.removed_at is not None:
            relisted += 1
        row.title, row.department, row.location, row.url = p.title, p.department, p.location, p.url
        row.category = categorise(p.title, p.department, p.team)
        row.published_at = p.published_at or row.published_at
        row.last_seen_at, row.removed_at = now, None
        row.meta = {**(row.meta or {}), "team": p.team, **({"board_token": token} if token else {})}
    gone = [r for r in existing.values() if r.removed_at is None and r.posting_id not in seen]
    # an empty or collapsed board is far more likely an API/ATS migration than a mass closure
    suspicious = active_before >= 10 and len(seen) < 0.2 * active_before
    if suspicious:
        log.warning("%s/%s: %d listed vs %d active before - not marking removals", lab_id, board,
                    len(seen), active_before)
    else:
        for r in gone:
            r.removed_at = now
    session.flush()
    return {"listed": len(seen), "new": new, "relisted": relisted,
            "removed": 0 if suspicious else len(gone), "suspicious": suspicious}


# ------------------------------------------------------------------------------- weekly summaries
def opened_at(row: JobPosting) -> datetime | None:
    """When a posting opened: the board's publish date, else when we first saw it.
    Undated postings from the very first snapshot are 'baseline' (opened before tracking); postings
    that surface long after their publish date count from when we first saw them."""
    meta = row.meta or {}
    if row.published_at is not None and not meta.get("late_listed"):
        return row.published_at
    return None if meta.get("baseline") else row.first_seen_at


def _location_tokens(location: str | None) -> list[str]:
    return [t.strip() for t in re.split(r"[;|]", location or "") if t.strip()]


def _first_seen_by(rows: list[JobPosting], key) -> dict[str, datetime]:
    first: dict[str, datetime] = {}
    for r in rows:
        opened = opened_at(r)
        for value in key(r):
            if opened is not None and (value not in first or opened < first[value]):
                first[value] = opened
    return first


def iso_label(day: datetime) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def _notable(postings: list[JobPosting], limit: int = 12) -> list[JobPosting]:
    def rank(r: JobPosting) -> tuple[int, str]:
        signal, senior = r.category in SIGNAL_CATEGORIES, bool(SENIOR.search(r.title or ""))
        return (0 if signal and senior else 1 if signal else 2, r.title or "")
    picked = [r for r in postings if r.category in SIGNAL_CATEGORIES or SENIOR.search(r.title or "")]
    return sorted(picked, key=rank)[:limit]


def _fmt_counts(counts: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def week_summary(rows: list[JobPosting], ws: datetime, *, lab_id: str, lab_name: str,
                 boards: list[str], board_url: str) -> IngestRecord | None:
    """Summary record for the ISO week starting `ws`, or None when nothing opened or closed."""
    we = ws + timedelta(days=7) - timedelta(seconds=1)
    new = [r for r in rows if (o := opened_at(r)) is not None and ws <= o <= we]
    removed = [r for r in rows if r.removed_at is not None and ws <= r.removed_at <= we]
    if not new and not removed:
        return None
    label = iso_label(ws)
    by_cat = Counter(r.category or "other" for r in new)
    earliest = min((o for r in rows if (o := opened_at(r)) is not None), default=ws)
    tracked_since = min(r.first_seen_at for r in rows)
    trailing: dict[str, float] = {}
    if earliest <= ws - timedelta(days=28):
        prior = Counter(r.category or "other" for r in rows
                        if (o := opened_at(r)) is not None and ws - timedelta(days=28) <= o < ws)
        trailing = {k: round(v / 4, 2) for k, v in prior.items()}
    shifts = [f"{c} {n} new vs {trailing.get(c, 0.0):.1f}/week trailing average"
              for c, n in by_cat.items() if trailing and n >= 3 and n >= 2 * trailing.get(c, 0.0)]
    first_dept = _first_seen_by(rows, lambda r: [r.department] if r.department else [])
    first_loc = _first_seen_by(rows, lambda r: _location_tokens(r.location))
    new_depts = sorted({r.department for r in new if r.department and ws <= first_dept[r.department] <= we})
    new_locs = sorted({t for r in new for t in _location_tokens(r.location) if ws <= first_loc[t] <= we})
    open_end = sum(1 for r in rows if (opened_at(r) or r.first_seen_at) <= we
                   and (r.removed_at is None or r.removed_at > we))
    notable = _notable(new)
    reconstructed = we < tracked_since
    lines = [
        (f"{lab_name} hiring activity, ISO week {label} ({ws:%Y-%m-%d} to {we:%Y-%m-%d}), "
         f"from its public job board ({', '.join(boards)})."),
        f"New postings: {len(new)}" + (f" (trailing 4-week average {sum(trailing.values()):.1f})" if trailing else "")
        + f". Removed postings: {len(removed) if not reconstructed else 'unknown (before tracking)'}."
        + f" Open postings on record at week end: {open_end}.",
    ]
    if by_cat:
        lines.append(f"New postings by category: {_fmt_counts(by_cat)}.")
    if removed:
        lines.append(f"Removed postings by category: {_fmt_counts(Counter(r.category or 'other' for r in removed))}.")
    if shifts:
        lines.append("Notable shifts: " + "; ".join(shifts) + ".")
    if new_depts:
        lines.append("Departments with their first posting on record: " + ", ".join(new_depts) + ".")
    if new_locs:
        lines.append("Locations with their first posting on record: " + ", ".join(new_locs) + ".")
    if notable:
        lines.append("Notable new roles:")
        lines += [f"- {r.title} [{r.category}; {r.department or 'no department'}; {r.location or 'n/a'}] {r.url or ''}"
                  .rstrip() for r in notable]
    if reconstructed:
        lines.append("Note: reconstructed from postings still listed when tracking began (dated by first "
                     "publication); roles opened and closed before then are not visible.")
    return IngestRecord(
        natural_key=f"hiring:{lab_id}:{label}",
        url=f"{board_url}?week={label}",
        title=f"{lab_name} hiring {label}: {len(new)} new postings"
              + (f", top category {by_cat.most_common(1)[0][0]}" if by_cat else ""),
        content_text="\n".join(lines),
        published_at=we,
        meta={"kind": "hiring_summary", "week": label, "week_start": ws.isoformat(), "week_end": we.isoformat(),
              "new_postings": len(new), "removed_postings": len(removed), "open_at_week_end": open_end,
              "by_category": dict(by_cat), "trailing_4w_avg": trailing, "shifts": shifts,
              "new_departments": new_depts, "new_locations": new_locs, "reconstructed": reconstructed,
              "boards": boards, "notable": [{"title": r.title, "url": r.url, "category": r.category,
                                             "department": r.department, "location": r.location} for r in notable]},
    )


def weekly_summaries(rows: list[JobPosting], *, since: datetime, until: datetime, lab_id: str,
                     lab_name: str, boards: list[str], board_url: str) -> list[IngestRecord]:
    """One record per completed ISO week whose end falls inside [since, until]."""
    if not rows:
        return []
    out = []
    ws = week_start(since)
    while ws + timedelta(days=7) - timedelta(seconds=1) <= until:
        if ws + timedelta(days=7) - timedelta(seconds=1) >= since:
            rec = week_summary(rows, ws, lab_id=lab_id, lab_name=lab_name, boards=boards, board_url=board_url)
            if rec is not None:
                out.append(rec)
        ws += timedelta(days=7)
    return out


@collector("job_board", description="Public ATS job boards -> job_posting snapshot + weekly hiring summaries")
def job_board(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    ats, board = source.config.get("ats"), source.config.get("board")
    if ats not in BOARD_URLS or not board:
        raise ValueError(f"{source.id}: config needs ats (greenhouse|ashby|lever) and board")
    if not source.lab_id:
        raise ValueError(f"{source.id}: job_board sources must belong to a lab")
    with httpx.Client(timeout=settings.http_timeout_s, headers={"User-Agent": settings.user_agent},
                      follow_redirects=True) as client:
        postings = fetch_board(client, ats, board, source.config)
    now = utcnow()
    with session_scope() as s:
        stats = upsert_snapshot(s, source.lab_id, ats, postings, now, token=board)
        rows = list(s.scalars(select(JobPosting).where(JobPosting.lab_id == source.lab_id)))
        lab = s.get(Lab, source.lab_id)
        boards = sorted({f"{r.board}:{(r.meta or {}).get('board_token') or board}" for r in rows})
        records = weekly_summaries(rows, since=window.since, until=min(window.until, now), lab_id=source.lab_id,
                                   lab_name=lab.name if lab else source.lab_id, boards=boards,
                                   board_url=BOARD_URLS[ats].format(board=board))
    log.info("jobs %s %s:%s: %s, %d weekly summaries", source.lab_id, ats, board, stats, len(records))
    return records[-window.max_items:] if window.max_items else records
