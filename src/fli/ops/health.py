"""System health: freshness SLOs, failing and never-run sources, silent parsers, failed and stalled
jobs, a silent worker, unreadable scoring models and budgets.

Every check is a read of ledgers the pipeline already writes (source health fields, job_run,
llm_call, api_usage, the deployed model files). Findings become OpsEvent rows keyed
'<kind>:<subject>:<YYYY-MM-DD>', so the 15-minute schedule never duplicates them; an event resolves
itself when its condition clears, and an older day's event for a still-active condition is superseded
by today's. Open critical events go to Slack once (re-sent daily while they persist); Healthchecks.io
gets a heartbeat, or /fail while critical events are open, so a dead worker is caught externally
(dead-man's switch).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from fli.config import Settings, get_settings
from fli.db.models import ApiUsage, Document, JobRun, LLMCall, OpsEvent, Source
from fli.db.session import session_scope
from fli.deliver import notify
from fli.timeutil import utcnow

log = logging.getLogger("fli.ops.health")

HC_SLUG = "fli-health"
SEVERITY_RANK = {"info": 0, "warn": 1, "crit": 2}
FAILING_AFTER = 3  # consecutive collector failures
PARSER_RECENT, PARSER_HISTORY = 3, 10  # empty successful runs in a row / runs that define "normal"
BUDGET_WARN, BUDGET_STOP = 0.8, 0.95
X_PROVIDERS = ("x", "x_api", "twitter")
JOB_PREFIXES = ("stage:", "worker:")
JOB_WINDOW = timedelta(hours=24)  # failed jobs are judged on the last day of runs
SLACK_MAX_ATTEMPTS = 3
NEVER_RUN_CADENCES = 2  # an enabled source that has not run within two cadences of being registered
NAMED_SOURCES = 8  # source ids spelled out in a grouped message
ACTIVE_WINDOW = timedelta(days=7)  # "a worker / ingestion is running in this deployment"
INGEST_RECENT = timedelta(hours=24)  # ingestion counts as running now (as the web's STALE_INGEST_HOURS)
# job_run names written by fli.ops.scheduler (worker:<job id>, plus one row per worker start)
WORKER_PIPELINE, WORKER_START = "worker:pipeline", "worker:start"
STALL_AFTER = timedelta(hours=2)  # a pipeline run takes minutes; a hung one blocks all later runs
SILENT_AFTER = timedelta(hours=2)  # four missed 30-minute pipeline runs
COMPLETED = ("ok", "partial")  # partial: a stage failed or the LLM budget stopped (reported on their own)
UNFINISHED = ("running", "interrupted")  # interrupted: closed when a killed worker's successor started


@dataclass(frozen=True)
class Condition:
    kind: str
    severity: str
    subject: str
    message: str
    active: bool = True  # False: happened but already recovered (recorded as resolved)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def _hours(td: timedelta) -> str:
    return f"{td.total_seconds() / 3600:.1f} h"


def _named(ids: list[str]) -> str:
    more = len(ids) - NAMED_SOURCES
    return ", ".join(ids[:NAMED_SOURCES]) + (f" and {more} more" if more > 0 else "")


# --------------------------------------------------------------------------------- checks
def check_sources(s: Session, now: datetime, settings: Settings | None = None) -> list[Condition]:
    """Failing (>= 3 consecutive errors), stale and never-run sources. Stale = no new item within the
    SLO, or (complementary, so a frozen feed is caught on day one rather than one SLO after the first
    backfill stamped it fresh) its newest *published* item is older than the SLO. Never run: see
    check_never_run."""
    newest = dict(s.execute(select(Document.source_id, func.max(Document.published_at))
                            .where(Document.published_at <= now + timedelta(days=1)).group_by(Document.source_id)).all())
    out, never = [], []
    for src in s.scalars(select(Source).where(Source.enabled.is_(True)).order_by(Source.id)):
        if (src.consecutive_failures or 0) >= FAILING_AFTER:
            out.append(Condition("source_failing", "crit", src.id, f"{src.consecutive_failures} consecutive failures; "
                                 f"last error: {(src.last_error or 'unknown')[:300]}"))
        if src.last_run_at is None:
            if src.created_at is not None and now - src.created_at > _cadences(src):
                never.append(src)
            continue
        slo, slo_days = timedelta(hours=src.freshness_slo_hours), src.freshness_slo_hours / 24
        ref = src.last_new_item_at or src.created_at
        published = newest.get(src.id)
        if ref is not None and now - ref > slo:
            out.append(Condition("source_stale", "warn", src.id,
                                 f"no new items for {(now - ref).total_seconds() / 86400:.1f} days (SLO {slo_days:.0f} days)"
                                 + ("" if src.last_new_item_at else "; it has never produced an item")))
        elif published is not None and now - published > slo:
            out.append(Condition("source_stale", "warn", src.id,
                                 f"newest published item is {(now - published).total_seconds() / 86400:.1f} days old "
                                 f"(SLO {slo_days:.0f} days): the source is quiet or its feed is frozen"))
    return out + check_never_run(s, now, never, settings or get_settings())


def _cadences(src: Source) -> timedelta:
    return timedelta(minutes=NEVER_RUN_CADENCES * (src.cadence_minutes or 0))


def check_never_run(s: Session, now: datetime, sources: list[Source], settings: Settings) -> list[Condition]:
    """Enabled sources with no run (last_run_at NULL) two cadences after they were registered: the
    runner skips them without a trace, so a whole source class can be silently absent. One warn
    event per cause, listing its sources, so 34 X accounts behind one missing token are one finding:

      setting:<NAMES>   a setting the adapter requires is unset (the event names it)
      adapter:<name>    no collector is registered under the adapter name
      unexplained       collector and settings are present and ingestion is running here (a run in
                        the last 24 h) and has been for two of the source's cadences, yet it never
                        ran (a hanging collector, or records that fail to persist and roll it back)

    Without a cause, a database where nothing ingests now (an imported snapshot, a stopped worker)
    is not flagged source by source: the summary counts those sources, worker_silent and the web
    report the stop."""
    if not sources:
        return []
    from fli.ingest.records import COLLECTORS  # lazy: the collector modules are only needed here
    from fli.ingest.runner import load_collectors

    load_collectors()
    active_since, last_active = s.execute(
        select(func.min(JobRun.started_at), func.max(JobRun.started_at))
        .where(or_(JobRun.job.like("ingest:%"), JobRun.job == WORKER_PIPELINE), JobRun.started_at >= now - ACTIVE_WINDOW)).one()
    ingesting = last_active is not None and now - last_active <= INGEST_RECENT
    groups: dict[str, list[Source]] = {}
    for src in sources:
        col = COLLECTORS.get(src.adapter)
        if col is None:
            key = f"adapter:{src.adapter}"
        elif missing := [r.upper() for r in col.requires if not getattr(settings, r, None)]:
            key = f"setting:{','.join(missing)}"
        elif ingesting and now - max(src.created_at, active_since) > _cadences(src):
            key = "unexplained"
        else:
            continue
        groups.setdefault(key, []).append(src)
    out = []
    for key, srcs in sorted(groups.items()):
        ids, adapters = sorted(x.id for x in srcs), ", ".join(sorted({x.adapter for x in srcs}))
        n = f"{len(ids)} enabled source{'s have' if len(ids) > 1 else ' has'} never run"
        cause, _, what = key.partition(":")
        if cause == "setting":
            names = what.split(",")
            one = len(names) == 1
            msg = (f"{n}: {' and '.join(names)} {'is' if one else 'are'} not set (required by adapter {adapters}): "
                   f"{_named(ids)}. Set {'it' if one else 'them'} in .env, or disable the source(s) in config/")
        elif cause == "adapter":
            msg = (f"{n}: no collector is registered for adapter '{what}' (a typo in config/, or its collector "
                   f"module failed to import; see the worker log): {_named(ids)}")
        else:
            msg = (f"{n}, although ingestion has been running here since {active_since:%Y-%m-%d %H:%M} UTC and "
                   f"the collector and settings are present (adapter {adapters}): {_named(ids)}. A hanging "
                   "collector or records that fail to persist leave no trace: try `fli ingest -s <id>`")
        out.append(Condition("source_never_run", "warn", key, msg))
    return out


def check_parsers(s: Session, now: datetime) -> list[Condition]:
    """A source that normally returns items but whose last few successful runs returned nothing:
    the feed moved or the page layout changed (a 'successful' silent failure)."""
    out = []
    for src in s.scalars(select(Source).where(Source.enabled.is_(True), Source.last_success_at.is_not(None))):
        runs = s.scalars(select(JobRun).where(JobRun.job == f"ingest:{src.id}", JobRun.status == "ok")
                         .order_by(JobRun.started_at.desc(), JobRun.id.desc()).limit(PARSER_RECENT + PARSER_HISTORY)).all()
        recent, prior = runs[:PARSER_RECENT], runs[PARSER_RECENT:]
        if len(recent) < PARSER_RECENT or not prior or any((r.items_in or 0) > 0 for r in recent):
            continue
        typical = statistics.median((r.items_in or 0) for r in prior)
        if typical > 0:
            out.append(Condition("parser_empty", "warn", src.id,
                                 f"last {PARSER_RECENT} successful runs returned 0 items, a typical run returns "
                                 f"{typical:g}: the feed URL or page layout may have changed"))
    return out


def check_jobs(s: Session, now: datetime) -> list[Condition]:
    """Pipeline stages and worker jobs that errored in the last 24 h: warn while failing, crit after
    two failures in a row, recorded as a resolved 'info' event when the job has since recovered."""
    runs = s.scalars(select(JobRun).where(or_(*[JobRun.job.like(f"{p}%") for p in JOB_PREFIXES]),
                                          JobRun.started_at >= now - JOB_WINDOW)
                     .order_by(JobRun.job, JobRun.started_at, JobRun.id)).all()
    by_job: dict[str, list[JobRun]] = {}
    for r in runs:
        by_job.setdefault(r.job, []).append(r)
    out = []
    for job, rs in by_job.items():
        errors = [r for r in rs if r.status == "error"]
        if not errors:
            continue
        trailing = 0
        for r in reversed([r for r in rs if r.status not in UNFINISHED]):
            if r.status != "error":
                break
            trailing += 1
        last = errors[-1]
        detail = f"{len(errors)} failed run(s) in 24 h, last at {last.started_at:%Y-%m-%d %H:%M} UTC: {(last.error or 'unknown')[:300]}"
        if trailing:
            out.append(Condition("job_failed", "crit" if trailing >= 2 else "warn", job, detail))
            continue
        seen = s.scalar(select(OpsEvent.id).where(OpsEvent.kind == "job_failed", OpsEvent.subject == job,
                                                  OpsEvent.created_at >= last.started_at).limit(1))
        if seen is None:  # failed and recovered between two health runs: keep a record
            out.append(Condition("job_failed", "info", job, detail + " (recovered)", active=False))
    return out


def check_worker(s: Session, now: datetime) -> list[Condition]:
    """The scheduler's pipeline job (job_run 'worker:pipeline'), once a worker has run here in the
    last 7 days (a snapshot-only instance without a worker is not flagged):

      job_stalled (crit)    its newest run has been 'running' for over 2 h: hung, or the worker died
                            mid-run. With max_instances=1 a hung run silently blocks every later one.
      worker_silent (warn)  no run has completed (ok, or partial: a failed stage or the budget, which
                            job_failed and budget report) in 2 h. Quiet while a run is in progress,
                            while job_failed already reports the newest run, and for 2 h after a
                            worker (re)start, so a fresh worker can finish its first run.

    Rows a killed worker left 'running' are closed as 'interrupted' when the next worker starts, and
    only the newest run counts, so an old orphan never reads as a hung run."""
    last_seen = s.scalar(select(func.max(JobRun.started_at)).where(JobRun.job.like("worker:%"),
                                                                    JobRun.started_at >= now - ACTIVE_WINDOW))
    if last_seen is None:
        return []
    latest = s.scalars(select(JobRun).where(JobRun.job == WORKER_PIPELINE)
                       .order_by(JobRun.started_at.desc(), JobRun.id.desc()).limit(1)).first()
    if latest is not None and latest.status == "running":
        age = now - latest.started_at
        if age <= STALL_AFTER:
            return []  # in progress
        return [Condition("job_stalled", "crit", WORKER_PIPELINE,
                          f"pipeline run #{latest.id} has been running for {_hours(age)} (since "
                          f"{latest.started_at:%Y-%m-%d %H:%M} UTC): it is hung, or the worker died mid-run. No later "
                          "run can start meanwhile, so nothing is ingested, scored or alerted")]
    if latest is not None and latest.status == "error" and now - latest.started_at <= JOB_WINDOW:
        return []  # job_failed reports the failing pipeline
    started = s.scalar(select(func.max(JobRun.started_at)).where(JobRun.job == WORKER_START))
    if started is not None and now - started <= SILENT_AFTER:
        return []
    done = s.scalar(select(func.max(func.coalesce(JobRun.finished_at, JobRun.started_at)))
                    .where(JobRun.job == WORKER_PIPELINE, JobRun.status.in_(COMPLETED)))
    if done is not None and now - done <= SILENT_AFTER:
        return []
    last = f"the last one at {done:%Y-%m-%d %H:%M} UTC" if done else "none ever has"
    return [Condition("worker_silent", "warn", WORKER_PIPELINE,
                      f"no pipeline run has completed in {_hours(SILENT_AFTER)} ({last}); the worker was last seen "
                      f"{last_seen:%Y-%m-%d %H:%M} UTC. It may be down or no longer scheduling the pipeline, so "
                      "nothing is ingested or alerted")]


def check_models(models_dir: Path | None = None) -> list[Condition]:
    """A deployed scoring model file that cannot be read halts scoring for its audience (rather than
    silently falling back to the unvalidated prior): crit until the file is fixed or removed. Run it
    outside an open write transaction: load_model records the same event (same dedup key) in a
    session of its own. Imported lazily, because scoring pulls in scikit-learn."""
    try:
        from fli.analyze import scoring

        problems = scoring.model_file_problems(models_dir)
        return [Condition(scoring.OPS_KIND, "crit", scoring.ops_subject(p.audience), str(p)[:2000]) for p in problems]
    except Exception:  # noqa: BLE001 - a broken scoring import must not stop the other checks
        log.exception("could not check the scoring model files")
        return []


def budget_status(s: Session, now: datetime, settings: Settings) -> dict:
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    total = float(s.scalar(select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0))) or 0.0)
    today = float(s.scalar(select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0))
                           .where(LLMCall.created_at >= midnight)) or 0.0)
    api = {p: float(c or 0.0) for p, c in s.execute(select(ApiUsage.provider, func.sum(ApiUsage.cost_usd))
                                                     .group_by(ApiUsage.provider))}
    x_spent = sum(v for k, v in api.items() if (k or "").lower() in X_PROVIDERS)

    def pct(v: float, b: float) -> float | None:
        return round(v / b, 4) if b else None

    return {"llm_spent_usd": round(total, 4), "llm_today_usd": round(today, 4),
            "llm_budget_total_usd": settings.llm_budget_total_usd, "llm_budget_daily_usd": settings.llm_budget_daily_usd,
            "llm_total_pct": pct(total, settings.llm_budget_total_usd),
            "llm_daily_pct": pct(today, settings.llm_budget_daily_usd),
            "x_spent_usd": round(x_spent, 4), "x_budget_total_usd": settings.x_budget_total_usd,
            "x_total_pct": pct(x_spent, settings.x_budget_total_usd),
            "metered_apis_usd": {k: round(v, 4) for k, v in sorted(api.items())}}


def check_budgets(s: Session, now: datetime, budget: dict) -> list[Condition]:
    out = []
    for subject, key, label in (("llm_total", "llm_total_pct", "LLM total"), ("llm_daily", "llm_daily_pct", "LLM daily"),
                                ("x_total", "x_total_pct", "X API total")):
        if (frac := budget[key]) is not None and frac > BUDGET_WARN:
            out.append(Condition("budget", "warn", subject, f"{label} spend is at {frac:.0%} of its budget"))
    # 'llm_budget' (crit) is also opened by fli.pipeline when the gateway refuses a call; keep it open
    # while spend stays in the danger zone, resolve it once the daily budget resets.
    refused_today = s.scalar(select(OpsEvent.id).where(OpsEvent.dedup_key == f"budget:llm_budget:{now:%Y-%m-%d}"))
    near = max(budget["llm_total_pct"] or 0.0, budget["llm_daily_pct"] or 0.0)
    if near >= BUDGET_STOP or (refused_today is not None and near > BUDGET_WARN):
        out.append(Condition("budget", "crit", "llm_budget",
                             f"LLM budget exhausted (spent ${budget['llm_spent_usd']:.2f} total, "
                             f"${budget['llm_today_usd']:.2f} today): model stages pause until it resets"))
    return out


# --------------------------------------------------------------------------------- events
def reconcile(s: Session, conditions: list[Condition], now: datetime) -> tuple[int, int]:
    """Upsert today's events, resolve cleared ones. Returns (created, resolved)."""
    day = f"{now:%Y-%m-%d}"
    live: set[tuple[str, str]] = set()
    created = 0
    for c in conditions:
        key = f"{c.kind}:{c.subject}:{day}"[:240]
        ev = s.scalar(select(OpsEvent).where(OpsEvent.dedup_key == key))
        if ev is None:
            ev = OpsEvent(kind=c.kind, severity=c.severity, subject=c.subject[:200], message=c.message,
                          dedup_key=key, delivered={}, created_at=now)
            s.add(ev)
            created += 1
        else:
            renotify = SEVERITY_RANK[c.severity] > SEVERITY_RANK.get(ev.severity, 0) or (
                c.active and ev.resolved_at is not None)
            if renotify:  # escalated or re-opened: announce again
                ev.delivered = {k: v for k, v in (ev.delivered or {}).items() if not k.startswith("slack")}
            ev.severity = max(ev.severity, c.severity, key=lambda x: SEVERITY_RANK.get(x, 0))
            ev.message = c.message
        if c.active:
            ev.resolved_at = None
            live.add((c.kind, c.subject))
        elif ev.resolved_at is None:
            ev.resolved_at = now
    s.flush()
    resolved = 0
    for ev in s.scalars(select(OpsEvent).where(OpsEvent.resolved_at.is_(None))):
        if (ev.kind, ev.subject) in live and ev.dedup_key.endswith(day):
            continue
        ev.resolved_at = now
        if (ev.kind, ev.subject) in live:
            ev.delivered = {**(ev.delivered or {}), "superseded_by": f"{ev.kind}:{ev.subject}:{day}"}
        else:
            resolved += 1
    return created, resolved


def _event_dict(ev: OpsEvent) -> dict:
    return {"id": ev.id, "kind": ev.kind, "severity": ev.severity, "subject": ev.subject, "message": ev.message,
            "since": _iso(ev.created_at), "resolved_at": _iso(ev.resolved_at)}


def summarize(s: Session, now: datetime, budget: dict, created: int, resolved: int) -> dict:
    open_events = sorted(s.scalars(select(OpsEvent).where(OpsEvent.resolved_at.is_(None))),
                         key=lambda e: (-SEVERITY_RANK.get(e.severity, 0), e.created_at))
    recent = s.scalars(select(OpsEvent).where(OpsEvent.resolved_at.is_not(None),
                                              OpsEvent.resolved_at >= now - timedelta(hours=24))
                       .order_by(OpsEvent.resolved_at.desc()).limit(20)).all()
    sources = s.scalars(select(Source)).all()
    enabled = [x for x in sources if x.enabled]
    latest_ids = select(func.max(JobRun.id)).where(or_(*[JobRun.job.like(f"{p}%") for p in JOB_PREFIXES])).group_by(JobRun.job)
    jobs = {r.job: {"status": r.status, "started_at": _iso(r.started_at), "finished_at": _iso(r.finished_at),
                    "error": (r.error or "")[:300] or None}
            for r in s.scalars(select(JobRun).where(JobRun.id.in_(latest_ids)).order_by(JobRun.job))}
    worst = max((SEVERITY_RANK.get(e.severity, 0) for e in open_events), default=0)
    return {
        "status": {2: "crit", 1: "warn"}.get(worst, "ok"),
        "checked_at": _iso(now),
        "open_events": [_event_dict(e) for e in open_events],
        "recently_resolved": [_event_dict(e) for e in recent],
        "new_events": created, "resolved_events": resolved,
        "sources": {"total": len(sources), "enabled": len(enabled),
                    "never_run": sum(1 for x in enabled if x.last_run_at is None),
                    "failing": sum(1 for x in enabled if (x.consecutive_failures or 0) >= FAILING_AFTER),
                    "stale": sum(1 for e in open_events if e.kind == "source_stale"),
                    "last_new_item_at": _iso(max((x.last_new_item_at for x in enabled if x.last_new_item_at), default=None))},
        "jobs": jobs,
        "budget": budget,
    }


def _notify(now: datetime, settings: Settings) -> dict:
    """Slack for open critical events not yet announced; Healthchecks heartbeat or /fail."""
    out: dict = {"slack": None, "healthchecks": None}
    with session_scope() as s:
        crit = s.scalars(select(OpsEvent).where(OpsEvent.resolved_at.is_(None), OpsEvent.severity == "crit")
                         .order_by(OpsEvent.created_at)).all()
        pending = [e for e in crit if not (e.delivered or {}).get("slack")
                   and int((e.delivered or {}).get("slack_attempts", 0)) < SLACK_MAX_ATTEMPTS]
        message = notify.ops_message(pending) if pending else None
        pending_ids, n_crit = [e.id for e in pending], len(crit)
        crit_text = "; ".join(f"{e.kind} {e.subject}" for e in crit)[:2000]
    if message and settings.slack_webhook_url:
        res = notify.post_slack(message)
        out["slack"] = res.ok
        with session_scope() as s:
            for ev in s.scalars(select(OpsEvent).where(OpsEvent.id.in_(pending_ids))):
                d = dict(ev.delivered or {})
                d["slack_attempts"] = int(d.get("slack_attempts", 0)) + 1
                d.update({"slack": _iso(now)} if res.ok else {"slack_error": res.error})
                ev.delivered = d
    if settings.healthchecks_ping_base:
        out["healthchecks"] = notify.ping_healthcheck(HC_SLUG, "fail" if n_crit else "",
                                                      body=f"{n_crit} open critical event(s): {crit_text}" if n_crit else "ok")
    return out


def run_health_checks(now: datetime | None = None, deliver: bool = True) -> dict:
    """Run every check, reconcile OpsEvents, notify. The summary feeds /status and /api/health."""
    now, settings = now or utcnow(), get_settings()
    models = check_models()  # before the session below: load_model writes its own event row
    with session_scope() as s:
        budget = budget_status(s, now, settings)
        conditions = [*check_sources(s, now, settings), *check_parsers(s, now), *check_jobs(s, now),
                      *check_worker(s, now), *models, *check_budgets(s, now, budget)]
        created, resolved = reconcile(s, conditions, now)
    notified = _notify(now, settings) if deliver else {}
    with session_scope() as s:
        summary = summarize(s, now, budget, created, resolved)
    summary["notified"] = notified
    log.info("health %s: %d open event(s), %d new, %d resolved", summary["status"], len(summary["open_events"]),
             created, resolved)
    return summary
