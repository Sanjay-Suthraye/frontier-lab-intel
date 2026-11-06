"""Worker: one long-running process that runs the whole system on a schedule.

    every 30 min   ingest (per-source cadence applies inside run_ingest), then pipeline stages
                   near_dedup ... alerts
    every 15 min   health checks (+ Healthchecks.io heartbeat)
    Monday 07:00   weekly digests (settings.timezone: Berlin morning, before the European open)

Each job is wrapped: a job_run row ('worker:<name>') that the health check reads, Healthchecks
start/success/fail pings, and exceptions logged instead of killing the scheduler. max_instances=1
and coalesce=True mean a slow run is never stacked and missed runs collapse into one (so a hung run
blocks the next ones: fli.ops.health reports it as job_stalled). SIGTERM lets running jobs finish;
every stage is idempotent, so even a hard kill only loses time. A starting worker writes a
'worker:start' row and closes the rows a killed predecessor left 'running' as 'interrupted'.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from fli.config import Settings, get_settings
from fli.db.models import JobRun
from fli.db.session import session_scope
from fli.deliver.notify import ping_healthcheck
from fli.timeutil import utcnow

log = logging.getLogger("fli.ops.scheduler")

PIPELINE_EVERY_MIN = 30
HEALTH_EVERY_MIN = 15
DIGEST_CRON = {"day_of_week": "mon", "hour": 7, "minute": 0}
START_JOB = "worker:start"  # job_run written once per worker start (fli.ops.health: restart grace)
INTERRUPTED = "interrupted: the worker restarted before this run finished (hard kill, crash or redeploy)"


# --------------------------------------------------------------------------------- jobs
def job_pipeline() -> dict:
    """Ingest due sources, then every downstream stage. ok=False when ingest or a stage failed.

    Ingest runs as the pipeline's 'ingest' stage, so a crash leaves a 'stage:ingest' error row (a
    job_failed ops event) rather than only a log line, and never starves the downstream stages."""
    from fli.pipeline import STAGES, run_pipeline, run_stage

    try:
        ingest: dict | None = run_stage("ingest")
    except Exception:  # noqa: BLE001 - recorded as stage:ingest; downstream stages still run on stored data
        log.exception("ingest failed; running downstream stages on what is already stored")
        ingest = None
    stages = run_pipeline(stages=[name for name, _ in STAGES if name != "ingest"])
    failed = (["ingest"] if ingest is None else []) + [
        n for n, r in stages.items() if isinstance(r, dict) and r.get("status") in ("error", "budget_exceeded")]
    per_source = [r for r in (ingest or {}).values() if isinstance(r, dict)]
    return {"ok": not failed, "failed_stages": failed, "sources_run": len(per_source),
            "source_errors": sum(1 for r in per_source if r.get("status") == "error")}


def job_health() -> dict:
    from fli.ops.health import run_health_checks

    summary = run_health_checks()  # pings Healthchecks itself (/fail while crit events are open)
    return {"ok": True, "status": summary["status"], "open_events": len(summary["open_events"])}


def job_digest() -> dict:
    from fli.deliver.digest import run_digest

    out = run_digest(audience="all")
    return {"ok": True, "reports": [f"{r['audience']}:{r['period_start']}:{r['status']}" for r in out["reports"]]}


def job_contributors() -> dict:
    from fli.register.contributors import run_contributor_scoring

    out = run_contributor_scoring()
    return {"ok": True, "people": out.get("people", out.get("scored"))}


# (job id, function, Healthchecks slug, whether the wrapper sends the success/fail ping)
JOBS: dict[str, tuple[Callable[[], dict], str, bool]] = {
    "pipeline": (job_pipeline, "fli-pipeline", True),
    "health": (job_health, "fli-health", False),
    "digest": (job_digest, "fli-digest", True),
    "contributors": (job_contributors, "fli-contributors", True),
}


def _record(run_id: int | None, status: str, details: dict | None = None, error: str | None = None) -> int | None:
    """Open (run_id None) or close a worker job_run row; never lets ledger trouble kill the job."""
    try:
        with session_scope() as s:
            if run_id is None:
                run = JobRun(job=f"worker:{details['job']}", status="running", details={})
                s.add(run)
                s.flush()
                return run.id
            run = s.get(JobRun, run_id)
            run.status, run.error, run.finished_at = status, error, utcnow()
            run.details = json.loads(json.dumps(details or {}, default=str))
            return run_id
    except Exception:  # database unavailable: the Healthchecks ping still reports the failure
        log.exception("could not record worker job run")
        return run_id


def record_start(settings: Settings) -> list[int]:
    """Ledger a worker (re)start: close the 'running' worker rows a previous process left behind (a
    hard kill mid-run never closes its row) as 'interrupted', and add a 'worker:start' row. Returns
    the ids it closed. Never raises: the ledger must not stop the worker from starting."""
    now = utcnow()
    try:
        with session_scope() as s:
            orphans = list(s.scalars(select(JobRun).where(JobRun.job.like("worker:%"), JobRun.status == "running")))
            for run in orphans:
                run.status, run.error, run.finished_at = "interrupted", INTERRUPTED, now
            closed = [r.id for r in orphans]
            s.add(JobRun(job=START_JOB, status="ok", started_at=now, finished_at=now,
                         details={"interrupted_runs": closed, "pid": os.getpid(), "timezone": settings.timezone}))
        if closed:
            log.warning("closed %d worker run(s) left running by a previous process: %s", len(closed), closed)
        return closed
    except Exception:  # noqa: BLE001 - database unavailable: the jobs report it themselves
        log.exception("could not record the worker start")
        return []


def run_job(name: str) -> dict:
    """Run one scheduled job with ledger row and Healthchecks start/success/fail pings."""
    fn, slug, ping_result = JOBS[name]
    ping_healthcheck(slug, "start")
    run_id = _record(None, "running", {"job": name})
    try:
        result = fn() or {}
    except Exception as exc:
        log.exception("worker job %s failed", name)
        error = f"{exc.__class__.__name__}: {exc}"[:2000]
        _record(run_id, "error", error=error)
        ping_healthcheck(slug, "fail", body=error)
        return {"ok": False, "error": error}
    ok = bool(result.get("ok", True))
    _record(run_id, "ok" if ok else "partial", details=result)
    if ping_result:
        ping_healthcheck(slug, "" if ok else "fail", body=json.dumps(result, default=str)[:2000])
    log.info("worker job %s finished: %s", name, result)
    return result


# --------------------------------------------------------------------------------- scheduler
def build_scheduler(settings: Settings | None = None, run_now: bool = True) -> BlockingScheduler:
    settings = settings or get_settings()
    tz = ZoneInfo(settings.timezone)
    policy = {"max_instances": 1, "coalesce": True, "misfire_grace_time": 15 * 60, "replace_existing": True}
    sched = BlockingScheduler(timezone=tz, executors={"default": ThreadPoolExecutor(max_workers=3)})
    first: dict[str, Any] = {"next_run_time": datetime.now(tz)} if run_now else {}
    sched.add_job(run_job, IntervalTrigger(minutes=PIPELINE_EVERY_MIN, timezone=tz), args=["pipeline"],
                  id="pipeline", name="ingest + pipeline", **policy, **first)
    sched.add_job(run_job, IntervalTrigger(minutes=HEALTH_EVERY_MIN, timezone=tz), args=["health"],
                  id="health", name="health checks", **policy, **first)
    sched.add_job(run_job, CronTrigger(timezone=tz, **DIGEST_CRON), args=["digest"], id="digest",
                  name="weekly digests", **policy)
    sched.add_job(run_job, CronTrigger(timezone=tz, day_of_week="sun", hour=22), args=["contributors"],
                  id="contributors", name="weekly contributor scores", **policy)
    return sched


def install_signal_handlers(sched: BlockingScheduler) -> Callable[[int, Any], None]:
    """First SIGTERM/SIGINT: stop scheduling and let running jobs finish. Second: exit now."""
    state = {"stopping": False}

    def handle(signum: int, _frame: Any) -> None:
        if state["stopping"]:
            log.warning("second %s: exiting immediately", signal.Signals(signum).name)
            os._exit(1)
        state["stopping"] = True
        log.info("%s received: finishing running jobs, then exiting", signal.Signals(signum).name)
        # shutdown(wait=True) blocks until jobs finish; run it off the signal handler's frame
        threading.Thread(target=sched.shutdown, kwargs={"wait": True}, name="fli-shutdown", daemon=True).start()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    return handle


def _init_sentry(settings: Settings) -> None:
    if not settings.sentry_dsn:
        return
    try:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.0, environment=os.environ.get("FLI_ENV", "prod"),
                        include_local_variables=False, send_default_pii=False)  # never ship settings/keys
    except Exception as exc:  # monitoring must never stop the worker
        log.warning("sentry init failed: %s", exc.__class__.__name__)


def run_worker(run_now: bool = True) -> None:
    """Entry point of `fli worker` (and the worker container)."""
    settings = get_settings()
    _init_sentry(settings)
    sched = build_scheduler(settings, run_now=run_now)
    install_signal_handlers(sched)
    record_start(settings)
    log.info("worker starting (timezone %s): %s", settings.timezone,
             "; ".join(f"{j.id} [{j.trigger}]" for j in sched.get_jobs()))
    sched.start()
    log.info("worker stopped")
