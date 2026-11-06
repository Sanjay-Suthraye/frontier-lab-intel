"""Pipeline orchestration.

Stages run in data-flow order. Every stage is an idempotent function that processes whatever is
pending (documents with status=new, insights without embeddings, ...), so the scheduler can call
them repeatedly and a crash mid-run loses nothing - the next run picks up where it stopped.

Stage contract: `def run_<x>(limit: int | None = None, **kwargs) -> dict` returning a small JSON
summary. Stages that call the LLM also accept `use_batch: bool` (Message Batches, 50% cheaper).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable

from fli.db.models import JobRun, OpsEvent
from fli.db.session import session_scope
from fli.llm.client import BudgetExceeded
from fli.timeutil import utcnow

log = logging.getLogger("fli.pipeline")

STAGES: list[tuple[str, str]] = [
    ("ingest", "fli.ingest.runner:run_ingest"),
    ("resolve_entities", "fli.register.entity_resolution:run_entity_resolution"),
    ("near_dedup", "fli.analyze.dedup:run_near_dedup"),
    ("triage", "fli.extract.triage:run_triage"),
    ("extract", "fli.extract.extraction:run_extraction"),
    ("embed", "fli.analyze.embed:run_embed"),
    ("cluster", "fli.analyze.cluster:run_clustering"),
    ("novelty", "fli.analyze.novelty:run_novelty"),
    ("exposure", "fli.analyze.exposure:run_exposure"),
    ("score", "fli.analyze.scoring:run_scoring"),
    ("lens", "fli.deliver.lens:run_lens"),
    ("alerts", "fli.deliver.alerts:run_alerts"),
]
STAGE_MAP = dict(STAGES)
LLM_STAGES = {"triage", "extract", "lens"}  # skipped (not fatal) once the LLM budget is exhausted


def resolve(path: str) -> Callable[..., dict]:
    module, fn = path.split(":")
    return getattr(importlib.import_module(module), fn)


def run_stage(name: str, **kwargs) -> dict:
    fn = resolve(STAGE_MAP[name])
    with session_scope() as s:
        run = JobRun(job=f"stage:{name}", details={"kwargs": {k: str(v) for k, v in kwargs.items()}})
        s.add(run)
        s.flush()
        run_id = run.id
    status, error, result = "ok", None, {}
    try:
        result = fn(**kwargs) or {}
        return result
    except BudgetExceeded as exc:
        status, error = "partial", str(exc)
        _ops_event("budget", "crit", "llm_budget", str(exc))
        raise
    except Exception as exc:
        status, error = "error", f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        with session_scope() as s:
            run = s.get(JobRun, run_id)
            run.status, run.error, run.finished_at = status, error, utcnow()
            if isinstance(result, dict):
                run.details = {**(run.details or {}), "result": _jsonable(result)}


def run_pipeline(stages: list[str] | None = None, stop_on_error: bool = False,
                 stage_kwargs: dict[str, dict] | None = None) -> dict:
    """Run the named stages (default: all) in order. Budget exhaustion stops the run cleanly."""
    names = stages or [n for n, _ in STAGES]
    out: dict[str, dict] = {}
    llm_blocked = False
    for name in names:
        if llm_blocked and name in LLM_STAGES:
            out[name] = {"status": "skipped_budget"}
            continue
        try:
            out[name] = run_stage(name, **((stage_kwargs or {}).get(name, {})))
        except BudgetExceeded as exc:
            # degrade, don't stop: deterministic stages (cluster, score, rule-based alerts) still run
            out[name] = {"status": "budget_exceeded", "error": str(exc)}
            log.error("LLM budget exhausted at %s; skipping remaining LLM stages: %s", name, exc)
            llm_blocked = True
        except Exception as exc:  # keep going so one broken stage does not starve the others
            log.exception("stage %s failed", name)
            out[name] = {"status": "error", "error": str(exc)}
            if stop_on_error:
                break
    return out


def _ops_event(kind: str, severity: str, subject: str, message: str) -> None:
    key = f"{kind}:{subject}:{utcnow():%Y-%m-%d}"
    with session_scope() as s:
        if s.query(OpsEvent).filter_by(dedup_key=key).first() is None:
            s.add(OpsEvent(kind=kind, severity=severity, subject=subject, message=message, dedup_key=key))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in list(obj.items())[:200]}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj[:200]]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return str(obj)
