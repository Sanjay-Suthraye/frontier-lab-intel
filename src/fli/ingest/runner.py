"""Ingestion runner: decides which sources are due, runs their collectors, and persists results
idempotently. Every run leaves a `job_run` row and updates the source's health fields, which is
what the freshness monitor (fli.ops.health) reads."""

from __future__ import annotations

import gzip
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from fli.config import get_settings
from fli.db.models import Document, JobRun, Source
from fli.db.session import session_scope
from fli.ingest.records import COLLECTORS, CollectorUnavailable, IngestRecord, SourceInfo, Window
from fli.textutil import canonical_url, clean_text, content_hash, truncate
from fli.timeutil import to_naive_utc, utcnow

log = logging.getLogger("fli.ingest.runner")

OVERLAP = timedelta(days=2)  # re-read a little history each run; upserts make it free


def load_collectors() -> None:
    """Import collector modules so they register themselves. Missing modules are logged, not fatal."""
    import importlib

    for mod in (
        "fli.ingest.collectors.feeds",
        "fli.ingest.collectors.research",
        "fli.ingest.collectors.code",
        "fli.ingest.collectors.social",
        "fli.ingest.collectors.jobs",
        "fli.ingest.collectors.market",
        "fli.ingest.collectors.press",
        "fli.ingest.collectors.filings",
        "fli.ingest.collectors.register",
    ):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:
            if exc.name == mod:
                log.debug("collector module %s not present", mod)
            else:
                log.warning("collector module %s failed to import: %s", mod, exc)


def _is_due(src: Source, now: datetime) -> bool:
    if src.last_run_at is None:
        return True
    return now - src.last_run_at >= timedelta(minutes=src.cadence_minutes)


def _info(src: Source) -> SourceInfo:
    return SourceInfo(
        id=src.id,
        adapter=src.adapter,
        kind=src.kind,
        source_class=src.source_class,
        url=src.url,
        config=dict(src.config or {}),
        lab_id=src.lab_id,
        person_id=src.person_id,
    )


def _snapshot(source_id: str, rec: IngestRecord) -> str:
    """Gzip the record as received so a citation can be re-checked later even if the page changes."""
    settings = get_settings()
    folder = Path(settings.raw_dir) / source_id.replace(":", "__").replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(rec.natural_key.encode()).hexdigest()[:20] + ".json.gz"
    path = folder / name
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(rec.model_dump_json())
    return str(path.relative_to(Path(settings.raw_dir).parent))


def persist_records(session: Session, src: Source, records: list[IngestRecord]) -> tuple[int, int]:
    """Upsert records for one source. Returns (new, duplicates_of_other_sources)."""
    settings = get_settings()
    new = dupes = 0
    window_start = datetime.combine(settings.backfill_start, datetime.min.time())
    for rec in records:
        published = to_naive_utc(rec.published_at)
        if published is not None and published < window_start:
            continue  # outside the tracked window
        text = truncate(clean_text(rec.content_text), settings.max_doc_chars)
        canon = canonical_url(rec.url)
        chash = content_hash(f"{rec.title or ''}\n{text}")
        existing = session.scalar(
            select(Document).where(Document.source_id == src.id, Document.natural_key == rec.natural_key)
        )
        if existing is not None:
            # refresh volatile metadata (stars, likes, downloads) without re-triggering the pipeline
            existing.meta = {**(existing.meta or {}), **rec.meta}
            # a full article may arrive after a short feed summary was stored; take it while the
            # document has not been processed downstream yet
            if text and (not existing.content_text
                         or (existing.status == "new" and len(text) > 2 * len(existing.content_text))):
                existing.content_text, existing.content_hash = text, chash
            continue
        doc = Document(
            source_id=src.id,
            natural_key=rec.natural_key,
            url=rec.url,
            canonical_url=canon,
            title=(rec.title or "").strip() or None,
            author=rec.author,
            published_at=published,
            fetched_at=utcnow(),
            content_text=text,
            content_hash=chash,
            lab_id=rec.lab_id or src.lab_id,
            person_id=rec.person_id or src.person_id,
            source_class=rec.source_class or src.source_class,
            meta=rec.meta,
            status="new",
        )
        # exact cross-source duplicate: same canonical URL, or same substantial text
        twin = session.scalar(
            select(Document.id)
            .where(
                Document.source_id != src.id,
                Document.duplicate_of_id.is_(None),
                or_(Document.canonical_url == canon, (Document.content_hash == chash) & (len(text) > 280)),
            )
            .limit(1)
        )
        if twin is not None:
            doc.duplicate_of_id, doc.status = twin, "duplicate"
            dupes += 1
        try:
            doc.raw_path = _snapshot(src.id, rec)
        except OSError as exc:  # snapshots are best-effort
            log.warning("snapshot failed for %s: %s", rec.natural_key, exc)
        session.add(doc)
        session.flush()
        new += 1
    return new, dupes


def _collect(info: SourceInfo, window: Window) -> list[IngestRecord]:
    col = COLLECTORS[info.adapter]
    return col.fn(info, window, get_settings())


def run_ingest(
    source_ids: list[str] | None = None,
    since: datetime | None = None,
    force: bool = False,
    max_items: int | None = None,
    workers: int = 6,
) -> dict:
    """Run due (or selected) sources. Returns a per-source summary."""
    load_collectors()
    settings = get_settings()
    now = utcnow()
    with session_scope() as s:
        q = select(Source).where(Source.enabled.is_(True))
        if source_ids:
            q = q.where(Source.id.in_(source_ids))
        sources = [src for src in s.scalars(q) if force or source_ids or _is_due(src, now)]
        plan: list[tuple[SourceInfo, Window]] = []
        skipped: dict[str, str] = {}
        for src in sources:
            col = COLLECTORS.get(src.adapter)
            if col is None:
                skipped[src.id] = f"no collector '{src.adapter}'"
                continue
            missing = [r for r in col.requires if not getattr(settings, r, None)]
            if missing:
                skipped[src.id] = f"missing settings: {', '.join(missing)}"
                continue
            start = since or (src.last_success_at - OVERLAP if src.last_success_at else None)
            floor = datetime.combine(settings.backfill_start, datetime.min.time())
            start = max(start, floor) if start else floor
            plan.append((_info(src), Window(since=start, until=now, max_items=max_items)))

    results: dict[str, dict] = {sid: {"status": "skipped", "reason": why} for sid, why in skipped.items()}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_collect, info, win): info for info, win in plan}
        for fut in as_completed(futures):
            info = futures[fut]
            results[info.id] = _finish(info, fut)
    return results


def _finish(info: SourceInfo, fut) -> dict:
    started = utcnow()
    with session_scope() as s:
        src = s.get(Source, info.id)
        run = JobRun(job=f"ingest:{info.id}", started_at=started)
        s.add(run)
        src.last_run_at = utcnow()
        try:
            records = fut.result()
        except CollectorUnavailable as exc:
            run.status, run.error, run.finished_at = "skipped", str(exc), utcnow()
            return {"status": "skipped", "reason": str(exc)}
        except Exception as exc:  # collector failure: record, count, surface via health checks
            log.exception("collector %s failed", info.id)
            src.consecutive_failures = (src.consecutive_failures or 0) + 1
            src.last_error = f"{exc.__class__.__name__}: {exc}"[:2000]
            run.status, run.error, run.finished_at = "error", src.last_error, utcnow()
            return {"status": "error", "error": src.last_error}
        new, dupes = persist_records(s, src, records)
        src.consecutive_failures = 0
        src.last_error = None
        src.last_success_at = utcnow()
        if new:
            src.last_new_item_at = utcnow()
        run.items_in, run.items_new = len(records), new
        run.details = {"duplicates": dupes}
        run.status, run.finished_at = "ok", utcnow()
        return {"status": "ok", "fetched": len(records), "new": new, "duplicates": dupes}
