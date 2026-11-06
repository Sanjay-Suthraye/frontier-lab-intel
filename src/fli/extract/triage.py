"""Triage: a cheap relevance / noise gate in front of extraction.

Every document with status 'new' (oldest first) gets one Haiku call on a compact view (source
metadata + the first ~5,000 characters; press items are headline-only). The structured result is
stored on `document.triage` and decides the status:

    relevant and noise_reason == 'none'  -> 'relevant'  (goes to extraction)
    otherwise                            -> 'skipped'   (kept, browsable, never extracted)
    LLM failure                          -> attempts += 1, stays 'new'; 'failed' after 3 attempts

Work is processed in chunks committed one at a time, so a crash or a budget stop loses at most
one chunk (whose responses are in the gateway's response cache and cost nothing to replay).

Re-triage after a prompt change: `retriage_candidates` picks the skipped documents an older prompt
version may have misjudged (prompt v1 dropped deprecations, retirements, breaking and default
changes as routine changelog lines), and `run_retriage` puts them back to 'new' with triage
cleared, so the next triage run spends one cheap call on each of those and nothing else.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable

from sqlalchemy import func, null, select, update
from sqlalchemy.orm import Session

from fli.config import get_settings
from fli.db.models import Document, Source
from fli.db.session import session_scope
from fli.extract import context as ctx
from fli.extract.schemas import clean_triage, triage_schema
from fli.llm.client import LLMResult
from fli.llm.prompts import load_prompt
from fli.timeutil import utcnow

log = logging.getLogger("fli.extract.triage")

PURPOSE = "triage"
PROMPT = "triage"
MAX_CHARS = 5_000
MAX_TOKENS = 1_500
EFFORT = "low"
CHUNK = 32
BATCH_CHUNK = 2_000

# Re-triage: changelog-like sources, and the words that announce a breaking or lifecycle change.
RETRIAGE_SOURCE_KINDS = ("docs_changelog", "release_notes", "github_org")
BREAKING_HINTS = re.compile(
    r"deprecat|retir|sunset|discontinu|shut(?:s|ting)?[\s-]*down|end[\s-]of[\s-](?:life|support)"
    r"|breaking|default|retention",
    re.IGNORECASE,
)
_NO_BREAKING = re.compile(  # release-note templates: "## Breaking Changes  No known breaking changes."
    r"(?:#{1,6}\s*breaking\s+changes?\s*:?\s*)?\b(?:no|without)\s+(?:known\s+)?breaking\s+changes?\b"
    r"|breaking\s+changes?\s*:?\s*(?:none|n/a)\b",
    re.IGNORECASE,
)
_PROMPT_MAJOR = re.compile(r"@v(\d+)\+")


def pending_ids(session: Session, limit: int | None = None) -> list[int]:
    q = (select(Document.id).where(Document.status == "new")
         .order_by(func.coalesce(Document.published_at, Document.fetched_at), Document.id))
    if limit is not None:
        q = q.limit(limit)
    return list(session.scalars(q))


def build_inputs(session: Session, ids: list[int], labs: dict[str, ctx.LabInfo]) -> list[tuple[int, str]]:
    """(document id, user message) for each still-pending document, in the given order."""
    rows = session.execute(
        select(Document, Source.kind).join(Source, Document.source_id == Source.id).where(Document.id.in_(ids))
    ).all()
    by_id = {doc.id: (doc, kind) for doc, kind in rows if doc.status == "new"}
    out = []
    for did in ids:
        if did in by_id:
            doc, kind = by_id[did]
            view = ctx.document_view(doc, kind, labs, max_chars=MAX_CHARS,
                                     headline_only=ctx.is_headline_only(doc, kind))
            out.append((did, "Triage this item.\n\n" + view))
    return out


def apply_result(doc: Document, res: LLMResult | None, lab_ids: list[str]) -> str:
    """Store one triage result on its document. Returns the outcome used in the run summary."""
    if res is None or not res.ok:
        return ctx.mark_failure(doc, PURPOSE, (res.error or res.status) if res else "no result returned")
    try:
        tri = clean_triage(res.data, lab_ids)
    except (TypeError, ValueError) as exc:
        return ctx.mark_failure(doc, PURPOSE, str(exc))
    doc.triage = {**tri, "prompt_version": res.prompt_version, "model": res.model,
                  "triaged_at": utcnow().isoformat(timespec="seconds")}
    status = "relevant" if tri["relevant"] and tri["noise_reason"] == "none" else "skipped"
    ctx.mark_done(doc, status)
    return status


def _apply_all(results: dict[int, LLMResult | None], lab_ids: list[str], summary: Counter) -> None:
    with session_scope() as s:
        for did, res in results.items():
            doc = s.get(Document, did)
            if doc is None or doc.status != "new":  # handled by a concurrent run meanwhile
                continue
            summary[apply_result(doc, res, lab_ids)] += 1
            if res is not None:
                summary["cached"] += res.status == "cached"
                summary["cost_usd"] += res.cost_usd


async def _run_async(gw, ids: list[int], prompt, schema: dict, labs: dict, summary: Counter,
                     chunk_size: int) -> None:
    model = get_settings().model_triage
    lab_ids = sorted(labs)
    for n, chunk in enumerate(ctx.chunked(ids, chunk_size)):  # one static schema: warm once
        with session_scope() as s:
            items = build_inputs(s, chunk, labs)
        calls = [{"purpose": PURPOSE, "prompt": prompt, "user": user, "schema": schema, "model": model,
                  "max_tokens": MAX_TOKENS, "effort": EFFORT, "ref": {"document_id": did}} for did, user in items]
        results = await ctx.call_many(gw, calls, warm=n == 0)
        _apply_all({did: res for (did, _), res in zip(items, results, strict=True)}, lab_ids, summary)


def _run_batch(gw, ids: list[int], prompt, schema: dict, labs: dict, summary: Counter) -> None:
    model = get_settings().model_triage
    lab_ids = sorted(labs)
    for chunk in ctx.chunked(ids, BATCH_CHUNK):
        with session_scope() as s:
            items = build_inputs(s, chunk, labs)
        if not items:
            continue
        out = gw.structured_batch(
            purpose=PURPOSE, prompt=prompt, schema=schema, model=model, max_tokens=MAX_TOKENS, effort=EFFORT,
            items=[(f"doc-{did}", user, {"document_id": did}) for did, user in items])
        _apply_all({did: out.get(f"doc-{did}") for did, _ in items}, lab_ids, summary)


def run_triage(limit: int | None = None, use_batch: bool = False, gateway=None,
               chunk_size: int = CHUNK, **_: object) -> dict:
    """Triage pending documents. Idempotent: only status 'new' is read, and each result moves the
    document out of 'new' (or counts an attempt)."""
    with session_scope() as s:
        labs = {lab.id: lab for lab in ctx.load_labs(s)}
        ids = pending_ids(s, limit)
    summary: Counter = Counter()
    if not ids:
        return {"pending": 0}
    gw = gateway or ctx.make_gateway()
    schema = triage_schema(sorted(labs))
    prompt = ctx.render_prompt(PROMPT, event_types=ctx.event_types_block(), labs=ctx.labs_block(list(labs.values())))
    if use_batch:
        _run_batch(gw, ids, prompt, schema, labs, summary)
    else:
        ctx.run_coro(_run_async(gw, ids, prompt, schema, labs, summary, chunk_size))
    out = {"pending": len(ids), **{k: v for k, v in summary.items() if k != "cost_usd"},
           "cost_usd": round(summary["cost_usd"], 5), "prompt_version": prompt.version_for(schema)}
    log.info("triage: %s", out)
    return out


# ------------------------------------------------------------------------------- re-triage
def _prompt_major(version: str | None) -> int:
    """'triage@v2+1a2b3c4d' -> 2; 0 when missing or unparsable (treated as an old version)."""
    match = _PROMPT_MAJOR.search(version or "")
    return int(match.group(1)) if match else 0


def retriage_candidates(session: Session) -> list[int]:
    """Ids (oldest first) of skipped documents worth one more triage call under the current prompt:
    from a changelog, release-notes or GitHub source (GitHub: releases only; a new repository's
    README is no changelog), judged by an older prompt version (a re-run under the same version
    would only replay the cached answer), and whose title, text as triage reads it (the first
    MAX_CHARS) or stored triage reason matches BREAKING_HINTS ("no breaking changes" does not)."""
    current = load_prompt(PROMPT).version
    rows = session.execute(
        select(Document.id, Document.natural_key, Source.kind, Document.title, Document.content_text,
               Document.triage)
        .join(Source, Document.source_id == Source.id)
        .where(Document.status == "skipped", Source.kind.in_(RETRIAGE_SOURCE_KINDS))
        .order_by(func.coalesce(Document.published_at, Document.fetched_at), Document.id)
    )
    out: list[int] = []
    for did, natural_key, kind, title, text, tri in rows:
        tri = tri if isinstance(tri, dict) else {}
        if kind == "github_org" and not (natural_key or "").startswith("release:"):
            continue
        if _prompt_major(tri.get("prompt_version")) >= current:
            continue
        hay = "\n".join((title or "", (text or "")[:MAX_CHARS], str(tri.get("reason") or "")))
        if BREAKING_HINTS.search(_NO_BREAKING.sub(" ", hay)):
            out.append(did)
    return out


def run_retriage(ids: Iterable[int]) -> dict:
    """Put the given skipped documents back to 'new' with triage cleared (SQL NULL, as on a document
    never triaged) so the next triage run judges them again. Only 'skipped' documents change:
    relevant/extracted ones may own insights, duplicates point at their canonical copy, and dead
    letters need their failure fixed first. A repeated call ignores documents already back to
    'new'. The summary keeps the old noise reasons (the cleared triage is gone afterwards)."""
    wanted = sorted({int(i) for i in ids})
    reset: list[int] = []
    previous: Counter = Counter()
    with session_scope() as s:
        for chunk in ctx.chunked(wanted, 500):
            rows = s.execute(select(Document.id, Document.triage).order_by(Document.id)
                             .where(Document.id.in_(chunk), Document.status == "skipped")).all()
            if not rows:
                continue
            chunk_ids = [r.id for r in rows]
            s.execute(update(Document).where(Document.id.in_(chunk_ids), Document.status == "skipped")
                      .values(status="new", triage=null(), attempts=0, error=None))
            reset += chunk_ids
            previous.update(str((r.triage if isinstance(r.triage, dict) else {}).get("noise_reason") or "unknown")
                            for r in rows)
    out = {"requested": len(wanted), "reset": len(reset), "ignored": len(wanted) - len(reset),
           "previous_noise_reasons": dict(previous.most_common()), "ids": reset}
    log.info("re-triage: %s", {k: v for k, v in out.items() if k != "ids"})
    return out
