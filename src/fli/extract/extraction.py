"""Extraction: turn a 'relevant' document into 0-3 structured, cited insights.

The model gets register context (lab ids and names in the system prompt; the tracked people whose
names or handles occur in this document as a candidate list) and the document. The person_ids
enum is the union of the candidate lists of the documents sent together (see union_groups), and
stored ids are filtered to the document's own list.

Code, not the model, sets everything a citation depends on: the document link, published_at (from
the document), is_official (source_class == 'official'), prompt version and model. Every evidence
quote is then verified against the stored text (fli.extract.verify); insights without a verified
quote are kept but flagged, and alerting ignores them.

Document status -> 'extracted' (also when nothing material was found). LLM failures count
attempts like triage and dead-letter the document after 3.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fli.config import get_settings
from fli.db.models import Document, Insight, Source
from fli.db.session import session_scope
from fli.extract import context as ctx
from fli.extract.schemas import clean_extraction, extraction_schema
from fli.extract.verify import verify_quotes
from fli.llm.client import LLMResult

log = logging.getLogger("fli.extract.extraction")

PURPOSE = "extract"
PROMPT = "extract"
MAX_CHARS = 24_000
MAX_TOKENS = 6_000
EFFORT = "medium"
CHUNK = 24
BATCH_CHUNK = 1_000
MAX_UNION = 100  # max size of the candidate-person enum shared by one group of calls


@dataclass
class Job:
    doc_id: int
    user: str
    person_ids: list[str]


def pending_ids(session: Session, limit: int | None = None) -> list[int]:
    q = (select(Document.id).where(Document.status == "relevant")
         .order_by(func.coalesce(Document.published_at, Document.fetched_at), Document.id))
    if limit is not None:
        q = q.limit(limit)
    return list(session.scalars(q))


def _meta_names(meta: dict | None) -> str:
    """Author lists in collector metadata (arXiv, papers) also count as 'appearing in the text'."""
    names: list[str] = []
    for key, value in (meta or {}).items():
        if "author" in key.lower() and isinstance(value, list):
            names += [v if isinstance(v, str) else str(v.get("name", "")) for v in value
                      if isinstance(v, str | dict)]
    return " ; ".join(names)


def build_jobs(session: Session, ids: list[int], labs: dict[str, ctx.LabInfo],
               people: ctx.PeopleIndex) -> list[Job]:
    rows = session.execute(
        select(Document, Source.kind).join(Source, Document.source_id == Source.id).where(Document.id.in_(ids))
    ).all()
    by_id = {doc.id: (doc, kind) for doc, kind in rows if doc.status == "relevant"}
    jobs = []
    for did in ids:
        if did not in by_id:
            continue
        doc, kind = by_id[did]
        headline_only = ctx.is_headline_only(doc, kind)
        scan = "\n".join([doc.title or "", ctx.shown_text(doc, MAX_CHARS, headline_only), doc.author or "",
                          _meta_names(doc.meta)])
        cands = people.candidates(scan, always=[doc.person_id])
        view = ctx.document_view(doc, kind, labs, max_chars=MAX_CHARS, headline_only=headline_only)
        user = ("Candidate people (the only ids allowed in person_ids):\n" + ctx.people_block(cands)
                + "\n\nExtract insights from this document.\n\n" + view)
        jobs.append(Job(did, user, [p.id for p in cands]))
    return jobs


def store_insights(session: Session, doc: Document, res: LLMResult | None, job: Job,
                   lab_ids: list[str]) -> tuple[str, int]:
    """Persist one extraction result. Returns (outcome for the run summary, insights stored)."""
    if res is None or not res.ok:
        return ctx.mark_failure(doc, PURPOSE, (res.error or res.status) if res else "no result returned"), 0
    try:
        no_material, insights = clean_extraction(res.data, lab_ids, job.person_ids)
    except (TypeError, ValueError) as exc:
        return ctx.mark_failure(doc, PURPOSE, str(exc)), 0
    existing = session.scalar(select(func.count(Insight.id)).where(Insight.document_id == doc.id)) or 0
    if existing:  # idempotency guard: never add a second set of insights to a document
        ctx.mark_done(doc, "extracted")
        return "already_extracted", 0
    source_text = ctx.verification_text(doc)
    n_verified = 0
    for ins in insights:
        ver = verify_quotes(ins["evidence_quotes"], source_text)
        n_verified += ver["status"] != "unverified"
        person_ids = ins["person_ids"]
        if doc.person_id and doc.source_class == "personal" and doc.person_id not in person_ids:
            person_ids = [doc.person_id, *person_ids]  # a personal post is attributed to its author
        session.add(Insight(
            document_id=doc.id, lab_id=ins["lab_id"] or doc.lab_id, person_ids=person_ids,
            event_type=ins["event_type"], title=ins["title"], claim=ins["claim"],
            evidence_quotes=ins["evidence_quotes"], quotes_verified=ver["all_verified"], verification=ver,
            artifacts=ins["artifacts"], entities={"orgs": ins["orgs"]} if ins["orgs"] else {},
            magnitude=ins["magnitude"], confidence=ins["confidence"], is_rumor=ins["is_rumor"],
            is_official=doc.source_class == "official", published_at=doc.published_at or doc.fetched_at,
            prompt_version=res.prompt_version, model=res.model,
        ))
    doc.triage = {**(doc.triage or {}), "extraction": {
        "no_material_insight": no_material, "insights": len(insights), "with_verified_quote": n_verified,
        "prompt_version": res.prompt_version}}
    ctx.mark_done(doc, "extracted")
    return ("extracted" if insights else "no_insight"), len(insights)


def _apply_all(jobs: list[Job], results: dict[int, LLMResult | None], lab_ids: list[str], summary: Counter) -> None:
    with session_scope() as s:
        for job in jobs:
            doc = s.get(Document, job.doc_id)
            if doc is None or doc.status != "relevant":
                continue
            res = results.get(job.doc_id)
            outcome, stored = store_insights(s, doc, res, job, lab_ids)
            summary[outcome] += 1
            summary["insights"] += stored
            if res is not None:
                summary["cached"] += res.status == "cached"
                summary["cost_usd"] += res.cost_usd


def _context(session: Session) -> tuple[dict[str, ctx.LabInfo], ctx.PeopleIndex]:
    return {lab.id: lab for lab in ctx.load_labs(session)}, ctx.PeopleIndex.load(session)


async def _run_async(gw, ids: list[int], prompt, summary: Counter, chunk_size: int) -> None:
    model = get_settings().model_extract
    prefix = ctx.PrefixTracker()
    for chunk in ctx.chunked(ids, chunk_size):
        with session_scope() as s:
            labs, people = _context(s)
            jobs = build_jobs(s, chunk, labs, people)
        lab_ids = sorted(labs)
        for group in union_groups(jobs):
            schema = extraction_schema(lab_ids, sorted({p for j in group for p in j.person_ids}))
            calls = [{"purpose": PURPOSE, "prompt": prompt, "user": j.user, "schema": schema, "model": model,
                      "max_tokens": MAX_TOKENS, "effort": EFFORT, "ref": {"document_id": j.doc_id}} for j in group]
            results = await ctx.call_many(gw, calls, warm=prefix.changed(schema))
            _apply_all(group, {j.doc_id: r for j, r in zip(group, results, strict=True)}, lab_ids, summary)


def union_groups(jobs: list[Job], max_union: int = MAX_UNION, max_items: int = BATCH_CHUNK) -> list[list[Job]]:
    """Calls in a group share one schema whose person enum is the union of the group's candidates.
    Why: the schema is part of the cached prompt prefix (per-document enums defeat prompt caching)
    and a batch takes one schema. Each document's own candidate list is in its user message and
    filters the stored person_ids, so attribution guarantees are unchanged."""
    groups: list[list[Job]] = []
    current: list[Job] = []
    union: set[str] = set()
    for job in jobs:
        grown = union | set(job.person_ids)
        if current and (len(grown) > max_union or len(current) >= max_items):
            groups.append(current)
            current, grown = [], set(job.person_ids)
        current.append(job)
        union = grown
    if current:
        groups.append(current)
    return groups


def _run_batch(gw, ids: list[int], prompt, summary: Counter) -> None:
    model = get_settings().model_extract
    for chunk in ctx.chunked(ids, BATCH_CHUNK):
        with session_scope() as s:
            labs, people = _context(s)
            jobs = build_jobs(s, chunk, labs, people)
        lab_ids = sorted(labs)
        for group in union_groups(jobs):
            union = sorted({p for j in group for p in j.person_ids})
            out = gw.structured_batch(
                purpose=PURPOSE, prompt=prompt, schema=extraction_schema(lab_ids, union), model=model,
                max_tokens=MAX_TOKENS, effort=EFFORT,
                items=[(f"doc-{j.doc_id}", j.user, {"document_id": j.doc_id}) for j in group])
            _apply_all(group, {j.doc_id: out.get(f"doc-{j.doc_id}") for j in group}, lab_ids, summary)


def run_extraction(limit: int | None = None, use_batch: bool = False, gateway=None,
                   chunk_size: int = CHUNK, **_: object) -> dict:
    """Extract insights from documents with status 'relevant'. Idempotent (status-driven, plus a
    guard that never stores a second set of insights for one document)."""
    with session_scope() as s:
        labs = ctx.load_labs(s)
        ids = pending_ids(s, limit)
    if not ids:
        return {"pending": 0}
    lab_ids = sorted(lab.id for lab in labs)
    prompt = ctx.render_prompt(PROMPT, template_schema=extraction_schema(lab_ids, None),
                               event_types=ctx.event_types_block(), labs=ctx.labs_block(labs))
    gw = gateway or ctx.make_gateway()
    summary: Counter = Counter()
    if use_batch:
        _run_batch(gw, ids, prompt, summary)
    else:
        ctx.run_coro(_run_async(gw, ids, prompt, summary, chunk_size))
    out = {"pending": len(ids), "insights": 0, **{k: v for k, v in summary.items() if k != "cost_usd"},
           "cost_usd": round(summary["cost_usd"], 5), "prompt_version": prompt.version_for()}
    log.info("extraction: %s", out)
    return out
