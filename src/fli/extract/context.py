"""Shared plumbing for the LLM stages (triage, extraction, lens).

* gateway access and a sync->async bridge (stages are sync; the gateway is async)
* prompt rendering: `{{placeholders}}` in src/fli/prompts/*.md are filled from the taxonomy and the
  register, so those lists have one source of truth (code/config), and the rendered text is what the
  prompt version hashes
* `TemplatedPrompt`: the prompt version hashes a *template* schema in which per-item enums
  (candidate people, candidate tickers) are replaced by a marker. Otherwise every document would
  get its own prompt version. Response-cache keys stay sound because each item's allowed values
  are written into its user message (which the key hashes) and stored output is filtered to them,
  whatever superset enum the schema carried.
* prompt-cache economics: `call_many` warms the cache with one call before fanning out, and
  `PrefixTracker` says when the schema (part of the cached prefix) changed.
* register context (labs, people candidates) and the untrusted-document view
* the per-document failure ledger (attempts -> dead letter)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from fli.db.models import Document, Identity, Lab, Person
from fli.llm.prompts import Prompt, load_prompt
from fli.taxonomy import EVENT_TYPES
from fli.textutil import truncate

log = logging.getLogger("fli.extract")

MAX_ATTEMPTS = 3  # failed LLM attempts per stage before a document is dead-lettered


# ------------------------------------------------------------------------------- runtime
def make_gateway():
    """Factory (monkeypatched in tests). Imported lazily: constructing it needs an API key."""
    from fli.llm.client import LLMGateway

    return LLMGateway()


def run_coro(coro):
    """asyncio.run, or run in a worker thread when called from inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def call_many(gw, calls: list[dict], warm: bool = False) -> list:
    """gateway.structured_many, except that with `warm` the first call runs alone. Measured: the
    cached prefix (system prompt + output schema) is ~3,500-3,900 tokens, and concurrent calls that
    start before a cache entry exists each pay the cache-write rate; one call first lets the rest
    read it (Haiku: $0.125/M write vs $0.01/M read)."""
    if warm and len(calls) > 1:
        first = await gw.structured(**calls[0])
        return [first, *await gw.structured_many(calls[1:])]
    return await gw.structured_many(calls) if calls else []


class PrefixTracker:
    """The output schema is part of the cached prefix, so a call needs warming only when the
    schema differs from the previous group's."""

    def __init__(self) -> None:
        self._last: str | None = None

    def changed(self, schema: dict) -> bool:
        key = json.dumps(schema, sort_keys=True)
        changed, self._last = key != self._last, key
        return changed


def chunked[T](items: Sequence[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), max(1, size)):
        yield list(items[i:i + size])


# ------------------------------------------------------------------------------- prompts
@dataclass(frozen=True)
class TemplatedPrompt(Prompt):
    template_schema: dict = field(default_factory=dict, hash=False)

    def version_for(self, schema: dict | None = None) -> str:
        return Prompt.version_for(self, self.template_schema)


def render_prompt(name: str, template_schema: dict | None = None, **blocks: str) -> Prompt:
    base = load_prompt(name)
    system = base.system
    for key, value in blocks.items():
        system = system.replace("{{" + key + "}}", value)
    if "{{" in system:
        log.warning("prompt %s has unfilled placeholders", name)
    if template_schema is None:
        return dataclasses.replace(base, system=system)
    return TemplatedPrompt(name=base.name, version=base.version, system=system, path=base.path,
                           template_schema=template_schema)


def event_types_block() -> str:
    return "\n".join(f"- {key}: {desc}" for key, desc in EVENT_TYPES.items())


# ------------------------------------------------------------------------------- register
@dataclass(frozen=True)
class LabInfo:
    id: str
    name: str
    kind: str
    tier: int
    description: str = ""


def load_labs(session: Session) -> list[LabInfo]:
    rows = session.scalars(select(Lab).order_by(Lab.id)).all()
    return [LabInfo(r.id, r.name, r.kind, r.tier, (r.description or "")[:140]) for r in rows]


def labs_block(labs: Sequence[LabInfo]) -> str:
    if not labs:
        return "- (register empty)"
    return "\n".join(f"- {lab.id}: {lab.name} ({lab.kind}, tier {lab.tier}). {lab.description}".rstrip()
                     for lab in labs)


def fold(text: str | None) -> str:
    """Lowercase, accents stripped, whitespace collapsed (name/handle matching)."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower())


def _find_token(haystack: str, needle: str) -> int:
    """First occurrence of needle not glued to surrounding letters/digits, else -1."""
    start = 0
    while (pos := haystack.find(needle, start)) >= 0:
        before = haystack[pos - 1] if pos > 0 else " "
        end = pos + len(needle)
        after = haystack[end] if end < len(haystack) else " "
        if not (before.isalnum() and needle[0].isalnum()) and not (after.isalnum() or after == "_"):
            return pos
        start = pos + 1
    return -1


def _handle_needles(platform: str, handle: str) -> list[str]:
    h = fold(handle).strip().lstrip("@")
    if len(h) < 3:
        return []
    if platform == "x":
        return [f"@{h}", f"x.com/{h}", f"twitter.com/{h}"]
    if platform == "bluesky":
        return [f"@{h}", f"bsky.app/profile/{h}"]
    if platform == "github":
        return [f"github.com/{h}"]
    if platform == "hf":
        return [f"huggingface.co/{h}"]
    if platform == "arxiv_name" and "," in h:
        last, first = (x.strip() for x in h.split(",", 1))
        return [f"{first} {last}", f"{last}, {first}"] if first and last else []
    return []


@dataclass(frozen=True)
class PersonRef:
    id: str
    name: str
    lab_id: str | None = None
    title: str | None = None


class PeopleIndex:
    """Finds tracked people whose name or handle appears in a text: the candidate list (and enum)
    the extractor may attribute an insight to. Recall over precision - the model decides."""

    def __init__(self, people: Iterable[PersonRef], identities: Iterable[tuple[str, str, str]] = ()):
        self.people = {p.id: p for p in people}
        self._needles: list[tuple[str, str]] = []
        for p in self.people.values():
            name = fold(p.name).strip()
            if " " in name or len(name) >= 6:
                self._needles.append((name, p.id))
        for person_id, platform, handle in identities:
            if person_id in self.people:
                self._needles += [(n, person_id) for n in _handle_needles(platform, handle)]

    @classmethod
    def load(cls, session: Session) -> PeopleIndex:
        people = [PersonRef(*row) for row in session.execute(
            select(Person.id, Person.name, Person.lab_id, Person.title))]
        idents = session.execute(
            select(Identity.person_id, Identity.platform, Identity.handle)
            .where(Identity.person_id.is_not(None), Identity.status != "rejected")).all()
        return cls(people, [tuple(r) for r in idents])

    def candidates(self, text: str, always: Iterable[str | None] = (), limit: int = 25) -> list[PersonRef]:
        folded = fold(text)
        first: dict[str, int] = {pid: -1 for pid in always if pid in self.people}
        for needle, pid in self._needles:
            if needle in folded and (pos := _find_token(folded, needle)) >= 0:
                first[pid] = min(pos, first.get(pid, pos))
        ordered = sorted(first, key=lambda pid: (first[pid], pid))
        return [self.people[pid] for pid in ordered[:limit]]


def people_block(people: Sequence[PersonRef]) -> str:
    if not people:
        return "- (none: return person_ids [])"
    return "\n".join(f"- {p.id}: {p.name}" + (f" ({p.title})" if p.title else "")
                     + (f" [lab: {p.lab_id}]" if p.lab_id else "") for p in people)


# ------------------------------------------------------------------------------- documents
_FENCE = re.compile(r"<\s*(/?)\s*(document|item)", re.IGNORECASE)


def fence(text: str) -> str:
    """Neutralise tag look-alikes so untrusted text cannot close our <document> block."""
    return _FENCE.sub(lambda m: f"<\u200b{m.group(1)}{m.group(2)}", text or "")


def compact_meta(meta: dict[str, Any] | None, max_chars: int = 500) -> str:
    """Scalar facts and short lists from collector metadata (stars, metrics, authors ...)."""
    parts: list[str] = []
    for key in sorted(meta or {}):
        value = (meta or {})[key]
        if key.startswith("_"):
            continue
        if isinstance(value, bool | int | float) or (isinstance(value, str) and 0 < len(value) <= 120):
            parts.append(f"{key}={value}")
        elif isinstance(value, list) and value and all(isinstance(v, str | int | float) for v in value):
            more = f" (+{len(value) - 8} more)" if len(value) > 8 else ""
            parts.append(f"{key}=" + ", ".join(str(v)[:60] for v in value[:8]) + more)
        elif isinstance(value, dict):
            parts += [f"{key}.{k}={v}" for k, v in list(value.items())[:6]
                      if isinstance(v, bool | int | float) or (isinstance(v, str) and len(v) <= 60)]
    return "; ".join(parts)[:max_chars]


def is_headline_only(doc: Document, source_kind: str | None) -> bool:
    return doc.source_class == "press" or source_kind == "press"


def shown_text(doc: Document, max_chars: int, headline_only: bool) -> str:
    return "" if headline_only else truncate(doc.content_text or "", max_chars)


def document_view(doc: Document, source_kind: str | None, labs: dict[str, LabInfo], *,
                  max_chars: int, headline_only: bool) -> str:
    """Trusted metadata in <item>; everything that came from the source inside <document>."""
    lab = labs.get(doc.lab_id or "")
    published = doc.published_at.strftime("%Y-%m-%d %H:%M UTC") if doc.published_at else "unknown"
    header = [
        f"source: {doc.source_id} | kind: {source_kind or 'unknown'} | class: {doc.source_class}",
        f"lab (from source): {lab.id + ' (' + lab.name + ')' if lab else 'none'}"
        + (f" | person (from source): {doc.person_id}" if doc.person_id else ""),
        f"published: {published}" + (" | headline-only press item" if headline_only else ""),
        f"url: {doc.url}",
    ]
    body = ["TITLE: " + (doc.title or "(untitled)")]
    if doc.author:
        body.append("AUTHOR: " + doc.author[:300])
    if meta := compact_meta(doc.meta):
        body.append("META: " + meta)
    text = shown_text(doc, max_chars, headline_only)
    body.append("TEXT:\n" + (text or "(no body text; judge the title and metadata)"))
    return ("<item>\n" + "\n".join(header) + "\n</item>\n<document>\n" + fence("\n".join(body))
            + "\n</document>")


def verification_text(doc: Document) -> str:
    """What evidence quotes are checked against: the stored title and full text."""
    return f"{doc.title or ''}\n{doc.content_text or ''}"


# ------------------------------------------------------------------------------- failure ledger
def mark_failure(doc: Document, stage: str, error: str | None) -> str:
    """Count a failed attempt; dead-letter (status 'failed') after MAX_ATTEMPTS. Returns outcome."""
    doc.attempts = (doc.attempts or 0) + 1
    doc.error = f"{stage}: {error or 'unknown error'}"[:2000]
    if doc.attempts >= MAX_ATTEMPTS:
        doc.status = "failed"
        log.warning("document %s dead-lettered after %d %s attempts: %s", doc.id, doc.attempts, stage, error)
        return "failed"
    return "retry"


def mark_done(doc: Document, status: str) -> None:
    """Stage succeeded: attempts count consecutive failures within one stage, so reset them."""
    doc.status, doc.attempts, doc.error = status, 0, None
