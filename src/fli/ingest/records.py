"""Collector contract.

A *collector* reads one kind of source through its official API, SDK or published feed and returns
`IngestRecord`s. Collectors never touch the database; the runner (`fli.ingest.runner`) owns
persistence, dedup, health bookkeeping and the job ledger.

Source policy: official APIs / SDKs and published feeds (RSS/Atom, sitemaps) only. No logged-in
scraping, no bypassing of access controls, and each provider's published rate limits apply.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from fli.config import Settings


class IngestRecord(BaseModel):
    """One published item. Dates/URLs/authors come from source metadata, never from an LLM."""

    natural_key: str = Field(min_length=1, max_length=512)  # stable id within the source
    url: str
    title: str | None = None
    content_text: str = ""  # cleaned plain text
    published_at: datetime | None = None  # naive UTC
    author: str | None = None
    lab_id: str | None = None  # attribution override (defaults to the source's lab)
    person_id: str | None = None  # attribution override (defaults to the source's person)
    source_class: str | None = None  # override (e.g. a personal post found via a lab source)
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SourceInfo:
    """Plain snapshot of a `Source` row handed to a collector."""

    id: str
    adapter: str
    kind: str
    source_class: str
    url: str | None
    config: dict[str, Any]
    lab_id: str | None = None
    person_id: str | None = None


@dataclass
class Window:
    since: datetime  # inclusive lower bound (naive UTC)
    until: datetime  # inclusive upper bound (naive UTC)
    max_items: int | None = None


class CollectorUnavailable(Exception):
    """Collector cannot run right now (missing credential, budget exhausted). Not a failure."""


CollectorFn = Callable[[SourceInfo, Window, Settings], list[IngestRecord]]


@dataclass
class Collector:
    name: str
    fn: CollectorFn
    requires: tuple[str, ...] = ()  # Settings attributes that must be set
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


COLLECTORS: dict[str, Collector] = {}


def collector(name: str, *, requires: tuple[str, ...] = (), description: str = ""):
    """Register a collector function under `name` (the `adapter` value in config/*.yaml)."""

    def wrap(fn: CollectorFn) -> CollectorFn:
        COLLECTORS[name] = Collector(name=name, fn=fn, requires=requires, description=description)
        return fn

    return wrap
