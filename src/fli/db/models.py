"""Database schema.

Portable across SQLite (local dev, tests, the committed demo snapshot) and Postgres (production):
JSON columns become JSONB on Postgres, embeddings are float32 bytes (corpus is small enough that
brute-force cosine in numpy beats operating pgvector), and all datetimes are naive UTC.

Layers, in data-flow order:
  register   lab, person, affiliation, identity, source
  raw        document, job_posting, http_cache
  extracted  insight, artifact
  analysed   development, ticker, exposure_link, score, lens
  delivered  alert, report
  learning   label
  ops        job_run, llm_call, response_cache, api_usage, ops_event
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from fli.timeutil import utcnow

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


# ---------------------------------------------------------------------------------------------
# Register: who and what we track
# ---------------------------------------------------------------------------------------------


class Lab(TimestampMixin, Base):
    """A frontier lab or alumni startup. First-class entity: most high-value signal is official."""

    __tablename__ = "lab"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # slug, e.g. "anthropic"
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(32))  # frontier_lab | big_tech_lab | open_weight_lab | alumni_startup
    tier: Mapped[int] = mapped_column(Integer, default=2)  # 1 = always material, 3 = watch
    parent: Mapped[str | None] = mapped_column(String(200))
    homepage: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    # [{ticker, relation, note}] - curated in config/labs.yaml, used by exposure mapping
    public_exposure: Mapped[list] = mapped_column(JSONType, default=list)
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)

    people: Mapped[list[Person]] = relationship(back_populates="lab")
    sources: Mapped[list[Source]] = relationship(back_populates="lab")


class Person(TimestampMixin, Base):
    """A tracked individual. `tier` and `why_key` make the inclusion decision auditable."""

    __tablename__ = "person"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)  # slug, e.g. "jeff-dean"
    name: Mapped[str] = mapped_column(String(200))
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"))  # current affiliation (register view)
    title: Mapped[str | None] = mapped_column(String(300))
    tier: Mapped[str] = mapped_column(String(1), default="C")  # A | B | C
    why_key: Mapped[str | None] = mapped_column(Text)
    inclusion_rule: Mapped[str | None] = mapped_column(String(64))  # seed_founder | flagship_contributor | ...
    status: Mapped[str] = mapped_column(String(32), default="active")  # active | departed | founded_startup | unknown
    contributor_score: Mapped[float | None] = mapped_column(Float)
    contributor_features: Mapped[dict] = mapped_column(JSONType, default=dict)
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)

    lab: Mapped[Lab | None] = relationship(back_populates="people")
    identities: Mapped[list[Identity]] = relationship(back_populates="person")
    affiliations: Mapped[list[Affiliation]] = relationship(back_populates="person")


class Affiliation(Base):
    """Dated affiliation history. A new row with a different lab is a talent-move signal."""

    __tablename__ = "affiliation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("person.id"), index=True)
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"))
    org_name: Mapped[str | None] = mapped_column(String(200))  # when the org is not in the register
    role: Mapped[str | None] = mapped_column(String(300))
    start_date: Mapped[datetime | None] = mapped_column(DateTime)
    end_date: Mapped[datetime | None] = mapped_column(DateTime)
    source_url: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    person: Mapped[Person] = relationship(back_populates="affiliations")


class Identity(Base):
    """One account/name of a person (or lab) on one platform, with how we know it is theirs.

    method: seed (curated) | self_asserted (one profile links the other) | fuzzy | llm_adjudicated | manual
    status: confirmed | candidate | rejected. Only confirmed identities attribute documents.
    """

    __tablename__ = "identity"
    __table_args__ = (UniqueConstraint("platform", "handle", name="uq_identity_platform_handle"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[str | None] = mapped_column(ForeignKey("person.id"), index=True)
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"), index=True)
    platform: Mapped[str] = mapped_column(String(32))  # x | github | hf | bluesky | arxiv_name | s2 | openalex | orcid | homepage | openreview | scholar
    handle: Mapped[str] = mapped_column(String(300))  # normalised: lowercase handle, or "lastname, firstname" for names
    url: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    method: Mapped[str] = mapped_column(String(32), default="seed")
    status: Mapped[str] = mapped_column(String(16), default="confirmed")
    evidence: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)  # e.g. platform user id (X numeric id)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    person: Mapped[Person | None] = relationship(back_populates="identities")


class Source(TimestampMixin, Base):
    """A pollable channel (feed, index page, account, org, board, query) handled by one adapter."""

    __tablename__ = "source"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # e.g. "anthropic:news"
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"), index=True)
    person_id: Mapped[str | None] = mapped_column(ForeignKey("person.id"), index=True)
    adapter: Mapped[str] = mapped_column(String(64))  # registry key, see fli.ingest.base.ADAPTERS
    kind: Mapped[str] = mapped_column(String(48))  # official_blog | research_blog | release_notes | model_cards | arxiv | hf_papers | github_org | hf_org | x_account | bluesky_account | job_board | press | sec | model_index
    source_class: Mapped[str] = mapped_column(String(24))  # official | personal | press | filing | aggregator | market_data
    url: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    cadence_minutes: Mapped[int] = mapped_column(Integer, default=360)
    freshness_slo_hours: Mapped[int] = mapped_column(Integer, default=24 * 14)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_new_item_at: Mapped[datetime | None] = mapped_column(DateTime)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)

    lab: Mapped[Lab | None] = relationship(back_populates="sources")


# ---------------------------------------------------------------------------------------------
# Raw layer
# ---------------------------------------------------------------------------------------------


class Document(TimestampMixin, Base):
    """One ingested item (post, paper, release, repo, article, filing, hiring digest).

    `published_at`, `url` and attribution come from source metadata - never from an LLM - so
    citations cannot be hallucinated. Pipeline state lives in `status`:
      new -> duplicate | skipped (triage: noise) | relevant -> extracted | failed
    """

    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("source_id", "natural_key", name="uq_document_source_key"),
        Index("ix_document_status", "status"),
        Index("ix_document_published", "published_at"),
        Index("ix_document_hash", "content_hash"),
        Index("ix_document_canonical", "canonical_url"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("source.id"), index=True)
    natural_key: Mapped[str] = mapped_column(String(512))  # arxiv id, post id, repo@tag, url ...
    url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    content_text: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str] = mapped_column(String(64))
    minhash: Mapped[bytes | None] = mapped_column(LargeBinary)
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"), index=True)
    person_id: Mapped[str | None] = mapped_column(ForeignKey("person.id"), index=True)
    source_class: Mapped[str] = mapped_column(String(24))
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)  # stars, upvotes, metrics, authors ...
    raw_path: Mapped[str | None] = mapped_column(Text)
    duplicate_of_id: Mapped[int | None] = mapped_column(ForeignKey("document.id"))
    status: Mapped[str] = mapped_column(String(16), default="new")
    triage: Mapped[dict | None] = mapped_column(JSONType)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    source: Mapped[Source] = relationship()
    insights: Mapped[list[Insight]] = relationship(back_populates="document")


class JobPosting(Base):
    """Snapshot rows from labs' public job boards; weekly diffs become 'hiring signal' documents."""

    __tablename__ = "job_posting"
    __table_args__ = (UniqueConstraint("lab_id", "board", "posting_id", name="uq_job_posting"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lab_id: Mapped[str] = mapped_column(ForeignKey("lab.id"), index=True)
    board: Mapped[str] = mapped_column(String(32))  # greenhouse | ashby | lever
    posting_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(Text)
    department: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64))  # our taxonomy: hardware, robotics, devices, ...
    url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    removed_at: Mapped[datetime | None] = mapped_column(DateTime)
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)


class HttpCache(Base):
    """Conditional-GET validators so polling unchanged feeds costs nothing."""

    __tablename__ = "http_cache"

    url: Mapped[str] = mapped_column(Text, primary_key=True)
    etag: Mapped[str | None] = mapped_column(Text)
    last_modified: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------------------------------------
# Extracted layer
# ---------------------------------------------------------------------------------------------


class Insight(Base):
    """A structured, attributed claim extracted from one document, with verbatim evidence."""

    __tablename__ = "insight"
    __table_args__ = (Index("ix_insight_published", "published_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"), index=True)
    development_id: Mapped[int | None] = mapped_column(ForeignKey("development.id"), index=True)
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"), index=True)
    person_ids: Mapped[list] = mapped_column(JSONType, default=list)
    event_type: Mapped[str] = mapped_column(String(48))  # see fli.taxonomy.EVENT_TYPES
    title: Mapped[str] = mapped_column(String(400))
    claim: Mapped[str] = mapped_column(Text)
    evidence_quotes: Mapped[list] = mapped_column(JSONType, default=list)
    quotes_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification: Mapped[dict] = mapped_column(JSONType, default=dict)
    artifacts: Mapped[list] = mapped_column(JSONType, default=list)  # model/product/repo names
    entities: Mapped[dict] = mapped_column(JSONType, default=dict)  # orgs/companies mentioned
    magnitude: Mapped[int | None] = mapped_column(Integer)  # extractor's 1-5 rubric judgement (a feature, not the score)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    is_official: Mapped[bool] = mapped_column(Boolean, default=False)
    is_rumor: Mapped[bool] = mapped_column(Boolean, default=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    novelty_entity: Mapped[float | None] = mapped_column(Float)
    novelty_global: Mapped[float | None] = mapped_column(Float)
    nearest_prior_insight_id: Mapped[int | None] = mapped_column(Integer)
    first_seen_artifacts: Mapped[list] = mapped_column(JSONType, default=list)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    prompt_version: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped[Document] = relationship(back_populates="insights")
    development: Mapped[Development | None] = relationship(back_populates="insights")


class Artifact(Base):
    """Registry of named things (models, products, repos, API model ids) for first-seen detection."""

    __tablename__ = "artifact"
    __table_args__ = (UniqueConstraint("name", "kind", name="uq_artifact_name_kind"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(300))  # normalised
    display_name: Mapped[str] = mapped_column(String(300))
    kind: Mapped[str] = mapped_column(String(32))  # model | product | repo | api_model_id | chip | dataset | other
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime)
    first_document_id: Mapped[int | None] = mapped_column(ForeignKey("document.id"))


# ---------------------------------------------------------------------------------------------
# Analysed layer: developments (cross-source events) are the unit of scoring and alerting
# ---------------------------------------------------------------------------------------------


class Development(TimestampMixin, Base):
    """One real-world event, clustered from insights across sources (blog + X + repo + press)."""

    __tablename__ = "development"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(400))
    event_type: Mapped[str] = mapped_column(String(48))
    lab_id: Mapped[str | None] = mapped_column(ForeignKey("lab.id"), index=True)
    person_ids: Mapped[list] = mapped_column(JSONType, default=list)
    artifacts: Mapped[list] = mapped_column(JSONType, default=list)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    insight_count: Mapped[int] = mapped_column(Integer, default=0)
    source_count: Mapped[int] = mapped_column(Integer, default=0)  # distinct sources
    source_classes: Mapped[list] = mapped_column(JSONType, default=list)
    has_official: Mapped[bool] = mapped_column(Boolean, default=False)
    has_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)
    summary: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="open")

    insights: Mapped[list[Insight]] = relationship(back_populates="development")


class Ticker(Base):
    """Public companies the exposure map can point at (closed list: the LLM may not invent tickers)."""

    __tablename__ = "ticker"

    symbol: Mapped[str] = mapped_column(String(24), primary_key=True)
    company: Mapped[str] = mapped_column(String(200))
    sector: Mapped[str | None] = mapped_column(String(100))
    themes: Mapped[list] = mapped_column(JSONType, default=list)
    bit_weight: Mapped[float | None] = mapped_column(Float)  # % of BIT Global Technology Leaders (dated in config)
    bit_funds: Mapped[list] = mapped_column(JSONType, default=list)
    notes: Mapped[str | None] = mapped_column(Text)


class ExposureLink(Base):
    __tablename__ = "exposure_link"
    __table_args__ = (UniqueConstraint("development_id", "ticker", name="uq_exposure_dev_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    development_id: Mapped[int] = mapped_column(ForeignKey("development.id"), index=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("ticker.symbol"))
    theme: Mapped[str | None] = mapped_column(String(64))
    direction: Mapped[str] = mapped_column(String(16), default="unclear")  # positive | negative | mixed | unclear
    rationale: Mapped[str | None] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(16), default="rule")  # rule | llm
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Score(Base):
    """Per-audience materiality probability with the features and contributions that produced it."""

    __tablename__ = "score"
    __table_args__ = (UniqueConstraint("development_id", "audience", "model_version", name="uq_score"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    development_id: Mapped[int] = mapped_column(ForeignKey("development.id"), index=True)
    audience: Mapped[str] = mapped_column(String(16))  # investment | ai_team
    p_material: Mapped[float] = mapped_column(Float)
    features: Mapped[dict] = mapped_column(JSONType, default=dict)
    contributions: Mapped[dict] = mapped_column(JSONType, default=dict)  # "why flagged"
    model_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Lens(Base):
    """Audience-specific last mile on top of the shared core: what it means and what to do."""

    __tablename__ = "lens"
    __table_args__ = (UniqueConstraint("development_id", "audience", name="uq_lens"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    development_id: Mapped[int] = mapped_column(ForeignKey("development.id"), index=True)
    audience: Mapped[str] = mapped_column(String(16))
    headline: Mapped[str] = mapped_column(String(400))
    so_what: Mapped[str] = mapped_column(Text)
    what_to_do: Mapped[str] = mapped_column(Text)
    tickers: Mapped[list] = mapped_column(JSONType, default=list)  # [{symbol, direction, why}]
    direction: Mapped[str | None] = mapped_column(String(16))
    horizon: Mapped[str | None] = mapped_column(String(32))
    action: Mapped[str | None] = mapped_column(String(32))  # investment: watch|review_position|... ; ai_team: adopt|trial|assess|hold
    artifacts: Mapped[list] = mapped_column(JSONType, default=list)  # ai_team: models/repos/tools to try
    checks: Mapped[dict] = mapped_column(JSONType, default=dict)  # deterministic faithfulness checks on the text
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    model: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------------------------------------
# Delivered layer
# ---------------------------------------------------------------------------------------------


class Alert(Base):
    __tablename__ = "alert"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    development_id: Mapped[int] = mapped_column(ForeignKey("development.id"), index=True)
    audience: Mapped[str] = mapped_column(String(16))
    severity: Mapped[str] = mapped_column(String(4))  # P1 | P2 | P3
    title: Mapped[str] = mapped_column(String(400))
    body_md: Mapped[str] = mapped_column(Text)
    rules: Mapped[list] = mapped_column(JSONType, default=list)  # which materiality rules fired
    dedup_key: Mapped[str] = mapped_column(String(200), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | acked | suppressed
    delivered: Mapped[dict] = mapped_column(JSONType, default=dict)
    acked_at: Mapped[datetime | None] = mapped_column(DateTime)
    acked_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class Report(Base):
    __tablename__ = "report"
    __table_args__ = (UniqueConstraint("audience", "period_start", name="uq_report_period"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audience: Mapped[str] = mapped_column(String(16))
    period_start: Mapped[datetime] = mapped_column(DateTime)
    period_end: Mapped[datetime] = mapped_column(DateTime)
    title: Mapped[str] = mapped_column(String(300))
    body_md: Mapped[str] = mapped_column(Text)
    items: Mapped[list] = mapped_column(JSONType, default=list)  # [{development_id, rank, p_material}]
    model: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------------------------
# Learning layer
# ---------------------------------------------------------------------------------------------


class Label(Base):
    """Ground truth for scoring. annotator = 'human:<name>' or 'llm:<model>' - never mixed silently."""

    __tablename__ = "label"
    __table_args__ = (UniqueConstraint("development_id", "audience", "annotator", name="uq_label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    development_id: Mapped[int] = mapped_column(ForeignKey("development.id"), index=True)
    audience: Mapped[str] = mapped_column(String(16))
    label: Mapped[int] = mapped_column(Integer)  # 0 noise | 1 worth knowing | 2 material (alert-worthy)
    reason: Mapped[str | None] = mapped_column(Text)
    annotator: Mapped[str] = mapped_column(String(64))
    rubric_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------------------------
# Ops layer
# ---------------------------------------------------------------------------------------------


class JobRun(Base):
    __tablename__ = "job_run"
    __table_args__ = (Index("ix_job_run_job_started", "job", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job: Mapped[str] = mapped_column(String(128))  # "ingest:anthropic:news", "triage", ...
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | ok | partial | error
    items_in: Mapped[int] = mapped_column(Integer, default=0)
    items_new: Mapped[int] = mapped_column(Integer, default=0)
    items_failed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSONType, default=dict)


class LLMCall(Base):
    """Every model call: the source of truth for the cost-per-workflow table."""

    __tablename__ = "llm_call"
    __table_args__ = (Index("ix_llm_call_purpose", "purpose"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    purpose: Mapped[str] = mapped_column(String(32))  # triage | extract | lens | digest | er | judge | ...
    model: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))  # ok | cached | refusal | max_tokens | error | invalid
    error: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(String(128))
    batch_id: Mapped[str | None] = mapped_column(String(128))
    ref: Mapped[dict] = mapped_column(JSONType, default=dict)  # {"document_id": 12} etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ResponseCache(Base):
    """Model outputs keyed by (model, prompt version, input) hash.

    Sampling parameters are rejected by the current API version, so reproducibility comes from
    persisting outputs rather than from temperature=0.
    """

    __tablename__ = "response_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    purpose: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(64))
    response: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ApiUsage(Base):
    """Metered third-party API spend (X reads are pay-per-use)."""

    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    endpoint: Mapped[str] = mapped_column(String(200))
    units: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class OpsEvent(Base):
    """System-health alerts (stale source, failing job, budget) - separate from product alerts."""

    __tablename__ = "ops_event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))  # source_stale | source_failing | parser_empty | job_failed | budget
    severity: Mapped[str] = mapped_column(String(8))  # info | warn | crit
    subject: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    dedup_key: Mapped[str] = mapped_column(String(240), unique=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    delivered: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
