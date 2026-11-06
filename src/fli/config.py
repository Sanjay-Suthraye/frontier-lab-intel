"""Runtime settings.

Everything that differs between a laptop, CI and production lives here and is read from the
environment (or `.env`). Model routing is configuration, not code: the design doc explains each
choice, and `fli costs` shows what each choice cost.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- storage -----------------------------------------------------------------------------
    database_url: str = Field(default=f"sqlite:///{DATA_DIR / 'fli.sqlite'}", repr=False)
    raw_dir: Path = DATA_DIR / "raw"  # gzipped raw payload snapshots (citations stay reproducible)

    # --- credentials (all optional; adapters degrade gracefully when missing) -----------------
    anthropic_api_key: str | None = Field(default=None, repr=False)
    x_bearer_token: str | None = Field(default=None, repr=False)
    github_token: str | None = Field(default=None, repr=False)
    hf_token: str | None = Field(default=None, repr=False)
    openalex_api_key: str | None = Field(default=None, repr=False)
    s2_api_key: str | None = Field(default=None, repr=False)
    sec_user_agent: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)
    slack_webhook_url: str | None = Field(default=None, repr=False)
    healthchecks_ping_base: str | None = Field(default=None, repr=False)
    sentry_dsn: str | None = Field(default=None, repr=False)

    # --- web ---------------------------------------------------------------------------------
    public_base_url: str | None = None  # used for deep links in Slack alerts
    fli_basic_auth: str | None = Field(default=None, repr=False)  # "user:pass" protects write endpoints
    fli_auth_all: bool = False  # also protect every page and API read
    fli_read_only: bool = False  # refuse every write (labels, acks): a viewing copy (deploy/vercel)

    # --- model routing (see docs/DESIGN.md "Model choices") ------------------------------------
    model_triage: str = "claude-haiku-5-5"
    model_extract: str = "claude-haiku-5-5"
    model_er: str = "claude-haiku-5-5"
    model_lens: str = "claude-sonnet-5-5"
    model_digest: str = "claude-opus-5-5"
    # retry model for an unusable digest output: a resend to the same model repeats the failure
    # (seen live: Opus drafts the digest in its thinking, then emits an empty entry)
    model_digest_retry: str = "claude-sonnet-5-5"
    model_judge: str = "claude-opus-5-5"
    llm_concurrency: int = 8

    # --- budgets (hard stops enforced in code) -------------------------------------------------
    llm_budget_total_usd: float = 45.0
    llm_budget_daily_usd: float = 15.0
    x_budget_total_usd: float = 10.0

    # --- behaviour ---------------------------------------------------------------------------
    backfill_start: date = date(2026, 8, 1)
    timezone: str = "Europe/Berlin"
    user_agent: str = "FrontierLabIntel/0.1 (+research prototype; polite crawler)"
    http_timeout_s: float = 30.0
    max_doc_chars: int = 40_000  # content kept per document (enough for posts, abstracts, cards)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
