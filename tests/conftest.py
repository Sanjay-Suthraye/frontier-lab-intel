"""Shared fixtures: every test gets an isolated SQLite database and raw-snapshot directory."""

from __future__ import annotations

import pytest

from fli import config as fli_config
from fli.db import session as db_session


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    """Tests must not inherit the developer's .env secrets/auth (env vars win over .env)."""
    for var in ("FLI_BASIC_AUTH", "SLACK_WEBHOOK_URL", "HEALTHCHECKS_PING_BASE", "SENTRY_DSN"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("FLI_AUTH_ALL", "0")
    fli_config.get_settings.cache_clear()
    yield
    fli_config.get_settings.cache_clear()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Fresh database with all tables. Yields a SQLAlchemy session factory."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.sqlite'}")
    monkeypatch.setenv("RAW_DIR", str(tmp_path / "raw"))
    fli_config.get_settings.cache_clear()
    db_session.configure()
    db_session.init_db()
    yield db_session.get_session
    db_session.get_engine().dispose()
    fli_config.get_settings.cache_clear()


@pytest.fixture()
def settings(db):
    return fli_config.get_settings()
