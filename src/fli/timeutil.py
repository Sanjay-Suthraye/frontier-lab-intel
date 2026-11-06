"""Time helpers.

Convention: every datetime stored in the database is *naive UTC*. SQLite drops tzinfo and mixing
aware/naive values raises at comparison time, so we normalise at the edges (adapters, UI).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from dateutil import parser as dtparser


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def to_naive_utc(value: datetime | str | None) -> datetime | None:
    """Parse/convert anything date-like to naive UTC. Returns None when unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = dtparser.parse(value)
        except (ValueError, OverflowError):
            return None
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value


def week_start(d: datetime | date) -> datetime:
    """Monday 00:00 (naive UTC) of the week containing d."""
    if isinstance(d, datetime):
        d = d.date()
    monday = d - timedelta(days=d.weekday())
    return datetime(monday.year, monday.month, monday.day)
