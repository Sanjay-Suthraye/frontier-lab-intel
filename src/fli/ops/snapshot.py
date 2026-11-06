"""Database snapshot: how the repo ships real data and how a Postgres deployment is seeded.

export  SQLite source: VACUUM INTO a temp file (a consistent copy even while the worker writes),
        then gzip. Postgres source: copy every table into a fresh SQLite file via SQLAlchemy Core.
import  SQLite target: gunzip next to the database, integrity-check, then atomically swap it in
        (stale -wal/-shm files removed first so SQLite cannot replay an old log onto it).
        Postgres target: create tables, bulk insert per table in dependency order, reset the id
        sequences. Import REPLACES the target database's contents.

The gzip header carries mtime=0, so an unchanged database yields a byte-identical file in git.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from sqlalchemy import Engine, Integer, create_engine, inspect, select, text, update
from sqlalchemy.engine import make_url

from fli.config import DATA_DIR, get_settings
from fli.db import session as db_session
from fli.db.models import Base

log = logging.getLogger("fli.ops.snapshot")

DEFAULT_PATH = DATA_DIR / "snapshot" / "fli_snapshot.sqlite.gz"
CHUNK = 1000


def run_snapshot(action: str, path: str | None = None) -> dict:
    """`fli snapshot export|import [--path ...]`."""
    target = Path(path).expanduser() if path else DEFAULT_PATH
    if action == "export":
        return export_snapshot(target)
    if action == "import":
        return import_snapshot(target)
    if action == "import-if-empty":  # container start-up: seed once, never overwrite live data
        from sqlalchemy import func, select

        from fli.db.models import Document
        from fli.db.session import init_db, session_scope

        init_db()
        with session_scope() as s:
            n = s.scalar(select(func.count()).select_from(Document)) or 0
        if n:
            return {"status": "skipped", "reason": f"database already has {n} documents"}
        if not target.exists():
            return {"status": "skipped", "reason": f"no snapshot at {target}"}
        return import_snapshot(target)
    raise ValueError(f"unknown snapshot action {action!r}; use 'export', 'import' or 'import-if-empty'")


def _sqlite_path(url: str) -> Path | None:
    """File path of a SQLite DATABASE_URL, None for other backends."""
    u = make_url(url)
    if u.get_backend_name() != "sqlite":
        return None
    if not u.database or u.database == ":memory:":
        raise ValueError("an in-memory SQLite database cannot be snapshotted")
    return Path(u.database).expanduser().resolve()


def _counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db))
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}
    finally:
        conn.close()


def _check_sqlite(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    try:
        result = conn.execute("PRAGMA quick_check").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"{db.name} is not a valid SQLite snapshot: {exc}") from exc
    finally:
        conn.close()
    if result != "ok":
        raise ValueError(f"snapshot failed its integrity check: {result}")


def _gzip(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fin, open(tmp, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        shutil.copyfileobj(fin, gz, 1 << 20)
    os.replace(tmp, dst)


def _gunzip(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, 1 << 20)


def _vacuum_into(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"database {src} does not exist")
    conn = sqlite3.connect(str(src), isolation_level=None, timeout=30)
    try:
        conn.execute("VACUUM INTO ?", (str(dst),))
    finally:
        conn.close()
    conn = sqlite3.connect(str(dst), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")  # a single self-contained file
    finally:
        conn.close()


def copy_database(src: Engine, dst: Engine, replace: bool = False) -> dict[str, int]:
    """Copy every model table from src to dst in FK dependency order, in chunks.

    Self-referencing FK columns (document.duplicate_of_id) are filled in a second pass so row order
    can never violate a constraint. Columns missing on the source (older snapshot) keep defaults."""
    Base.metadata.create_all(dst)
    src_insp = inspect(src)
    src_tables = set(src_insp.get_table_names())
    counts: dict[str, int] = {}
    with src.connect() as sconn, dst.begin() as dconn:
        if replace:
            for table in reversed(Base.metadata.sorted_tables):
                dconn.execute(table.delete())
        for table in Base.metadata.sorted_tables:
            if table.name not in src_tables:
                counts[table.name] = 0
                continue
            present = {c["name"] for c in src_insp.get_columns(table.name)}
            cols = [c for c in table.columns if c.name in present]
            self_ref = [fk.parent.name for fk in table.foreign_keys
                        if fk.column.table is table and fk.parent.name in present]
            pk = [c.name for c in table.primary_key.columns]
            deferred: list[tuple[dict, dict]] = []
            n = 0
            result = sconn.execution_options(yield_per=CHUNK).execute(select(*cols).order_by(*table.primary_key.columns))
            for part in result.mappings().partitions(CHUNK):
                rows = [dict(r) for r in part]
                for r in rows:
                    refs = {c: r[c] for c in self_ref if r.get(c) is not None}
                    if refs:
                        deferred.append(({k: r[k] for k in pk}, refs))
                        r.update({c: None for c in refs})
                dconn.execute(table.insert(), rows)
                n += len(rows)
            for key, refs in deferred:
                dconn.execute(update(table).where(*[table.c[k] == v for k, v in key.items()]).values(**refs))
            counts[table.name] = n
    return counts


def reset_sequences(engine: Engine) -> None:
    """After inserting explicit ids into Postgres, point each serial sequence past max(id)."""
    if engine.dialect.name != "postgresql":
        return
    quote = engine.dialect.identifier_preparer.quote
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            pk = list(table.primary_key.columns)
            if len(pk) != 1 or not isinstance(pk[0].type, Integer):
                continue
            tbl, col = quote(table.name), quote(pk[0].name)
            conn.execute(text(f"SELECT setval(pg_get_serial_sequence(:t, :c), "
                              f"COALESCE((SELECT MAX({col}) FROM {tbl}), 0) + 1, false)"),
                         {"t": tbl, "c": pk[0].name})


def export_snapshot(target: Path) -> dict:
    url = get_settings().database_url
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".snapshot-") as tmp:
        tmp_db = Path(tmp) / "snapshot.sqlite"
        db_path = _sqlite_path(url)
        if db_path is not None:
            _vacuum_into(db_path, tmp_db)
        else:
            dst = create_engine(f"sqlite:///{tmp_db}")
            try:
                copy_database(db_session.get_engine(), dst)
            finally:
                dst.dispose()
            _vacuum_into(tmp_db, Path(tmp) / "compact.sqlite")
            tmp_db = Path(tmp) / "compact.sqlite"
        counts = _counts(tmp_db)
        raw_bytes = tmp_db.stat().st_size
        _gzip(tmp_db, target)
    result = {"action": "export", "path": str(target), "bytes": target.stat().st_size, "sqlite_bytes": raw_bytes,
              "rows": sum(counts.values()), "tables": counts}
    log.info("snapshot exported: %s (%d rows, %.1f MB gzipped)", target, result["rows"], result["bytes"] / 1e6)
    return result


def import_snapshot(source: Path) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"snapshot {source} not found (run `fli snapshot export` first)")
    url = get_settings().database_url
    db_path = _sqlite_path(url)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = db_path.with_name(db_path.name + ".importing")
        try:
            _gunzip(source, tmp)
            _check_sqlite(tmp)
            db_session.get_engine().dispose()  # no pooled connection may keep the old file open
            for suffix in ("-wal", "-shm"):
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)
            os.replace(tmp, db_path)
        finally:
            tmp.unlink(missing_ok=True)
        db_session.init_db()  # adds tables introduced after the snapshot was taken
        counts = _counts(db_path)
    else:
        with tempfile.TemporaryDirectory(prefix="fli-snapshot-") as tmpdir:
            tmp_db = Path(tmpdir) / "snapshot.sqlite"
            _gunzip(source, tmp_db)
            _check_sqlite(tmp_db)
            src = create_engine(f"sqlite:///{tmp_db}")
            try:
                counts = copy_database(src, db_session.get_engine(), replace=True)
            finally:
                src.dispose()
        reset_sequences(db_session.get_engine())
    result = {"action": "import", "path": str(source), "target": make_url(url).render_as_string(hide_password=True),
              "rows": sum(counts.values()), "tables": counts}
    log.info("snapshot imported into %s: %d rows", result["target"], result["rows"])
    return result
