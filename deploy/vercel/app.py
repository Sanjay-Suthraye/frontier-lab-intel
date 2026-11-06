"""Vercel entrypoint: the web UI over the committed real-data snapshot.

Vercel Functions can only write to /tmp, so a cold start unpacks the bundled snapshot
(data/snapshot/fli_snapshot.sqlite.gz) to /tmp and serves it from there. The bundle carries only
the web dependencies (requirements.txt): the web process never imports the pipeline's ML stack.

Writes (labels, alert acks) land in that per-instance copy and are lost on the next cold start, so
this deployment is a read-mostly demo; the AWS deployment (deploy/aws) is the persistent one.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "data" / "snapshot" / "fli_snapshot.sqlite.gz"
DB = Path(os.environ.get("FLI_VERCEL_DB", "/tmp/fli.sqlite"))
sys.path.insert(0, str(ROOT / "src"))


def _unpack() -> None:
    if DB.exists():  # warm instance
        return
    part = DB.with_name(f"{DB.name}.{os.getpid()}.part")
    with gzip.open(SNAPSHOT, "rb") as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(part, DB)  # atomic: a concurrent cold start never sees a half-written file


_unpack()
os.environ["DATABASE_URL"] = f"sqlite:///{DB}"

from fli.web.app import app as _fli_app  # noqa: E402

app = _fli_app
