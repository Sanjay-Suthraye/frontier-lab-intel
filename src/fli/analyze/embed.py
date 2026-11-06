"""Sentence embeddings for insights (clustering into developments, novelty).

Model: BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU, 384-d, ~67 MB quantised download). Small
enough to run inside the worker with no GPU and no API spend; the corpus is thousands of short
claims, so a hosted embedding API would add a dependency and cost for no measurable gain.

Vectors are L2-normalised float32 stored as little-endian bytes, so cosine similarity is a dot
product and the stored format is identical on SQLite and Postgres.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from sqlalchemy import select

from fli.config import DATA_DIR
from fli.db.models import Insight
from fli.db.session import session_scope

log = logging.getLogger("fli.events.embed")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
BATCH_SIZE = 64
CACHE_DIR = DATA_DIR / "models" / "fastembed"
_DTYPE = np.dtype("<f4")


@lru_cache(maxsize=1)
def _model() -> Any:
    """Load the ONNX model once per process (first call downloads it into CACHE_DIR)."""
    from fastembed import TextEmbedding  # imported lazily: tests never load the model

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("loading embedding model %s from %s", MODEL_NAME, CACHE_DIR)
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))


def warm_up() -> None:
    """Download/load the model ahead of time (image build, worker start) so the first stage run
    does not pay for it: `uv run python -c "from fli.analyze.embed import warm_up; warm_up()"`."""
    encode(["warm-up"])


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows stay zero)."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise one vector."""
    return normalize_rows(vec)[0]


def encode(texts: Sequence[str]) -> np.ndarray:
    """Embed texts -> (n, 384) L2-normalised float32 matrix."""
    items = [t or "" for t in texts]
    if not items:
        return np.zeros((0, DIM), dtype=np.float32)
    vectors = np.asarray(list(_model().embed(items, batch_size=BATCH_SIZE)), dtype=np.float32)
    return normalize_rows(vectors)


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=_DTYPE).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    """Decode a stored vector (a writable float32 copy)."""
    return np.frombuffer(blob, dtype=_DTYPE).astype(np.float32)


def insight_text(ins: Insight) -> str:
    """'title. claim. artifact, artifact' - the claim carries the meaning, artifact names anchor it."""
    artifacts = ", ".join(a for a in (ins.artifacts or []) if isinstance(a, str) and a.strip())
    parts = [(ins.title or "").strip(), (ins.claim or "").strip(), artifacts]
    return ". ".join(p.rstrip(" .") for p in parts if p.rstrip(" ."))


def run_embed(limit: int | None = None, batch_size: int = BATCH_SIZE, **_: object) -> dict:
    """Embed insights whose embedding is NULL, committing per batch so a crash loses one batch."""
    embedded = batches = 0
    while limit is None or embedded < limit:
        take = batch_size if limit is None else min(batch_size, limit - embedded)
        with session_scope() as s:
            rows = list(s.scalars(
                select(Insight).where(Insight.embedding.is_(None)).order_by(Insight.id).limit(take)
            ))
            if not rows:
                break
            vectors = encode([insight_text(r) for r in rows])
            if len(vectors) != len(rows):
                raise RuntimeError(f"encoder returned {len(vectors)} vectors for {len(rows)} texts")
            for row, vec in zip(rows, vectors, strict=True):
                row.embedding = to_bytes(vec)
        embedded += len(rows)
        batches += 1
    log.info("embedded %d insights in %d batches", embedded, batches)
    return {"embedded": embedded, "batches": batches, "model": MODEL_NAME}
