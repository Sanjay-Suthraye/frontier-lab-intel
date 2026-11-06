"""Prompt files live in src/fli/prompts/<name>.md with a small front matter:

    ---
    version: 3
    ---
    <system prompt text>

The stored prompt version is `<name>@v<version>+<hash of text and output schema>`, so any edit to
the wording or the schema changes the version recorded on every output row and in the cache key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from fli.config import PROMPTS_DIR


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    system: str
    path: Path

    def version_for(self, schema: dict | None = None) -> str:
        payload = self.system + "\x00" + json.dumps(schema or {}, sort_keys=True)
        return f"{self.name}@v{self.version}+{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


@lru_cache(maxsize=64)
def load_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name}.md"
    raw = path.read_text(encoding="utf-8")
    version = 1
    body = raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        meta = yaml.safe_load(front) or {}
        version = int(meta.get("version", 1))
    return Prompt(name=name, version=version, system=body.strip(), path=path)
