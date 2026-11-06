"""Artifact registry: named things (models, products, repos, API model ids, chips, datasets).

Why a registry: "the first time anyone mentions X" is one of the strongest novelty signals (a new
model id in a repo, model card or changelog often precedes the launch post). Names are normalised
so that "OpenAI's GPT-5.1", "gpt 5.1" and "GPT5.1" collide, and the first sighting is keyed by
*document*, so every insight extracted from that document can claim it.

Known limitation (documented in the design doc): the tracked window starts on 2026-08-01, so older
artifacts mentioned in the first weeks look "first seen". Scoring should combine
`first_seen_artifacts` with a release-type event rather than use it alone.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from fli.db.models import Artifact, Insight
from fli.db.session import session_scope
from fli.timeutil import utcnow

log = logging.getLogger("fli.events.artifacts")

# Prefixes that name the maker rather than the thing ("OpenAI GPT-5" -> "gpt 5"). Brand names that
# are part of the artifact name (DeepSeek-V3, Mistral Large, Qwen3) are deliberately absent.
# Multi-word prefixes come first so "google deepmind" wins over "google".
VENDOR_PREFIXES: tuple[str, ...] = (
    "google deepmind", "deepmind", "google", "openai", "anthropic", "meta ai", "meta", "xai", "x ai",
    "mistral ai", "alibaba cloud", "alibaba", "moonshot ai", "microsoft", "nvidia", "amd", "amazon",
    "aws", "apple", "thinking machines lab", "thinking machines", "reflection ai", "periodic labs",
    "safe superintelligence", "the",  # not "a": "A100" normalises to "a 100"
)
GENERIC_SUFFIXES: frozenset[str] = frozenset({"model", "models", "family", "series"})

_URL_HOST = re.compile(r"^(?:https?://)?(?:www\.)?(?:github\.com|huggingface\.co|hf\.co)/", re.IGNORECASE)
_REPO_PATH = re.compile(r"^[\w.-]+/[\w.-]+$")
_VERSION_DASH = re.compile(r"(?<![\d.])(\d)-(\d)(?![\d.])")  # claude-opus-4-5 -> 4.5, not 2025-10-01
_POSSESSIVE = re.compile(r"['’]s\b")
_NON_WORD = re.compile(r"[^\w.]+")
_LONE_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")  # keep dots only inside version numbers
_LETTER_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)")  # gpt5 -> gpt 5, so GPT-5 / GPT5 / gpt 5 collide
_WS = re.compile(r"\s+")

# Kind rules, applied in this order to the raw name (first match wins).
_CHIP = re.compile(
    r"\b(tpus?|trainium\d*|inferentia\d*|mtia|maia|axion|graviton\d*|[hb]\d{3}|gb\d{3}|mi\d{3}x?|"
    r"blackwell|rubin|hopper|gaudi\d*|asics?|accelerators?|chips?|xpus?|npus?|lpus?|wafer|ironwood)\b",
    re.IGNORECASE,
)
_DATASET = re.compile(r"(bench\b|benchmark|dataset|corpus|\bevals?\b|leaderboard|\barena\b)", re.IGNORECASE)
_API_ID = re.compile(r"^[a-z][a-z0-9]*(?:[-.][a-z0-9]+)+$")  # gpt-5-mini, claude-opus-4-5, gemini-2.5-pro
_DATE_SUFFIX = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")
_PARAM_COUNT = re.compile(r"\d+(\.\d+)?[bm]\b", re.IGNORECASE)
_PRODUCT = re.compile(
    r"\b(app|apps|api|studio|chat|chatgpt|agent|agents|assistant|cli|sdk|platform|browser|extension|"
    r"search|device|glasses|plugin|copilot|console|playground|workspace|enterprise|code|notebooklm|"
    r"operator|atlas|pulse|cowork|for (chrome|excel|slack|word|sheets))\b",
    re.IGNORECASE,
)
_MODEL = re.compile(
    r"\b(gpt|o\d|claude|opus|sonnet|haiku|gemini|gemma|veo|imagen|genie|lyria|llama|grok|mistral|"
    r"mixtral|magistral|devstral|codestral|ministral|pixtral|deepseek|qwen\d*|qwq|kimi|k\d|sora|dall-?e|"
    r"whisper|phi|nova|olmo|glm|minimax|ernie|hunyuan)\b|\b\d+(\.\d+)?[bm]\b|\b[a-z]+-?\d+(\.\d+)*\b",
    re.IGNORECASE,
)


def normalize_artifact(name: str | None) -> str:
    """Matching key for an artifact name: lowercase, punctuation and vendor prefixes stripped,
    version numbers kept ("Claude Opus 4.5" == "claude-opus-4-5" == "Anthropic's Claude Opus 4.5")."""
    text = unicodedata.normalize("NFKC", name or "").casefold().strip()
    text = _URL_HOST.sub("", text).strip("/")
    if _REPO_PATH.match(text):  # org/repo (GitHub, Hugging Face): the org is the vendor
        text = text.split("/", 1)[1]
    text = _POSSESSIVE.sub("", text)
    text = _VERSION_DASH.sub(r"\1.\2", text)
    text = _NON_WORD.sub(" ", text).replace("_", " ")
    text = _LONE_DOT.sub(" ", text)
    text = _LETTER_DIGIT.sub(" ", text)
    tokens = _WS.sub(" ", text).strip().split(" ")
    tokens = _strip_vendor(tokens)
    if len(tokens) > 1 and tokens[-1] in GENERIC_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(t for t in tokens if t)[:300]


def _strip_vendor(tokens: list[str]) -> list[str]:
    """Drop leading vendor/article tokens while something meaningful remains."""
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        for prefix in VENDOR_PREFIXES:
            words = prefix.split(" ")
            if len(tokens) > len(words) and tokens[: len(words)] == words:
                tokens, changed = tokens[len(words):], True
                break
    return tokens


def guess_kind(name: str) -> str:
    """Cheap rule-based kind: chip | dataset | api_model_id | repo | product | model | other."""
    raw = (name or "").strip()
    if not raw:
        return "other"
    if _CHIP.search(raw):
        return "chip"
    if _DATASET.search(raw):
        return "dataset"
    model_like = bool(_MODEL.search(raw)) and not _PARAM_COUNT.search(raw)  # gpt-oss-120b is a model name
    if _API_ID.match(raw) and (_DATE_SUFFIX.search(raw) or model_like):
        return "api_model_id"
    if "github.com/" in raw.lower() or _REPO_PATH.match(raw):
        repo = raw.rsplit("/", 1)[-1]
        return "model" if _MODEL.search(repo) and not _PRODUCT.search(repo) else "repo"
    if _PRODUCT.search(raw):
        return "product"
    if _MODEL.search(raw):
        return "model"
    return "other"


def unique_normalized(names: Iterable[str] | None) -> list[tuple[str, str]]:
    """[(normalised, display)] in first-appearance order, empty names dropped."""
    out: dict[str, str] = {}
    for raw in names or []:
        if not isinstance(raw, str):
            continue
        key = normalize_artifact(raw)
        if key and key not in out:
            out[key] = raw.strip()[:300]
    return list(out.items())


# ------------------------------------------------------------------ matching (used by clustering)
SUBJECT_KINDS = frozenset({"model", "api_model_id", "product"})

# Umbrella brands name a whole product line, not one thing: "ChatGPT" covers ads, apps from Oracle,
# teen-safety policy and usage statistics alike, and "Gemini" tags any paper that used the model.
# Sharing one is no evidence of the same development (unrelated same-lab posts merged on it), so
# they are never link keys. They stay subjects: dropping them there would remove the subject
# conflict that keeps "ChatGPT" (ads test) and "ChatGPT Business" (seats) apart below 0.92 cosine
# (a full re-clustering of the snapshot merged 5 unrelated pairs that way).
# Versioned or qualified names ("Gemini 4 Carbon", "Claude Opus 5.5", "ChatGPT Atlas") still link.
# Matched on normalised keys, so vendor prefixes collapse too ("Amazon Bedrock", "Meta AI" -> "ai").
UMBRELLA_NAMES: tuple[str, ...] = (
    "ChatGPT", "Gemini", "Claude", "Codex", "Claude Code", "Grok", "Llama", "Meta AI", "Copilot",
    "Bedrock", "Vertex AI",
)
UMBRELLA_KEYS: frozenset[str] = frozenset(normalize_artifact(n) for n in UMBRELLA_NAMES)


def link_keys(names: Iterable[str] | None) -> tuple[set[str], set[str]]:
    """(linkable, subjects) keys of an insight's artifacts.

    Benchmarks/datasets are not linkable: "SWE-bench Verified" appears in unrelated launches.
    Umbrella brands (UMBRELLA_KEYS) are not linkable either: "ChatGPT" alone links unrelated posts.
    Subjects (models, API ids, products) are what a development is about; two different subjects
    (Claude Opus 5.5 vs Claude Haiku 5.5) mark sibling releases that must not merge on cosine alone.
    """
    linkable: set[str] = set()
    subjects: set[str] = set()
    for key, display in unique_normalized(names):
        kind = guess_kind(display)
        if kind == "dataset":
            continue
        if key not in UMBRELLA_KEYS:
            linkable.add(key)
        if kind in SUBJECT_KINDS:
            subjects.add(key)
    return linkable, subjects


def version_prefixes(key: str) -> list[str]:
    """Proper token prefixes ending in a version: "qwen 3.5 72b" -> ["qwen 3.5"]."""
    tokens = key.split(" ")
    return [" ".join(tokens[:n]) for n in range(2, len(tokens)) if any(c.isdigit() for c in tokens[n - 1])]


def same_artifact(a: str, b: str) -> bool:
    """Equal, or one names the versioned family of the other ("gpt 5.5" ~ "gpt 5.5 mini"). Two
    variants of a family ("qwen 3 coder" vs "qwen 3 vl") are not the same artifact."""
    return a == b or a in version_prefixes(b) or b in version_prefixes(a)


def related(xs: set[str], ys: set[str]) -> bool:
    return bool(xs & ys) or any(same_artifact(x, y) for x in xs for y in ys)


def _insight_time(ins: Insight) -> datetime:
    return ins.published_at or ins.created_at or utcnow()


def _load_rows(session: Session, names: Iterable[str]) -> dict[str, Artifact]:
    """Registry rows by normalised name (one row per name; kind is only a hint)."""
    names = sorted(set(names))
    rows: dict[str, Artifact] = {}
    for i in range(0, len(names), 500):
        chunk = names[i : i + 500]
        for row in session.scalars(select(Artifact).where(Artifact.name.in_(chunk)).order_by(Artifact.id)):
            rows.setdefault(row.name, row)
    return rows


def _first_seen_for(ins: Insight, rows: dict[str, Artifact]) -> list[str]:
    return [
        display
        for key, display in unique_normalized(ins.artifacts)
        if key in rows and rows[key].first_document_id == ins.document_id
    ]


def register_insight_artifacts(session: Session, insights: Sequence[Insight]) -> dict:
    """Insert unseen artifacts (in chronological order) and set `first_seen_artifacts`.

    Idempotent: rows are looked up by normalised name, and `first_seen_artifacts` is derived from
    the registry, so re-running yields the same values. If an earlier document turns up later
    (backfill), the row moves to it and the previous first document's insights are corrected.
    """
    ordered = sorted(insights, key=lambda i: (_insight_time(i), i.id or 0))
    keys = {key for ins in ordered for key, _ in unique_normalized(ins.artifacts)}
    rows = _load_rows(session, keys)
    created = moved = 0
    displaced_docs: set[int] = set()
    for ins in ordered:
        seen_at = _insight_time(ins)
        for key, display in unique_normalized(ins.artifacts):
            row = rows.get(key)
            if row is None:
                row = Artifact(name=key, display_name=display, kind=guess_kind(display), lab_id=ins.lab_id,
                               first_seen_at=seen_at, first_document_id=ins.document_id)
                session.add(row)
                rows[key] = row
                created += 1
            elif seen_at < row.first_seen_at and row.first_document_id != ins.document_id:
                if row.first_document_id is not None:
                    displaced_docs.add(row.first_document_id)
                row.first_seen_at, row.first_document_id = seen_at, ins.document_id
                moved += 1
    batch_ids = {ins.id for ins in ordered}
    for ins in ordered:
        firsts = _first_seen_for(ins, rows)
        if list(ins.first_seen_artifacts or []) != firsts:
            ins.first_seen_artifacts = firsts
    if displaced_docs:
        _refresh_documents(session, displaced_docs, exclude=batch_ids)
    session.flush()
    if created or moved:
        log.info("artifacts: %d new, %d moved to an earlier document", created, moved)
    return {"artifacts_new": created, "artifacts_moved": moved}


def _refresh_documents(session: Session, doc_ids: set[int], exclude: set[int]) -> None:
    """Recompute `first_seen_artifacts` for insights of documents that lost a first sighting."""
    others = [
        i for i in session.scalars(select(Insight).where(Insight.document_id.in_(sorted(doc_ids))))
        if i.id not in exclude
    ]
    rows = _load_rows(session, {k for i in others for k, _ in unique_normalized(i.artifacts)})
    for ins in others:
        firsts = _first_seen_for(ins, rows)
        if list(ins.first_seen_artifacts or []) != firsts:
            ins.first_seen_artifacts = firsts


def run_artifacts(limit: int | None = None, **_: object) -> dict:
    """Rebuild pass over all insights in chronological order (use after a backfill).

    The incremental path is `run_novelty`, which registers artifacts for the insights it scores.
    """
    with session_scope() as s:
        q = select(Insight).order_by(Insight.published_at.is_(None), Insight.published_at, Insight.id)
        if limit:
            q = q.limit(limit)
        insights = list(s.scalars(q))
        out = register_insight_artifacts(s, insights)
        out["insights"] = len(insights)
    return out
