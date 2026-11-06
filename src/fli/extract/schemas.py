"""Output schemas for the LLM stages and the normalisers that turn model output into rows.

Structured outputs accept a subset of JSON Schema: every object must set
`additionalProperties: false`, and numeric / string-length / array-size keywords (minimum,
maxLength, maxItems, ...) are not supported. So small ranges are integer enums (materiality 0-3,
magnitude 1-5), and everything else (word limits, list sizes, 0-1 confidence) is stated in the
prompt and clamped here. Normalisers are lenient on purpose: a parsed response is cached by the
gateway, so a strict validator that rejected it would fail identically on every retry.

Dynamic enums (candidate people, candidate tickers) are sorted for deterministic cache keys and
fall back to [NO_CANDIDATE] when empty (an empty enum is not a valid schema). Passing `None`
builds the *template* schema used for the prompt version (see context.TemplatedPrompt).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

from fli.taxonomy import (
    AI_TEAM_ACTIONS,
    AUDIENCES,
    DIRECTIONS,
    EVENT_TYPES,
    INVESTMENT_ACTIONS,
    NOISE_REASONS,
)

HORIZONS: tuple[str, ...] = ("days", "weeks", "quarters", "years")
MATERIALITY_LEVELS: tuple[int, ...] = (0, 1, 2, 3)
MAGNITUDES: tuple[int, ...] = (1, 2, 3, 4, 5)
NO_CANDIDATE = "none"
TEMPLATE_MARK = "<dynamic>"
OTHER_LAB = "other"  # triage: about an AI org outside the register
NO_LAB = "none"  # extraction: insight not about a register lab

TRIAGE_KEYS = ("reason", "event_type", "about_labs", "noise_reason", "materiality_guess", "audiences", "relevant")
MAX_INSIGHTS = 3
MAX_QUOTES = 3
MAX_QUOTE_CHARS = 300
MAX_TICKERS = 6
MAX_LIST = 12

# keywords structured outputs reject; tests assert none of our schemas use them
UNSUPPORTED_KEYWORDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
})


# ------------------------------------------------------------------------------- schema helpers
def _obj(props: dict[str, dict], description: str | None = None) -> dict:
    out: dict[str, Any] = {"type": "object", "properties": props, "required": list(props),
                           "additionalProperties": False}
    if description:
        out["description"] = description
    return out


def _str(description: str) -> dict:
    return {"type": "string", "description": description}


def _enum(values: Iterable, description: str, kind: str = "string") -> dict:
    return {"type": kind, "enum": list(values), "description": description}


def _arr(items: dict, description: str) -> dict:
    return {"type": "array", "items": items, "description": description}


def dynamic_enum(values: Iterable[str] | None) -> list[str]:
    """Sorted unique values; [NO_CANDIDATE] when empty; [TEMPLATE_MARK] for the template schema."""
    if values is None:
        return [TEMPLATE_MARK]
    vals = sorted({v for v in values if v and v != NO_CANDIDATE})
    return vals or [NO_CANDIDATE]


def triage_schema(lab_ids: Sequence[str]) -> dict:
    return _obj({
        "reason": _str("<= 25 words: the concrete fact that drove the decision"),
        "event_type": _enum(EVENT_TYPES, "best-fitting event type"),
        "about_labs": _arr(_enum([*sorted(lab_ids), OTHER_LAB], "register lab id"), "labs the item is about"),
        "noise_reason": _enum(NOISE_REASONS, "'none' if and only if relevant"),
        "materiality_guess": _enum(MATERIALITY_LEVELS, "0 noise .. 3 potentially material", "integer"),
        "audiences": _arr(_enum(AUDIENCES, "audience"), "who would want this; [] if not relevant"),
        "relevant": {"type": "boolean"},
    })


def extraction_schema(lab_ids: Sequence[str], person_ids: Iterable[str] | None) -> dict:
    insight = _obj({
        "evidence_quotes": _arr(_str("verbatim span from TITLE or TEXT, <= 300 chars"), "1-3 verbatim quotes"),
        "claim": _str("<= 40 words, factual, attributed"),
        "title": _str("<= 12 words"),
        "event_type": _enum(EVENT_TYPES, "event type"),
        "lab_id": _enum([*sorted(lab_ids), NO_LAB], "register lab the development is about"),
        "person_ids": _arr(_enum(dynamic_enum(person_ids), "candidate person id"), "tracked people involved"),
        "artifacts": _arr(_str("model / product / repo / chip / dataset name"), "named artifacts"),
        "orgs": _arr(_str("company or organisation name"), "other organisations involved"),
        "magnitude": _enum(MAGNITUDES, "1 routine .. 5 landmark", "integer"),
        "is_rumor": {"type": "boolean"},
        "confidence": {"type": "number", "description": "0-1"},
    })
    return _obj({
        "no_material_insight": {"type": "boolean"},
        "insights": _arr(insight, "0-3 insights"),
    })


def lens_investment_schema(symbols: Iterable[str] | None) -> dict:
    ticker = _obj({
        "symbol": _enum(dynamic_enum(symbols), "candidate symbol"),
        "direction": _enum(DIRECTIONS, "impact on this company"),
        "why": _str("<= 20 words"),
    })
    return _obj({
        "headline": _str("<= 12 words, investor angle"),
        "so_what": _str("<= 70 words: implication for positions / theses, naming holdings or themes"),
        "tickers": _arr(ticker, "0-5 candidate tickers"),
        "direction": _enum(DIRECTIONS, "net direction for exposed holdings"),
        "horizon": _enum(HORIZONS, "when it shows up"),
        "action": _enum(INVESTMENT_ACTIONS, "recommended action"),
        "what_to_do": _str("<= 40 words, concrete"),
        "confidence": {"type": "number", "description": "0-1"},
    })


def lens_ai_team_schema() -> dict:
    return _obj({
        "headline": _str("<= 12 words, engineering angle"),
        "so_what": _str("<= 70 words, technical: what changes for how we build"),
        "action": _enum(AI_TEAM_ACTIONS, "tech-radar ring"),
        "artifacts_to_try": _arr(_str("model / repo / API / paper name from the input"), "things to try"),
        "what_to_do": _str("<= 40 words, concrete next step"),
        "confidence": {"type": "number", "description": "0-1"},
    })


def strictness_errors(schema: dict, path: str = "$") -> list[str]:
    """Problems that would make the API reject the schema (empty list = OK)."""
    errors = [f"{path}: unsupported keyword {k}" for k in schema if k in UNSUPPORTED_KEYWORDS]
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            errors.append(f"{path}: additionalProperties must be false")
        if sorted(schema.get("required", [])) != sorted(props):
            errors.append(f"{path}: required must list every property")
        for name, sub in props.items():
            errors += strictness_errors(sub, f"{path}.{name}")
    if "enum" in schema and not schema["enum"]:
        errors.append(f"{path}: empty enum")
    if schema.get("type") == "array":
        errors += strictness_errors(schema.get("items", {}), f"{path}[]")
    return errors


# ------------------------------------------------------------------------------- normalisers
_WS = re.compile(r"\s+")


def clean_str(value: Any) -> str:
    return _WS.sub(" ", str(value or "")).strip()


def clip_words(text: str, max_words: int) -> str:
    """Guard against runaway output; limits in prompts are tighter than these."""
    words = clean_str(text).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + " ..."


def clip_chars(text: str, max_chars: int) -> str:
    """Cut at a word boundary without adding characters, so a verbatim quote stays verbatim."""
    text = clean_str(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars * 0.6 else cut).rstrip()


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else min(1.0, max(0.0, x))


def str_list(values: Any, limit: int = MAX_LIST, max_chars: int = 200) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values if isinstance(values, list) else []:
        s = clip_chars(str(v), max_chars) if v is not None else ""
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:limit]


def pick(value: Any, allowed: Sequence, default):
    return value if value in allowed else default


def _require(data: Any, keys: Iterable[str]) -> dict:
    if not isinstance(data, dict):
        raise TypeError("model output is not an object")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"model output missing {missing}")
    return data


def clean_triage(data: Any, lab_ids: Iterable[str]) -> dict:
    d = _require(data, TRIAGE_KEYS)
    labs = set(lab_ids) | {OTHER_LAB}
    try:
        materiality = int(d["materiality_guess"])
    except (TypeError, ValueError):
        materiality = 0
    return {
        "relevant": bool(d["relevant"]),
        "event_type": pick(d["event_type"], tuple(EVENT_TYPES), "other"),
        "audiences": [a for a in AUDIENCES if a in (d["audiences"] or [])],
        "noise_reason": pick(d["noise_reason"], NOISE_REASONS, "off_topic"),
        "materiality_guess": min(3, max(0, materiality)),
        "about_labs": [x for x in str_list(d["about_labs"]) if x in labs],
        "reason": clip_words(d["reason"], 40),
    }


def clean_insight(raw: Any, lab_ids: Iterable[str], person_ids: Iterable[str]) -> dict:
    d = _require(raw, ("evidence_quotes", "claim", "title", "event_type"))
    quotes = [clip_chars(str(q).strip().strip('"“”'), MAX_QUOTE_CHARS)
              for q in (d.get("evidence_quotes") or []) if isinstance(q, str)]
    lab = d.get("lab_id")
    try:
        magnitude = min(5, max(1, int(d.get("magnitude") or 1)))
    except (TypeError, ValueError):
        magnitude = 1
    allowed_people = set(person_ids)
    return {
        "evidence_quotes": str_list([q for q in quotes if q], MAX_QUOTES, MAX_QUOTE_CHARS),
        "claim": clip_words(d["claim"], 70),
        "title": clip_chars(clip_words(d["title"], 20), 400),
        "event_type": pick(d["event_type"], tuple(EVENT_TYPES), "other"),
        "lab_id": lab if lab in set(lab_ids) else None,
        "person_ids": [p for p in str_list(d.get("person_ids")) if p in allowed_people],
        "artifacts": str_list(d.get("artifacts")),
        "orgs": str_list(d.get("orgs")),
        "magnitude": magnitude,
        "is_rumor": bool(d.get("is_rumor")),
        "confidence": clamp01(d.get("confidence")),
    }


def clean_extraction(data: Any, lab_ids: Iterable[str], person_ids: Iterable[str]) -> tuple[bool, list[dict]]:
    """Returns (no_material_insight, insights). Duplicate claims are dropped; at most 3 kept."""
    d = _require(data, ("no_material_insight", "insights"))
    lab_ids, person_ids = list(lab_ids), list(person_ids)
    out: list[dict] = []
    seen: set[str] = set()
    for raw in d["insights"] if isinstance(d["insights"], list) else []:
        ins = clean_insight(raw, lab_ids, person_ids)
        key = ins["claim"].lower()
        if ins["claim"] and key not in seen:
            seen.add(key)
            out.append(ins)
    return bool(d["no_material_insight"]), out[:MAX_INSIGHTS]


def clean_lens(data: Any, audience: str, symbols: Iterable[str]) -> dict:
    """Normalised lens fields. An abstaining investment lens (action no_action) keeps no tickers and
    no net direction: a listed ticker would make the development count as exposed to that company
    (materiality's held_exposure). What was dropped is returned as `dropped_tickers` for the audit
    trail in Lens.checks."""
    d = _require(data, ("headline", "so_what", "what_to_do", "action", "confidence"))
    base = {
        "headline": clip_chars(clip_words(d["headline"], 20), 400),
        "so_what": clip_words(d["so_what"], 110),
        "what_to_do": clip_words(d["what_to_do"], 70),
        "confidence": clamp01(d["confidence"]),
        "dropped_tickers": [],
    }
    if audience == "ai_team":
        return {**base, "action": pick(d["action"], AI_TEAM_ACTIONS, "hold"), "tickers": [],
                "artifacts_to_try": str_list(d.get("artifacts_to_try"), 6), "direction": None, "horizon": None}
    allowed = set(symbols) - {NO_CANDIDATE}
    tickers: list[dict] = []
    for t in d.get("tickers") or []:
        if isinstance(t, dict) and t.get("symbol") in allowed and t["symbol"] not in {x["symbol"] for x in tickers}:
            tickers.append({"symbol": t["symbol"], "direction": pick(t.get("direction"), DIRECTIONS, "unclear"),
                            "why": clip_words(t.get("why", ""), 35)})
    action = pick(d["action"], INVESTMENT_ACTIONS, "watch")
    direction = pick(d.get("direction"), DIRECTIONS, "unclear")
    dropped: list[str] = []
    if action == "no_action":  # abstain: the prompt asks for tickers [], enforced here
        dropped, tickers, direction = [t["symbol"] for t in tickers], [], "unclear"
    return {**base, "action": action, "tickers": tickers[:MAX_TICKERS], "dropped_tickers": dropped,
            "artifacts_to_try": [], "direction": direction, "horizon": pick(d.get("horizon"), HORIZONS, None)}
