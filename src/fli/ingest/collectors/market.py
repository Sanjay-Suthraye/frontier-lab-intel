"""Market-data collectors: the OpenRouter model index (public, no key).

A model appearing on OpenRouter is a dated, machine-readable launch signal with list prices, which
is exactly what the AI team needs ("a cheaper or better model") and what makes price pressure
visible to the investment team. One record per model created inside the tracked window; the
latest pricing is refreshed into `meta` on every run (the runner merges meta on existing rows).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fli.config import Settings
from fli.ingest.records import IngestRecord, SourceInfo, Window, collector
from fli.timeutil import utcnow

log = logging.getLogger("fli.ingest.market")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# provider prefix -> register lab id ("meta" replaced "meta-llama" for the Muse family in 2026)
PROVIDER_LABS: dict[str, str] = {
    "openai": "openai", "anthropic": "anthropic", "google": "google-deepmind", "meta-llama": "meta-ai",
    "meta": "meta-ai", "x-ai": "xai", "mistralai": "mistral", "deepseek": "deepseek", "qwen": "qwen",
    "moonshotai": "moonshot", "reflection": "reflection-ai",
}


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


@retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _get_json(client: httpx.Client, url: str) -> Any:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


def per_mtok(value: Any) -> float | None:
    """OpenRouter quotes USD per token as strings; convert to USD per million tokens."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return round(price * 1_000_000, 6) if price >= 0 else None  # -1 marks "variable" pricing


def _pricing(p: dict) -> dict[str, float | None]:
    return {"prompt": per_mtok(p.get("prompt")), "completion": per_mtok(p.get("completion")),
            "cache_read": per_mtok(p.get("input_cache_read"))}


def _variants(models: list[dict]) -> dict[str, dict[str, dict]]:
    """':batch' / ':free' variants folded into their base model (base id -> variant -> pricing)."""
    out: dict[str, dict[str, dict]] = {}
    for m in models:
        mid = str(m.get("id") or "")
        if ":" in mid:
            base, variant = mid.split(":", 1)
            out.setdefault(base, {})[variant] = _pricing(m.get("pricing") or {})
    return out


def _fmt_price(value: float | None) -> str:
    """$4.00, $0.15, and $0.003 for sub-dime prices (cached-input reads are often fractions of a cent)."""
    if value is None:
        return "n/a"
    return f"${value:,.2f}" if value >= 0.1 or value == 0 else f"${value:.3g}"


def split_overrides(pricing: dict) -> tuple[dict | None, list[dict]]:
    """OpenRouter price overrides are either long-context tiers (min_prompt_tokens) or
    time-of-day windows (utc_start/utc_end/utc_days, e.g. off-peak discounts)."""
    long_ctx, timed = None, []
    for o in pricing.get("overrides") or []:
        if not isinstance(o, dict):
            continue
        if o.get("min_prompt_tokens") is not None and long_ctx is None:
            long_ctx = {"min_prompt_tokens": int(o["min_prompt_tokens"]), **_pricing(o)}
        elif any(k in o for k in ("utc_start", "utc_end", "utc_days")):
            timed.append({k: o.get(k) for k in ("utc_days", "utc_start", "utc_end")} | _pricing(o))
    return long_ctx, timed


def model_to_record(m: dict, lab_id: str | None, variants: dict[str, dict]) -> IngestRecord:
    mid, name = m["id"], m.get("name") or m["id"]
    created = datetime.fromtimestamp(int(m["created"]), UTC).replace(tzinfo=None)
    price = _pricing(m.get("pricing") or {})
    top = m.get("top_provider") or {}
    arch = m.get("architecture") or {}
    long_ctx, timed = split_overrides(m.get("pricing") or {})
    lines = [
        f"{name} ({mid}) became available on OpenRouter on {created:%Y-%m-%d}.",
        f"List price per million tokens: {_fmt_price(price['prompt'])} input, "
        f"{_fmt_price(price['completion'])} output"
        + (f", {_fmt_price(price['cache_read'])} cached input" if price["cache_read"] is not None else "") + ".",
    ]
    if "batch" in variants:
        b = variants["batch"]
        lines.append(f"Batch variant: {_fmt_price(b['prompt'])} input, {_fmt_price(b['completion'])} output.")
    if long_ctx:
        lines.append(f"Prompts above {long_ctx['min_prompt_tokens']:,} tokens: {_fmt_price(long_ctx['prompt'])} "
                     f"input, {_fmt_price(long_ctx['completion'])} output.")
    prompts = [t["prompt"] for t in timed if t["prompt"] is not None]
    if prompts:
        lines.append(f"Time-of-day pricing ({len(timed)} UTC windows): input {_fmt_price(min(prompts))} to "
                     f"{_fmt_price(max(prompts))} per million tokens.")
    ctx = m.get("context_length")
    if ctx:
        lines.append(f"Context window {int(ctx):,} tokens"
                     + (f"; max output {int(top['max_completion_tokens']):,} tokens"
                        if top.get("max_completion_tokens") else "")
                     + (f"; modality {arch['modality']}" if arch.get("modality") else "") + ".")
    if m.get("hugging_face_id"):
        lines.append(f"Open weights on Hugging Face: {m['hugging_face_id']}.")
    if m.get("description"):
        lines.append("")
        lines.append(str(m["description"]).strip())
    return IngestRecord(
        natural_key=f"or:{mid}",
        url=f"https://openrouter.ai/{mid}",
        title=f"OpenRouter: {name}",
        content_text="\n".join(lines),
        published_at=created,
        lab_id=lab_id,
        meta={"model_id": mid, "provider": mid.split("/", 1)[0], "canonical_slug": m.get("canonical_slug"),
              "hugging_face_id": m.get("hugging_face_id") or None, "open_weights": bool(m.get("hugging_face_id")),
              "prompt_usd_per_mtok": price["prompt"], "completion_usd_per_mtok": price["completion"],
              "cache_read_usd_per_mtok": price["cache_read"], "long_context_pricing": long_ctx,
              "time_of_day_pricing": timed,
              "variants": variants, "context_length": ctx, "max_completion_tokens": top.get("max_completion_tokens"),
              "modality": arch.get("modality"), "created": created.isoformat(),
              "pricing_checked_at": utcnow().isoformat(timespec="seconds")},
    )


def select_models(models: list[dict], *, since: datetime, until: datetime, providers: dict[str, str | None],
                  ) -> list[tuple[dict, str | None]]:
    """Base models (no ':variant', no '~alias') from known providers, created inside [since, until]."""
    out = []
    for m in models:
        mid = str(m.get("id") or "")
        prefix = mid.split("/", 1)[0]
        if not mid or ":" in mid or prefix.startswith("~") or prefix not in providers:
            continue
        try:
            created = datetime.fromtimestamp(int(m["created"]), UTC).replace(tzinfo=None)
        except (KeyError, TypeError, ValueError, OverflowError):
            log.warning("openrouter: %s has no usable created timestamp", mid)
            continue
        if since <= created <= until:
            out.append((m, providers[prefix]))
    return out


@collector("openrouter_models", description="OpenRouter model index: new models with list prices")
def openrouter_models(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    providers: dict[str, str | None] = {**PROVIDER_LABS, **(source.config.get("providers") or {})}
    for extra in source.config.get("extra_providers") or []:  # untracked open-weight leaders (lab_id None)
        providers.setdefault(str(extra), None)
    # one request returns the whole index, so always cover the full tracked window: models created
    # before this run's incremental window still get their latest pricing refreshed into meta
    floor = min(window.since, datetime.combine(settings.backfill_start, datetime.min.time()))
    with httpx.Client(timeout=settings.http_timeout_s, headers={"User-Agent": settings.user_agent}) as client:
        data = _get_json(client, source.url or OPENROUTER_MODELS_URL)
    models = data.get("data") or []
    variants = _variants(models)
    picked = select_models(models, since=floor, until=window.until, providers=providers)
    records = []
    for m, lab in picked:
        try:
            records.append(model_to_record(m, lab, variants.get(m["id"], {})))
        except (KeyError, TypeError, ValueError) as exc:  # one odd entry must not fail the whole index
            log.warning("openrouter: skipping %s: %s", m.get("id"), exc)
    records.sort(key=lambda r: r.published_at or datetime.min, reverse=True)
    log.info("openrouter: %d models listed, %d in window from tracked providers", len(models), len(records))
    return records[: window.max_items] if window.max_items else records
