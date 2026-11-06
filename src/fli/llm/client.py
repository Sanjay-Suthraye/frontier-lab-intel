"""LLM gateway - the only module that calls the model API.

What every caller gets for free:
  * JSON-schema structured outputs (output_config.format) parsed into dicts
  * a response cache keyed by (model, prompt version, input): sampling parameters are rejected by
    the current API version, so reproducibility comes from persisting outputs
  * a cost ledger (one llm_call row per call) - the source of the design doc's cost table
  * hard budget stops (total and daily) checked before each call
  * refusal / max_tokens handling, and server-side refusal fallbacks on Sonnet/Opus
  * a Message Batches path (50% cheaper) for backfills
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from sqlalchemy import func, select

from fli.config import Settings, get_settings
from fli.db.models import LLMCall, ResponseCache
from fli.db.session import session_scope
from fli.llm.prompts import Prompt
from fli.timeutil import utcnow

log = logging.getLogger("fli.llm")

# USD per million tokens (Anthropic first-party pricing). Haiku 5.5 bills a second rate card when
# the prompt exceeds 100K tokens; our prompts stay far below that by design.
PRICES: dict[str, dict] = {
    "claude-haiku-5-5": {
        "in": 0.10, "out": 0.50, "cache_read": 0.01,
        "long": {"in": 0.50, "out": 2.50, "cache_read": 0.05}, "long_threshold": 100_000,
    },
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.20},
    "claude-opus-5-5": {"in": 4.00, "out": 20.00, "cache_read": 0.20},
    "claude-fable-5-1": {"in": 10.00, "out": 50.00, "cache_read": 0.25},
}
CACHE_WRITE_MULT = 1.25
BATCH_DISCOUNT = 0.5
# Fail fast on a hung connection (observed once in a backfill); the SDK retries 3 times.
API_TIMEOUT = anthropic.Timeout(180.0, connect=15.0)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}  # Haiku 5.5 has no server-side fallback


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict | None
    status: str  # ok | cached | refusal | max_tokens | invalid | error
    cost_usd: float = 0.0
    model: str = ""
    prompt_version: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.data is not None


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0,
             cache_write: int = 0, batch: bool = False) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    rates = p
    if "long" in p and (input_tokens + cache_read + cache_write) > p["long_threshold"]:
        rates = {**p, **p["long"]}
    usd = (
        input_tokens * rates["in"]
        + cache_write * rates["in"] * CACHE_WRITE_MULT
        + cache_read * rates["cache_read"]
        + output_tokens * rates["out"]
    ) / 1_000_000
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def cache_key(model: str, prompt_version: str, user: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt_version}\x00{user}".encode()).hexdigest()


class LLMGateway:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; LLM stages cannot run")
        self._key = self.settings.anthropic_api_key
        self.sync_client = anthropic.Anthropic(api_key=self._key, max_retries=3, timeout=API_TIMEOUT)
        # The async client's connection pool and the semaphore bind to the event loop that first
        # uses them, so they are created per running loop (one gateway may serve several
        # asyncio.run() calls from the CLI or the scheduler).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: anthropic.AsyncAnthropic | None = None
        self._sem: asyncio.Semaphore | None = None
        self._reserved = 0.0  # estimated cost of calls in flight
        self._budget_lock = threading.Lock()

    def _async_parts(self) -> tuple[anthropic.AsyncAnthropic, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._client is None or self._sem is None:
            self._loop = loop
            self._client = anthropic.AsyncAnthropic(api_key=self._key, max_retries=3, timeout=API_TIMEOUT)
            self._sem = asyncio.Semaphore(self.settings.llm_concurrency)
        return self._client, self._sem

    # ------------------------------------------------------------------ budget
    def spent_usd(self, since=None) -> float:
        with session_scope() as s:
            q = select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0))
            if since is not None:
                q = q.where(LLMCall.created_at >= since)
            return float(s.scalar(q) or 0.0)

    def check_budget(self, estimate: float) -> None:
        total = self.spent_usd()
        if total + estimate > self.settings.llm_budget_total_usd:
            raise BudgetExceeded(
                f"LLM total budget ${self.settings.llm_budget_total_usd:.2f} reached (spent ${total:.2f})"
            )
        midnight = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today = self.spent_usd(since=midnight)
        if today + estimate > self.settings.llm_budget_daily_usd:
            raise BudgetExceeded(
                f"LLM daily budget ${self.settings.llm_budget_daily_usd:.2f} reached (spent ${today:.2f} today)"
            )

    @staticmethod
    def estimate(model: str, system: str, user: str, max_tokens: int, batch: bool = False) -> float:
        approx_in = int((len(system) + len(user)) / 3.5)
        return cost_usd(model, approx_in, int(max_tokens * 0.5), batch=batch)

    # ------------------------------------------------------------------ cache + ledger
    def _cache_get(self, key: str) -> dict | None:
        with session_scope() as s:
            row = s.get(ResponseCache, key)
            return dict(row.response) if row is not None else None

    def _record(self, *, purpose: str, model: str, version: str, status: str, usage=None,
                cost: float = 0.0, latency_ms: int = 0, error: str | None = None, request_id: str | None = None,
                batch_id: str | None = None, ref: dict | None = None, cache_put: tuple[str, dict] | None = None) -> None:
        with session_scope() as s:
            s.add(LLMCall(
                purpose=purpose, model=model, prompt_version=version, status=status,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cost_usd=cost, latency_ms=latency_ms, error=(error or None) and error[:2000],
                request_id=request_id, batch_id=batch_id, ref=ref or {},
            ))
            if cache_put is not None:
                key, data = cache_put
                row = s.get(ResponseCache, key)
                if row is None:
                    s.add(ResponseCache(key=key, purpose=purpose, model=model, prompt_version=version, response=data))
                else:  # a validated fresh answer (use_cache=False retry) replaces a stale cached one
                    row.response, row.created_at = data, utcnow()

    # ------------------------------------------------------------------ request building
    def _params(self, model: str, prompt: Prompt, user: str, schema: dict, max_tokens: int, effort: str) -> dict:
        return {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": prompt.system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": effort},
        }

    @staticmethod
    def _parse(resp) -> tuple[dict | None, str, str | None]:
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            return None, "refusal", f"refusal: {getattr(details, 'category', None)}"
        if resp.stop_reason == "max_tokens":
            return None, "max_tokens", "hit max_tokens before finishing"
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if not text:
            return None, "invalid", "no text block in response"
        try:
            return json.loads(text), "ok", None
        except json.JSONDecodeError as exc:
            return None, "invalid", f"json decode: {exc}"

    # ------------------------------------------------------------------ synchronous-style call
    async def structured(self, *, purpose: str, prompt: Prompt, user: str, schema: dict, model: str,
                         max_tokens: int = 4096, effort: str = "low", ref: dict | None = None,
                         use_cache: bool = True, validate: Callable[[dict], bool] | None = None) -> LLMResult:
        version = prompt.version_for(schema)
        key = cache_key(model, version, user)
        if use_cache and (cached := self._cache_get(key)) is not None:
            self._record(purpose=purpose, model=model, version=version, status="cached", ref=ref)
            return LLMResult(cached, "cached", 0.0, model, version)

        params = self._params(model, prompt, user, schema, max_tokens, effort)
        client, sem = self._async_parts()
        for attempt in (1, 2):
            # Reserve the estimate inside the semaphore so a fan-out of concurrent calls cannot all
            # pass the check against the same stale total (the budget is a hard stop).
            est = self.estimate(model, prompt.system, user, max_tokens)
            async with sem:
                self._reserve(est)
                try:
                    t0 = time.monotonic()
                    try:
                        if model in FALLBACK_MODELS:
                            resp = await client.beta.messages.create(
                                **params, betas=[FALLBACK_BETA], fallbacks="default"
                            )
                        else:
                            resp = await client.messages.create(**params)
                    except anthropic.BadRequestError as exc:
                        self._record(purpose=purpose, model=model, version=version, status="error",
                                     error=f"400: {exc.message}", ref=ref)
                        return LLMResult(None, "error", 0.0, model, version, f"400: {exc.message}")
                    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                        self._record(purpose=purpose, model=model, version=version, status="error",
                                     error=f"{exc.__class__.__name__}: {exc}", ref=ref)
                        return LLMResult(None, "error", 0.0, model, version, str(exc))
                    latency = int((time.monotonic() - t0) * 1000)
                    usage = resp.usage
                    served = getattr(resp, "model", None) or model  # differs when a refusal fallback answered
                    cost = cost_usd(served if served in PRICES else model, usage.input_tokens or 0,
                                    usage.output_tokens or 0, getattr(usage, "cache_read_input_tokens", 0) or 0,
                                    getattr(usage, "cache_creation_input_tokens", 0) or 0)
                    data, status, err = self._parse(resp)
                    if data is not None and validate is not None and not validate(data):
                        data, status, err = None, "invalid", "rejected by caller validation (not cached)"
                    self._record(purpose=purpose, model=served, version=version, status=status, usage=usage,
                                 cost=cost, latency_ms=latency, error=err,
                                 request_id=getattr(resp, "_request_id", None), ref=ref,
                                 cache_put=(key, data) if data is not None else None)
                finally:
                    self._release(est)
            if status == "max_tokens" and attempt == 1 and max_tokens < 32_000:
                params["max_tokens"] = max_tokens = min(32_000, max_tokens * 2)  # thinking counts toward max_tokens
                continue
            return LLMResult(data, status, cost, served, version, err)
        return LLMResult(None, "max_tokens", 0.0, model, version, "max_tokens after retry")

    def _reserve(self, est: float) -> None:
        with self._budget_lock:
            self.check_budget(est + self._reserved)
            self._reserved += est

    def _release(self, est: float) -> None:
        with self._budget_lock:
            self._reserved = max(0.0, self._reserved - est)

    async def structured_many(self, calls: list[dict]) -> list[LLMResult]:
        """Run many `structured(**kwargs)` calls concurrently (bounded by llm_concurrency).
        A BudgetExceeded in any call stops the remaining ones."""

        async def one(kw: dict) -> LLMResult:
            return await self.structured(**kw)

        tasks = [asyncio.create_task(one(kw)) for kw in calls]
        results: list[LLMResult] = []
        try:
            for t in tasks:
                results.append(await t)
        except BudgetExceeded:
            for t in tasks:
                t.cancel()
            raise
        return results

    # ------------------------------------------------------------------ batch path (backfills)
    def structured_batch(self, *, purpose: str, prompt: Prompt, items: list[tuple[str, str, dict]],
                         schema: dict, model: str, max_tokens: int = 4096, effort: str = "low",
                         poll_seconds: int = 20, timeout_s: int = 6 * 3600, use_cache: bool = True,
                         validate: Callable[[dict], bool] | None = None) -> dict[str, LLMResult]:
        """items: (custom_id, user_text, ref). custom_id must match ^[A-Za-z0-9_-]{1,64}$.
        Fallbacks are not available on the Batches API; Haiku has none anyway."""
        version = prompt.version_for(schema)
        out: dict[str, LLMResult] = {}
        pending: list[tuple[str, str, dict, str]] = []
        for cid, user, ref in items:
            key = cache_key(model, version, user)
            if use_cache and (cached := self._cache_get(key)) is not None:
                self._record(purpose=purpose, model=model, version=version, status="cached", ref=ref)
                out[cid] = LLMResult(cached, "cached", 0.0, model, version)
            else:
                pending.append((cid, user, ref, key))
        if not pending:
            return out
        self.check_budget(sum(self.estimate(model, prompt.system, u, max_tokens, batch=True) for _, u, _, _ in pending))
        by_id = {cid: (ref, key) for cid, _, ref, key in pending}
        requests = [
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                **self._params(model, prompt, user, schema, max_tokens, effort)))
            for cid, user, _, _ in pending
        ]
        batch = self.sync_client.messages.batches.create(requests=requests)
        log.info("submitted batch %s with %d requests (%s)", batch.id, len(requests), purpose)
        started = time.monotonic()
        while True:
            batch = self.sync_client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() - started > timeout_s:
                self.sync_client.messages.batches.cancel(batch.id)
                raise TimeoutError(f"batch {batch.id} did not finish in {timeout_s}s")
            time.sleep(poll_seconds)
        for result in self.sync_client.messages.batches.results(batch.id):
            ref, key = by_id.get(result.custom_id, ({}, None))
            if result.result.type != "succeeded":
                err = f"batch {result.result.type}"
                self._record(purpose=purpose, model=model, version=version, status="error", error=err,
                             batch_id=batch.id, ref=ref)
                out[result.custom_id] = LLMResult(None, "error", 0.0, model, version, err)
                continue
            msg = result.result.message
            usage = msg.usage
            cost = cost_usd(model, usage.input_tokens or 0, usage.output_tokens or 0,
                            getattr(usage, "cache_read_input_tokens", 0) or 0,
                            getattr(usage, "cache_creation_input_tokens", 0) or 0, batch=True)
            data, status, err = self._parse(msg)
            if data is not None and validate is not None and not validate(data):
                data, status, err = None, "invalid", "rejected by caller validation (not cached)"
            self._record(purpose=purpose, model=model, version=version, status=status, usage=usage, cost=cost,
                         error=err, batch_id=batch.id, ref=ref,
                         cache_put=(key, data) if (data is not None and key) else None)
            out[result.custom_id] = LLMResult(data, status, cost, model, version, err)
        return out


def run_async(coro):
    """Run a coroutine from sync code (CLI, scheduler jobs)."""
    return asyncio.run(coro)
