"""The LLM budget must be a hard stop even when many calls fan out concurrently (regression test
for a reviewer's reproduction: 100 concurrent calls previously spent 4x a $0.05 budget)."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import func, select

from fli.db.models import LLMCall
from fli.llm.client import BudgetExceeded, LLMGateway
from fli.llm.prompts import Prompt


class _Usage:
    input_tokens = 1000
    output_tokens = 500
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Block:
    type = "text"
    text = '{"ok": true}'


class _Resp:
    stop_reason = "end_turn"
    usage = _Usage()
    content = [_Block()]
    model = "claude-haiku-5-5"
    _request_id = "req_test"


class _Messages:
    async def create(self, **_kw):
        await asyncio.sleep(0.005)
        return _Resp()


class _FakeClient:
    def __init__(self, **_kw):
        self.messages = _Messages()


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def test_budget_is_hard_under_concurrency(db, monkeypatch):
    monkeypatch.setattr("fli.llm.client.anthropic.AsyncAnthropic", _FakeClient)
    gw = LLMGateway()
    gw.settings = gw.settings.model_copy(update={"anthropic_api_key": "test", "llm_budget_total_usd": 0.01,
                                                 "llm_budget_daily_usd": 1.0, "llm_concurrency": 8})
    prompt = Prompt(name="t", version=1, system="classify", path=None)  # type: ignore[arg-type]
    calls = [dict(purpose="test", prompt=prompt, user=f"item {i}", schema=SCHEMA, model="claude-haiku-5-5",
                  max_tokens=4096, effort="low") for i in range(100)]
    with pytest.raises(BudgetExceeded):
        asyncio.run(gw.structured_many(calls))
    with db() as s:
        spent = s.scalar(select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)))
    assert 0 < spent <= 0.01


def test_validate_rejects_and_does_not_cache(db, monkeypatch):
    monkeypatch.setattr("fli.llm.client.anthropic.AsyncAnthropic", _FakeClient)
    gw = LLMGateway()
    gw.settings = gw.settings.model_copy(update={"anthropic_api_key": "test"})
    prompt = Prompt(name="t", version=1, system="classify", path=None)  # type: ignore[arg-type]
    kw = dict(purpose="test", prompt=prompt, user="x", schema=SCHEMA, model="claude-haiku-5-5", max_tokens=512)
    first = asyncio.run(gw.structured(**kw, validate=lambda d: False))
    assert first.status == "invalid" and first.data is None
    second = asyncio.run(gw.structured(**kw))  # not served from cache: the rejected output was never stored
    assert second.status == "ok"
