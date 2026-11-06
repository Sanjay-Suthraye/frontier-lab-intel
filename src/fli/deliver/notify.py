"""Outbound notifications: Slack incoming webhooks and Healthchecks.io pings.

Both are best-effort by design: a failure is logged and returned to the caller (which records it
and may retry), but never raises into a pipeline stage. Webhook URLs and ping keys are secrets, so
they never appear in logs, exceptions we store, or the database.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import httpx

from fli.config import get_settings

log = logging.getLogger("fli.deliver.notify")

TIMEOUT_S = 10.0
AUDIENCE_LABELS = {"investment": "Investment", "ai_team": "AI team"}
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class SendResult:
    ok: bool
    error: str | None = None


def public_base_url() -> str | None:
    """Base URL of the web UI for deep links (FLI_PUBLIC_URL); optional."""
    url = os.environ.get("FLI_PUBLIC_URL") or getattr(get_settings(), "public_base_url", None)
    return url.rstrip("/") if url else None


def _scrub(message: str, secret: str) -> str:
    return message.replace(secret, "<redacted>") if secret else message


def md_to_mrkdwn(md: str) -> str:
    """Slack mrkdwn is not Markdown: *bold*, <url|text> links, no headings."""
    text = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1).replace('|', '/')}>", md)
    text = _BOLD.sub(r"*\1*", text)
    return _HEADING.sub(r"*\1*", text)


def post_slack(text: str, blocks: list[dict] | None = None, *, webhook_url: str | None = None,
               timeout: float = TIMEOUT_S) -> SendResult:
    """POST a message to the configured Slack incoming webhook."""
    url = webhook_url or get_settings().slack_webhook_url
    if not url:
        return SendResult(False, "slack not configured")
    payload: dict = {"text": text[:3000]}
    if blocks:
        payload["blocks"] = blocks
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
    except httpx.HTTPError as exc:
        err = _scrub(f"{exc.__class__.__name__}: {exc}", url)[:300]
        log.warning("slack delivery failed: %s", err)
        return SendResult(False, err)
    if resp.status_code == 200:
        return SendResult(True)
    err = f"HTTP {resp.status_code}: {resp.text[:200]}"
    log.warning("slack delivery failed: %s", err)
    return SendResult(False, err)


def alert_message(alert) -> dict:
    """Slack payload (text + blocks) for an Alert row."""
    label = AUDIENCE_LABELS.get(alert.audience, alert.audience)
    header = f"[{alert.severity}] {label}: {alert.title}"
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": header[:150]}},
        {"type": "section", "text": {"type": "mrkdwn", "text": md_to_mrkdwn(alert.body_md)[:2900]}},
    ]
    if base := public_base_url():
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"<{base}/alerts/{alert.id}|Open in Frontier Lab Intelligence>"}]})
    return {"text": header[:3000], "blocks": blocks}


def ops_message(events: list) -> str:
    """One Slack message for a batch of new critical ops events (avoids a burst of pings)."""
    lines = [f"*FLI health: {len(events)} new critical event{'s' if len(events) != 1 else ''}*"]
    lines += [f"- `{e.kind}` {e.subject}: {e.message[:300]}" for e in events[:20]]
    if len(events) > 20:
        lines.append(f"... and {len(events) - 20} more (see /status)")
    return "\n".join(lines)


def ping_healthcheck(slug: str, signal: str = "", body: str | None = None, *, timeout: float = TIMEOUT_S) -> bool:
    """Ping Healthchecks.io: '<base>/<slug>' (success), '/start' or '/fail'.

    `?create=1` auto-provisions the check on first ping when the base uses a project ping key.
    Returns False when not configured or when the ping failed (logged, never raised)."""
    base = get_settings().healthchecks_ping_base
    if not base:
        return False
    url = f"{base.rstrip('/')}/{slug}" + (f"/{signal}" if signal else "")
    try:
        resp = httpx.post(url, params={"create": "1"}, content=(body or "")[:10_000].encode(), timeout=timeout)
    except httpx.HTTPError as exc:
        log.warning("healthchecks ping %s %s failed: %s", slug, signal or "success", exc.__class__.__name__)
        return False
    if resp.status_code >= 300:
        log.warning("healthchecks ping %s %s returned HTTP %s", slug, signal or "success", resp.status_code)
        return False
    return True
