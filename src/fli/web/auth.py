"""HTTP Basic auth for the web UI and API, failing closed.

  FLI_BASIC_AUTH='user:pass'  protects write endpoints (labels, alert acks)
  FLI_AUTH_ALL=1              additionally protects every page and API read

Without FLI_BASIC_AUTH only a *local* client may write: a loopback peer that did not come through a
reverse proxy (a laptop running `fli serve`). Every other client gets 403, so a deployment that
forgot to configure credentials refuses writes instead of accepting self-asserted 'human' labels
from the internet. FLI_AUTH_ALL=1 without credentials likewise refuses non-local requests rather
than silently serving everything.

/healthz and /static stay public: load-balancer probes cannot send credentials, and the stylesheet
must load on the error page. Settings are read per request, so rotating them needs no code change
and tests can toggle them with monkeypatch. Writes also get a same-origin check: browsers resend
Basic credentials automatically, so without it any site could post labels on a logged-in user's
behalf (CSRF).
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import logging
import os
import secrets
from collections.abc import Iterable
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("fli.web.auth")

REALM = 'Basic realm="Frontier Lab Intelligence", charset="UTF-8"'
PUBLIC_PREFIXES = ("/static/", "/healthz")
# Set by reverse proxies (Caddy sets X-Forwarded-For/-Host/-Proto). A proxy on the same host connects
# from 127.0.0.1, so a loopback peer that carries any of these is a remote visitor, not a local user.
PROXY_HEADERS = frozenset({"forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                           "x-real-ip"})
NO_CREDENTIALS = ("Writes from this address need authentication, but the server has no credentials: "
                  "set FLI_BASIC_AUTH=user:password (without it only a local client may write)")
LOCKED = "FLI_AUTH_ALL=1 is set but FLI_BASIC_AUTH is not: refusing non-local requests"


def _settings():
    from fli.config import get_settings

    return get_settings()


def configured_credentials() -> tuple[str, str] | None:
    """Environment first (containers), then .env via Settings (local runs)."""
    raw = (os.environ.get("FLI_BASIC_AUTH") or _settings().fli_basic_auth or "").strip()
    user, sep, password = raw.partition(":")
    return (user, password) if sep and user and password else None


def auth_all() -> bool:
    env = os.environ.get("FLI_AUTH_ALL")
    if env is not None:
        return env.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_settings().fli_auth_all)


def check_basic(header: str | None, creds: tuple[str, str]) -> str | None:
    """User name when the Authorization header carries the configured credentials, else None."""
    if not header or header[:6].lower() != "basic ":
        return None
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    user, sep, password = decoded.partition(":")
    if not sep:
        return None
    # compare both parts unconditionally so timing does not reveal which one was wrong
    ok_user = secrets.compare_digest(user.encode(), creds[0].encode())
    ok_pass = secrets.compare_digest(password.encode(), creds[1].encode())
    return user if ok_user and ok_pass else None


def is_local_client(client: tuple[str, int] | None, header_names: Iterable[str]) -> bool:
    """True for a loopback peer that did not come through a reverse proxy. Fails closed: a missing
    peer (e.g. a Unix socket behind a proxy), a host name or any forwarding header counts as remote."""
    if PROXY_HEADERS & {h.lower() for h in header_names}:
        return False
    host = (client[0] if client else "") or ""
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1
    return ip.is_loopback


def request_is_local(request: Request) -> bool:
    client = (request.client.host, request.client.port) if request.client else None
    return is_local_client(client, request.headers.keys())


def current_user(request: Request) -> str | None:
    """The authenticated login of this request, if credentials are configured and were sent."""
    creds = configured_credentials()
    return check_basic(request.headers.get("authorization"), creds) if creds else None


def _check_same_origin(request: Request) -> None:
    """Reject browser writes whose Origin is another site. Non-browser clients send no Origin."""
    origin = request.headers.get("origin")
    if not origin:
        return
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    if origin == "null" or urlsplit(origin).netloc.lower() != host.split(",")[0].strip().lower():
        log.warning("rejected cross-origin write from %s to %s", origin, request.url.path)
        raise HTTPException(403, "Cross-origin write rejected")


def require_writer(request: Request) -> str | None:
    """FastAPI dependency for write endpoints. Returns the authenticated user name, or None for a
    local client when no credentials are configured; any other unauthenticated write is refused."""
    _check_same_origin(request)
    creds = configured_credentials()
    if creds is None:
        if request_is_local(request):
            return None
        log.warning("refused unauthenticated write to %s from %s (FLI_BASIC_AUTH not set)",
                    request.url.path, request.client.host if request.client else "unknown client")
        raise HTTPException(403, NO_CREDENTIALS)
    user = check_basic(request.headers.get("authorization"), creds)
    if user is None:
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": REALM})
    return user


class BasicAuthMiddleware:
    """Gate every request when FLI_AUTH_ALL=1 (pure ASGI, so streaming responses are untouched)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] != "http" or str(scope.get("path", "")).startswith(PUBLIC_PREFIXES)
                or not auth_all()):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        creds = configured_credentials()
        if creds is None:  # protection was asked for but cannot be enforced: only local use
            names = [k.decode("latin-1") for k in headers]
            if is_local_client(scope.get("client"), names):
                await self.app(scope, receive, send)
                return
            log.warning("FLI_AUTH_ALL=1 without FLI_BASIC_AUTH: refused %s", scope.get("path"))
            await PlainTextResponse(LOCKED, status_code=403)(scope, receive, send)
            return
        if check_basic(headers.get(b"authorization", b"").decode("latin-1"), creds) is None:
            response = PlainTextResponse("Authentication required", status_code=401,
                                         headers={"WWW-Authenticate": REALM})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
