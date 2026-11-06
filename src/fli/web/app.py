"""Web UI and JSON API.

Server-rendered Jinja2 pages for the investment team and the AI team, plus a small JSON API for
BIT's "Signals & Alerts" platform layer. Handlers stay thin: `queries` returns plain dicts and the
templates format them. No frontend build and no CDN: one stylesheet and one small script in /static.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import markdown
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from markupsafe import Markup
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from fli.config import get_settings
from fli.db.session import get_session, init_db
from fli.taxonomy import AUDIENCES, EVENT_TYPES
from fli.timeutil import to_naive_utc, utcnow
from fli.web import queries as q
from fli.web.auth import BasicAuthMiddleware, configured_credentials, current_user, require_writer

log = logging.getLogger("fli.web")
WEB_DIR = Path(__file__).resolve().parent
COOKIE = "fli_annotator"
PAGE_SIZE = 40
SLUGS = {"investment": "investment", "ai-team": "ai_team", "ai_team": "ai_team"}
AUDIENCE_LABELS = {"investment": "Investment", "ai_team": "AI team"}
ALERT_STATUSES = ("open", "acked", "suppressed")
NAME_RE = re.compile(r"^[\w.@ -]{1,40}$")
TRUE = ("1", "on", "true", "yes")
NAV = [("/", "Overview", "overview"), ("/feed/investment", "Investment", "investment"),
       ("/feed/ai-team", "AI team", "ai_team"), ("/alerts", "Alerts", "alerts"),
       ("/reports", "Reports", "reports"), ("/register/labs", "Labs", "labs"),
       ("/register/people", "People", "people"), ("/label", "Label", "label"),
       ("/status", "Status", "status")]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_db()  # idempotent create_all: `fli serve` also works on a fresh database
    except SQLAlchemyError:  # pragma: no cover - e.g. a read-only snapshot; serve what exists
        log.exception("init_db failed; serving the existing schema")
    # No scoring-module warm-up: the live model version is read from data/models/*.json
    # (queries.live_versions), so the web process never imports scikit-learn.
    if configured_credentials() is None:
        log.warning("FLI_BASIC_AUTH is not set: only local clients can write (labels, alert acks)")
    yield


# Swagger UI would load its assets from a CDN; /api is a self-contained reference page instead.
app = FastAPI(title="Frontier Lab Intelligence", version="0.1.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url="/api/openapi.json")
app.add_middleware(BasicAuthMiddleware)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")
# cache-busting suffix for /static links: changes whenever the stylesheet or script changes
_STATIC_FILES = sorted((WEB_DIR / "static").glob("*.*"))
STATIC_V = hashlib.sha1(b"".join(f.read_bytes() for f in _STATIC_FILES)).hexdigest()[:10]


def get_db() -> Iterator[Session]:
    s = get_session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


DB = Annotated[Session, Depends(get_db)]
Writer = Annotated[str | None, Depends(require_writer)]


# ------------------------------------------------------------------------------------ rendering
_SAFE_HREF = re.compile(r"^(https?:|mailto:|/(?![/\\])|#)", re.IGNORECASE)  # browsers read /\\x as //x
_LOCAL_PATH = re.compile(r"^/(?![/\\])")


class _SafeLinks(Treeprocessor):
    """Keep links (reports cite sources) but drop javascript:/data: targets; LLM text is untrusted."""

    def run(self, root):
        for el in root.iter("a"):
            href = (el.get("href") or "").strip()
            if not _SAFE_HREF.match(href):
                el.attrib.pop("href", None)
            elif href.lower().startswith("http"):
                el.set("rel", "noopener noreferrer")
                el.set("target", "_blank")
        for el in root.iter("img"):
            if not re.match(r"^https?:", el.get("src") or "", re.IGNORECASE):
                el.attrib.pop("src", None)


class _NoRawHtml(Extension):
    def extendMarkdown(self, md):
        md.preprocessors.deregister("html_block")  # raw HTML becomes escaped text
        md.inlinePatterns.deregister("html")
        md.treeprocessors.register(_SafeLinks(md), "fli_safe_links", -10)  # after inline + unescape


def render_md(source: str | None) -> Markup:
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", _NoRawHtml()])
    return Markup(md.convert(source or ""))


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(get_settings().timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def fmt_dt(value: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Stored datetimes are naive UTC; the UI shows the team's timezone."""
    if not isinstance(value, datetime):
        return "–"
    return value.replace(tzinfo=UTC).astimezone(_tz()).strftime(fmt)


def ago(value: datetime | None) -> str:
    if not isinstance(value, datetime):
        return "never"
    secs = (utcnow() - value).total_seconds()
    if secs < 90:
        return "just now"
    if secs < 5400:
        return f"{int(secs // 60)} min ago"
    return f"{int(secs // 3600)} h ago" if secs < 172800 else f"{int(secs // 86400)} d ago"


def money(value: float | None) -> str:
    v = float(value or 0.0)
    return f"${v:,.2f}" if v >= 1 or v == 0 else f"${v:.4f}"


def num(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, list | tuple):
        return ", ".join(num(v) for v in value[:8]) + (" …" if len(value) > 8 else "")
    return str(value)


def signed(value: float) -> str:
    return f"{value:+.3f}"


def safe_href(value: str | None) -> str:
    """Ingested URLs are untrusted: only http(s) and same-site paths become links."""
    url = (value or "").strip()
    return url if re.match(r"^https?://", url, re.IGNORECASE) or _LOCAL_PATH.match(url) else ""


for _name, _fn in {"dt": fmt_dt, "day": lambda v: fmt_dt(v, "%d %b"), "date": lambda v: fmt_dt(v, "%Y-%m-%d"),
                   "ago": ago, "money": money, "num": num, "signed": signed, "md": render_md,
                   "href": safe_href,
                   "slug": lambda a: "ai-team" if a == "ai_team" else a,
                   "audl": lambda a: AUDIENCE_LABELS.get(a, a or "–")}.items():
    templates.env.filters[_name] = _fn


def render(request: Request, template: str, /, status_code: int = 200, headers: dict | None = None,
           **ctx: Any) -> Response:
    # with auth on, labels are filed under the login, so the forms show it instead of a name field
    auth_on = configured_credentials() is not None
    login = current_user(request) if auth_on else None
    base = {"now": utcnow(), "tz": get_settings().timezone, "audiences": AUDIENCES, "nav": NAV,
            "audience_labels": AUDIENCE_LABELS, "event_types": EVENT_TYPES, "static_v": STATIC_V,
            "annotator": request.cookies.get(COOKIE, ""), "active": "", "auth_on": auth_on,
            "login": login_annotator(login) if login else None}
    return templates.TemplateResponse(request, template, {**base, **ctx}, status_code=status_code,
                                      headers=headers)


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _float(value: str | None, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value))) if value not in (None, "") else default
    except ValueError:
        return default


def _audience(slug: str | None, default: str | None = None) -> str:
    aud = SLUGS.get(slug or "") or default
    if aud is None:
        raise HTTPException(404, f"Unknown audience {slug!r}")
    return aud


def _strict_int(value: Any) -> int:
    """JSON ints or digit strings only: 1.5 or true must not silently become a label."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    raise ValueError(value)


def _not_found(what: str) -> HTTPException:
    return HTTPException(404, f"{what} not found")


# ------------------------------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
def overview_page(request: Request, s: DB):
    return render(request, "overview.html", active="overview", o=q.overview(s))


@app.get("/feed/{slug}", include_in_schema=False)
def feed_page(request: Request, slug: str, s: DB):
    """Pushed alerts first, then the lens's recommended action, then relevance. What the lens
    dismissed (investment no_action, AI-team hold) is hidden unless ?dismissed=1, and counted."""
    audience, p = _audience(slug), request.query_params
    page = max(1, _int(p.get("page"), 1))
    show_dismissed = p.get("dismissed") in TRUE
    f = q.FeedFilters(audience=audience, week=p.get("week") or "7d", lab=p.get("lab") or None,
                      event_type=p.get("event_type") if p.get("event_type") in EVENT_TYPES else None,
                      ticker=(p.get("ticker") or "").strip().upper() or None,
                      text=(p.get("q") or "").strip()[:80] or None,
                      min_p=_float(p.get("min_p"), q.DEFAULT_MIN_P.get(audience, 0.0)),
                      bit_only=p.get("bit") in TRUE, hide_dismissed=not show_dismissed, order="triage",
                      limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
    keep = {k: v for k, v in p.items() if k != "page" and v}
    no_ticker = urlencode({k: v for k, v in keep.items() if k != "ticker"})
    with_dismissed = urlencode({**{k: v for k, v in keep.items() if k != "dismissed"}, "dismissed": "1"})
    data = q.feed(s, f)
    return render(request, "feed.html", active=audience, audience=audience, f=f, page=page, data=data,
                  labs=q.lab_choices(s), weeks=q.week_options(utcnow()), page_size=PAGE_SIZE,
                  base_qs=urlencode(keep), no_ticker_qs=no_ticker, with_dismissed_qs=with_dismissed,
                  show_dismissed=show_dismissed, default_min_p=q.DEFAULT_MIN_P.get(audience, 0.0),
                  dismissed_actions=q.DISMISSED.get(audience, ()),
                  prior_live=data["live_versions"].get(audience) == q.PRIOR_VERSION)


def _detail_with_why(s: Session, dev_id: int) -> dict | None:
    """Development detail plus 'why flagged' per audience (materiality.explain, extended)."""
    d = q.development_detail(s, dev_id)
    if d is not None:
        d["why_flagged"] = q.explain_many(s, [dev_id]).get(dev_id, {})
    return d


@app.get("/developments/{dev_id}", include_in_schema=False)
def development_page(request: Request, dev_id: int, s: DB):
    d = _detail_with_why(s, dev_id)
    if d is None:
        raise _not_found(f"Development {dev_id}")
    return render(request, "development.html", d=d, fact_help=q.FACT_HELP)


@app.get("/register", include_in_schema=False)
def register_root():
    return RedirectResponse("/register/labs", status_code=307)


@app.get("/register/labs", include_in_schema=False)
def labs_page(request: Request, s: DB):
    return render(request, "labs.html", active="labs", labs=q.labs_overview(s, utcnow()))


@app.get("/register/labs/{lab_id}", include_in_schema=False)
def lab_page(request: Request, lab_id: str, s: DB):
    d = q.lab_detail(s, lab_id, utcnow())
    if d is None:
        raise _not_found(f"Lab {lab_id!r}")
    return render(request, "lab.html", active="labs", d=d)


@app.get("/register/people", include_in_schema=False)
def people_page(request: Request, s: DB):
    p = request.query_params
    filt = {"lab": p.get("lab") or None, "tier": p.get("tier") if p.get("tier") in ("A", "B", "C") else None,
            "q": (p.get("q") or "").strip()[:80] or None}
    return render(request, "people.html", active="people", people=q.people_rows(s, **filt), filt=filt,
                  labs=q.lab_choices(s))


@app.get("/register/people/{person_id}", include_in_schema=False)
def person_page(request: Request, person_id: str, s: DB):
    d = q.person_detail(s, person_id, utcnow())
    if d is None:
        raise _not_found(f"Person {person_id!r}")
    return render(request, "person.html", active="people", d=d)


@app.get("/alerts", include_in_schema=False)
def alerts_page(request: Request, s: DB):
    p = request.query_params
    status = p.get("status") if p.get("status") in (*ALERT_STATUSES, "all") else "open"
    filt = {"audience": SLUGS.get(p.get("audience") or ""), "status": status,
            "severity": p.get("severity") if p.get("severity") in ("P1", "P2", "P3") else None}
    alerts = q.alert_signals(s, q.alert_rows(s, **filt, limit=300))
    return render(request, "alerts.html", active="alerts", alerts=alerts, filt=filt)


@app.get("/reports", include_in_schema=False)
def reports_page(request: Request, s: DB):
    audience = SLUGS.get(request.query_params.get("audience") or "")
    reports = q.report_rows(s, audience)
    return render(request, "reports.html", active="reports", reports=reports, audience=audience)


@app.get("/reports/{report_id}", include_in_schema=False)
def report_page(request: Request, report_id: int, s: DB):
    r = q.report_detail(s, report_id)
    if r is None:
        raise _not_found(f"Report {report_id}")
    return render(request, "report.html", active="reports", r=r, body=render_md(r["body_md"]))


@app.get("/label", include_in_schema=False)
def label_page(request: Request, s: DB):
    p = request.query_params
    audience = _audience(p.get("audience"), "investment")
    if configured_credentials() is not None:  # labels are filed under the login, never a typed name
        login = current_user(request)
        name = login_annotator(login) if login else ""
    else:
        name = clean_annotator(request.cookies.get(COOKIE)) or ""
    others = p.get("others") == "1" and bool(name)
    annotator = f"human:{name}" if name else None
    data = q.label_queue(s, audience, annotator, max(0, _int(p.get("pos"), 0)), include_others=others)
    return render(request, "label.html", active="label", audience=audience, data=data, name=name,
                  others=others, rubric=q.RUBRIC_VERSION)


@app.get("/api", include_in_schema=False)
def api_reference(request: Request):
    return render(request, "api.html", active="", base=_base(request))


@app.get("/status", include_in_schema=False)
def status_page(request: Request, s: DB):
    return render(request, "status.html", active="status", st=q.status_page(s, utcnow()))


# ------------------------------------------------------------------------------------ writes
def clean_annotator(raw: str | None) -> str | None:
    """'human:<name>' is the label provenance; keep names short, plain and case-insensitive."""
    name = " ".join((raw or "").split()).lower().removeprefix("human:")
    return name if NAME_RE.match(name) else None


def login_annotator(user: str) -> str:
    """The authenticated login as an annotator name. Any login maps to a valid name, so labelling
    never fails because of how the account is spelled."""
    plain = re.sub(r"[^\w.@ -]+", "-", " ".join(user.split()).lower().removeprefix("human:"))[:40].strip()
    return clean_annotator(user) or plain or "user"


def _actor(request: Request, user: str | None) -> str | None:
    """Who performed a write: the login when auth is configured, else the remembered local name."""
    if user is not None:
        return f"human:{login_annotator(user)}"
    name = clean_annotator(request.cookies.get(COOKIE))
    return f"human:{name}" if name else None


def _safe_next(value: Any, default: str) -> str:
    """Only same-site relative redirects (no open redirect via `next`, incl. '/<tab>/evil' tricks)."""
    nxt = str(value or "")
    return nxt if _LOCAL_PATH.match(nxt) and not any(c.isspace() or c == "\\" for c in nxt) else default


async def _payload(request: Request) -> tuple[dict, bool]:
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "Invalid JSON body") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "Expected a JSON object")
        return data, True
    return dict(await request.form()), False


@app.post("/api/labels", tags=["write"])
async def post_label(request: Request, s: DB, user: Writer):
    """Record a human label (0 noise, 1 worth knowing, 2 material). JSON or form; upserts per annotator.
    With auth configured the annotator is the authenticated login, whatever the body says, so nobody
    can file labels (which `fli train` prefers over the LLM judge) under someone else's name."""
    data, is_json = await _payload(request)
    if user is not None:
        name = login_annotator(user)
    else:
        name = clean_annotator(str(data.get("annotator") or request.cookies.get(COOKIE) or ""))
    if name is None:
        raise HTTPException(400, "Annotator name required: letters, digits, space . _ - @ (max 40)")
    audience = SLUGS.get(str(data.get("audience") or ""))
    if audience is None:
        raise HTTPException(400, f"audience must be one of {', '.join(AUDIENCES)}")
    try:
        dev_id, value = _strict_int(data.get("development_id")), _strict_int(data.get("label"))
    except (TypeError, ValueError):
        raise HTTPException(400, "development_id and label must be integers") from None
    if value not in (0, 1, 2):
        raise HTTPException(400, "label must be 0, 1 or 2")
    if not q.development_exists(s, dev_id):
        raise _not_found(f"Development {dev_id}")
    reason = str(data.get("reason") or "").strip()[:2000] or None
    row = q.save_label(s, dev_id, audience, value, f"human:{name}", reason)
    s.commit()  # before redirecting, so the next page already sees the label
    body = {"ok": True, "label": {"id": row.id, "development_id": dev_id, "audience": audience,
                                  "label": value, "annotator": row.annotator,
                                  "rubric_version": row.rubric_version}}
    resp = JSONResponse(body) if is_json else RedirectResponse(
        _safe_next(data.get("next"), f"/developments/{dev_id}"), status_code=303)
    resp.set_cookie(COOKIE, name, max_age=365 * 86400, samesite="lax", httponly=True)
    return resp


@app.post("/api/alerts/{alert_id}/ack", tags=["write"])
async def post_ack(request: Request, alert_id: int, s: DB, user: Writer):
    """Acknowledge an alert (or set status=open|suppressed); records who (the login under auth)."""
    data, is_json = await _payload(request)
    status = str(data.get("status") or "acked")
    if status not in ALERT_STATUSES:
        raise HTTPException(400, f"status must be one of {', '.join(ALERT_STATUSES)}")
    a = q.ack_alert(s, alert_id, status, by=_actor(request, user))
    if a is None:
        raise _not_found(f"Alert {alert_id}")
    s.commit()
    if is_json:
        return JSONResponse({"ok": True, "id": a.id, "status": a.status, "acked_by": a.acked_by})
    return RedirectResponse(_safe_next(data.get("next"), "/alerts"), status_code=303)


# ------------------------------------------------------------------------------------ JSON API
def jsonable(obj: Any) -> Any:
    """Naive-UTC datetimes go out as ISO 8601 with an explicit Z."""
    if isinstance(obj, datetime):
        return obj.replace(microsecond=0).isoformat() + "Z"
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [jsonable(v) for v in (sorted(obj) if isinstance(obj, set | frozenset) else obj)]
    return obj


def _api_dt(value: str | None, name: str) -> datetime | None:
    if not value:
        return None
    parsed = to_naive_utc(value)
    if parsed is None:
        raise HTTPException(422, f"{name} must be an ISO 8601 date or datetime")
    return parsed


def _api_audience(value: str | None) -> str | None:
    if not value:
        return None
    if value not in SLUGS:
        raise HTTPException(422, f"audience must be one of {', '.join(AUDIENCES)}")
    return SLUGS[value]


def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _api_dev(c: dict, docs: dict, base: str) -> dict:
    keep = ("id", "title", "event_type", "lab_id", "lab_name", "first_seen_at", "last_seen_at", "updated_at",
            "source_classes", "source_count", "insight_count", "has_official", "has_verified", "unconfirmed",
            "unconfirmed_reason", "artifacts", "scores", "lenses", "tickers", "held_tickers", "lab_relations",
            "curated_sources", "alerts")
    # relevance = P(worth knowing); p_material is the same number under its old name, kept for pollers
    return {**{k: c[k] for k in keep}, "relevance": c["p"], "p_material": c["p"],
            "url": f"{base}/developments/{c['id']}", "sources": docs.get(c["id"], [])}


@app.get("/api/developments", tags=["api"])
def api_developments(request: Request, s: DB, audience: str | None = None, since: str | None = None,
                     min_p: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0, lab: str | None = None,
                     event_type: str | None = None, updated_since: str | None = None,
                     limit: Annotated[int, Query(ge=1, le=500)] = 100):
    """Ranked developments. `since` filters on first_seen_at (default: last 7 days unless
    `updated_since` is given); `updated_since` lets a poller pick up re-scored or grown items."""
    aud, updated = _api_audience(audience), _api_dt(updated_since, "updated_since")
    lo = _api_dt(since, "since") or (None if updated else utcnow() - timedelta(days=7))
    f = q.FeedFilters(audience=aud or AUDIENCES[0], week="all", since=lo, lab=lab, event_type=event_type,
                      min_p=min_p, updated_since=updated, limit=limit)
    if aud:
        items = q.feed(s, f)["items"]
    else:  # both audiences: rank by the higher of the two scores
        merged = {c["id"]: c for a in AUDIENCES for c in q.feed(s, replace(f, audience=a))["items"]}
        items = q.build_cards(s, list(merged))
        items.sort(key=lambda c: -(c["p"] or 0.0))
        items = items[:limit]
    docs = q.dev_documents(s, [c["id"] for c in items])
    base = _base(request)
    return JSONResponse(jsonable({"audience": aud, "since": lo, "count": len(items),
                                  "items": [_api_dev(c, docs, base) for c in items]}))


@app.get("/api/developments/{dev_id}", tags=["api"])
def api_development(request: Request, dev_id: int, s: DB):
    """Full detail, incl. why_flagged per audience: the materiality decision with the rules that fired
    (and the facts they matched), the guards, every rule that did not fire with its first failing
    condition, and the score the rules read with its top contributions."""
    d = _detail_with_why(s, dev_id)
    if d is None:
        raise _not_found(f"Development {dev_id}")
    return JSONResponse(jsonable({**d, "url": f"{_base(request)}/developments/{dev_id}"}))


@app.get("/api/alerts", tags=["api"])
def api_alerts(request: Request, s: DB, audience: str | None = None, since: str | None = None,
               severity: Annotated[str | None, Query(pattern="^P[123]$")] = None, status: str = "all",
               limit: Annotated[int, Query(ge=1, le=1000)] = 200):
    """Alerts newest first (default: last 7 days, any status), with routing fields: relevance, the
    lens action and horizon for the alert's audience, tickers [{symbol, direction, bit_weight, basis}]
    (exposure as the materiality rules count it), held_tickers (the rules' BIT holdings), lab_relations
    (context only) and why_flagged (the decision for the alert's audience, evaluated now)."""
    if status not in (*ALERT_STATUSES, "all"):
        raise HTTPException(422, "status must be open, acked, suppressed or all")
    lo = _api_dt(since, "since") or utcnow() - timedelta(days=7)
    rows = q.alert_signals(s, q.alert_rows(s, audience=_api_audience(audience), severity=severity,
                                           status=status, since=lo, limit=limit))
    base = _base(request)
    for r in rows:
        r["url"] = f"{base}/developments/{r['development_id']}"
    return JSONResponse(jsonable({"since": lo, "count": len(rows), "items": rows}))


@app.get("/api/health", tags=["api"])
def api_health(s: DB):
    """Source freshness, last ingest, open ops events and spend vs budget."""
    try:
        return JSONResponse(jsonable(q.health_summary(s)))
    except SQLAlchemyError as exc:
        log.error("health query failed: %s", exc.__class__.__name__)
        return JSONResponse({"status": "down", "error": "database unavailable"}, status_code=503)


@app.get("/healthz", include_in_schema=False)
def healthz(s: DB):
    """Liveness probe for load balancers (public even when FLI_AUTH_ALL=1)."""
    try:
        s.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse({"ok": False}, status_code=503)
    return {"ok": True}


@app.get("/api/reports", tags=["api"])
def api_reports(request: Request, s: DB, audience: str | None = None):
    rows = q.report_rows(s, _api_audience(audience))
    base = _base(request)
    return JSONResponse(jsonable({"count": len(rows), "items": [{**r, "url": f"{base}/reports/{r['id']}",
                                                                 "api_url": f"{base}/api/reports/{r['id']}"}
                                                                for r in rows]}))


@app.get("/api/reports/{report_id}", tags=["api"])
def api_report(request: Request, report_id: int, s: DB):
    r = q.report_detail(s, report_id)
    if r is None:
        raise _not_found(f"Report {report_id}")
    return JSONResponse(jsonable({**r, "url": f"{_base(request)}/reports/{report_id}",
                                  "body_html": str(render_md(r["body_md"]))}))


# ------------------------------------------------------------------------------------ errors
def _wants_json(request: Request) -> bool:
    browser_form = request.method == "POST" and "text/html" in request.headers.get("accept", "")
    return request.url.path.startswith("/api/") and not browser_form


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    headers = getattr(exc, "headers", None)
    if _wants_json(request):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=headers)
    return render(request, "error.html", status_code=exc.status_code, headers=headers, code=exc.status_code,
                  message=exc.detail)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if _wants_json(request):
        return JSONResponse({"detail": jsonable_encoder(exc.errors())}, status_code=422)
    where = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', ()))}: {e.get('msg')}" for e in exc.errors())
    return render(request, "error.html", status_code=404, code=404, message=f"Not found ({where})")
