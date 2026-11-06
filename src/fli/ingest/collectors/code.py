"""Code collectors: GitHub org repositories/releases and Hugging Face org model releases.

Both are official, high-signal channels ("a lab shipped code or weights") buried in noise
(mirrors, patch releases, quantized re-uploads). Noise is collapsed deterministically here, and the
GitHub collector is budgeted to work unauthenticated (60 requests/hour shared by every org source).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from sqlalchemy import select

from fli.config import Settings
from fli.db.models import Document
from fli.db.session import session_scope
from fli.ingest.collectors.research import TransientError, hf_client, retrying
from fli.ingest.records import CollectorUnavailable, IngestRecord, SourceInfo, Window, collector
from fli.timeutil import to_naive_utc as ts

log = logging.getLogger("fli.ingest.code")

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"  # file host: no robots.txt, not metered like the API
HF_BASE = "https://huggingface.co"
README_CHARS = 6000
RELEASE_CHARS = 8000
BACKFILL_SPAN = timedelta(days=3)  # windows longer than the runner overlap (2 d) + slack are backfills
_EPOCH = datetime.min


class RateLimited(Exception):
    """GitHub quota (primary or secondary limit) exhausted."""


def clean_markdown(text: str, limit: int) -> str:
    """Drop images/badges and HTML tags that carry no meaning for extraction; cap the length."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text or "")
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"<[^>]+>", " ", text)).strip()
    return text[:limit].rstrip() + ("\n[truncated]" if len(text) > limit else "")


def _org(source: SourceInfo) -> str:
    return str((source.config or {}).get("org") or (source.url or "").rstrip("/").rsplit("/", 1)[-1])


# ===================================================================================== GitHub
class RateGate:
    """Process-wide view of the GitHub core quota, shared by concurrently running org sources."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.remaining: int | None = None
        self.reset_at: float | None = None

    def update(self, headers: httpx.Headers) -> None:
        if headers.get("x-ratelimit-resource", "core") != "core" or "x-ratelimit-remaining" not in headers:
            return
        with self._lock:
            self.remaining = int(headers["x-ratelimit-remaining"])
            self.reset_at = float(headers.get("x-ratelimit-reset") or 0) or None

    def check(self, reserve: int) -> None:
        with self._lock:
            if self.remaining is None or self.remaining > reserve:
                return
            if self.reset_at is not None and time.time() >= self.reset_at:
                self.remaining = None  # the hourly window rolled over
                return
            when = time.strftime("%H:%M UTC", time.gmtime(self.reset_at)) if self.reset_at else "unknown"
            raise RateLimited(f"GitHub quota at {self.remaining} (reserve {reserve}); resets {when}")


GATE = RateGate()


class GitHub:
    """Thin REST client: version-pinned headers, optional token, quota gate, retries on 5xx."""

    def __init__(self, settings: Settings, reserve: int) -> None:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        self.authenticated, self.reserve, self.calls = bool(settings.github_token), reserve, 0
        ua = {"User-Agent": settings.user_agent}
        self.client = httpx.Client(
            base_url=GITHUB_API, headers={**headers, **ua}, timeout=settings.http_timeout_s
        )
        self.raw = httpx.Client(headers=ua, timeout=settings.http_timeout_s)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        GATE.check(self.reserve)
        for attempt in retrying():
            with attempt:
                try:
                    resp = self.client.get(path, params=params)
                except httpx.TransportError as exc:
                    raise TransientError(str(exc)) from exc
                self.calls += 1
                GATE.update(resp.headers)
                limited = (
                    resp.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in resp.text.lower()
                )
                if resp.status_code in (403, 429) and limited:
                    raise RateLimited(f"GitHub rate limit (HTTP {resp.status_code}) on {path}")
                if resp.status_code >= 500:
                    raise TransientError(f"HTTP {resp.status_code} on {path}", resp.status_code)
                resp.raise_for_status()
                return resp.json()
        return None  # pragma: no cover

    def readme(self, repo: dict) -> str:
        """README.md from the raw file host (does not consume API quota). Best effort."""
        try:
            resp = self.raw.get(
                f"{GITHUB_RAW}/{repo['full_name']}/{repo.get('default_branch') or 'main'}/README.md"
            )
        except httpx.HTTPError:
            return ""
        return clean_markdown(resp.text, README_CHARS) if resp.status_code == 200 else ""

    def close(self) -> None:
        self.client.close()
        self.raw.close()


def list_repos(gh: GitHub, org: str, since: datetime, always_created: bool) -> dict[str, dict]:
    """Repos sorted by push time; the created-sorted listing is fetched only when the pushed page is
    saturated (100 repos all pushed in the window), because otherwise every non-fork repo created
    in the window is already on it (created_at <= pushed_at)."""
    pushed = gh.get(f"/orgs/{org}/repos", {"sort": "pushed", "direction": "desc", "per_page": 100}) or []
    repos = {r["full_name"]: r for r in pushed}
    saturated = len(pushed) >= 100 and all((ts(r.get("pushed_at")) or _EPOCH) >= since for r in pushed)
    if always_created or saturated:
        created = gh.get(f"/orgs/{org}/repos", {"sort": "created", "direction": "desc", "per_page": 100})
        for r in created or []:
            repos.setdefault(r["full_name"], r)
    return repos


def repo_meta(repo: dict) -> dict:
    return {
        "repo": repo["full_name"],
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "language": repo.get("language"),
        "topics": repo.get("topics") or [],
        "license": (repo.get("license") or {}).get("spdx_id"),
        "fork": bool(repo.get("fork")),
        "archived": bool(repo.get("archived")),
        "homepage": repo.get("homepage") or None,
        "created_at": repo.get("created_at"),
        "pushed_at": repo.get("pushed_at"),
    }


def _join(parts: list[str], sep: str) -> str:
    return sep.join(p for p in parts if p)


def repo_record(repo: dict, org: str, readme: str = "") -> IngestRecord:
    full, desc = repo["full_name"], " ".join((repo.get("description") or "").split())
    created = ts(repo.get("created_at"))
    facts = _join(
        [
            f"Language: {repo['language']}" if repo.get("language") else "",
            f"Topics: {', '.join(repo['topics'])}" if repo.get("topics") else "",
            f"Stars at ingestion: {repo.get('stargazers_count', 0)}",
            "Fork of another repository" if repo.get("fork") else "",
            f"Homepage: {repo['homepage']}" if repo.get("homepage") else "",
        ],
        " | ",
    )
    head = f"New public GitHub repository {full} created by {org}" + (
        f" on {created:%Y-%m-%d}." if created else "."
    )
    content = _join([head, f"Description: {desc}" if desc else "", facts], "\n")
    return IngestRecord(
        natural_key=f"repo:{full}",
        url=repo.get("html_url") or f"https://github.com/{full}",
        title=(f"{full}: {desc}" if desc else full)[:300],
        content_text=content + (f"\n\nREADME:\n{readme}" if readme else ""),
        published_at=created,
        author=org,
        meta={**repo_meta(repo), "kind": "repo"},
    )


def release_record(repo: dict, rel: dict) -> IngestRecord:
    full, tag = repo["full_name"], str(rel["tag_name"])
    name = " ".join((rel.get("name") or "").split())
    desc = " ".join((repo.get("description") or "").split())
    head = (
        f"Release {tag} of {full}"
        + (f" ({desc})" if desc else "")
        + (" [pre-release]" if rel.get("prerelease") else "")
    )
    extra = {
        "kind": "release",
        "tag": tag,
        "release_name": name or None,
        "prerelease": bool(rel.get("prerelease")),
    }
    return IngestRecord(
        natural_key=f"release:{full}@{tag}",
        url=rel.get("html_url") or f"https://github.com/{full}/releases/tag/{tag}",
        title=f"{full} {name if name and tag in name else f'{tag} {name}'.strip()}"[:300],
        content_text=f"{head}.\n\n{clean_markdown(rel.get('body') or '', RELEASE_CHARS)}",
        published_at=ts(rel.get("published_at") or rel.get("created_at")),
        author=(rel.get("author") or {}).get("login"),
        meta={**repo_meta(repo), **extra},
    )


def release_history(source_id: str) -> set[str]:
    """Repos that published releases before (our own documents): checked first, since most
    top-starred repos (cookbooks, courses) never publish releases."""
    with session_scope() as s:
        q = select(Document.natural_key).where(Document.source_id == source_id)
        keys = s.scalars(q.where(Document.natural_key.like("release:%")))
        return {k.removeprefix("release:").split("@", 1)[0] for k in keys}


def _new_repo_records(
    gh: GitHub, repos: dict[str, dict], org: str, window: Window, floor: datetime, cfg: dict
) -> list[IngestRecord]:
    """Repos created since the tracked-window floor (older ones only refresh stars via meta merge);
    READMEs only for repos new in this run's window, most-starred first."""
    fresh = [r for r in repos.values() if (ts(r.get("created_at")) or _EPOCH) >= floor]
    if not cfg.get("include_forks", True):
        fresh = [r for r in fresh if not r.get("fork")]
    fresh.sort(key=lambda r: -(r.get("stargazers_count") or 0))
    budget, out = int(cfg.get("readme_max_per_run", 20)), []
    for r in fresh:
        readme = ""
        if budget > 0 and (ts(r.get("created_at")) or _EPOCH) >= window.since:
            readme, budget = gh.readme(r), budget - 1
        out.append(repo_record(r, org, readme))
    return out


def _release_records(
    gh: GitHub, source: SourceInfo, repos: dict[str, dict], window: Window, cfg: dict
) -> tuple[list[IngestRecord], int, int]:
    """Releases of the top-k active repos. Returns (records, repos_checked, repos_planned)."""
    history = release_history(source.id)
    active = [
        r
        for r in repos.values()
        if not r.get("archived") and (ts(r.get("pushed_at")) or _EPOCH) >= window.since
    ]
    active.sort(key=lambda r: (r["full_name"] in history, r.get("stargazers_count") or 0), reverse=True)
    k = int(cfg.get("top_repos", 10)) if gh.authenticated else int(cfg.get("top_repos_unauthenticated", 2))
    plan, include_pre = active[:k], bool(cfg.get("include_prereleases", False))
    per_repo, per_page = int(cfg.get("max_releases_per_repo", 5)), int(cfg.get("releases_per_page", 10))
    out, checked = [], 0
    for repo in plan:
        try:
            rels = gh.get(f"/repos/{repo['full_name']}/releases", {"per_page": per_page}) or []
        except RateLimited as exc:
            log.warning("%s: release checks stopped at %d/%d: %s", source.id, checked, len(plan), exc)
            break
        checked += 1
        kept = [
            rel
            for rel in rels
            if not rel.get("draft")
            and (include_pre or not rel.get("prerelease"))
            and (ts(rel.get("published_at")) or _EPOCH) >= window.since
        ]
        out.extend(release_record(repo, rel) for rel in kept[:per_repo])
    return out, checked, len(plan)


@collector("github_org", description="GitHub REST: new repos and releases of the top repos of a lab org")
def github_org(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    cfg, org = source.config or {}, _org(source)
    floor = datetime.combine(settings.backfill_start, datetime.min.time())
    gh = GitHub(settings, reserve=int(cfg.get("rate_reserve", 2)))
    try:
        try:
            repos = list_repos(gh, org, window.since, bool(cfg.get("always_list_created", False)))
        except RateLimited as exc:
            raise CollectorUnavailable(str(exc)) from exc
        records = _new_repo_records(gh, repos, org, window, floor, cfg)
        releases, checked, planned = _release_records(gh, source, repos, window, cfg)
        if checked < planned and window.until - window.since > BACKFILL_SPAN:
            # a cut-short backfill is retried whole; an incremental run's gap is re-read by the overlap
            raise CollectorUnavailable(
                f"GitHub quota ran out during backfill ({checked}/{planned} release checks)"
            )
        counts = (len(repos), len(records), len(releases), checked, gh.calls)
        log.info("%s: %d repos, %d repo records, %d releases from %d repos, %d API calls", source.id, *counts)
    finally:
        gh.close()
    records = sorted(records + releases, key=lambda r: r.published_at or _EPOCH, reverse=True)
    return records[: window.max_items] if window.max_items else records


# ========================================================================= Hugging Face models
DEFAULT_NOISE = tuple(  # quantization / format re-uploads of an existing release
    "gguf awq gptq mlx fp8 fp4 nvfp4 mxfp4 int4 int8 w4a16 w8a8 w4a8 bnb 4bit 8bit exl2 nf4 onnx".split()
    + ["quantized", "qat", "q[2-8]"]
)
_SIZE_TOKEN = re.compile(
    r"(?:\d+x)?\d+(?:\.\d+)?[kmbt]"  # 7B, 2.4T, 270m, 8x7B, 128k
    r"|a\d+(?:\.\d+)?[mb]|e\d+(?:\.\d+)?b|\d+e"  # active (A3B), effective (E4B) params, experts (16E)
    r"|(?:ti|s|b|l|h|g|so\d+m?)\d{1,2}|so\d+m|patch\d+"  # ViT size/patch: B16, so400m14, patch14
    r"|xs|s|m|l|xl|xxl|\d{3,4}",  # size letters (MobileMoE-S), resolution / date codes (384, 0905)
    re.IGNORECASE,
)
_VARIANT_TOKENS = set("base instruct chat it pt sft hf dpo lowres highres jax pytorch flax tf keras".split())
_EXPAND = "createdAt downloads likes pipeline_tag tags cardData gated library_name safetensors".split()


def noise_patterns(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern]:
    return [re.compile(rf"(?<![a-z0-9])(?:{p})(?![a-z0-9])", re.IGNORECASE) for p in patterns]


def is_noise(name: str, noise: list[re.Pattern]) -> bool:
    return any(rx.search(name) for rx in noise)


def family_of(name: str, noise: list[re.Pattern]) -> str:
    """'Qwen3.8-2.4T-A95B-FP8' -> 'Qwen3.8', 'gemma-3n-E4B-it' -> 'gemma-3n': drop size, variant
    and format tokens so the sizes of one release collapse into one record."""
    toks = [t for t in re.split(r"[-_]", name) if t]
    keep = [
        t
        for t in toks
        if not (_SIZE_TOKEN.fullmatch(t) or t.lower() in _VARIANT_TOKENS or is_noise(t, noise))
    ]
    return "-".join(keep) or name


def _card(model: Any, key: str) -> Any:
    card = getattr(model, "card_data", None)
    return card.get(key) if card is not None and hasattr(card, "get") else None


def _license(model: Any) -> str | None:
    tags = (t.split(":", 1)[1] for t in (model.tags or []) if t.startswith("license:"))
    return str(_card(model, "license") or next(tags, "")) or None


def _representative(models: list[Any]) -> Any:
    """Most-liked (then most-downloaded, then shortest-named) variant: its card describes the family."""
    return min(models, key=lambda m: (-(m.likes or 0), -(m.downloads or 0), len(m.id), m.id))


def list_new_models(api: HfApi, org: str, since: datetime, max_models: int) -> list[Any]:
    """Models created since `since`, newest first (the API sorts by createdAt descending)."""
    out = []
    for m in api.list_models(author=org, sort="created_at", limit=max_models, expand=_EXPAND):
        if (created := ts(m.created_at)) is not None and created < since:
            break
        out.append(m)
    return out


def fetch_card(client: httpx.Client, repo_id: str, limit: int) -> str:
    """Model card body without YAML front matter. Best effort (gated repos may refuse)."""
    try:
        resp = client.get(f"{HF_BASE}/{repo_id}/resolve/main/README.md")
    except httpx.HTTPError:
        return ""
    if resp.status_code != 200:
        return ""
    text = resp.text
    if text.startswith("---") and (end := text.find("\n---", 3)) != -1:
        text = text[end + 4 :]
    return clean_markdown(text, limit)


def family_record(
    org: str, family: str, day: date, keep: list[Any], skipped: list[Any], card: str
) -> IngestRecord:
    rep, ids = _representative(keep), sorted(m.id for m in keep)
    params = max((m.safetensors.total for m in keep if getattr(m, "safetensors", None)), default=None)
    meta = {
        "org": org,
        "family": family,
        "day": day.isoformat(),
        "variants": ids,
        "skipped_variants": sorted(m.id for m in skipped),
        "representative": rep.id,
        "downloads": sum(m.downloads or 0 for m in keep),
        "likes": sum(m.likes or 0 for m in keep),
        "pipeline_tag": rep.pipeline_tag,
        "license": _license(rep),
        "library_name": rep.library_name,
        "gated": any(bool(m.gated) for m in keep),
        "max_params": params,
        "base_models": _card(rep, "base_model"),
        "hf_url": f"{HF_BASE}/{rep.id}",
        "tags": [t for t in (rep.tags or []) if not t.startswith(("region:", "endpoints"))][:20],
    }
    facts = _join(
        [
            f"Pipeline: {rep.pipeline_tag}" if rep.pipeline_tag else "",
            f"License: {meta['license']}" if meta["license"] else "",
            f"Library: {rep.library_name}" if rep.library_name else "",
            f"Largest variant: {params / 1e9:.1f}B parameters" if params else "",
            "Gated access" if meta["gated"] else "",
        ],
        " | ",
    )
    plural = "s" if len(ids) > 1 else ""
    content = _join(
        [
            f"Hugging Face model release by {org}: {family} ({len(ids)} variant{plural}, created {day}).",
            "Variants: " + ", ".join(ids),
            facts,
            ("Quantized/format re-uploads collapsed: " + ", ".join(meta["skipped_variants"]))
            if skipped
            else "",
        ],
        "\n",
    )
    return IngestRecord(
        natural_key=f"hf-family:{org}/{family}@{day.isoformat()}",
        url=meta["hf_url"],
        title=rep.id if len(ids) == 1 else f"{org}/{family} ({len(ids)} variants)",
        content_text=content + (f"\n\nModel card ({rep.id}):\n{card}" if card else ""),
        published_at=min((ts(m.created_at) for m in keep if m.created_at), default=None),
        author=org,
        meta=meta,
    )


@collector(
    "hf_org_models", description="HF Hub: new models of a lab org; sizes collapsed, re-uploads skipped"
)
def hf_org_models(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    cfg, org = source.config or {}, _org(source)
    noise = noise_patterns(cfg.get("noise_patterns") or DEFAULT_NOISE)
    api = HfApi(token=settings.hf_token or False)
    try:
        models = list_new_models(api, org, window.since, int(cfg.get("max_models", 300)))
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            raise CollectorUnavailable(f"Hugging Face rate limit: {exc}") from exc
        raise
    groups: dict[tuple[date, str], list[Any]] = defaultdict(list)
    for m in models:
        if (created := ts(m.created_at)) is not None:
            groups[(created.date(), family_of(m.id.split("/", 1)[-1], noise))].append(m)
    records, cards_left = [], int(cfg.get("cards_max_per_run", 40))
    with hf_client(settings) as client:
        for (day, family), members in sorted(groups.items(), reverse=True):
            keep = [m for m in members if not is_noise(m.id.split("/", 1)[-1], noise)]
            if not keep:
                continue  # only re-uploads (e.g. an FP8 copy of an older release)
            limit = int(cfg.get("card_chars", README_CHARS))
            card = fetch_card(client, _representative(keep).id, limit) if cards_left > 0 else ""
            cards_left -= 1
            records.append(family_record(org, family, day, keep, [m for m in members if m not in keep], card))
    log.info("%s: %d new models -> %d family records", source.id, len(models), len(records))
    return records[: window.max_items] if window.max_items else records
