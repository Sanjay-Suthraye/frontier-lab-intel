"""Code collectors (GitHub orgs, Hugging Face org models) against trimmed recorded responses.

tests/fixtures/code holds real API responses (anthropics repos sorted by push, claude-agent-sdk-python
releases, Qwen models, one Qwen model card) cut down to a few items. One edit: release v0.2.176 is
flagged prerelease to exercise the filter. All HTTP is served by respx; nothing hits the network.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import respx
import yaml

from fli.config import CONFIG_DIR
from fli.db.models import Document, Lab, Source
from fli.ingest.collectors import code
from fli.ingest.records import COLLECTORS, CollectorUnavailable, SourceInfo, Window
from fli.ingest.runner import persist_records

FIX = Path(__file__).parent / "fixtures" / "code"
REPOS = json.loads((FIX / "github_orgs_anthropics_repos_pushed.json").read_text())
RELEASES = json.loads((FIX / "github_releases_claude_agent_sdk_python.json").read_text())
BACKFILL = Window(since=datetime(2026, 8, 1), until=datetime(2026, 11, 4, 12))
GH_SOURCE = SourceInfo(
    id="anthropic:github",
    adapter="github_org",
    kind="github_org",
    source_class="official",
    url="https://github.com/anthropics",
    config={"org": "anthropics"},
    lab_id="anthropic",
)


@pytest.fixture(autouse=True)
def fresh_gate():
    code.GATE.remaining = code.GATE.reset_at = None
    yield
    code.GATE.remaining = code.GATE.reset_at = None


def _quota(remaining: int) -> dict:
    return {
        "x-ratelimit-remaining": str(remaining),
        "x-ratelimit-reset": str(int(time.time()) + 3600),
        "x-ratelimit-resource": "core",
    }


def _github(
    mock: respx.MockRouter, remaining_after_list: int = 50, remaining_after_release: int = 49
) -> dict:
    """Mount the recorded org listing, releases (only claude-agent-sdk-python has any) and one README."""
    routes = {
        "list": mock.get(f"{code.GITHUB_API}/orgs/anthropics/repos").mock(
            return_value=httpx.Response(200, json=REPOS, headers=_quota(remaining_after_list))
        )
    }

    def releases(request: httpx.Request) -> httpx.Response:
        body = RELEASES if "/claude-agent-sdk-python/" in request.url.path else []
        return httpx.Response(200, json=body, headers=_quota(remaining_after_release))

    routes["releases"] = mock.get(url__regex=rf"{code.GITHUB_API}/repos/anthropics/[^/]+/releases").mock(
        side_effect=releases
    )
    routes["readme"] = mock.get(url__regex=rf"{code.GITHUB_RAW}/anthropics/.+/README.md").mock(
        side_effect=lambda req: (
            httpx.Response(200, text="# OSS Scanner\n![ci](https://x/badge.svg)\nScans dependencies.")
            if "oss-scanner" in req.url.path
            else httpx.Response(404)
        )
    )
    return routes


def _unauth(settings):
    return settings.model_copy(update={"github_token": None})


# ------------------------------------------------------------------------------------ GitHub
def test_github_org_unauthenticated_budget_and_records(db, settings):
    with respx.mock() as mock:
        routes = _github(mock)
        records = code.github_org(GH_SOURCE, BACKFILL, _unauth(settings))
        params = routes["list"].calls[0].request.url.params
        assert (params["sort"], params["per_page"]) == ("pushed", "100")
        assert routes["list"].call_count == 1  # page not saturated -> no created-sorted listing
        checked = sorted(c.request.url.path.split("/")[3] for c in routes["releases"].calls)
        assert checked == ["claude-agent-sdk-python", "claude-code"]  # top-2 active repos by stars
        assert "authorization" not in routes["list"].calls[0].request.headers
    keys = [r.natural_key for r in records]
    assert sorted(k for k in keys if k.startswith("repo:")) == [
        "repo:anthropics/commerce-agents",
        "repo:anthropics/oss-scanner",
        "repo:anthropics/xls",
    ]  # created >= Aug 1
    assert sorted(k for k in keys if k.startswith("release:")) == [
        "release:anthropics/claude-agent-sdk-python@v0.2.175",
        "release:anthropics/claude-agent-sdk-python@v0.2.177",
        "release:anthropics/claude-agent-sdk-python@v0.2.178",
    ]  # v0.2.176 is a prerelease
    scanner = next(r for r in records if r.natural_key == "repo:anthropics/oss-scanner")
    assert "README:\n# OSS Scanner" in scanner.content_text and "badge.svg" not in scanner.content_text
    assert scanner.meta["stars"] == 1690 and scanner.published_at == datetime(2026, 10, 8, 17, 25, 19)
    assert scanner.title == "anthropics/oss-scanner"  # no description: the README carries the content
    rel = next(r for r in records if r.natural_key.endswith("@v0.2.177"))
    assert rel.url.endswith("/releases/tag/v0.2.177") and "on_failure" in rel.content_text
    assert rel.meta["kind"] == "release" and rel.author == "github-actions[bot]"
    assert [r.published_at for r in records] == sorted((r.published_at for r in records), reverse=True)


def test_github_org_authenticated_checks_more_repos_and_prefers_release_history(db, settings):
    with db() as s:
        s.add(Lab(id="anthropic", name="Anthropic", kind="frontier_lab", tier=1))
        s.add(Source(id=GH_SOURCE.id, adapter="github_org", kind="github_org", source_class="official"))
        s.flush()
        s.add(
            Document(
                source_id=GH_SOURCE.id,
                natural_key="release:anthropics/commerce-agents@v1.0",
                url="u",
                canonical_url="u",
                content_hash="h",
                source_class="official",
            )
        )
        s.commit()
    with respx.mock() as mock:
        routes = _github(mock)
        code.github_org(GH_SOURCE, BACKFILL, settings.model_copy(update={"github_token": "test-token"}))
        assert routes["list"].calls[0].request.headers["authorization"] == "Bearer test-token"
        checked = [c.request.url.path.split("/")[3] for c in routes["releases"].calls]
    # all 5 active, non-archived repos; the one with past releases is checked first
    assert checked[0] == "commerce-agents" and len(checked) == 5 and "courses" not in checked
    with respx.mock() as mock:
        routes = _github(mock)
        code.github_org(GH_SOURCE, BACKFILL, _unauth(settings))
        checked = [c.request.url.path.split("/")[3] for c in routes["releases"].calls]
    assert checked == ["commerce-agents", "claude-code"]  # unauthenticated top-2: history first, then stars


def test_github_org_quota_handling(db, settings):
    code.GATE.remaining, code.GATE.reset_at = 1, time.time() + 600
    with respx.mock(), pytest.raises(CollectorUnavailable, match="quota"):
        code.github_org(GH_SOURCE, BACKFILL, _unauth(settings))  # refuses before spending a call
    code.GATE.remaining = None
    with respx.mock() as mock:  # quota reaches the reserve after the first of two release checks
        routes = _github(mock, remaining_after_list=3, remaining_after_release=2)
        with pytest.raises(CollectorUnavailable, match="backfill"):
            code.github_org(GH_SOURCE, BACKFILL, _unauth(settings))
        assert routes["releases"].call_count == 1
    code.GATE.remaining = None
    incremental = Window(since=datetime(2026, 11, 2), until=datetime(2026, 11, 4, 12))
    with respx.mock(
        assert_all_called=False
    ) as mock:  # incremental: keep partial results, overlap re-reads the gap
        routes = _github(mock, remaining_after_list=3, remaining_after_release=2)
        keys = {r.natural_key for r in code.github_org(GH_SOURCE, incremental, _unauth(settings))}
    assert "repo:anthropics/oss-scanner" in keys and not any(k.startswith("release:") for k in keys)
    assert not routes["readme"].called  # READMEs only for repos created inside this run's window
    code.GATE.remaining = None
    with respx.mock() as mock:
        mock.get(f"{code.GITHUB_API}/orgs/anthropics/repos").mock(
            return_value=httpx.Response(
                403, json={"message": "API rate limit exceeded"}, headers={"x-ratelimit-remaining": "0"}
            )
        )
        with pytest.raises(CollectorUnavailable, match="rate limit"):
            code.github_org(GH_SOURCE, BACKFILL, _unauth(settings))


def test_list_repos_fetches_created_listing_only_when_saturated():
    class FakeGH:
        def __init__(self, pushed):
            self.pushed, self.calls = pushed, []

        def get(self, path, params=None):
            self.calls.append(params["sort"])
            return self.pushed if params["sort"] == "pushed" else [{"full_name": "o/new", "pushed_at": None}]

    busy = [{"full_name": f"o/r{i}", "pushed_at": "2026-10-01T00:00:00Z"} for i in range(100)]
    gh = FakeGH(busy)
    assert "o/new" in code.list_repos(gh, "o", datetime(2026, 8, 1), always_created=False) and gh.calls == [
        "pushed",
        "created",
    ]
    gh = FakeGH(busy[:99])
    code.list_repos(gh, "o", datetime(2026, 8, 1), always_created=False)
    assert gh.calls == ["pushed"]


def test_github_records_persist_idempotently(db, settings):
    with respx.mock() as mock:
        _github(mock)
        records = code.github_org(GH_SOURCE, BACKFILL, _unauth(settings))
    with db() as s:
        s.add(Lab(id="anthropic", name="Anthropic", kind="frontier_lab", tier=1))
        row = Source(
            id=GH_SOURCE.id,
            lab_id="anthropic",
            adapter="github_org",
            kind="github_org",
            source_class="official",
        )
        s.add(row)
        s.flush()
        assert persist_records(s, row, records) == (6, 0)
        records[0].meta["stars"] = 99_999  # a later run refreshes volatile metadata only
        assert persist_records(s, row, records) == (0, 0)
        s.commit()
        doc = s.query(Document).filter_by(natural_key=records[0].natural_key).one()
        assert doc.meta["stars"] == 99_999 and doc.lab_id == "anthropic"


# ------------------------------------------------------------------------------ HF org models
NOISE = code.noise_patterns(code.DEFAULT_NOISE)


@pytest.mark.parametrize(
    "name,family,noise",
    [
        ("Qwen3.8-2.4T-A95B", "Qwen3.8", False),
        ("Qwen3.8-2.4T-A95B-FP8", "Qwen3.8", True),
        ("gemma-3n-E4B-it", "gemma-3n", False),
        ("Llama-4-Scout-17B-16E-Instruct", "Llama-4-Scout", False),
        ("gpt-oss-120b", "gpt-oss", False),
        ("tipsv1-so400m14", "tipsv1", False),
        ("MoEViE-L16-384", "MoEViE", False),
        ("MobileMoE-S-Base", "MobileMoE", False),
        ("tabfm-1.0.0-jax", "tabfm-1.0.0", False),
        ("Qwen3.5-35B-A3B-GPTQ-Int4", "Qwen3.5", True),
        ("gemma-4-12B-it-qat-q4_0-gguf", None, True),
        ("Mistral-Large-4", "Mistral-Large-4", False),
        ("Qwen-Image-2.1-PE-I2I", "Qwen-Image-2.1-PE-I2I", False),
    ],
)
def test_family_and_noise_rules(name, family, noise):
    assert code.is_noise(name, NOISE) is noise
    if family:
        assert code.family_of(name, NOISE) == family


def _hf(mock: respx.MockRouter) -> dict:
    card = (FIX / "hf_card_qwen3.8-flash-next.md").read_text()
    return {
        "models": mock.get(f"{code.HF_BASE}/api/models").mock(
            return_value=httpx.Response(200, json=json.loads((FIX / "hf_models_qwen.json").read_text()))
        ),
        "cards": mock.get(url__regex=rf"{code.HF_BASE}/Qwen/.+/resolve/main/README.md").mock(
            side_effect=lambda req: (
                httpx.Response(200, text=card)
                if "Qwen3.8-Flash-Next/" in req.url.path
                else httpx.Response(401)
            )
        ),  # e.g. a gated repo: metadata-only record
    }


HF_SOURCE = SourceInfo(
    id="qwen:hf",
    adapter="hf_org_models",
    kind="hf_org",
    source_class="official",
    url="https://huggingface.co/Qwen",
    config={"org": "Qwen"},
    lab_id="qwen",
)


def test_hf_org_models_collapses_families_and_skips_reuploads(db, settings):
    with respx.mock() as mock:
        routes = _hf(mock)
        records = code.hf_org_models(HF_SOURCE, BACKFILL, settings)
        params = routes["models"].calls[0].request.url.params
        assert (params["author"], params["sort"], params["limit"]) == ("Qwen", "createdAt", "300")
        assert "cardData" in params.get_list("expand")
        assert routes["cards"].call_count == 4
    assert [r.natural_key for r in records] == [
        "hf-family:Qwen/Qwen-Drive-1.0@2026-08-27",
        "hf-family:Qwen/Qwen3.8-Flash-Next@2026-08-24",
        "hf-family:Qwen/Qwen3.8@2026-08-08",
        "hf-family:Qwen/Qwen3.8@2026-08-05",
    ]  # 08-13 had only an FP8 copy
    flash = records[1]
    assert flash.meta["variants"] == ["Qwen/Qwen3.8-Flash-Next"]
    assert flash.meta["skipped_variants"] == ["Qwen/Qwen3.8-Flash-Next-FP8"]
    assert (
        "Model card (Qwen/Qwen3.8-Flash-Next)" in flash.content_text
        and "license_link" not in flash.content_text
    )
    assert (
        flash.meta["license"] == "other"
        and flash.meta["max_params"]
        and flash.url.endswith("/Qwen/Qwen3.8-Flash-Next")
    )
    assert "Model card" not in records[0].content_text  # card refused (401) -> metadata only
    assert all(r.published_at >= BACKFILL.since for r in records)


def test_hf_org_models_rate_limit_is_unavailable_not_failure(db, settings):
    with respx.mock() as mock:
        mock.get(f"{code.HF_BASE}/api/models").mock(
            return_value=httpx.Response(429, json={"error": "slow down"})
        )
        with pytest.raises(CollectorUnavailable):
            code.hf_org_models(HF_SOURCE, BACKFILL, settings)


# ------------------------------------------------------------------------------- config file
def test_sources_research_yaml_is_consistent():
    import fli.ingest.collectors.research  # noqa: F401  (registers its collectors)

    specs = yaml.safe_load((CONFIG_DIR / "sources_research.yaml").read_text())["sources"]
    labs = {lab["id"] for lab in yaml.safe_load((CONFIG_DIR / "labs.yaml").read_text())["labs"]}
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids))
    kinds = {
        "hf_daily_papers": "hf_papers",
        "arxiv_authors": "arxiv",
        "github_org": "github_org",
        "hf_org_models": "hf_org",
    }
    for spec in specs:
        assert spec["adapter"] in COLLECTORS and kinds[spec["adapter"]] == spec["kind"]
        assert spec.get("lab") is None or spec["lab"] in labs
        assert spec["cadence_minutes"] > 0 and spec["freshness_slo_hours"] >= 24
        if spec["adapter"] in ("github_org", "hf_org_models"):
            assert spec["config"]["org"] and spec["url"].rstrip("/").endswith(spec["config"]["org"])
