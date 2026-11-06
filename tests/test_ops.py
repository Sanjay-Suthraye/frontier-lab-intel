"""Health events (stale, failing, never-run and parser-empty sources, job failures, a stalled or silent
worker, unreadable scoring models, budgets; resolution and day-level dedup), Healthchecks pings,
snapshot round trip on SQLite, and scheduler wiring. No network."""

from __future__ import annotations

import gzip
import signal
from datetime import datetime, timedelta

import httpx
import pytest
import respx
from sqlalchemy import create_engine, select

from fli import config as fli_config
from fli.db.models import (
    ApiUsage,
    Development,
    Document,
    Insight,
    JobRun,
    Lab,
    LLMCall,
    OpsEvent,
    Source,
)

NOW = datetime(2026, 11, 4, 12, 0, 0)
WEBHOOK = "https://hooks.slack.test/services/T000/B000/XXXX"
HC_BASE = "https://hc-ping.test/ping-key"


@pytest.fixture(autouse=True)
def _no_outbound(monkeypatch):
    for var in ("SLACK_WEBHOOK_URL", "HEALTHCHECKS_PING_BASE"):
        monkeypatch.setenv(var, "")
    fli_config.get_settings.cache_clear()
    yield
    fli_config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_deployed_models(monkeypatch, tmp_path):
    """The health check reads the deployed scoring model files: never the repo's data/models here."""
    from fli.analyze import scoring

    monkeypatch.setattr(scoring, "MODELS_DIR", tmp_path / "models")


def _set_env(monkeypatch, **values):
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    fli_config.get_settings.cache_clear()


def _source(sid: str, **fields) -> Source:
    base = dict(id=sid, adapter="rss", kind="official_blog", source_class="official", enabled=True,
                freshness_slo_hours=24 * 14, created_at=NOW - timedelta(days=60))
    return Source(**{**base, **fields})


def _events(db, **filters) -> list[OpsEvent]:
    with db() as s:
        return list(s.scalars(select(OpsEvent).filter_by(**filters).order_by(OpsEvent.id)))


def _open(out: dict, kind: str) -> dict[str, dict]:
    return {e["subject"]: e for e in out["open_events"] if e["kind"] == kind}


def _run(job: str, start: datetime, status: str = "ok", minutes: int = 5) -> JobRun:
    end = None if status == "running" else start + timedelta(minutes=minutes)
    return JobRun(job=job, started_at=start, finished_at=end, status=status)


def _worker_events(out: dict) -> list[tuple[str, str]]:
    return [(e["kind"], e["severity"]) for e in out["open_events"] if e["subject"] == "worker:pipeline"]


# ------------------------------------------------------------------------------- health
def test_stale_and_failing_sources_open_then_resolve(db, monkeypatch):
    from fli.ops.health import run_health_checks

    _set_env(monkeypatch, SLACK_WEBHOOK_URL=WEBHOOK)
    with db() as s:
        s.add_all([
            _source("lab:stale", last_run_at=NOW - timedelta(hours=1), last_success_at=NOW - timedelta(hours=1),
                    last_new_item_at=NOW - timedelta(days=20)),
            _source("lab:failing", last_run_at=NOW, consecutive_failures=3, last_error="HTTPStatusError: 503",
                    last_new_item_at=NOW - timedelta(days=1)),
            _source("lab:fresh", last_run_at=NOW, last_success_at=NOW, last_new_item_at=NOW - timedelta(days=2)),
            _source("x:never"),
        ])
        s.commit()
    with respx.mock(assert_all_called=False) as router:
        slack = router.post(WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        first = run_health_checks(now=NOW)
        assert first["status"] == "crit" and first["new_events"] == 2
        assert {(e["kind"], e["subject"], e["severity"]) for e in first["open_events"]} == {
            ("source_failing", "lab:failing", "crit"), ("source_stale", "lab:stale", "warn")}
        assert "20.0 days (SLO 14 days)" in next(e["message"] for e in first["open_events"] if e["kind"] == "source_stale")
        assert first["sources"]["never_run"] == 1 and first["sources"]["failing"] == 1
        assert slack.call_count == 1 and "lab:failing" in slack.calls[0].request.content.decode()

        again = run_health_checks(now=NOW + timedelta(minutes=15))
        assert again["new_events"] == 0 and slack.call_count == 1  # deduped, not re-announced
        assert len(_events(db)) == 2

        with db() as s:
            s.get(Source, "lab:failing").consecutive_failures = 0
            s.commit()
        fixed = run_health_checks(now=NOW + timedelta(minutes=30))
        assert fixed["status"] == "warn" and fixed["resolved_events"] == 1
        assert _events(db, kind="source_failing")[0].resolved_at == NOW + timedelta(minutes=30)

    next_day = run_health_checks(now=NOW + timedelta(days=1))
    stale = _events(db, kind="source_stale")
    assert len(stale) == 2 and stale[0].resolved_at is not None and stale[1].resolved_at is None
    assert stale[0].delivered["superseded_by"].endswith(f"{NOW + timedelta(days=1):%Y-%m-%d}")
    assert stale[1].dedup_key == f"source_stale:lab:stale:{NOW + timedelta(days=1):%Y-%m-%d}"
    assert [e["subject"] for e in next_day["open_events"]] == ["lab:stale"]


def test_parser_empty_needs_a_history_of_items(db):
    from fli.ops.health import run_health_checks

    with db() as s:
        s.add(_source("lab:feed", last_run_at=NOW, last_success_at=NOW, last_new_item_at=NOW - timedelta(days=1)))
        s.add(_source("lab:quiet", last_run_at=NOW, last_success_at=NOW, last_new_item_at=NOW - timedelta(days=1)))
        for i in range(8):  # 5 normal runs, then 3 empty ones
            at = NOW - timedelta(hours=8 - i)
            s.add(JobRun(job="ingest:lab:feed", started_at=at, status="ok", items_in=12 if i < 5 else 0))
            s.add(JobRun(job="ingest:lab:quiet", started_at=at, status="ok", items_in=1 if i == 0 else 0))
        s.commit()
    out = run_health_checks(now=NOW)
    events = [(e["kind"], e["subject"]) for e in out["open_events"]]
    assert events == [("parser_empty", "lab:feed")]  # a quiet blog (median 0) is not a broken parser
    assert "typical run returns 12" in out["open_events"][0]["message"]


def test_failed_jobs_warn_escalate_and_recover(db):
    from fli.ops.health import run_health_checks

    with db() as s:
        s.add_all([
            JobRun(job="stage:extract", started_at=NOW - timedelta(hours=2), status="error", error="KeyError: x"),
            JobRun(job="stage:extract", started_at=NOW - timedelta(hours=1), status="error", error="KeyError: x"),
            JobRun(job="stage:embed", started_at=NOW - timedelta(hours=3), status="error", error="OSError: disk"),
            JobRun(job="stage:embed", started_at=NOW - timedelta(hours=2), status="ok"),
            JobRun(job="stage:triage", started_at=NOW - timedelta(hours=30), status="error"),  # outside 24 h
            JobRun(job="ingest:lab:feed", started_at=NOW, status="error"),  # sources have their own check
        ])
        s.commit()
    out = run_health_checks(now=NOW)
    assert [(e["subject"], e["severity"]) for e in out["open_events"]] == [("stage:extract", "crit")]
    recovered = _events(db, subject="stage:embed")
    assert len(recovered) == 1 and recovered[0].severity == "info" and recovered[0].resolved_at == NOW
    assert out["jobs"]["stage:embed"]["status"] == "ok"
    with db() as s:
        s.add(JobRun(job="stage:extract", started_at=NOW + timedelta(minutes=5), status="ok"))
        s.commit()
    out = run_health_checks(now=NOW + timedelta(minutes=15))
    assert out["status"] == "ok" and out["open_events"] == []
    assert len(_events(db, subject="stage:embed")) == 1  # the recovered failure is not recorded twice


def test_never_run_sources_are_grouped_by_cause_and_name_the_missing_setting(db, monkeypatch):
    from fli.ops.health import run_health_checks

    _set_env(monkeypatch, X_BEARER_TOKEN="", SEC_USER_AGENT="")
    x = dict(adapter="x_timeline", kind="x_account", source_class="personal", cadence_minutes=720)
    sec = dict(adapter="edgar_fts", kind="sec", source_class="filing", cadence_minutes=1440)
    with db() as s:
        s.add_all([
            _source("x:a", **x), _source("x:b", **x),
            _source("sec:fts", **sec),
            _source("sec:new", **sec, created_at=NOW - timedelta(hours=40)),  # younger than two cadences: not yet
            _source("lab:typo", adapter="rsss"),
            _source("lab:quiet", cadence_minutes=360),  # collector and settings fine; nothing ingests here
            _source("x:off", **x, enabled=False),
            _source("lab:ran", last_run_at=NOW - timedelta(hours=1), last_success_at=NOW - timedelta(hours=1),
                    last_new_item_at=NOW - timedelta(days=1)),
        ])
        s.commit()
    out = run_health_checks(now=NOW)
    never = _open(out, "source_never_run")
    assert set(never) == {"setting:X_BEARER_TOKEN", "setting:SEC_USER_AGENT", "adapter:rsss"}  # 3 causes, not 5 events
    assert {e["severity"] for e in never.values()} == {"warn"} and out["status"] == "warn"
    assert never["setting:X_BEARER_TOKEN"]["message"].startswith(
        "2 enabled sources have never run: X_BEARER_TOKEN is not set (required by adapter x_timeline): x:a, x:b.")
    assert never["setting:SEC_USER_AGENT"]["message"].startswith(
        "1 enabled source has never run: SEC_USER_AGENT is not set (required by adapter edgar_fts): sec:fts.")
    assert "lab:typo" in never["adapter:rsss"]["message"] and "no collector is registered" in never["adapter:rsss"]["message"]
    assert out["sources"]["never_run"] == 6  # the summary still counts all of them, lab:quiet and sec:new included
    assert _events(db, subject="setting:SEC_USER_AGENT")[0].dedup_key == f"source_never_run:setting:SEC_USER_AGENT:{NOW:%Y-%m-%d}"

    _set_env(monkeypatch, X_BEARER_TOKEN="test-token")  # the token is set: that cause clears
    out = run_health_checks(now=NOW + timedelta(minutes=15))
    assert set(_open(out, "source_never_run")) == {"setting:SEC_USER_AGENT", "adapter:rsss"}
    assert _events(db, subject="setting:X_BEARER_TOKEN")[0].resolved_at == NOW + timedelta(minutes=15)

    with db() as s:  # ingestion started an hour ago: x:a, x:b and lab:quiet get two cadences to run
        s.add(_run("ingest:lab:ran", NOW - timedelta(hours=1)))
        s.commit()
    assert "unexplained" not in _open(run_health_checks(now=NOW + timedelta(minutes=20)), "source_never_run")
    with db() as s:  # ... it has in fact been running for three days and they still never ran
        s.add(_run("ingest:lab:ran", NOW - timedelta(days=3)))
        s.commit()
    unexplained = _open(run_health_checks(now=NOW + timedelta(minutes=30)), "source_never_run")["unexplained"]
    assert unexplained["message"].startswith(
        f"3 enabled sources have never run, although ingestion has been running here since {NOW - timedelta(days=3):%Y-%m-%d %H:%M} UTC")
    assert "(adapter rss, x_timeline): lab:quiet, x:a, x:b." in unexplained["message"]

    with db() as s:  # two of them run: the same event now lists only the one left
        for sid in ("x:a", "x:b"):
            s.get(Source, sid).last_run_at = NOW + timedelta(minutes=40)
        s.commit()
    unexplained = _open(run_health_checks(now=NOW + timedelta(minutes=45)), "source_never_run")["unexplained"]
    assert "1 enabled source has never run, although" in unexplained["message"] and "(adapter rss): lab:quiet." in unexplained["message"]
    with db() as s:
        s.get(Source, "lab:quiet").last_run_at = NOW + timedelta(minutes=50)
        s.commit()
    out = run_health_checks(now=NOW + timedelta(hours=1))
    assert set(_open(out, "source_never_run")) == {"setting:SEC_USER_AGENT", "adapter:rsss"}
    assert [e.resolved_at for e in _events(db, subject="unexplained")] == [NOW + timedelta(hours=1)]

    with db() as s:  # a day later ingestion has stopped (worker_silent and the web report that) ...
        s.add(_source("lab:late", created_at=NOW - timedelta(days=2)))
        s.commit()
    out = run_health_checks(now=NOW + timedelta(days=1, hours=2))
    assert set(_open(out, "source_never_run")) == {"setting:SEC_USER_AGENT", "adapter:rsss"}  # ... so no lab:late
    sec_events = _events(db, subject="setting:SEC_USER_AGENT")  # day-level key: today's event supersedes yesterday's
    assert [e.resolved_at is None for e in sec_events] == [False, True]
    assert sec_events[0].delivered["superseded_by"] == f"source_never_run:setting:SEC_USER_AGENT:{NOW + timedelta(days=1):%Y-%m-%d}"
    assert "sec:fts, sec:new." in sec_events[1].message  # sec:new is now two cadences old too


def test_never_run_lists_at_most_eight_sources(db, monkeypatch):
    from fli.ops.health import run_health_checks

    _set_env(monkeypatch, X_BEARER_TOKEN="")
    with db() as s:
        s.add_all([_source(f"x:{i:02d}", adapter="x_timeline", kind="x_account", source_class="personal")
                   for i in range(34)])
        s.commit()
    event = _open(run_health_checks(now=NOW), "source_never_run")["setting:X_BEARER_TOKEN"]
    assert event["message"].startswith("34 enabled sources have never run")
    assert "x:00, x:01, x:02, x:03, x:04, x:05, x:06, x:07 and 26 more." in event["message"]
    assert len(_events(db)) == 1  # one finding, not 34


def test_stalled_and_silent_worker_pipeline(db):
    from fli.ops.health import run_health_checks

    assert _worker_events(run_health_checks(now=NOW)) == []  # no worker ever ran here (snapshot instance)
    with db() as s:
        s.add(_run("worker:pipeline", NOW - timedelta(hours=20), status="running"))  # an old orphan
        s.add_all([_run("worker:pipeline", NOW - timedelta(hours=h)) for h in (5.5, 5, 4.5, 4, 3.5)])
        s.add(_run("worker:pipeline", NOW - timedelta(hours=3), status="running"))  # hung
        s.add_all([_run("worker:health", NOW - timedelta(minutes=m), minutes=1) for m in (45, 30, 15)])
        s.commit()
    out = run_health_checks(now=NOW)
    assert _worker_events(out) == [("job_stalled", "crit")] and out["status"] == "crit"  # worker_silent stays quiet
    stalled = _open(out, "job_stalled")["worker:pipeline"]
    assert "has been running for 3.0 h (since " + f"{NOW - timedelta(hours=3):%Y-%m-%d %H:%M} UTC)" in stalled["message"]
    assert _events(db, kind="job_stalled")[0].dedup_key == f"job_stalled:worker:pipeline:{NOW:%Y-%m-%d}"
    assert _worker_events(run_health_checks(now=NOW + timedelta(minutes=15))) == [("job_stalled", "crit")]
    assert len(_events(db, kind="job_stalled")) == 1  # deduplicated within the day

    with db() as s:  # the hung run finally finishes
        run = s.scalars(select(JobRun).where(JobRun.status == "running").order_by(JobRun.started_at.desc())).first()
        run.status, run.finished_at = "ok", NOW + timedelta(minutes=20)
        s.commit()
    assert _worker_events(run_health_checks(now=NOW + timedelta(minutes=30))) == []
    assert _events(db, kind="job_stalled")[0].resolved_at == NOW + timedelta(minutes=30)

    with db() as s:  # the worker keeps checking health, but no pipeline run completes any more
        s.add(_run("worker:health", NOW + timedelta(hours=2, minutes=15), minutes=1))
        s.commit()
    assert _worker_events(run_health_checks(now=NOW + timedelta(hours=2))) == []  # 1 h 40 min: not yet
    out = run_health_checks(now=NOW + timedelta(hours=2, minutes=30))
    assert _worker_events(out) == [("worker_silent", "warn")]
    assert _open(out, "worker_silent")["worker:pipeline"]["message"].startswith(
        f"no pipeline run has completed in 2.0 h (the last one at {NOW + timedelta(minutes=20):%Y-%m-%d %H:%M} UTC); "
        f"the worker was last seen {NOW + timedelta(hours=2, minutes=15):%Y-%m-%d %H:%M} UTC")

    with db() as s:  # a partial run (a stage failed: job_failed reports that) still proves the worker works
        s.add(_run("worker:pipeline", NOW + timedelta(hours=2, minutes=35), status="partial"))
        s.commit()
    assert _worker_events(run_health_checks(now=NOW + timedelta(hours=2, minutes=45))) == []
    assert _events(db, kind="worker_silent")[0].resolved_at == NOW + timedelta(hours=2, minutes=45)

    with db() as s:  # the newest run errored: job_failed owns it, worker_silent does not repeat it
        s.add(_run("worker:pipeline", NOW + timedelta(hours=3), status="error"))
        s.commit()
    out = run_health_checks(now=NOW + timedelta(hours=5))
    assert _worker_events(out) == [("job_failed", "warn")]


def test_budget_events(db):
    from fli.ops.health import run_health_checks

    with db() as s:
        s.add(LLMCall(purpose="extract", model="claude-haiku-5-5", prompt_version="v", status="ok", cost_usd=37.0,
                      created_at=NOW - timedelta(days=2)))
        s.add(ApiUsage(provider="x", endpoint="/2/users/:id/tweets", units=9000, cost_usd=9.0, created_at=NOW))
        s.commit()
    out = run_health_checks(now=NOW)
    assert {(e["subject"], e["severity"]) for e in out["open_events"]} == {("llm_total", "warn"), ("x_total", "warn")}
    assert out["budget"]["llm_total_pct"] == pytest.approx(37 / 45, abs=1e-3) and out["budget"]["x_spent_usd"] == 9.0
    with db() as s:
        s.add(LLMCall(purpose="digest", model="claude-opus-5-5", prompt_version="v", status="ok", cost_usd=14.5,
                      created_at=NOW))
        s.commit()
    out = run_health_checks(now=NOW + timedelta(minutes=15))
    subjects = {(e["subject"], e["severity"]) for e in out["open_events"]}
    assert {("llm_budget", "crit"), ("llm_daily", "warn")} <= subjects and out["status"] == "crit"


def test_healthchecks_heartbeat_and_fail(db, monkeypatch):
    from fli.ops.health import run_health_checks

    _set_env(monkeypatch, HEALTHCHECKS_PING_BASE=HC_BASE)
    with respx.mock(assert_all_called=False) as router:
        ok = router.post(f"{HC_BASE}/fli-health").mock(return_value=httpx.Response(200, text="OK"))
        fail = router.post(f"{HC_BASE}/fli-health/fail").mock(return_value=httpx.Response(200, text="OK"))
        assert run_health_checks(now=NOW)["notified"]["healthchecks"] is True
        assert ok.call_count == 1 and fail.call_count == 0
        assert ok.calls[0].request.url.params["create"] == "1"
        with db() as s:
            s.add(_source("lab:failing", last_run_at=NOW, consecutive_failures=5, last_error="boom"))
            s.commit()
        run_health_checks(now=NOW + timedelta(minutes=15))
        assert fail.call_count == 1 and b"source_failing lab:failing" in fail.calls[0].request.content


def test_unreadable_scoring_model_is_a_crit_event_until_fixed(db, monkeypatch):
    from fli.analyze import scoring
    from fli.ops.health import check_models, run_health_checks

    _set_env(monkeypatch, SLACK_WEBHOOK_URL=WEBHOOK)
    monkeypatch.setattr(scoring, "utcnow", lambda: NOW)  # load_model records its event under the same day key
    path = scoring.model_path("investment")  # in tmp (autouse fixture)
    path.parent.mkdir(parents=True)
    path.write_text('{"audience": "investment", "version": "lr-1", "coef": [0.1, ')  # half-written
    with respx.mock(assert_all_called=False) as router:
        slack = router.post(WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        out = run_health_checks(now=NOW)
        assert out["status"] == "crit"
        event = _open(out, scoring.OPS_KIND)[scoring.ops_subject("investment")]
        assert event["severity"] == "crit" and "is unreadable (JSONDecodeError" in event["message"]
        assert "re-run `fli train`" in event["message"] and set(_open(out, scoring.OPS_KIND)) == {"scoring_model:investment"}
        assert slack.call_count == 1 and "model_unreadable` scoring_model:investment" in slack.calls[0].request.content.decode()
        run_health_checks(now=NOW + timedelta(minutes=15))
        assert slack.call_count == 1  # announced once per day
    rows = _events(db, kind=scoring.OPS_KIND)  # the row load_model opened is the one the health check keeps
    assert len(rows) == 1 and rows[0].dedup_key == f"model_unreadable:scoring_model:investment:{NOW:%Y-%m-%d}"

    path.unlink()  # removed (or retrained): the condition clears and the event resolves
    out = run_health_checks(now=NOW + timedelta(minutes=30))
    assert out["status"] == "ok" and _events(db, kind=scoring.OPS_KIND)[0].resolved_at == NOW + timedelta(minutes=30)

    def broken(models_dir=None):
        raise ImportError("sklearn")

    monkeypatch.setattr(scoring, "model_file_problems", broken)
    assert check_models() == []  # a broken scoring import never stops the other checks


# ------------------------------------------------------------------------------- snapshot
def _seed_snapshot_rows(s) -> None:
    s.add(Lab(id="openai", name="OpenAI", kind="frontier_lab", tier=1, public_exposure=[{"ticker": "MSFT"}]))
    s.add(_source("openai:news", lab_id="openai"))
    s.flush()
    s.add(Document(id=1, source_id="openai:news", natural_key="a", url="https://o.example/a",
                   canonical_url="https://o.example/a", content_text="hello", content_hash="h1",
                   source_class="official", meta={"stars": 3}, minhash=b"\x00\x01\x02"))
    s.add(Document(id=2, source_id="openai:news", natural_key="b", url="https://o.example/b",
                   canonical_url="https://o.example/b", content_text="hello", content_hash="h1",
                   source_class="official", duplicate_of_id=1, status="duplicate"))
    s.add(Development(id=1, title="Dev", event_type="model_release", first_seen_at=NOW, last_seen_at=NOW,
                      source_classes=["official"]))
    s.flush()
    s.add(Insight(document_id=1, development_id=1, event_type="model_release", title="t", claim="c",
                  evidence_quotes=["q"], verification={"status": "verified"}, published_at=NOW,
                  prompt_version="p", model="m"))
    s.add(OpsEvent(kind="budget", severity="warn", subject="llm_total", message="m", dedup_key="budget:x:1"))


def test_snapshot_round_trip_sqlite(db, tmp_path):
    from fli.ops.snapshot import run_snapshot

    with db() as s:
        _seed_snapshot_rows(s)
        s.commit()
    path = tmp_path / "snap" / "fli_snapshot.sqlite.gz"
    exported = run_snapshot("export", str(path))
    assert exported["tables"]["document"] == 2 and exported["tables"]["insight"] == 1
    with gzip.open(path, "rb") as fh:
        assert fh.read(16) == b"SQLite format 3\x00"
    with db() as s:  # damage the live database
        s.query(Insight).delete()
        s.get(Lab, "openai").name = "changed"
        s.commit()
    imported = run_snapshot("import", str(path))
    assert imported["tables"]["insight"] == 1 and imported["rows"] == exported["rows"]
    with db() as s:
        assert s.get(Lab, "openai").name == "OpenAI" and s.get(Lab, "openai").public_exposure == [{"ticker": "MSFT"}]
        doc = s.get(Document, 2)
        assert doc.duplicate_of_id == 1 and s.get(Document, 1).minhash == b"\x00\x01\x02"
        ins = s.scalar(select(Insight))
        assert ins.verification == {"status": "verified"} and ins.published_at == NOW
        s.add(Insight(document_id=1, event_type="other", title="new", claim="c", prompt_version="p", model="m"))
        s.commit()  # the database is writable after the swap
    again = tmp_path / "again.sqlite.gz"
    run_snapshot("export", str(again))
    with gzip.open(again, "rb") as fh:
        assert fh.read(16) == b"SQLite format 3\x00"


def test_copy_database_preserves_self_references(db, tmp_path):
    """The Postgres seeding path (copy_database + sequence reset), exercised SQLite to SQLite."""
    from fli.db.session import get_engine
    from fli.ops.snapshot import copy_database, reset_sequences

    with db() as s:
        _seed_snapshot_rows(s)
        s.commit()
    dst = create_engine(f"sqlite:///{tmp_path / 'copy.sqlite'}")
    counts = copy_database(get_engine(), dst)
    reset_sequences(dst)  # no-op off Postgres
    assert counts["document"] == 2 and counts["ops_event"] == 1 and counts["lab"] == 1
    with dst.connect() as conn:
        assert conn.exec_driver_sql("SELECT duplicate_of_id FROM document WHERE id = 2").scalar() == 1
    again = copy_database(get_engine(), dst, replace=True)  # replace: no duplicate-key errors
    assert again == counts
    dst.dispose()


def test_snapshot_import_rejects_garbage(db, tmp_path):
    from fli.ops.snapshot import run_snapshot

    bad = tmp_path / "bad.sqlite.gz"
    with gzip.open(bad, "wb") as fh:
        fh.write(b"not a database" * 100)
    with pytest.raises(ValueError):
        run_snapshot("import", str(bad))
    with pytest.raises(ValueError):
        run_snapshot("explode")


# ------------------------------------------------------------------------------- scheduler
def test_scheduler_jobs_and_triggers(db):
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    from fli.ops.scheduler import build_scheduler

    sched = build_scheduler(fli_config.get_settings(), run_now=False)
    jobs = {j.id: j for j in sched.get_jobs()}
    assert set(jobs) == {"pipeline", "health", "digest", "contributors"}
    assert isinstance(jobs["pipeline"].trigger, IntervalTrigger) and jobs["pipeline"].trigger.interval == timedelta(minutes=30)
    assert jobs["health"].trigger.interval == timedelta(minutes=15)
    cron = jobs["digest"].trigger
    assert isinstance(cron, CronTrigger) and str(cron.timezone) == "Europe/Berlin"
    fields = {f.name: str(f) for f in cron.fields}
    assert (fields["day_of_week"], fields["hour"], fields["minute"]) == ("mon", "7", "0")
    assert all(j.max_instances == 1 and j.coalesce for j in jobs.values())


def test_run_job_records_ledger_and_pings(db, monkeypatch):
    from fli.ops import scheduler

    pings: list[tuple[str, str]] = []
    monkeypatch.setattr(scheduler, "ping_healthcheck", lambda slug, signal="", body=None: pings.append((slug, signal)))
    monkeypatch.setitem(scheduler.JOBS, "pipeline", (lambda: {"ok": True, "n": 1}, "fli-pipeline", True))
    assert scheduler.run_job("pipeline") == {"ok": True, "n": 1}

    def boom():
        raise RuntimeError("database is locked")

    monkeypatch.setitem(scheduler.JOBS, "digest", (boom, "fli-digest", True))
    assert scheduler.run_job("digest")["ok"] is False
    assert pings == [("fli-pipeline", "start"), ("fli-pipeline", ""), ("fli-digest", "start"), ("fli-digest", "fail")]
    with db() as s:
        runs = {r.job: r for r in s.scalars(select(JobRun))}
    assert runs["worker:pipeline"].status == "ok" and runs["worker:pipeline"].details == {"ok": True, "n": 1}
    assert runs["worker:digest"].status == "error" and "database is locked" in runs["worker:digest"].error

    from fli.ops import health

    assert health.WORKER_PIPELINE in runs  # the health check watches the row name the scheduler really writes
    events = health.run_health_checks()["open_events"]  # the failed worker job surfaces as an ops event
    assert [(e["kind"], e["subject"]) for e in events] == [("job_failed", "worker:digest")]


def test_pipeline_job_runs_stages_after_ingest(db, monkeypatch):
    from fli import pipeline
    from fli.ingest import runner
    from fli.ops.scheduler import job_pipeline

    seen: dict = {}
    monkeypatch.setattr(runner, "run_ingest", lambda: {"a": {"status": "ok"}, "b": {"status": "error"}})
    monkeypatch.setattr(pipeline, "run_pipeline", lambda stages: seen.setdefault("stages", stages) and
                        {"triage": {"status": "error"}, "alerts": {"created": 1}})
    out = job_pipeline()
    assert seen["stages"][0] == "resolve_entities" and seen["stages"][-1] == "alerts" and "ingest" not in seen["stages"]
    assert out == {"ok": False, "failed_stages": ["triage"], "sources_run": 2, "source_errors": 1}
    with db() as s:  # ingest is ledgered like any other stage
        assert [(r.job, r.status) for r in s.scalars(select(JobRun))] == [("stage:ingest", "ok")]


def test_an_ingest_crash_in_the_worker_becomes_a_job_failure(db, monkeypatch):
    from fli import pipeline
    from fli.ingest import runner
    from fli.ops.health import run_health_checks
    from fli.ops.scheduler import job_pipeline

    def crash():
        raise RuntimeError('relation "source" does not exist')

    monkeypatch.setattr(runner, "run_ingest", crash)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda stages: {"alerts": {"created": 0}})
    assert job_pipeline() == {"ok": False, "failed_stages": ["ingest"], "sources_run": 0, "source_errors": 0}
    events = run_health_checks()["open_events"]  # it used to be a log line only
    assert [(e["kind"], e["subject"], e["severity"]) for e in events] == [("job_failed", "stage:ingest", "warn")]
    assert 'relation "source" does not exist' in events[0]["message"]


def test_worker_start_closes_orphaned_runs_and_gets_a_grace_period(db, monkeypatch):
    from fli.ops import health, scheduler

    assert scheduler.START_JOB == health.WORKER_START
    monkeypatch.setattr(scheduler, "utcnow", lambda: NOW)
    with db() as s:
        s.add(_run("worker:pipeline", NOW - timedelta(days=2)))
        s.add(_run("worker:pipeline", NOW - timedelta(hours=6), status="running"))  # killed mid-run by a redeploy
        s.add(_run("stage:extract", NOW - timedelta(hours=6), status="running"))  # not the worker's own row
        s.commit()
    closed = scheduler.record_start(fli_config.get_settings())
    with db() as s:
        runs = list(s.scalars(select(JobRun).order_by(JobRun.id)))
    by_id = {r.id: r for r in runs}
    assert len(closed) == 1 and by_id[closed[0]].status == "interrupted" and by_id[closed[0]].finished_at == NOW
    assert "the worker restarted before this run finished" in by_id[closed[0]].error
    start = [r for r in runs if r.job == "worker:start"]
    assert len(start) == 1 and start[0].details["interrupted_runs"] == closed and start[0].started_at == NOW
    assert [r.status for r in runs if r.job == "stage:extract"] == ["running"]

    # a fresh worker gets 2 h to finish its first run; the closed orphan never reads as a hung run
    assert _worker_events(health.run_health_checks(now=NOW + timedelta(minutes=30))) == []
    out = health.run_health_checks(now=NOW + timedelta(hours=2, minutes=30))
    assert _worker_events(out) == [("worker_silent", "warn")]
    assert "(the last one at " + f"{NOW - timedelta(days=2) + timedelta(minutes=5):%Y-%m-%d %H:%M} UTC)" in \
        _open(out, "worker_silent")["worker:pipeline"]["message"]


def test_record_start_never_raises(monkeypatch):
    from fli.ops import scheduler

    def no_db():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(scheduler, "session_scope", no_db)
    assert scheduler.record_start(fli_config.get_settings()) == []


def test_sigterm_shuts_down_gracefully():
    from fli.ops.scheduler import install_signal_handlers

    class FakeScheduler:
        def __init__(self):
            self.calls = []

        def shutdown(self, wait=True):
            self.calls.append(wait)

    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        fake = FakeScheduler()
        handler = install_signal_handlers(fake)
        assert signal.getsignal(signal.SIGTERM) is handler
        handler(signal.SIGTERM, None)
        import threading

        for t in threading.enumerate():
            if t.name == "fli-shutdown":
                t.join(timeout=5)
        assert fake.calls == [True]
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def test_frozen_feed_is_stale_on_day_one(db):
    """A first backfill stamps last_new_item_at=now; the newest *published* date still exposes a frozen feed."""
    from fli.ops.health import run_health_checks

    with db() as s:
        s.add(_source("lab:frozen", last_run_at=NOW, last_success_at=NOW, last_new_item_at=NOW))
        s.add(_source("lab:live", last_run_at=NOW, last_success_at=NOW, last_new_item_at=NOW))
        s.flush()
        for sid, age in (("lab:frozen", 30), ("lab:live", 2)):
            s.add(Document(source_id=sid, natural_key="k", url=f"https://{sid}.example", canonical_url=f"https://{sid}.example",
                           content_hash=sid, source_class="official", published_at=NOW - timedelta(days=age)))
        s.commit()
    out = run_health_checks(now=NOW)
    assert [(e["kind"], e["subject"]) for e in out["open_events"]] == [("source_stale", "lab:frozen")]
    assert "newest published item is 30.0 days old (SLO 14 days)" in out["open_events"][0]["message"]
