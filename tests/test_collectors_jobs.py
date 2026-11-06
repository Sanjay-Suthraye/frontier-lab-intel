"""Job-board collector: ATS parsers, categorisation, snapshot bookkeeping, weekly hiring summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import respx

from fli.db.models import Document, JobPosting, Lab, Source
from fli.ingest.collectors import jobs
from fli.ingest.collectors.jobs import Posting, categorise, upsert_snapshot, weekly_summaries
from fli.ingest.records import SourceInfo, Window
from fli.ingest.runner import persist_records

FIX = Path(__file__).parent / "fixtures" / "jobs"


def load(name: str):
    return json.loads((FIX / name).read_text())


# ------------------------------------------------------------------------------- parsers
def test_parse_greenhouse_departments_payload():
    posts = jobs.parse_greenhouse(load("greenhouse_departments_xai.json"))
    assert len(posts) == 6
    by_title = {p.title: p for p in posts}
    power = by_title["Power Generation Engineer - Memphis"]
    assert power.department == "Data Center" and power.location == "Southaven, MS; Memphis, TN"
    assert power.url.startswith("https://job-boards.greenhouse.io/xai/jobs/")
    assert power.published_at == datetime(2025, 9, 10, 2, 52, 53)  # 2025-09-09T22:52:53-04:00 as naive UTC
    assert {p.department for p in posts} == {"Data Center", "Engineering", "Information Security"}


def test_parse_greenhouse_jobs_payload_joins_departments():
    posts = jobs.parse_greenhouse(load("greenhouse_jobs_anthropic.json"))
    assert [p.department for p in posts] == ["Compute", "Compute", "Sales"]
    assert posts[0].title == "Capacity Deployment Lead - Data Center Operations"
    assert all(p.published_at is not None for p in posts)


def test_parse_ashby_includes_team_and_secondary_locations():
    posts = jobs.parse_ashby(load("ashby_openai.json"))
    assert len(posts) == 7
    white = next(p for p in posts if p.title == "White Space Manager")
    assert (white.department, white.team) == ("Scaling", "Datacenter Design")
    assert white.location == "San Francisco; Seattle; US - Remote"
    assert white.url == f"https://jobs.ashbyhq.com/openai/{white.posting_id}"


def test_parse_lever_epoch_millis_and_all_locations():
    posts = jobs.parse_lever(load("lever_postings.json"))
    assert [p.published_at.date().isoformat() for p in posts] == ["2026-10-20", "2026-10-22", "2026-09-01"]
    assert posts[0].location == "Paris; London" and posts[1].location == "Paris"
    assert jobs.parse_lever(load("lever_mistral_empty.json")) == []


@pytest.mark.parametrize("title, department, team, expected", [
    ("ML Silicon Performance Lead", "Scaling", "Hardware", "hardware_chips"),
    ("Manufacturing Test Engineer", "Scaling", "Hardware", "hardware_chips"),
    ("Engineering Technician, Actuators", "Research", "Robotics", "robotics"),
    ("White Space Manager", "Scaling", "Datacenter Design", "infrastructure_datacenter"),
    ("Optical Engineer - Memphis", "Data Center", None, "infrastructure_datacenter"),
    ("Lead Engineer, Battery Storage - Memphis", "Data Center", None, "infrastructure_datacenter"),
    ("Gas Supply and Transport Lead, US and Canada", "Compute", None, "infrastructure_datacenter"),
    ("Software Engineer, HSM Infrastructure Security, Consumer Devices", "Consumer Devices", "Software", "devices"),
    ("Account Director, Manufacturing- Tokyo", "Go To Market", "Sales", "go_to_market"),
    ("Applied AI Architect, OpenAI for Government", "OpenAI for Gov", "OpenAI for Gov", "policy"),
    ("Staff + Software Security Engineer, Secure Frameworks", "Security", None, "security"),
    ("Research Engineer, Machine Learning (Reinforcement Learning)", "AI Research & Engineering", None, "research"),
    ("Customer Success Manager, Top Accounts", "Sales", None, "go_to_market"),
    ("Staff+ Software Engineer, Full-stack", "Engineering & Design - Product", None, "applied_product"),
    ("Software Engineer, Applied Emerging Talent (2027)", "Applied AI", "Emerging Talent", "applied_product"),
    ("Product Counsel", "Legal", None, "other"),
    ("Cook, Sushi", "Operations", None, "other"),
])
def test_categorise_real_titles(title, department, team, expected):
    assert categorise(title, department, team) == expected


# ------------------------------------------------------------------------------- snapshot table
def _lab(db, lab_id="openai", name="OpenAI"):
    with db() as s:
        s.add(Lab(id=lab_id, name=name, kind="frontier_lab", tier=1))
        s.commit()


def P(pid, title, published, dept="Research", loc="San Francisco", team=None):
    return Posting(pid, title, dept, team, loc, f"https://jobs.example/{pid}", published)


def test_snapshot_lifecycle_new_removed_relisted(db):
    _lab(db)
    t1, t2, t3 = datetime(2026, 10, 1), datetime(2026, 10, 8), datetime(2026, 10, 15)
    with db() as s:
        assert upsert_snapshot(s, "openai", "ashby", [P("a", "Research Engineer", None), P("b", "SRE", t1)], t1)["new"] == 2
        stats = upsert_snapshot(s, "openai", "ashby", [P("b", "SRE", t1), P("c", "Sales Lead", t2)], t2)
        assert (stats["new"], stats["removed"]) == (1, 1)
        stats = upsert_snapshot(s, "openai", "ashby", [P("a", "Research Engineer", None), P("b", "SRE", t1),
                                                      P("c", "Sales Lead", t2)], t3)
        assert stats["relisted"] == 1
        s.commit()
        rows = {r.posting_id: r for r in s.query(JobPosting)}
    assert rows["a"].removed_at is None and rows["a"].meta["baseline"] is True  # undated, first snapshot
    assert rows["a"].first_seen_at == t1 and rows["b"].last_seen_at == t3
    assert rows["c"].category == "go_to_market" and rows["b"].category == "infrastructure_datacenter"


def test_snapshot_collapse_is_not_treated_as_mass_closure(db):
    _lab(db)
    with db() as s:
        upsert_snapshot(s, "openai", "ashby", [P(str(i), f"Engineer {i}", datetime(2026, 9, 1)) for i in range(20)],
                        datetime(2026, 10, 1))
        stats = upsert_snapshot(s, "openai", "ashby", [P("0", "Engineer 0", datetime(2026, 9, 1))],
                                datetime(2026, 10, 2))
        s.commit()
        assert stats["suspicious"] and stats["removed"] == 0
        assert s.query(JobPosting).filter(JobPosting.removed_at.is_not(None)).count() == 0


# ------------------------------------------------------------------------------- weekly summaries
def _history(db):
    """Tracking starts Wed 2026-10-21 (W43); W40-W43 are reconstructed from publish dates."""
    _lab(db)
    d = datetime
    before = [P("r1", "Research Engineer", d(2026, 9, 28)), P("r2", "Research Scientist", d(2026, 10, 1)),
              P("r3", "Researcher, Training", d(2026, 10, 6)),
              P("g1", "Account Executive", d(2026, 10, 7), dept="Go To Market"),
              P("r4", "Research Engineer, Evals", d(2026, 10, 14)), P("r5", "Researcher, RL", d(2026, 10, 20))]
    week44 = [P("dc1", "Data Center Technician", d(2026, 10, 26), dept="Datacenter Design", loc="Abilene, TX"),
              P("dc2", "Data Center Technician II", d(2026, 10, 26), dept="Datacenter Design", loc="Abilene, TX"),
              P("dc3", "Data Center Controls Engineer", d(2026, 10, 27), dept="Datacenter Design",
                loc="Abilene, TX | San Francisco"),
              P("dc4", "Head of Datacenter Construction", d(2026, 10, 27), dept="Datacenter Design"),
              P("hw1", "ASIC Physical Design Engineer", d(2026, 10, 27), dept="Hardware"),
              P("r6", "Research Engineer, Pretraining", d(2026, 10, 28))]
    with db() as s:
        upsert_snapshot(s, "openai", "ashby", before, d(2026, 10, 21, 9))
        upsert_snapshot(s, "openai", "ashby", [p for p in before if p.posting_id != "r1"] + week44, d(2026, 10, 28, 9))
        s.commit()
        return list(s.query(JobPosting))


def test_weekly_summaries_counts_shifts_and_firsts(db):
    rows = _history(db)
    recs = weekly_summaries(rows, since=datetime(2026, 8, 1), until=datetime(2026, 11, 4, 12), lab_id="openai",
                            lab_name="OpenAI", boards=["ashby:openai"], board_url="https://jobs.ashbyhq.com/openai")
    assert [r.natural_key for r in recs] == [f"hiring:openai:2026-W{w}" for w in (40, 41, 42, 43, 44)]  # W45 open
    w44 = recs[-1]
    m = w44.meta
    assert w44.published_at == datetime(2026, 11, 1, 23, 59, 59)
    assert w44.url == "https://jobs.ashbyhq.com/openai?week=2026-W44"
    assert m["new_postings"] == 6 and m["removed_postings"] == 1 and not m["reconstructed"]
    assert m["by_category"] == {"infrastructure_datacenter": 4, "hardware_chips": 1, "research": 1}
    assert m["trailing_4w_avg"] == {"research": 1.25, "go_to_market": 0.25}
    assert m["shifts"] == ["infrastructure_datacenter 4 new vs 0.0/week trailing average"]
    assert m["new_departments"] == ["Datacenter Design", "Hardware"]
    assert m["new_locations"] == ["Abilene, TX"]
    assert m["notable"][0]["title"] == "Head of Datacenter Construction"
    assert all(n["category"] != "research" for n in m["notable"])
    assert m["open_at_week_end"] == 11  # 12 postings opened by Nov 1, r1 closed on Oct 28
    assert "New postings by category: infrastructure_datacenter 4, hardware_chips 1, research 1." in w44.content_text
    assert "https://jobs.example/hw1" in w44.content_text  # notable roles carry citable posting URLs
    w40 = recs[0]
    assert w40.meta["reconstructed"] and "unknown (before tracking)" in w40.content_text
    assert w40.meta["trailing_4w_avg"] == {}  # no history before the first posting


def test_weekly_summaries_respect_window(db):
    rows = _history(db)
    recs = weekly_summaries(rows, since=datetime(2026, 10, 30), until=datetime(2026, 11, 2, 6), lab_id="openai",
                            lab_name="OpenAI", boards=["ashby:openai"], board_url="https://jobs.ashbyhq.com/openai")
    assert [r.natural_key for r in recs] == ["hiring:openai:2026-W44"]
    assert weekly_summaries([], since=datetime(2026, 8, 1), until=datetime(2026, 11, 4), lab_id="x", lab_name="X",
                            boards=[], board_url="u") == []


# ------------------------------------------------------------------------------- collector end to end
def test_job_board_collector_is_idempotent(db, settings, monkeypatch):
    _lab(db)
    with db() as s:
        s.add(Source(id="openai:jobs", lab_id="openai", adapter="job_board", kind="job_board",
                     source_class="official", config={"ats": "ashby", "board": "openai"}))
        s.commit()
    info = SourceInfo(id="openai:jobs", adapter="job_board", kind="job_board", source_class="official", url=None,
                      config={"ats": "ashby", "board": "openai"}, lab_id="openai")
    window = Window(since=datetime(2026, 8, 1), until=datetime(2026, 11, 4, 12))
    monkeypatch.setattr(jobs, "utcnow", lambda: datetime(2026, 11, 4, 12))
    for _ in range(2):
        with respx.mock() as router:
            route = router.get("https://api.ashbyhq.com/posting-api/job-board/openai").mock(
                return_value=httpx.Response(200, json=load("ashby_openai.json")))
            recs = jobs.job_board(info, window, settings)
            assert route.called
        with db() as s:
            persist_records(s, s.get(Source, "openai:jobs"), recs)
            s.commit()
    with db() as s:
        assert s.query(JobPosting).count() == 7
        docs = s.query(Document).filter_by(source_id="openai:jobs").all()
        keys = sorted(d.natural_key for d in docs)
    assert len(keys) == len(set(keys)) == len(recs) > 0
    assert keys[-1] == "hiring:openai:2026-W44"  # 2026-W45 (Nov 2-8) is not complete yet
    assert all(k.startswith("hiring:openai:2026-W") for k in keys)


def test_job_board_requires_ats_and_lab(settings):
    bad = SourceInfo(id="x:jobs", adapter="job_board", kind="job_board", source_class="official", url=None,
                     config={"ats": "workday", "board": "x"}, lab_id="x")
    with pytest.raises(ValueError, match="ats"):
        jobs.job_board(bad, Window(since=datetime(2026, 8, 1), until=datetime(2026, 11, 4)), settings)


def test_two_boards_on_one_ats_do_not_remove_each_other(db):
    _lab(db)
    t1, t2 = datetime(2026, 10, 1), datetime(2026, 10, 2)
    with db() as s:
        upsert_snapshot(s, "openai", "greenhouse", [P("a1", "Researcher", t1)], t1, token="board-a")
        upsert_snapshot(s, "openai", "greenhouse", [P("b1", "Recruiter", t1)], t1, token="board-b")
        stats = upsert_snapshot(s, "openai", "greenhouse", [P("a1", "Researcher", t1)], t2, token="board-a")
        s.commit()
        assert stats["removed"] == 0
        assert s.query(JobPosting).filter(JobPosting.removed_at.is_not(None)).count() == 0
        assert {r.meta["board_token"] for r in s.query(JobPosting)} == {"board-a", "board-b"}


def test_late_listed_posting_counts_in_the_week_it_appears(db):
    _lab(db)
    with db() as s:
        upsert_snapshot(s, "openai", "ashby", [P("a", "Research Engineer", datetime(2026, 9, 1))], datetime(2026, 10, 1))
        # appears on Oct 21 with a September publish date (re-opened role)
        upsert_snapshot(s, "openai", "ashby", [P("a", "Research Engineer", datetime(2026, 9, 1)),
                                               P("b", "Robotics Engineer", datetime(2026, 9, 2))], datetime(2026, 10, 21))
        s.commit()
        rows = list(s.query(JobPosting))
    assert next(r for r in rows if r.posting_id == "b").meta["late_listed"] is True
    recs = weekly_summaries(rows, since=datetime(2026, 8, 31), until=datetime(2026, 10, 26), lab_id="openai",
                            lab_name="OpenAI", boards=["ashby:openai"], board_url="u")
    assert {r.meta["week"]: r.meta["new_postings"] for r in recs} == {"2026-W36": 1, "2026-W43": 1}
