"""Register collector: curated affiliation changes become curated documents (one per event).

The fixture database mirrors the real register's moves of the window (Discovery Loop founders, the
Google DeepMind reshuffle, Zoph, Tulloch, Dayiheng Liu) plus synthetic edge cases with fictional people
and example.org sources. No network and no LLM: the collector only reads the local database.
"""

from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml
from sqlalchemy import select

from fli.config import CONFIG_DIR
from fli.db.models import Affiliation, Document, Lab, Person, Source
from fli.ingest.collectors import register as reg
from fli.ingest.records import COLLECTORS, SourceInfo, Window
from fli.register import loader
from fli.taxonomy import SOURCE_CLASSES
from fli.textutil import canonical_url

NOW = datetime(2026, 11, 5, 12, 0)
WINDOW = Window(since=datetime(2026, 8, 1), until=NOW)
CURATED_ON = datetime(2026, 8, 1)  # fixture rows were recorded before any incremental window opened
SRC = SourceInfo(id="register:affiliations", adapter="register_affiliations", kind="register",
                 source_class="curated", url=None, config={})

TECHREPUBLIC = "https://techrepublic.com/article/news-google-ai-leadership-hassabis-jeff-dean"
GEEKWIRE = ("https://www.geekwire.com/2026/the-startup-idea-that-convinced-a-uw-computer-science-legend-"
            "to-leave-google-after-27-years/")
GIGAZINE = "https://gigazine.net/gsc_news/en/20260806-google-jeff-dean-discovery-loop/"
NINE_TO_FIVE = "https://9to5google.com/2026/08/05/demis-hassabis-deepmind/"
BUSINESS_TODAY = "https://www.businesstoday.in/technology/news/story/google-deepmind-leadership-changes-547553-2026-08-06"
THE_STAR = "https://www.thestar.com.my/tech/tech-news/2026/08/27/thinking-machines-lab-co-founder-barret-zoph-joins-google"
SEMAFOR = "https://www.semafor.com/article/09/09/2026/ai-researcher-andrew-tulloch-is-leaving-meta"
TECHNEWS = "https://technews.tw/2026/09/11/andrew-tulloch-departs-meta-joins-anthropic/"
KR36 = "https://eu.36kr.com/en/p/3993126854186888"


def ex(slug: str) -> str:
    return f"https://example.org/news/{slug}"


LABS = [("google-deepmind", "Google DeepMind", 1), ("discovery-loop", "Discovery Loop", 3),
        ("meta-ai", "Meta AI (Superintelligence Labs / FAIR)", 1), ("anthropic", "Anthropic", 1),
        ("qwen", "Qwen (Alibaba)", 2), ("xai", "xAI", 1), ("thinking-machines", "Thinking Machines Lab", 3)]
PEOPLE = [("jeff-dean", "Jeff Dean", "A"), ("sanjay-ghemawat", "Sanjay Ghemawat", "A"),
          ("oriol-vinyals", "Oriol Vinyals", "A"), ("quoc-le", "Quoc Le", "A"),
          ("demis-hassabis", "Demis Hassabis", "A"), ("koray-kavukcuoglu", "Koray Kavukcuoglu", "A"),
          ("barret-zoph", "Barret Zoph", "B"), ("andrew-tulloch", "Andrew Tulloch", "B"),
          ("dayiheng-liu", "Dayiheng Liu", "B"),
          # fictional people for the edge cases
          ("alex-mover", "Alex Mover", "B"), ("pat-leaver", "Pat Leaver", "C"), ("sam-leaver", "Sam Leaver", "A"),
          ("yan-founder", "Yan Founder", "A"), ("no-source", "No Source", "A"), ("old-move", "Old Move", "A"),
          ("future-move", "Future Move", "A")]
# (id, person, lab, org, role, start, end, source_url, evidence, confidence)
AFFILIATIONS = [
    (149, "jeff-dean", "google-deepmind", None, "Chief Scientist, Google DeepMind and Google Research", None,
     "2026-08-05", TECHREPUBLIC, None, 1.0),
    (150, "jeff-dean", "discovery-loop", None, "Co-founder", "2026-08-05", None, GEEKWIRE, None, 1.0),
    (151, "sanjay-ghemawat", None, "Google", "Google Fellow", None, "2026-08-05", GIGAZINE, None, 1.0),
    (152, "sanjay-ghemawat", "discovery-loop", None, "Co-founder", "2026-08-05", None, GIGAZINE, None, 1.0),
    (153, "oriol-vinyals", "google-deepmind", None, "VP Research, Gemini co-lead", None, "2026-08-05", GIGAZINE,
     None, 1.0),
    (154, "oriol-vinyals", "discovery-loop", None, "Co-founder", "2026-08-05", None, GIGAZINE, None, 1.0),
    (155, "quoc-le", "google-deepmind", None, "Google Fellow", None, "2026-08-05", GIGAZINE, None, 1.0),
    (156, "quoc-le", "discovery-loop", None, "Co-founder", "2026-08-05", None, GIGAZINE, None, 1.0),
    (52, "demis-hassabis", "google-deepmind", None, "Co-founder and CEO", None, "2026-08-05", NINE_TO_FIVE, None, 1.0),
    (53, "demis-hassabis", "google-deepmind", None, "Chair; Alphabet Chief Scientist", "2026-08-05", None,
     NINE_TO_FIVE, None, 1.0),
    (54, "koray-kavukcuoglu", "google-deepmind", None, "CTO; Google Chief AI Architect", None, "2026-08-05",
     BUSINESS_TODAY, None, 1.0),
    (55, "koray-kavukcuoglu", "google-deepmind", None, "SVP, Google DeepMind", "2026-08-05", None, BUSINESS_TODAY,
     None, 1.0),
    (60, "barret-zoph", "google-deepmind", None, "VP Research, RL and post-training", "2026-08-26", None, THE_STAR,
     "announced on X; confirmed by a Google spokesperson", 1.0),
    (45, "andrew-tulloch", "thinking-machines", None, "Co-founder", "2025-02-18", "2025-10-01", ex("tml"), None, 1.0),
    (46, "andrew-tulloch", "meta-ai", None, "TBD Lab researcher", "2025-10-01", "2026-09-09", SEMAFOR, None, 1.0),
    (47, "andrew-tulloch", "anthropic", None, "Inference and performance (reported)", "2026-09-14", None, TECHNEWS,
     "TechNews/HTX report; not confirmed by Anthropic", 0.7),
    (119, "dayiheng-liu", "qwen", None, "Pretraining lead", None, "2026-09-21", KR36, None, 1.0),
    (120, "dayiheng-liu", "qwen", None, "Head of Qwen LLM project", "2026-09-21", None, KR36,
     "insider reports; no official Alibaba announcement", 0.7),
    # synthetic edge cases
    (300, "alex-mover", None, "Microsoft", "President, AI Platform", None, "2026-10-08", ex("mover"), None, 1.0),
    (301, "alex-mover", "anthropic", None, "AI infrastructure lead", "2026-10-08", None, ex("mover"), None, 1.0),
    (310, "pat-leaver", "xai", None, "Co-founder (training)", "2023-07-12", "2026-10-15", ex("xai-exits"), None, 1.0),
    (311, "sam-leaver", "xai", None, "Co-founder (research)", "2023-07-12", "2026-10-15", ex("xai-exits"), None, 1.0),
    (320, "yan-founder", "meta-ai", None, "Chief AI Scientist", None, "2026-10-20", ex("founder"), None, 1.0),
    (321, "yan-founder", None, "Example Labs (EL)", "Founder and Executive Chairman", "2026-10-20", None,
     ex("founder"), None, 1.0),
    (330, "no-source", "anthropic", None, "Researcher", "2026-09-01", None, None, None, 1.0),  # uncited
    (331, "old-move", "anthropic", None, "Researcher", "2026-06-01", None, ex("old"), None, 1.0),  # before the window
    (332, "future-move", "anthropic", None, "Researcher", "2026-12-01", None, ex("future"), None, 1.0),  # after it
]
EXPECTED_KEYS = ["aff:150:start", "aff:53:start", "aff:60:start", "aff:46:end", "aff:47:start", "aff:120:start",
                 "aff:301:start", "aff:310:end", "aff:321:start"]


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def add_affiliations(session_factory, rows, created_at: datetime = CURATED_ON) -> None:
    with session_factory() as s:
        for aid, person, lab, org, role, start, end, url, evidence, conf in rows:
            s.add(Affiliation(id=aid, person_id=person, lab_id=lab, org_name=org, role=role, start_date=_dt(start),
                              end_date=_dt(end), source_url=url, evidence=evidence, confidence=conf,
                              created_at=created_at))
        s.commit()


@pytest.fixture()
def register_db(db):
    with db() as s:
        s.add_all(Lab(id=i, name=n, kind="frontier_lab", tier=t) for i, n, t in LABS)
        s.add_all(Person(id=i, name=n, tier=t) for i, n, t in PEOPLE)
        s.commit()
    add_affiliations(db, AFFILIATIONS)
    return db


def collect(settings, window: Window = WINDOW) -> list:
    return reg.register_affiliations(SRC, window, settings)


def by_key(records) -> dict:
    return {r.natural_key: r for r in records}


def narrative(text: str) -> list[str]:
    """The sentences above the structured register block."""
    return text.split("\n\n", 1)[0].splitlines()


# --------------------------------------------------------------------------------- events
def test_one_record_per_event_in_window(register_db, settings):
    recs = collect(settings)
    assert [r.natural_key for r in recs] == EXPECTED_KEYS  # sorted by day, then key
    assert {r.source_class for r in recs} == {"curated"}
    assert all(r.published_at.time() == datetime.min.time() for r in recs)


def test_cofounding_team_is_one_record(register_db, settings):
    recs = collect(settings)
    assert sum("Discovery Loop" in r.title for r in recs) == 1  # four founders, one event, one document
    r = by_key(recs)["aff:150:start"]
    assert r.title == ("Jeff Dean, Sanjay Ghemawat, Oriol Vinyals and Quoc Le leave Google DeepMind and Google "
                       "to co-found Discovery Loop")
    assert (r.url, r.person_id, r.lab_id) == (GEEKWIRE, "jeff-dean", "discovery-loop")
    assert r.published_at == datetime(2026, 8, 5)
    lines = narrative(r.content_text)
    assert lines[0] == ("Jeff Dean, Sanjay Ghemawat, Oriol Vinyals and Quoc Le left Google DeepMind and Google to "
                        "co-found Discovery Loop on 2026-08-05.")
    assert ("Jeff Dean left Google DeepMind (Chief Scientist, Google DeepMind and Google Research) on 2026-08-05 "
            "to co-found Discovery Loop.") in lines
    assert "Sanjay Ghemawat left Google (Google Fellow) on 2026-08-05 to co-found Discovery Loop." in lines
    assert "Status: confirmed in the register (confidence 1)." in lines
    assert all(len(line) <= 300 for line in lines)  # quotable: the extractor's quote limit
    for url in (TECHREPUBLIC, GEEKWIRE, GIGAZINE):
        assert f"source: {url}" in r.content_text
    assert ("- Sanjay Ghemawat | tier A | ended | Google (not in the lab register) | role: Google Fellow | "
            f"until 2026-08-05 | confidence 1 | source: {GIGAZINE}") in r.content_text
    assert "tier A = founders, chief scientists and heads of research" in r.content_text
    assert r.meta == {
        "affiliation_id": 150, "change": "start", "role": "Co-founder", "tier": "A", "source_url": GEEKWIRE,
        "event": "co_founding", "affiliation_ids": list(range(149, 157)),
        "people": ["jeff-dean", "sanjay-ghemawat", "oriol-vinyals", "quoc-le"], "person_tiers": ["A"],
        "labs": ["discovery-loop", "google-deepmind"], "sources": [TECHREPUBLIC, GEEKWIRE, GIGAZINE],
        "confidence": 1.0,
    }


def test_same_day_role_changes_are_one_reorganisation_not_departures(register_db, settings):
    r = by_key(collect(settings))["aff:53:start"]
    assert r.title == "Demis Hassabis and Koray Kavukcuoglu change roles at Google DeepMind"
    assert "leave" not in r.title.lower() and " left " not in r.content_text
    assert (r.lab_id, r.person_id, r.url, r.meta["event"]) == ("google-deepmind", "demis-hassabis", NINE_TO_FIVE,
                                                              "role_change")
    assert ("Demis Hassabis changed role at Google DeepMind on 2026-08-05, from Co-founder and CEO to "
            "Chair; Alphabet Chief Scientist.") in narrative(r.content_text)
    assert r.meta["affiliation_ids"] == [52, 53, 54, 55]


def test_single_reported_role_change(register_db, settings):
    r = by_key(collect(settings))["aff:120:start"]
    assert r.title == "Dayiheng Liu reportedly becomes Head of Qwen LLM project at Qwen (was Pretraining lead)"
    lines = narrative(r.content_text)
    assert lines[0] == ("Dayiheng Liu reportedly changed role at Qwen on 2026-09-21, from Pretraining lead to "
                        "Head of Qwen LLM project.")
    assert ("Register note on Dayiheng Liu, Head of Qwen LLM project at Qwen: insider reports; "
            "no official Alibaba announcement.") in lines
    assert lines[-1] == ("Status: reported by secondary sources only, not confirmed: Dayiheng Liu, Head of Qwen LLM "
                         "project at Qwen (register confidence 0.7).")
    assert r.lab_id == "qwen" and r.meta["confidence"] == 0.7


def test_leave_and_later_reported_join_are_separate_records(register_db, settings):
    recs = by_key(collect(settings))
    leave, join = recs["aff:46:end"], recs["aff:47:start"]
    assert leave.title == "Andrew Tulloch leaves Meta AI (TBD Lab researcher)"
    assert (leave.lab_id, leave.url, leave.published_at) == ("meta-ai", SEMAFOR, datetime(2026, 9, 9))
    assert "Status: confirmed in the register (confidence 1)." in leave.content_text
    assert "| 2025-10-01 to 2026-09-09 |" in leave.content_text  # the affiliation's dates
    assert join.title == "Andrew Tulloch reportedly joins Anthropic as Inference and performance"
    assert (join.lab_id, join.url, join.published_at) == ("anthropic", TECHNEWS, datetime(2026, 9, 14))
    lines = narrative(join.content_text)
    assert lines[0] == "Andrew Tulloch reportedly joined Anthropic as Inference and performance on 2026-09-14."
    assert "TechNews/HTX report; not confirmed by Anthropic." in lines[1]
    assert "not confirmed" in lines[-1] and "register confidence 0.7" in lines[-1]
    assert "aff:45" not in " ".join(recs)  # the 2025 Thinking Machines exit is outside the window


def test_join_carries_the_curated_evidence(register_db, settings):
    r = by_key(collect(settings))["aff:60:start"]
    assert r.title == "Barret Zoph joins Google DeepMind as VP Research, RL and post-training"
    assert narrative(r.content_text) == [
        "Barret Zoph joined Google DeepMind as VP Research, RL and post-training on 2026-08-26.",
        "Register note on Barret Zoph, VP Research, RL and post-training at Google DeepMind: announced on X; "
        "confirmed by a Google spokesperson.",
        "Status: confirmed in the register (confidence 1).",
    ]
    assert ("- Barret Zoph | tier B | started | Google DeepMind (register lab google-deepmind) | "
            "role: VP Research, RL and post-training | from 2026-08-26 | confidence 1 | "
            f"source: {THE_STAR}") in r.content_text
    assert "tier B = named leads of flagship model work" in r.content_text
    assert r.meta["tier"] == "B" and r.meta["event"] == "join"


def test_move_from_an_untracked_org_is_attributed_to_the_register_lab(register_db, settings):
    r = by_key(collect(settings))["aff:301:start"]
    assert r.title == "Alex Mover leaves Microsoft to join Anthropic as AI infrastructure lead"
    assert r.lab_id == "anthropic" and r.meta["event"] == "move"
    assert narrative(r.content_text)[0] == ("Alex Mover left Microsoft (President, AI Platform) on 2026-10-08 to "
                                            "join Anthropic as AI infrastructure lead.")
    assert "Microsoft (not in the lab register)" in r.content_text


def test_people_leaving_one_lab_on_one_day_are_one_record_most_senior_first(register_db, settings):
    r = by_key(collect(settings))["aff:310:end"]  # key: the lowest id, so it is stable as members are added
    assert r.title == "Sam Leaver and Pat Leaver leave xAI"
    assert r.person_id == "sam-leaver" and r.meta["people"] == ["sam-leaver", "pat-leaver"]
    assert r.lab_id == "xai" and r.meta["person_tiers"] == ["A", "C"]
    assert narrative(r.content_text)[0] == "Sam Leaver and Pat Leaver left xAI on 2026-10-15."
    assert "tier A = founders" in r.content_text and "tier C = high-signal contributors" in r.content_text


def test_founding_an_untracked_org_keeps_the_tracked_lab_left(register_db, settings):
    r = by_key(collect(settings))["aff:321:start"]
    assert r.title == "Yan Founder leaves Meta AI to found Example Labs"  # "Founder", not "Co-founder"
    assert r.lab_id == "meta-ai" and r.meta["event"] == "co_founding"
    assert narrative(r.content_text)[0] == ("Yan Founder left Meta AI (Chief AI Scientist) on 2026-10-20 to found "
                                            "Example Labs (Founder and Executive Chairman).")


def test_cofounding_group_key_survives_a_member_added_later(register_db, settings):
    with register_db() as s:
        s.add(Person(id="late-founder", name="Late Founder", tier="B"))
        s.commit()
    add_affiliations(register_db, [(400, "late-founder", "discovery-loop", None, "Co-founder", "2026-08-05", None,
                                    ex("dl"), None, 1.0)])
    recs = collect(settings)
    assert [r.natural_key for r in recs] == EXPECTED_KEYS  # same documents: nothing re-ingested
    r = by_key(recs)["aff:150:start"]
    assert r.meta["people"][-1] == "late-founder"  # tier B after the tier-A founders
    # not every founder left a register lab that day, so the headline no longer claims they all did
    assert r.title == "Jeff Dean, Sanjay Ghemawat, Oriol Vinyals, Quoc Le and Late Founder co-found Discovery Loop"
    assert "Late Founder co-founded Discovery Loop on 2026-08-05." in narrative(r.content_text)


def test_degenerate_rows(register_db, settings):
    add_affiliations(register_db, [
        (500, "pat-leaver", "anthropic", None, "Advisor", "2026-10-01", "2026-10-01", ex("zero"), None, 1.0),
        (501, "sam-leaver", None, None, "Advisor", "2026-10-02", None, ex("no-org"), None, 1.0),
        (502, "old-move", "anthropic", None, "Researcher", "2026-10-03", None, "  ", None, 1.0),
    ])
    keys = [r.natural_key for r in collect(settings)]
    assert "aff:500:start" in keys and "aff:500:end" not in keys  # zero-length row: the start only
    assert not any(k.startswith(("aff:501:", "aff:502:")) for k in keys)  # no organisation / blank source
    assert by_key(collect(settings))["aff:500:start"].title == "Pat Leaver joins Anthropic as Advisor"


def test_namesakes_are_both_named_and_organisations_once(register_db, settings):
    with register_db() as s:
        s.add_all([Person(id="wei-zhang-1", name="Wei Zhang", tier="B"),
                   Person(id="wei-zhang-2", name="Wei Zhang", tier="B")])
        s.commit()
    add_affiliations(register_db, [
        (700, "wei-zhang-1", "qwen", None, "Researcher", None, "2026-10-28", ex("qwen-exit"), None, 1.0),
        (701, "wei-zhang-2", "qwen", None, "Engineer", None, "2026-10-28", ex("qwen-exit"), None, 1.0),
    ])
    r = by_key(collect(settings))["aff:700:end"]
    assert r.title == "Wei Zhang and Wei Zhang leave Qwen"  # two people, one organisation
    assert narrative(r.content_text)[0] == "Wei Zhang and Wei Zhang left Qwen on 2026-10-28."
    assert r.meta["people"] == ["wei-zhang-1", "wei-zhang-2"]


# --------------------------------------------------------------------------------- window
def test_uncited_and_out_of_window_rows_are_skipped(register_db, settings):
    keys = " ".join(r.natural_key for r in collect(settings))
    for aid in (330, 331, 332):  # no source_url / before the window / after `until`
        assert f"aff:{aid}:" not in keys


def test_window_is_whole_days(register_db, settings):
    recs = collect(settings, Window(since=datetime(2026, 9, 9, 8, 15), until=datetime(2026, 9, 21, 0, 1)))
    assert [r.natural_key for r in recs] == ["aff:46:end", "aff:47:start", "aff:120:start"]


@pytest.mark.parametrize("day, created, expected", [
    (date(2026, 9, 9), None, True),                         # on the first day of the window (since 08:15)
    (date(2026, 11, 6), datetime(2026, 11, 5), False),      # after `until`: collected once its day arrives
    (date(2026, 9, 1), datetime(2026, 9, 10), True),        # recorded after the window opened, in the tracked period
    (date(2026, 9, 1), datetime(2026, 9, 8), False),        # recorded before the window: an earlier run had it
    (date(2026, 7, 31), datetime(2026, 9, 10), False),      # before the tracked period (backfill_start)
])
def test_in_window(day, created, expected):
    window = Window(since=datetime(2026, 9, 9, 8, 15), until=datetime(2026, 11, 5, 12))
    assert reg.in_window(day, created, window, floor=date(2026, 8, 1)) is expected


def test_move_recorded_late_is_collected_by_the_incremental_run(register_db, settings):
    with register_db() as s:
        s.add_all([Person(id="late-recorded", name="Late Recorded", tier="B"),
                   Person(id="too-old", name="Too Old", tier="A")])
        s.commit()
    add_affiliations(register_db, [
        (600, "late-recorded", "anthropic", None, "Pretraining lead", "2026-09-30", None, ex("late"), None, 1.0),
        (601, "too-old", "anthropic", None, "Researcher", "2026-07-15", None, ex("too-old"), None, 1.0),
    ], created_at=datetime(2026, 11, 4, 9))
    recs = collect(settings, Window(since=datetime(2026, 11, 3, 6), until=NOW))  # a scheduled run: 2-day overlap
    assert [r.natural_key for r in recs] == ["aff:600:start"]
    assert recs[0].published_at == datetime(2026, 9, 30)  # dated at the move, not at the curation


def test_max_items_keeps_the_most_recent(register_db, settings):
    recs = collect(settings, Window(since=WINDOW.since, until=NOW, max_items=2))
    assert [r.natural_key for r in recs] == EXPECTED_KEYS[-2:]


# --------------------------------------------------------------------------------- wiring
@pytest.mark.parametrize("name, short", [
    ("Meta AI (Superintelligence Labs / FAIR)", "Meta AI"), ("Qwen (Alibaba)", "Qwen"), ("xAI", "xAI"),
    ("Example Labs (EL)", "Example Labs"), ("(odd)", "(odd)"), (None, ""),
])
def test_short_name(name, short):
    assert reg.short_name(name) == short


def _register_spec() -> dict:
    return yaml.safe_load((CONFIG_DIR / "sources_register.yaml").read_text(encoding="utf-8"))["sources"]


def test_sources_register_yaml_declares_the_curated_source():
    (spec,) = _register_spec()
    assert spec == {"id": "register:affiliations", "lab": None, "adapter": "register_affiliations", "kind": "register",
                    "source_class": "curated", "cadence_minutes": 1440, "freshness_slo_hours": 8760}
    assert spec["source_class"] in SOURCE_CLASSES
    assert COLLECTORS["register_affiliations"].fn is reg.register_affiliations
    assert COLLECTORS["register_affiliations"].requires == ()  # no credential: it never leaves the database


def test_runner_persists_curated_documents_idempotently(register_db, tmp_path, monkeypatch):
    from fli.ingest import runner

    config = tmp_path / "config"
    config.mkdir()
    shutil.copy(CONFIG_DIR / "sources_register.yaml", config / "sources_register.yaml")
    monkeypatch.setattr(loader, "CONFIG_DIR", config)
    assert loader.load_register()["sources"] == 1  # the path `fli register` takes
    monkeypatch.setattr(runner, "utcnow", lambda: NOW)
    first = runner.run_ingest(source_ids=["register:affiliations"], since=datetime(2026, 8, 1))
    second = runner.run_ingest(source_ids=["register:affiliations"], since=datetime(2026, 8, 1))
    assert first["register:affiliations"] == {"status": "ok", "fetched": 9, "new": 9, "duplicates": 0}
    assert second["register:affiliations"] == {"status": "ok", "fetched": 9, "new": 0, "duplicates": 0}
    with register_db() as s:
        src = s.get(Source, "register:affiliations")
        assert (src.kind, src.source_class, src.cadence_minutes, src.freshness_slo_hours) == ("register", "curated",
                                                                                              1440, 8760)
        docs = {d.natural_key: d for d in s.scalars(select(Document).where(Document.source_id == src.id))}
        assert sorted(docs) == sorted(EXPECTED_KEYS)
        assert {d.source_class for d in docs.values()} == {"curated"}
        assert {d.status for d in docs.values()} == {"new"}  # flows into triage like any document
        dl = docs["aff:150:start"]
        assert (dl.lab_id, dl.person_id, dl.published_at) == ("discovery-loop", "jeff-dean", datetime(2026, 8, 5))
        assert dl.canonical_url == canonical_url(GEEKWIRE) and dl.meta["event"] == "co_founding"
        assert dl.raw_path and (Path(tmp_path) / dl.raw_path).exists()  # snapshot under the test RAW_DIR
