"""Research collectors (HF daily papers, arXiv author feeds) against trimmed recorded responses.

Fixtures in tests/fixtures/research are real API responses (Sep-Oct 2026) cut down to a few items;
no test touches the network: HTTP goes through respx, arXiv through a fake requests session.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import arxiv
import httpx
import pytest
import respx

from fli.db.models import Document, Identity, Lab, Person, Source
from fli.ingest.collectors import research
from fli.ingest.records import SourceInfo, Window
from fli.ingest.runner import persist_records

FIX = Path(__file__).parent / "fixtures" / "research"


def _seed(db) -> None:
    """Minimal register: three labs, Google's HF identity, tracked people with arXiv/HF identities."""
    with db() as s:
        for lab_id, name in [
            ("google-deepmind", "Google DeepMind"),
            ("discovery-loop", "Discovery Loop"),
            ("meta-ai", "Meta AI (Superintelligence Labs / FAIR)"),
            ("xai", "xAI"),
        ]:
            s.add(Lab(id=lab_id, name=name, kind="frontier_lab", tier=1))
        s.flush()
        s.add_all(
            [
                Person(id="jakob-foerster", name="Jakob Foerster", lab_id="meta-ai", tier="B"),
                Person(id="jeff-dean", name="Jeff Dean", lab_id="discovery-loop", tier="A"),
                Person(id="josef-dean", name="Josef Dean", lab_id="google-deepmind", tier="C"),
                Person(id="song-han", name="Song Han", lab_id="discovery-loop", tier="B"),
                Person(
                    id="junjie-wang", name="J. Wang", lab_id="meta-ai", tier="C"
                ),  # matched only via HF username
            ]
        )
        s.flush()
        s.add_all(
            [
                Identity(platform="hf", handle="google", lab_id="google-deepmind"),
                Identity(platform="hf", handle="wanng", person_id="junjie-wang"),
                Identity(platform="arxiv_name", handle="foerster, jakob", person_id="jakob-foerster"),
                Identity(platform="arxiv_name", handle="dean, jeff", person_id="jeff-dean"),
                Identity(platform="arxiv_name", handle="dean, josef", person_id="josef-dean"),
            ]
        )
        s.commit()


def _source(sid: str, adapter: str, kind: str, config: dict) -> SourceInfo:
    return SourceInfo(id=sid, adapter=adapter, kind=kind, source_class="research", url=None, config=config)


# ------------------------------------------------------------------------------------ names
def test_name_matching_handles_order_initials_and_namesakes():
    jeff = research.TrackedName.from_name("dean, jeff", "jeff-dean", None, "A")
    assert (jeff.first, jeff.last, jeff.query) == ("jeff", "dean", "jeff dean")
    assert jeff.matches("Jeff Dean") and not jeff.matches("Josef Dean")
    assert jeff.matches("J. Dean", allow_initial=True) and not jeff.matches("J. Dean")
    jakob = research.TrackedName.from_name("Jakob N. Foerster", "jf", None, "B")
    assert jakob.query == "jakob foerster"  # initials dropped from the arXiv query
    assert jakob.matches("Jakob Nicolaus Foerster")
    assert research.name_tokens("Łukasz Kaiser") == ["lukasz", "kaiser"]
    assert research.TrackedName.from_name("Madonna", "m", None, None) is None


def test_org_matching_uses_handles_not_display_names(db):
    _seed(db)
    index = research.load_org_index({"facebook": "meta-ai"})
    assert research.match_org({"name": "google", "fullname": "Google"}, index)["via"] == "hf_identity"
    assert research.match_org({"name": "facebook"}, index) == {
        "lab_id": "meta-ai",
        "via": "alias",
        "value": "facebook",
    }
    assert research.match_org({"name": "GoogleDeepMind"}, index)["via"] == "lab_name"
    assert research.match_org({"name": "GoogIe", "fullname": "Google"}, index) is None  # impostor org
    assert research.match_org({"name": "XAI", "fullname": "XAI"}, index) is None  # short names never match


# ---------------------------------------------------------------------------- hf_daily_papers
def _daily_router(mock: respx.MockRouter) -> respx.Route:
    files = {p.stem.removeprefix("hf_daily_"): json.loads(p.read_text()) for p in FIX.glob("hf_daily_*.json")}

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.params["limit"] == "100"
        return httpx.Response(200, json=files.get(request.url.params["date"], []))

    return mock.get(research.HF_DAILY_PAPERS_API).mock(side_effect=answer)


def test_hf_daily_papers_selects_lab_author_and_popular_papers(db, settings):
    _seed(db)
    src = _source(
        "hf:daily_papers",
        "hf_daily_papers",
        "hf_papers",
        {"org_aliases": {"facebook": "meta-ai"}, "min_upvotes_unmatched": 150},
    )
    window = Window(since=datetime(2026, 9, 14), until=datetime(2026, 10, 14))
    with respx.mock(assert_all_called=False) as mock:
        route = _daily_router(mock)
        records = research.hf_daily_papers(src, window, settings)
    assert route.call_count == 31  # one call per day; no day reached the 100-item page size
    by_id = {r.natural_key: r for r in records}
    assert set(by_id) == {"2609.14858", "2609.15818", "2609.14708", "2609.14302", "2610.08077", "2610.09484"}
    google = by_id["2609.14858"]
    assert (google.lab_id, google.meta["lab_match"]["via"], google.meta["selection"]) == (
        "google-deepmind",
        "hf_identity",
        "lab",
    )
    assert google.url == "https://arxiv.org/abs/2609.14858" and google.published_at == datetime(2026, 9, 14)
    assert google.content_text.startswith(google.title) and "machine-generated" in google.content_text
    assert by_id["2610.09484"].lab_id == "meta-ai" and by_id["2610.09484"].meta["lab_match"]["via"] == "alias"
    assert by_id["2609.15818"].lab_id is None and by_id["2609.15818"].meta["selection"] == "upvotes"
    by_name = by_id["2609.14708"]  # "Song Han" is an author
    assert (by_name.person_id, by_name.lab_id, by_name.meta["selection"]) == (
        "song-han",
        "discovery-loop",
        "lab",
    )
    assert by_name.meta["lab_match"]["via"] == "tracked_author"
    assert by_id["2609.14302"].person_id == "junjie-wang"  # claimed HF account "wanng"
    # impostor org "GoogIe" (33 upvotes) and alibabagroup (128 < 150) are dropped
    assert "2610.09326" not in by_id and "2610.13166" not in by_id


def test_hf_daily_papers_follows_pages_and_persists_idempotently(db, settings):
    _seed(db)
    item = json.loads((FIX / "hf_daily_2026-09-15.json").read_text())[1]  # the google paper
    full_page = [json.loads(json.dumps(item).replace("2609.14858", f"2609.{90000 + i}")) for i in range(100)]
    pages = {"0": full_page, "1": full_page[:3]}
    with respx.mock() as mock:
        route = mock.get(research.HF_DAILY_PAPERS_API).mock(
            side_effect=lambda req: httpx.Response(200, json=pages.get(req.url.params["p"], []))
        )
        with httpx.Client() as client:
            assert len(research.fetch_daily(client, datetime(2026, 9, 15).date())) == 103
        assert route.call_count == 2  # stopped on the short page despite the API's "next" link
    src = _source("hf:daily_papers", "hf_daily_papers", "hf_papers", {})
    with respx.mock() as mock:
        _daily_router(mock)
        records = research.hf_daily_papers(
            src, Window(since=datetime(2026, 9, 15), until=datetime(2026, 9, 15)), settings
        )
    with db() as s:
        row = Source(id=src.id, adapter=src.adapter, kind=src.kind, source_class="research")
        s.add(row)
        s.flush()
        assert persist_records(s, row, records) == (len(records), 0)
        assert persist_records(s, row, records) == (0, 0)
        s.commit()
        assert s.query(Document).filter_by(source_id=src.id, lab_id="google-deepmind").count() == 1


# ------------------------------------------------------------------------------ arxiv_authors
class FakeArxivSession:
    """Stands in for the requests.Session inside arxiv.Client; serves the recorded Atom feed."""

    def __init__(self, payload: bytes):
        self.payload, self.urls = payload, []

    def get(self, url: str, headers=None):
        self.urls.append(url)
        return SimpleNamespace(status_code=200, content=self.payload)


@pytest.fixture()
def fake_arxiv(monkeypatch):
    client = arxiv.Client(page_size=100, delay_seconds=0, num_retries=0)
    client._session = FakeArxivSession((FIX / "arxiv_authors_feed.xml").read_bytes())
    monkeypatch.setattr(research, "_arxiv_shared", client)
    return client._session


def _arxiv_source(**config) -> SourceInfo:
    return _source("arxiv:authors", "arxiv_authors", "arxiv", {"delay_seconds": 0, **config})


def test_arxiv_authors_attributes_window_papers(db, settings, fake_arxiv):
    _seed(db)
    records = research.arxiv_authors(
        _arxiv_source(), Window(since=datetime(2026, 8, 1), until=datetime(2026, 11, 4)), settings
    )
    assert len(fake_arxiv.urls) == 1  # 3 tracked names fit one OR query; the feed was not re-paged
    query = httpx.URL(fake_arxiv.urls[0]).params["search_query"]
    assert 'au:"jakob foerster"' in query and 'au:"jeff dean"' in query and " OR " in query
    assert httpx.URL(fake_arxiv.urls[0]).params["sortBy"] == "submittedDate"
    assert [r.natural_key for r in records] == ["2610.39576", "2610.28753", "2610.28086", "2608.03644"]
    first = records[0]
    assert (first.person_id, first.lab_id, first.meta["version"]) == ("jakob-foerster", "meta-ai", "v2")
    assert first.url == "https://arxiv.org/abs/2610.39576" and first.source_class is None
    assert records[2].meta["matched_authors"] == ["Jakob N. Foerster"]  # middle initial tolerated


def test_arxiv_authors_merges_coauthors_and_filters_categories(db, settings, fake_arxiv):
    _seed(db)
    window = Window(since=datetime(2026, 4, 1), until=datetime(2026, 11, 4))
    records = {r.natural_key: r for r in research.arxiv_authors(_arxiv_source(), window, settings)}
    diloco = records["2604.21428"]  # co-authored by tracked Jeff Dean (tier A) and Josef Dean (tier C)
    assert diloco.person_id == "jeff-dean" and set(diloco.meta["tracked_people"]) == {
        "jeff-dean",
        "josef-dean",
    }
    only_cv = research.arxiv_authors(_arxiv_source(category_prefixes=["cs.CV"]), window, settings)
    assert [r.natural_key for r in only_cv] == ["2610.28086"]
    capped = research.arxiv_authors(_arxiv_source(max_results_per_author=2), window, settings)
    assert sum(1 for r in capped if r.person_id == "jakob-foerster") == 2


def test_arxiv_authors_without_identities_makes_no_request(db, settings, fake_arxiv):
    assert (
        research.arxiv_authors(
            _arxiv_source(), Window(since=datetime(2026, 8, 1), until=datetime(2026, 11, 4)), settings
        )
        == []
    )
    assert fake_arxiv.urls == []


def test_window_results_stops_after_a_streak_of_old_items():
    def res(day: int):
        return SimpleNamespace(published=datetime(2026, 7, day))

    stream = [res(30), res(20), res(29), res(10), res(9), res(8), res(7), res(6), res(31)]
    kept = list(research.window_results(iter(stream), since=datetime(2026, 7, 25)))
    assert [r.published.day for r in kept] == [30, 29]  # the late "31" is beyond the stop streak
