"""Register: loader round trip, entity resolution (attribution, adjudication, GitHub links) and the
contributor score. No network and no LLM: HTTP goes through httpx.MockTransport, the LLM gateway is a
fake, and every run passes use_github/use_llm/use_citations explicitly (.env may hold real keys)."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import httpx
import pytest
import yaml
from sqlalchemy import func, select

from fli.config import CONFIG_DIR, DATA_DIR
from fli.db.models import Affiliation, Document, Identity, Insight, Lab, Person, Source
from fli.db.session import session_scope
from fli.llm.client import LLMResult
from fli.register import contributors as contrib
from fli.register import entity_resolution as er
from fli.register import loader

FIX = Path(__file__).parent / "fixtures" / "register"


@pytest.fixture()
def register(db, monkeypatch):
    """Fixture register (9 people, 4 labs) loaded into the per-test database."""
    monkeypatch.setattr(loader, "CONFIG_DIR", FIX / "config")
    loader.load_register()
    return db


def add_doc(
    key: str,
    *,
    kind: str = "hf_papers",
    authors: list[str] | None = None,
    author: str | None = None,
    lab: str | None = None,
    meta: dict | None = None,
    published=None,
) -> int:
    source_id = f"test:{kind}"
    with session_scope() as s:
        if s.get(Source, source_id) is None:
            s.add(Source(id=source_id, adapter="test", kind=kind, source_class="research"))
        doc = Document(
            source_id=source_id,
            natural_key=key,
            url=f"https://example.org/{key}",
            canonical_url=f"https://example.org/{key}",
            content_hash=key,
            source_class="research",
            author=author,
            lab_id=lab,
            published_at=published,
            meta={**({"authors": authors} if authors else {}), **(meta or {})},
        )
        s.add(doc)
        s.flush()
        return doc.id


def get_doc(doc_id: int) -> Document:
    with session_scope() as s:
        return s.get(Document, doc_id)


def run_er(**kw) -> dict:
    return er.run_entity_resolution(**{"use_github": False, "use_llm": False, **kw})


# ------------------------------------------------------------------------------------- loader
def test_loader_round_trip_fixture(register):
    with session_scope() as s:
        assert s.scalar(select(func.count(Person.id))) == 9
        assert s.get(Person, "jerry-tworek").lab_id is None
        assert s.get(Person, "andrej-karpathy").meta["revealed_preference"] is None
        x = s.scalar(select(Identity).where(Identity.platform == "x", Identity.handle == "sama"))
        assert x.person_id == "sam-altman" and x.method == "seed" and x.status == "confirmed"
        src = s.get(Source, "x:sama")
        assert src.person_id == "sam-altman" and src.kind == "x_account" and src.source_class == "personal"
        n_aff, n_ident = (
            s.scalar(select(func.count(Affiliation.id))),
            s.scalar(select(func.count(Identity.id))),
        )
    loader.load_register()  # idempotent: no duplicated affiliations or identities
    with session_scope() as s:
        assert s.scalar(select(func.count(Affiliation.id))) == n_aff
        assert s.scalar(select(func.count(Identity.id))) == n_ident


def test_real_people_yaml_is_valid(db, tmp_path, monkeypatch):
    """config/people.yaml loads with FKs enforced and meets the register's own rules."""
    for name in ("labs.yaml", "people.yaml"):
        shutil.copy(CONFIG_DIR / name, tmp_path / name)
    monkeypatch.setattr(loader, "CONFIG_DIR", tmp_path)
    counts = loader.load_register()
    people = yaml.safe_load((CONFIG_DIR / "people.yaml").read_text())["people"]
    assert 70 <= counts["people"] == len(people) <= 110
    with session_scope() as s:
        labs = set(s.scalars(select(Lab.id)))
        current = set(s.scalars(select(Person.lab_id).where(Person.lab_id.is_not(None))))
        tracked = s.scalar(select(func.count(Source.id)).where(Source.person_id.is_not(None)))
    assert labs == current, f"labs without a current person: {labs - current}"
    assert 20 <= tracked <= 30
    for p in people:
        meta = p["meta"]
        assert p["tier"] in {"A", "B", "C"} and p["status"] in {"active", "departed", "founded_startup"}
        assert meta.get("revealed_preference") in (True, False, None)
        if meta.get("revealed_preference"):
            assert str(meta.get("revealed_preference_source", "")).startswith("http"), p["id"]
        if p["tier"] in {"A", "B"}:
            assert meta.get("verified_on") == "2026-11-04", p["id"]
        assert set(meta.get("roles") or []) <= contrib.LEADERSHIP_ROLES | {"flagship_author", "maintainer"}


# ------------------------------------------------------------------------- name matching
def test_name_score_rules():
    assert er.name_score("quoc v le", "quoc le") == er.INITIAL_VARIANT_SCORE
    assert er.name_score("tom a brown", "tom b brown") < er.FUZZY_MIN  # conflicting middle initials
    assert er.name_score("j dean", "jeff dean") == er.FIRST_INITIAL_SCORE
    assert er.name_score("zhilin yang", "yang zhilin") == 100.0  # word order
    assert er.norm_name("Łukasz  Kaiser") == "lukasz kaiser"
    assert er.norm_name("Hyung-Won Chung") == "hyung won chung"


def test_attribution_exact_fuzzy_and_negative(register):
    exact = add_doc("exact", authors=["Alice Smith", "Jeffrey Dean"], meta={"categories": ["cs.LG"]})
    fuzzy = add_doc(
        "fuzzy",
        authors=["Jeffrey A. Dean", "Bob Jones"],
        lab="google-deepmind",
        meta={"lab_match": {"lab_id": "google-deepmind", "via": "hf_identity"}},
    )
    common = add_doc("common", authors=["Mark Chen", "Wei Li"])
    common_ok = add_doc("common-ok", authors=["Quoc Le", "Carol White"], lab="google-deepmind")
    off_domain = add_doc("offdomain", authors=["Jeffrey Dean"], meta={"categories": ["q-bio.PE"]})
    no_context = add_doc("nocontext", authors=["Jeffrey A. Dean"])
    circular = add_doc(
        "circular", authors=["Mark Chen"], lab="openai", meta={"tracked_people": ["mark-chen"]}
    )
    senior = add_doc("senior", authors=["Neel Nanda", "Anca D. Dragan", "Jeffrey Dean"])
    out = run_er()
    assert out["documents_scanned"] == 8 and out["attributed"] == 4
    assert get_doc(exact).person_id == "jeff-dean"
    f = get_doc(fuzzy)
    assert f.person_id == "jeff-dean" and f.meta["people_matched"][0]["method"] == "fuzzy"
    assert get_doc(common_ok).person_id == "quoc-le"  # common name, corroborated by the lab
    for doc_id in (common, off_domain, no_context, circular):
        d = get_doc(doc_id)
        assert d.person_id is None and d.meta["people_matched"] == [] and d.meta["er"]["reg"]
    s = get_doc(senior)
    assert s.person_id == "jeff-dean"  # tier A beats B and C
    assert [m["person_id"] for m in s.meta["people_matched"]] == ["jeff-dean", "anca-dragan", "neel-nanda"]


def test_handles_bylines_and_bots(register):
    post = add_doc("post", kind="x_account", author="SamA", meta={"platform": "x"})
    release = add_doc("release", kind="github_org", author="karpathy", meta={"kind": "release"})
    bot = add_doc("bot", kind="github_org", author="github-actions[bot]", meta={"kind": "release"})
    byline = add_doc(
        "byline", kind="official_blog", author="Anca Dragan; Raluca Ada Popa", lab="google-deepmind"
    )
    wrong_platform = add_doc("wrong", kind="x_account", author="colah", meta={"platform": "x"})
    run_er()
    assert get_doc(post).person_id == "sam-altman"
    assert get_doc(release).meta["people_matched"][0]["method"] == "handle"
    assert get_doc(release).person_id == "andrej-karpathy"
    assert get_doc(bot).person_id is None
    assert get_doc(byline).person_id == "anca-dragan"
    assert get_doc(wrong_platform).person_id is None  # colah is his GitHub, not his X handle


def test_er_is_idempotent_and_incremental(register):
    add_doc("a", authors=["Jeffrey Dean"])
    unmatched = add_doc("b", authors=["Jerry  Tworek-Smith", "Someone Else"])
    first = run_er()
    second = run_er()
    assert first["documents_scanned"] == 2
    assert second["documents_scanned"] == 0 and second["skipped_resolved"] == 1
    with session_scope() as s:  # a register change (new name form) re-opens unmatched documents
        s.add(Identity(platform="arxiv_name", handle="jerry tworek-smith", person_id="jerry-tworek"))
    third = run_er()
    assert third["register_fingerprint"] != first["register_fingerprint"]
    assert third["documents_scanned"] == 1 and get_doc(unmatched).person_id == "jerry-tworek"


# -------------------------------------------------------------------- gray-zone adjudication
RECORD_A = (
    "name: Jeff Dean\ncurrent lab: discovery-loop\n"
    "affiliation: google-deepmind | Chief Scientist | until 2026-08\n"
    "tracked colleagues (shared lab): Anca Dragan, Sanjay Ghemawat"
)
RECORD_B = (
    "author as printed: J. Dean\ndocument lab: google-deepmind\nother authors: Sanjay Ghemawat\n"
    "tracked co-author: Sanjay Ghemawat (register name: Sanjay Ghemawat; labs: discovery-loop)"
)


@pytest.mark.parametrize(
    ("data", "accepted"),
    [
        ({"verdict": "same", "overlapping_facts": ["Sanjay Ghemawat"]}, True),  # shared tracked co-author
        ({"verdict": "same", "overlapping_facts": ["sanjay ghemawat."]}, True),  # normalised comparison
        ({"verdict": "same", "overlapping_facts": ["google-deepmind"]}, False),  # the lab is the precondition
        ({"verdict": "same", "overlapping_facts": ["Chief Scientist"]}, False),  # only in record A
        ({"verdict": "same", "overlapping_facts": ["Dean"]}, False),  # the compared name is not evidence
        ({"verdict": "same", "overlapping_facts": ["Ana Lopez"]}, False),  # invented: in neither record
        ({"verdict": "same", "overlapping_facts": []}, False),
        ({"verdict": "different", "overlapping_facts": ["Sanjay Ghemawat"]}, False),
        (None, False),
    ],
)
def test_adjudication_acceptance_rule(data, accepted):
    labs = ("google-deepmind", "discovery-loop")
    ok, facts = er.accept_adjudication(data, RECORD_A, RECORD_B, ("J. Dean", "Jeff Dean"), labs)
    assert ok is accepted and (bool(facts) is accepted)


class FakeGateway:
    def __init__(self, *responses: dict | None):
        self.responses, self.calls = list(responses), []

    async def structured_many(self, calls: list[dict]) -> list[LLMResult]:
        self.calls += calls
        return [LLMResult(self.responses.pop(0), "ok") for _ in calls]


def test_gray_zone_with_mocked_llm(register, settings):
    lab = {"lab_match": {"lab_id": "google-deepmind", "via": "hf_identity"}}
    accepted = add_doc("gray-yes", authors=["J. Dean", "Anca D. Dragan"], lab="google-deepmind", meta=lab)
    gw = FakeGateway(
        {"verdict": "same", "overlapping_facts": ["google-deepmind", "Anca Dragan"], "reason": "x"}
    )
    out = run_er(use_llm=True, gateway=gw)
    assert out["gray_pairs"] == 1 and out["llm_accepted"] == 1
    call = gw.calls[0]
    assert call["purpose"] == "er" and call["model"] == settings.model_er and "RECORD A" in call["user"]
    assert "register name: Anca Dragan" in call["user"] and "tracked colleagues" in call["user"]
    doc = get_doc(accepted)
    assert doc.person_id == "jeff-dean"  # tier A outranks the exactly matched tier-B co-author
    assert [m["method"] for m in doc.meta["people_matched"]] == ["llm_adjudicated", "exact"]
    assert doc.meta["er"]["gray"] == []
    with session_scope() as s:
        ident = s.scalar(
            select(Identity).where(Identity.platform == "arxiv_name", Identity.handle == "j. dean")
        )
        assert (ident.method, ident.status, ident.person_id) == ("llm_adjudicated", "candidate", "jeff-dean")

    rejected = add_doc("gray-no", authors=["J. Dean"], lab="google-deepmind", meta=lab)
    lab_only = {"verdict": "same", "overlapping_facts": ["google-deepmind"], "reason": "same lab"}
    gw2 = FakeGateway(lab_only)
    out = run_er(use_llm=True, gateway=gw2)
    assert out["llm_rejected"] == 1 and get_doc(rejected).person_id is None
    # the candidate created above must not be shown to the model as a confirmed name form
    assert "publishes as: j. dean" not in gw2.calls[0]["user"]
    assert "publishes as: jeffrey dean" in gw2.calls[0]["user"]
    assert run_er(use_llm=True, gateway=FakeGateway())["documents_scanned"] == 0  # decided, not retried


def test_gray_zone_deferred_without_llm(register, settings, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    doc_id = add_doc(
        "gray",
        authors=["J. Dean"],
        lab="google-deepmind",
        meta={"lab_match": {"lab_id": "google-deepmind", "via": "hf_identity"}},
    )
    assert run_er(use_llm=True)["gray_deferred"] == 1
    assert get_doc(doc_id).meta["er"]["gray"][0]["person_id"] == "jeff-dean"
    assert run_er(use_llm=True)["documents_scanned"] == 1  # pending pairs are retried


# ---------------------------------------------------------------------- GitHub self-assertion
def github_transport(seen: list[str]) -> httpx.MockTransport:
    profiles = {
        "karpathy": json.loads((FIX / "github_user_karpathy.json").read_text()),
        "neelnanda-io": {
            "login": "neelnanda-io",
            "twitter_username": "NeelNanda5",
            "blog": "https://www.neelnanda.io",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        login = request.url.path.rsplit("/", 1)[-1]
        seen.append(login)
        return (
            httpx.Response(200, json=profiles[login]) if login in profiles else httpx.Response(404, json={})
        )

    return httpx.MockTransport(handler)


def test_github_links_x_identities(register, settings):
    seen: list[str] = []
    counts: Counter = Counter()
    client = httpx.Client(transport=github_transport(seen))
    er.link_github(settings, counts, client=client)
    assert sorted(seen) == ["colah", "karpathy", "neelnanda-io"]
    assert (
        counts["x_self_asserted"] == 1
        and counts["x_candidates_added"] == 1
        and counts["github_not_found"] == 1
    )
    with session_scope() as s:

        def ident(platform: str, handle: str) -> Identity:
            return s.scalar(select(Identity).where(Identity.platform == platform, Identity.handle == handle))

        kx = ident("x", "karpathy")  # seed X handle confirmed by the blog link on his GitHub profile
        assert (kx.method, kx.confidence, kx.status) == ("self_asserted", 1.0, "confirmed")
        assert "blog=karpathy" in kx.evidence
        nx = ident("x", "neelnanda5")
        assert (nx.person_id, nx.status, nx.method) == ("neel-nanda", "candidate", "self_asserted")
        assert ident("homepage", "neelnanda.io").status == "candidate"
        assert ident("github", "colah").meta["github"]["found"] is False
    er.link_github(settings, counts, client=client)  # refreshed monthly, so no new calls
    assert len(seen) == 3


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://twitter.com/karpathy", ("x", "karpathy")),
        ("x.com/@NeelNanda5/", ("x", "neelnanda5")),
        ("https://github.com/nat", ("github", "nat")),  # natfriedman's blog field points at his main account
        ("https://scholar.google.com/citations?user=JWmiQR0AAAAJ&hl=en", ("scholar", "jwmiqr0aaaaj")),
        ("http://www.horace.io/", ("homepage", "horace.io")),
        ("lilianweng.github.io", ("homepage", "lilianweng.github.io")),
    ],
)
def test_classify_profile_urls(url, expected):
    assert er.classify_url(url) == expected


def test_github_clash_is_left_for_review(register):
    counts: Counter = Counter()
    with session_scope() as s:
        gh = s.scalar(select(Identity).where(Identity.platform == "github", Identity.handle == "colah"))
        er.apply_github_profile(s, gh.id, "chris-olah", "colah", {"twitter_username": "OpenAI"}, counts)
    assert counts["x_conflicts"] == 1
    with session_scope() as s:
        lab_x = s.scalar(select(Identity).where(Identity.platform == "x", Identity.handle == "openai"))
        assert lab_x.lab_id == "openai" and lab_x.person_id is None


# ----------------------------------------------------------------------------- ER evaluation
def test_evaluate_er_on_labelled_pairs():
    m = er.evaluate_er(DATA_DIR / "eval" / "er_pairs.yaml")
    assert 40 <= m["n"] <= 60
    assert m["precision"] >= 0.9 and m["recall"] >= 0.85
    assert m["recall_if_gray_accepted"] >= m["recall"]


def test_evaluate_er_counts(tmp_path):
    pairs = {
        "pairs": [
            {"id": 1, "person": "jeff-dean", "mention": "Jeffrey Dean", "label": "same"},
            {"id": 2, "person": "mark-chen", "mention": "Mark Chen", "label": "different"},
            {"id": 3, "person": "mark-chen", "mention": "Mark Chen", "lab": "openai", "label": "same"},
            {"id": 4, "person": "jeff-dean", "mention": "J. Dean", "lab": "google-deepmind", "label": "same"},
            {"id": 5, "person": "sam-altman", "mention": "sama", "platform": "x", "label": "same"},
        ]
    }
    path = tmp_path / "pairs.yaml"
    path.write_text(yaml.safe_dump(pairs))
    m = er.evaluate_er(path, FIX / "config" / "people.yaml")
    assert (m["tp"], m["fp"], m["fn"], m["tn"], m["gray_zone"]) == (3, 0, 1, 1, 1)
    assert m["precision"] == 1.0 and m["recall"] == 0.75


# ------------------------------------------------------------------------- contributor score
def make_people(n_pos: int, n_neg: int) -> None:
    """Synthetic register: positives have more citations and recent papers (with noise/overlap)."""
    with session_scope() as s:
        s.add(Lab(id="lab", name="Lab", kind="frontier_lab"))
        s.add(Source(id="test:papers", adapter="test", kind="arxiv", source_class="research"))
        s.flush()
        for i in range(n_pos + n_neg):
            pos = i < n_pos
            cites = (8000 + 900 * i) if pos else (300 + 150 * i)
            if i in (1, n_pos + 3):  # overlap, so the task is not trivially separable
                cites = 2500
            s.add(
                Person(
                    id=f"p{i}",
                    name=f"Person {i}",
                    lab_id="lab",
                    tier="C",
                    meta={"revealed_preference": pos, "roles": ["research_lead"] if i % 3 == 0 else []},
                    contributor_features={
                        "cite_found": True,
                        "cite_citations": cites,
                        "cite_h_index": cites // 400,
                        "cite_source": "s2",
                        "cite_fetched_at": "2026-11-01T00:00:00",
                    },
                )
            )
        s.flush()
        for i in range(n_pos):
            s.add(
                Document(
                    source_id="test:papers",
                    natural_key=f"paper{i}",
                    url=f"https://arxiv.org/abs/{i}",
                    canonical_url=f"https://arxiv.org/abs/{i}",
                    content_hash=f"h{i}",
                    source_class="research",
                    person_id=f"p{i}",
                    published_at=contrib.utcnow(),
                    meta={"arxiv_id": f"2610.{i:05d}", "people_matched": [{"person_id": f"p{i + 1}"}]},
                )
            )
        s.flush()
        doc_id = s.scalar(select(Document.id).limit(1))
        for event_type in ("research_paper", "talent_move"):
            s.add(
                Insight(
                    document_id=doc_id,
                    event_type=event_type,
                    title="t",
                    claim="c",
                    person_ids=["p0"],
                    prompt_version="v",
                    model="m",
                )
            )


def test_contributor_scoring_logistic(db, tmp_path):
    make_people(14, 22)
    out = contrib.run_contributor_scoring(use_citations=False, metrics_path=tmp_path / "metrics.json")
    assert out["method"] == "logistic_regression" and out["n_labelled"] == 36
    assert out["loo_auc"] > 0.8 and out["precision_at_20"] >= 0.6
    assert out["baseline"]["name"] == "citations_only" and out["baseline"]["auc"] is not None
    assert json.loads((tmp_path / "metrics.json").read_text())["loo_auc"] == out["loo_auc"]
    with session_scope() as s:
        p0 = s.get(Person, "p0")
        assert 0.0 <= p0.contributor_score <= 1.0
        f = p0.contributor_features
        assert f["insight_mentions"] == 1.0  # the talent_move insight is excluded (it is the label event)
        assert f["papers_12m"] == 1.0 and f["tracked_coauthors"] == 1.0 and f["leadership_roles"] == 1.0
        assert "contrib_log_citations" in f and f["cite_citations"] == 8000  # cache survives re-scoring
        scores = dict(s.execute(select(Person.id, Person.contributor_score)).all())
    assert all(v is not None for v in scores.values())


def test_contributor_scoring_falls_back_with_few_labels(db, tmp_path):
    make_people(3, 20)
    out = contrib.run_contributor_scoring(use_citations=False, metrics_path=tmp_path / "m.json")
    assert out["method"] == "rank_average" and "only 3 positive" in out["fallback_reason"]
    with session_scope() as s:
        assert s.get(Person, "p0").contributor_features["score_method"] == "rank_average"


def test_citation_refresh_falls_back_to_openalex(db, settings):
    with session_scope() as s:
        s.add(Person(id="jeff-dean", name="Jeff Dean", tier="A", meta={}))
    s2_429 = (FIX / "s2_rate_limited.json").read_text()
    openalex = json.loads((FIX / "openalex_authors_jeff_dean.json").read_text())
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "api.semanticscholar.org":
            return httpx.Response(429, text=s2_429)
        return httpx.Response(200, json=openalex)

    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with session_scope() as s:
        people = list(s.scalars(select(Person)))
        stats = contrib.refresh_citations(people, {}, settings, client=client, sleep=sleeps.append)
    assert hosts == ["api.semanticscholar.org", "api.semanticscholar.org", "api.openalex.org"]
    assert stats["cite_rate_limited"] == 2 and stats["cite_found"] == 1 and sum(sleeps) >= 15
    with session_scope() as s:
        f = s.get(Person, "jeff-dean").contributor_features
        assert (f["cite_source"], f["cite_citations"], f["cite_h_index"]) == ("openalex", 47693, 28)
        # 3 split "Jeff Dean" profiles + "J Paul Dean" (lists "Jeff Dean" as an alternative name);
        # "Jeffry L. Dean" is no match, and the runner-up has < 20% of the top citations
        assert f["cite_candidates"] == 4 and f["cite_ambiguous"] is False
        people = list(s.scalars(select(Person)))
        assert contrib.refresh_citations(people, {}, settings, client=client)["cite_calls"] == 0  # cached


def test_pick_author_handles_order_and_namesakes():
    cands = contrib.S2.candidates(
        {
            "data": [
                {
                    "authorId": "1",
                    "name": "Yang Zhilin",
                    "citationCount": 900,
                    "hIndex": 10,
                    "paperCount": 20,
                },
                {
                    "authorId": "2",
                    "name": "Zhilin Yang",
                    "citationCount": 40000,
                    "hIndex": 40,
                    "paperCount": 80,
                },
                {
                    "authorId": "3",
                    "name": "Zhiling Yang",
                    "citationCount": 99999,
                    "hIndex": 90,
                    "paperCount": 300,
                },
            ]
        }
    )
    out = contrib.pick_author(cands, ["Yang Zhilin", "Zhilin Yang"])
    assert out["cite_author_id"] == "2" and out["cite_candidates"] == 2 and out["cite_ambiguous"] is False
    assert contrib.pick_author(cands, ["Nobody Here"]) == {"cite_found": False, "cite_candidates": 0}
