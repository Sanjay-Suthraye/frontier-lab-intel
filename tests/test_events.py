"""Event layer: near-dedup, embeddings, clustering, novelty, artifact registry.

Synthetic data only: hand-made unit vectors stand in for embeddings (the model is never loaded)."""

from __future__ import annotations

import importlib.util
import itertools
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select

from fli.analyze import artifacts, cluster, dedup, embed, novelty
from fli.db.models import (
    Alert,
    Artifact,
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Label,
    Lens,
    Score,
    Source,
    Ticker,
)
from fli.textutil import content_hash

T0 = datetime(2026, 10, 1, 12, 0)
DAY = timedelta(days=1)
DIM = 8
_seq = itertools.count()
VOCAB = [f"{a}{b}" for a in ("model", "chip", "agent", "token", "lab", "train", "infer", "scale")
         for b in ("ing", "er", "s", "ed", "ion", "al", "ive", "ly", "ity", "ward")]


def e(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def mix(a: np.ndarray, i: int, cos: float) -> np.ndarray:
    """Unit vector with cosine `cos` to unit vector `a`, tilted towards axis i (orthogonal to a)."""
    return (cos * a + np.sqrt(1 - cos**2) * e(i)).astype(np.float32)


def words(seed: int, n: int = 120) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def tweak(text: str, *positions: int) -> str:
    """Same text with the words at `positions` replaced (each change touches <= 5 shingles)."""
    tokens = text.split(" ")
    for pos in positions:
        tokens[pos] = "zz"
    return " ".join(tokens)


@pytest.fixture()
def world(db):
    with db() as s:
        s.add_all([Lab(id="openai", name="OpenAI", kind="frontier_lab"),
                   Lab(id="anthropic", name="Anthropic", kind="frontier_lab")])
        s.add_all([
            Source(id="openai:news", lab_id="openai", adapter="rss", kind="official_blog",
                   source_class="official"),
            Source(id="anthropic:news", lab_id="anthropic", adapter="rss", kind="official_blog",
                   source_class="official"),
            Source(id="press:a", adapter="rss", kind="press", source_class="press"),
            Source(id="press:b", adapter="rss", kind="press", source_class="press"),
        ])
        s.commit()
    return db


def add_doc(s, source_id: str, text: str = "body", title: str = "T", published: datetime = T0,
            status: str = "new") -> Document:
    n = next(_seq)
    src = s.get(Source, source_id)
    doc = Document(source_id=source_id, natural_key=f"k{n}", url=f"https://example.com/{n}",
                   canonical_url=f"https://example.com/{n}", title=title, published_at=published,
                   content_text=text, content_hash=content_hash(f"{title}\n{text}"), lab_id=src.lab_id,
                   source_class=src.source_class, status=status)
    s.add(doc)
    s.flush()
    return doc


def add_insight(s, doc: Document, vec: np.ndarray | None = None, *, lab_id: str | None = "doc",
                event_type: str = "model_release", title: str = "t", artifacts_: tuple = (),
                persons: tuple = (), published: datetime | None = None, verified: bool = False,
                dev: Development | None = None) -> Insight:
    ins = Insight(document_id=doc.id, lab_id=doc.lab_id if lab_id == "doc" else lab_id,
                  person_ids=list(persons),
                  event_type=event_type, title=title, claim=f"claim of {title}", artifacts=list(artifacts_),
                  is_official=doc.source_class == "official", quotes_verified=verified,
                  published_at=published or doc.published_at,
                  embedding=embed.to_bytes(vec) if vec is not None else None,
                  development_id=dev.id if dev else None, prompt_version="test", model="test")
    s.add(ins)
    s.flush()
    return ins


def add_dev(s, lab_id: str | None, t: datetime = T0) -> Development:
    dev = Development(title="d", event_type="other", lab_id=lab_id, first_seen_at=t, last_seen_at=t)
    s.add(dev)
    s.flush()
    return dev


# ------------------------------------------------------------------------------------ near-dedup
def test_minhash_roundtrip_is_portable():
    mh = dedup.minhash_of(words(1))
    blob = dedup.minhash_to_bytes(mh)
    assert len(blob) == 8 + 4 + 1 + 4 * dedup.NUM_PERM
    assert dedup.minhash_from_bytes(blob).jaccard(mh) == 1.0


def test_near_duplicate_across_sources_points_at_earliest(world):
    text = words(1)
    tweaked = tweak(text, 30) + " reported first by our desk"
    with world() as s:
        first = add_doc(s, "openai:news", text, published=T0, status="extracted")
        copy_b = add_doc(s, "press:b", tweaked, published=T0 + timedelta(hours=3))
        copy_a = add_doc(s, "press:a", text, published=T0 + timedelta(hours=2))
        other = add_doc(s, "press:a", words(2), published=T0 + timedelta(hours=1))
        s.commit()
        ids = first.id, copy_a.id, copy_b.id, other.id
    out = dedup.run_near_dedup()
    assert out["duplicates"] == 2 and out["checked"] == 3 and out["minhash_backfilled"] == 1
    with world() as s:
        first, copy_a, copy_b, other = (s.get(Document, i) for i in ids)
        assert (copy_a.status, copy_a.duplicate_of_id) == ("duplicate", first.id)
        assert (copy_b.status, copy_b.duplicate_of_id) == ("duplicate", first.id)
        assert other.status == "new" and other.minhash is not None
        assert first.status == "extracted" and first.minhash is not None
    again = dedup.run_near_dedup()  # idempotent: nothing new to mark
    assert again["duplicates"] == 0 and again["checked"] == 1


def test_new_documents_dedup_among_themselves_in_time_order(world):
    text = words(3)
    with world() as s:
        later = add_doc(s, "press:a", text, published=T0 + timedelta(days=1))
        earliest = add_doc(s, "press:b", text + " update", published=T0)
        s.commit()
        later_id, earliest_id = later.id, earliest.id
    dedup.run_near_dedup()
    with world() as s:
        assert s.get(Document, later_id).duplicate_of_id == earliest_id
        assert s.get(Document, earliest_id).status == "new"


def test_same_source_requires_identical_text(world):
    text = words(4)
    with world() as s:
        a = add_doc(s, "press:a", text, published=T0)
        near = add_doc(s, "press:a", tweak(text, 20, 90), published=T0 + timedelta(hours=1))
        same = add_doc(s, "press:a", text, published=T0 + timedelta(hours=2))
        s.commit()
        ids = a.id, near.id, same.id
    dedup.run_near_dedup()
    with world() as s:
        a, near, same = (s.get(Document, i) for i in ids)
        assert a.status == "new" and near.status == "new"
        assert (same.status, same.duplicate_of_id) == ("duplicate", a.id)


def test_short_old_failed_and_processed_documents(world):
    text, short = words(5), "Short identical teaser about a launch."
    with world() as s:
        add_doc(s, "press:a", text, published=T0 - timedelta(days=40), status="extracted")
        add_doc(s, "press:b", words(6), published=T0, status="failed")
        processed_later = add_doc(s, "openai:news", words(7), published=T0 + timedelta(days=2),
                                  status="relevant")
        new_old_copy = add_doc(s, "press:b", text, published=T0)
        new_failed_copy = add_doc(s, "press:a", words(6), published=T0 + timedelta(hours=1))
        early_copy = add_doc(s, "press:a", words(7), published=T0 + timedelta(days=1))
        short_a = add_doc(s, "press:a", short, title="x", published=T0)
        short_b = add_doc(s, "press:b", short, title="x", published=T0 + timedelta(hours=1))
        s.commit()
        ids = [d.id for d in (new_old_copy, new_failed_copy, early_copy, short_a, short_b, processed_later)]
    out = dedup.run_near_dedup()
    assert out["short"] == 2
    with world() as s:
        old_copy, failed_copy, early, sa, sb, processed = (s.get(Document, i) for i in ids)
        assert old_copy.status == "new"  # the match is 40 days older: outside the window
        assert failed_copy.status == "new"  # a failed document is not a canonical copy
        assert (early.status, early.duplicate_of_id) == ("duplicate", processed.id)
        assert processed.status == "relevant"  # processed documents are never relabelled
        assert sa.status == sb.status == "new" and sa.minhash is None


# ------------------------------------------------------------------------------------ embeddings
def test_vector_bytes_roundtrip_and_normalisation():
    m = embed.normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(m[0], [0.6, 0.8]) and np.allclose(m[1], 0.0)
    v = embed.normalize(np.arange(1, 9, dtype=np.float32))
    back = embed.from_bytes(embed.to_bytes(v))
    assert back.dtype == np.float32 and np.allclose(back, v) and back.flags.writeable


def test_run_embed_is_incremental(world, monkeypatch):
    seen: list[str] = []

    def fake_encode(texts):
        seen.extend(texts)
        return embed.normalize_rows(np.ones((len(texts), DIM)))

    monkeypatch.setattr(embed, "encode", fake_encode)
    with world() as s:
        doc = add_doc(s, "openai:news")
        add_insight(s, doc, title="GPT-6 launches.", artifacts_=("GPT-6", "Codex"))
        for i in range(4):
            add_insight(s, doc, title=f"t{i}")
        s.commit()
    assert embed.run_embed(limit=2, batch_size=1) == {"embedded": 2, "batches": 2, "model": embed.MODEL_NAME}
    assert seen[0] == "GPT-6 launches. claim of GPT-6 launches. GPT-6, Codex"
    assert embed.run_embed()["embedded"] == 3
    assert embed.run_embed()["embedded"] == 0
    with world() as s:
        assert s.scalar(select(Insight).where(Insight.embedding.is_(None))) is None


# ------------------------------------------------------------------------------------ clustering
def test_cosine_join_and_development_aggregates(world):
    with world() as s:
        p = add_insight(s, add_doc(s, "press:a", published=T0), e(0), lab_id="openai",
                        title="Press: GPT-6 soon",
                        event_type="capability_claim", artifacts_=("GPT-6",), persons=("sam-altman",))
        o = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(hours=1)), mix(e(0), 1, 0.9),
                        title="Introducing GPT-6", artifacts_=("Stargate",), persons=("greg-brockman",),
                        verified=True)  # no shared artifact: joins on cosine 0.9 alone
        q = add_insight(s, add_doc(s, "press:b", published=T0 + timedelta(hours=2)), mix(e(0), 2, 0.85),
                        lab_id=None, title="GPT-6 is here", artifacts_=("gpt6",))  # lab-less: via artifact
        s.commit()
        ids = p.id, o.id, q.id
    assert cluster.run_clustering() == {"assigned": 3, "joined": 2, "created": 1, "developments_touched": 1,
                                        "repaired": 0}
    with world() as s:
        members = [s.get(Insight, i) for i in ids]
        assert len({m.development_id for m in members}) == 1
        dev = s.get(Development, members[0].development_id)
        assert dev.title == "Introducing GPT-6"  # official beats earlier press
        assert dev.event_type == "model_release"  # official vote wins over the majority of coverage
        assert dev.lab_id == "openai" and dev.person_ids == ["greg-brockman", "sam-altman"]
        assert dev.artifacts == ["GPT-6", "Stargate"]
        assert (dev.first_seen_at, dev.last_seen_at) == (T0, T0 + timedelta(hours=2))
        assert (dev.insight_count, dev.source_count, dev.source_classes) == (3, 3, ["official", "press"])
        assert dev.has_official and dev.has_verified
        expected = embed.normalize(np.mean([e(0), mix(e(0), 1, 0.9), mix(e(0), 2, 0.85)], axis=0))
        assert np.allclose(embed.from_bytes(dev.centroid), expected, atol=1e-6)


def test_low_similarity_other_lab_and_stale_developments_split(world):
    with world() as s:
        add_insight(s, add_doc(s, "openai:news", published=T0), e(0))
        add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(hours=1)), mix(e(0), 1, 0.5))
        add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(hours=2)), e(0))
        add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=9)), e(0))
        s.commit()
    assert cluster.run_clustering()["created"] == 4


def test_structural_join_needs_shared_artifact_and_compatible_family(world):
    with world() as s:
        a = add_insight(s, add_doc(s, "openai:news", published=T0), e(0), artifacts_=("GPT-6",))
        b = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=1)), e(1),
                        event_type="pricing_change", artifacts_=("gpt6",))
        c = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=1)), e(2),
                        event_type="talent_move", artifacts_=("GPT-6",))
        d = add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(days=2)), e(3), lab_id=None,
                        event_type="product_launch", artifacts_=("OpenAI GPT-6",))
        loose = add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(days=2)), e(0), lab_id=None)
        s.commit()
        ids = a.id, b.id, c.id, d.id, loose.id
    cluster.run_clustering()
    with world() as s:
        a, b, c, d, loose = (s.get(Insight, i) for i in ids)
        assert a.development_id == b.development_id == d.development_id  # lab-less press joins via artifact
        assert c.development_id != a.development_id  # talent move is not part of the launch
        assert loose.development_id not in {a.development_id, c.development_id}  # no lab, no overlap
        assert s.get(Development, a.development_id).lab_id == "openai"


def test_sibling_releases_need_high_cosine(world):
    """bge-small puts same-week sibling launches at ~0.88: different subjects must not merge on that."""
    with world() as s:
        opus = add_insight(s, add_doc(s, "anthropic:news", published=T0), e(0),
                           artifacts_=("Claude Opus 5.5",))
        haiku = add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(hours=1)),
                            mix(e(0), 1, 0.88), artifacts_=("Claude Haiku 5.5",))
        unnamed = add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(hours=2)),
                              mix(e(0), 2, 0.88))
        near_copy = add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(hours=3)), e(0),
                                artifacts_=("Claude Sonnet 5.5",))
        s.commit()
        ids = opus.id, haiku.id, unnamed.id, near_copy.id
    cluster.run_clustering()
    with world() as s:
        opus, haiku, unnamed, near_copy = (s.get(Insight, i) for i in ids)
        assert haiku.development_id != opus.development_id  # conflict: 0.88 < 0.92
        assert unnamed.development_id == opus.development_id  # no subject named: plain 0.78 bar
        assert near_copy.development_id == opus.development_id  # cosine 0.97 clears the conflict bar


def test_mixed_development_does_not_snowball(world):
    """Two different same-lab events (pairwise 0.74) joined by an artifact: their renormalised centroid
    is 0.79 from any further same-lab insight, but the mean similarity to members stays 0.74."""
    a, b = np.sqrt(0.74), np.sqrt(0.26)
    with world() as s:
        x = add_insight(s, add_doc(s, "openai:news", published=T0), a * e(0) + b * e(1),
                        artifacts_=("Sora 3",))
        y = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(hours=1)), a * e(0) + b * e(2),
                        event_type="product_launch", artifacts_=("Sora 3",))
        z = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(hours=2)), a * e(0) + b * e(3))
        s.commit()
        ids = x.id, y.id, z.id
    cluster.run_clustering()
    with world() as s:
        x, y, z = (s.get(Insight, i) for i in ids)
        centroid = embed.from_bytes(s.get(Development, x.development_id).centroid)
        assert x.development_id == y.development_id
        assert float(centroid @ embed.from_bytes(z.embedding)) > cluster.JOIN_COSINE  # what the centroid says
        assert z.development_id != x.development_id  # what the members say


def test_versioned_family_names_link_but_variants_do_not(world):
    with world() as s:
        a = add_insight(s, add_doc(s, "openai:news", published=T0), e(0), artifacts_=("GPT-6",))
        b = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=1)), e(1),
                        event_type="product_launch", artifacts_=("GPT-6 mini",))
        c = add_insight(s, add_doc(s, "openai:news", published=T0), e(2), artifacts_=("o9 mini",))
        d = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=1)), e(3),
                        artifacts_=("o9 nano",))
        bench_a = add_insight(s, add_doc(s, "openai:news", published=T0), e(4), event_type="capability_claim",
                              artifacts_=("SWE-bench Verified",))
        bench_b = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(hours=1)), e(5),
                              event_type="capability_claim", artifacts_=("SWE-bench Verified",))
        s.commit()
        ids = a.id, b.id, c.id, d.id, bench_a.id, bench_b.id
    cluster.run_clustering()
    with world() as s:
        a, b, c, d, bench_a, bench_b = (s.get(Insight, i) for i in ids)
        assert a.development_id == b.development_id  # "gpt 6" is the versioned family of "gpt 6 mini"
        assert c.development_id != d.development_id  # two variants of a family are different subjects
        assert bench_a.development_id != bench_b.development_id  # a shared benchmark is not a link


def test_umbrella_brands_are_subjects_but_not_link_keys():
    assert artifacts.link_keys(["ChatGPT"]) == (set(), {"chatgpt"})
    assert artifacts.link_keys(["Gemini 4 Carbon", "Gemini"]) == ({"gemini 4 carbon"}, {"gemini 4 carbon", "gemini"})
    for name in ("Amazon Bedrock", "Google Vertex AI", "OpenAI Codex", "Claude Code", "Meta AI", "xAI Grok",
                 "Llama", "Microsoft Copilot", "Claude"):
        assert artifacts.link_keys([name])[0] == set(), name
    for name in ("ChatGPT Atlas", "Claude Opus 5.5", "Llama 5", "Grok 5", "Codex CLI"):
        assert artifacts.link_keys([name])[0] == {artifacts.normalize_artifact(name)}, name


def test_umbrella_brands_do_not_merge_unrelated_items(world):
    """Dev 416 (OpenAI-Oracle apps in ChatGPT) absorbed a ChatGPT violence-prevention post on the
    shared artifact 'ChatGPT' alone; a paper that used 'Gemini' joined an unrelated Gemini launch."""
    with world() as s:
        s.add(Lab(id="google-deepmind", name="Google DeepMind", kind="frontier_lab"))
        s.add(Source(id="gdm:blog", lab_id="google-deepmind", adapter="rss", kind="official_blog",
                     source_class="official"))
        s.flush()
        apps = add_insight(s, add_doc(s, "openai:news", published=T0), e(0), event_type="partnership",
                           artifacts_=("ChatGPT",))
        safety = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=1)), e(1),
                             event_type="safety_policy", artifacts_=("ChatGPT",))
        safety_echo = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=1, hours=2)),
                                  mix(e(1), 6, 0.9), event_type="safety_policy", artifacts_=("ChatGPT",))
        business = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=2)), e(4),
                               event_type="product_launch", artifacts_=("ChatGPT Business",))
        ads = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=2, hours=1)),
                          mix(e(4), 5, 0.88), event_type="product_launch", artifacts_=("ChatGPT",))
        carbon = add_insight(s, add_doc(s, "gdm:blog", published=T0), e(2), artifacts_=("Gemini 4 Carbon", "Gemini"))
        carbon_press = add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(days=1)), e(3), lab_id=None,
                                   event_type="product_launch", artifacts_=("Google Gemini 4 Carbon",))
        paper = add_insight(s, add_doc(s, "press:b", published=T0 + timedelta(days=1)), e(7), lab_id=None,
                            event_type="research_paper", artifacts_=("Gemini",))
        s.commit()
        ids = apps.id, safety.id, safety_echo.id, business.id, ads.id, carbon.id, carbon_press.id, paper.id
    cluster.run_clustering()
    with world() as s:
        apps, safety, safety_echo, business, ads, carbon, carbon_press, paper = (s.get(Insight, i) for i in ids)
        assert safety.development_id != apps.development_id  # same lab and family, only 'ChatGPT' shared
        assert safety_echo.development_id == safety.development_id  # cosine still joins umbrella-only items
        assert ads.development_id != business.development_id  # still a subject conflict: 0.88 < 0.92
        assert carbon_press.development_id == carbon.development_id  # a versioned name still links
        assert paper.development_id != carbon.development_id  # 'Gemini' alone is no link


@pytest.fixture()
def launch_world(world):
    """Adds Reflection AI (official Hugging Face org), a lab-less market listing source and arXiv."""
    with world() as s:
        s.add(Lab(id="reflection-ai", name="Reflection AI", kind="alumni_startup"))
        s.add_all([
            Source(id="reflection-ai:hf", lab_id="reflection-ai", adapter="hf", kind="hf_org", source_class="official"),
            Source(id="openrouter:models", adapter="openrouter", kind="model_index", source_class="market_data"),
            Source(id="arxiv:cs", adapter="arxiv", kind="arxiv", source_class="research"),
        ])
        s.commit()
    return world


def test_versioned_keys_name_specific_artifacts():
    names = ["Gemini 4 Carbon", "Gemini", "Beam", "Beam-1.0", "DeepSeek-V4.1-Flash", "Qwen3.8-27B", "H100",
             "TPU v7", "SWE-bench Verified", "ChatGPT", "Claude Code", "Stargate"]
    assert cluster.versioned_keys(names) == {"gemini 4 carbon", "beam 1.0", "deepseek v 4.1 flash", "qwen 3.8 27b"}
    sig = cluster.Signals.of(None, ["Beam-1.0-501B-A23B", "Beam"], [], "open_weight_release")
    assert sig.versioned == {"beam 1.0 501b a 23b"} and sig.linkable == {"beam 1.0 501b a 23b", "beam"}


def test_launch_seen_again_within_21_days_joins_on_a_versioned_artifact(launch_world):
    """Reflection's Beam: Hugging Face weights, the OpenRouter listing 9 days later and the paper 10 days
    after that were 3 developments (and alerts). Templates differ, so cosine cannot join them."""
    with launch_world() as s:
        hf = add_insight(s, add_doc(s, "reflection-ai:hf", published=T0), e(0), event_type="open_weight_release",
                         title="Reflection releases Beam-1.0", artifacts_=("Beam-1.0-501B-A23B",), verified=True)
        listing = add_insight(s, add_doc(s, "openrouter:models", published=T0 + 9 * DAY), e(1),
                              lab_id="reflection-ai", event_type="model_release", artifacts_=("Beam 1.0 501B A23B",))
        paper = add_insight(s, add_doc(s, "arxiv:cs", published=T0 + 19 * DAY), e(2), lab_id="reflection-ai",
                            event_type="research_paper", artifacts_=("Beam-1.0-501B-A23B", "Beam"))
        s.commit()
        ids = hf.id, listing.id, paper.id
    assert cluster.run_clustering() == {"assigned": 3, "joined": 2, "created": 1, "developments_touched": 1,
                                        "repaired": 0}
    with launch_world() as s:
        dev = s.get(Development, s.get(Insight, ids[0]).development_id)
        assert {s.get(Insight, i).development_id for i in ids} == {dev.id}
        assert (dev.title, dev.event_type, dev.insight_count) == ("Reflection releases Beam-1.0", "open_weight_release", 3)
        assert dev.source_classes == ["market_data", "official", "research"]
        assert (dev.first_seen_at, dev.last_seen_at) == (T0, T0 + 19 * DAY)


def test_late_join_window_is_anchored_at_the_launch(launch_world):
    """21 days from the development's start, not from its last member: a model's later news cannot chain
    onto its launch for months (GPT-6 Astra items kept arriving for 6 weeks)."""
    with launch_world() as s:
        on_time = add_insight(s, add_doc(s, "reflection-ai:hf", published=T0), e(0), event_type="open_weight_release",
                              artifacts_=("Timestral-209M",))
        paper_20 = add_insight(s, add_doc(s, "arxiv:cs", published=T0 + 20 * DAY), e(1), lab_id="reflection-ai",
                               event_type="research_paper", artifacts_=("Timestral-209M",))
        launch = add_insight(s, add_doc(s, "reflection-ai:hf", published=T0), e(2), event_type="open_weight_release",
                             artifacts_=("Pixtral-3-12B",))
        listing = add_insight(s, add_doc(s, "openrouter:models", published=T0 + 5 * DAY), e(4),
                              lab_id="reflection-ai", event_type="model_release", artifacts_=("Pixtral-3-12B",))
        paper_23 = add_insight(s, add_doc(s, "arxiv:cs", published=T0 + 23 * DAY), e(3), lab_id="reflection-ai",
                               event_type="research_paper", artifacts_=("Pixtral-3-12B",))
        s.commit()
        ids = on_time.id, paper_20.id, launch.id, listing.id, paper_23.id
    cluster.run_clustering()
    with launch_world() as s:
        on_time, paper_20, launch, listing, paper_23 = (s.get(Insight, i) for i in ids)
        assert paper_20.development_id == on_time.development_id
        assert listing.development_id == launch.development_id  # within 7 days: the plain structural join
        # 18 days after the last member but 23 after the launch started: a development of its own
        assert paper_23.development_id not in {launch.development_id, on_time.development_id}


def test_late_join_guards_keep_other_events_apart(launch_world):
    """Each case names the launch's versioned artifact 10 days in, yet is a development of its own; each
    is a wrong merge a bare 'shared versioned name within 21 days' rule made on the 2026-11 corpus."""
    cases = {}
    with launch_world() as s:
        def launch(k: int, *names: str, event_type: str = "model_release", vec=None) -> Insight:
            return add_insight(s, add_doc(s, "openai:news", published=T0 + 60 * k * DAY), e(0) if vec is None else vec,
                               event_type=event_type, artifacts_=names)

        def later(k: int, *names: str, event_type: str = "model_release", source: str = "openai:news",
                  lab_id: str | None = "doc", vec=None) -> Insight:
            return add_insight(s, add_doc(s, source, published=T0 + (60 * k + 10) * DAY), e(1) if vec is None else vec,
                               event_type=event_type, artifacts_=names, lab_id=lab_id)

        cases["control: the weights of the same model"] = (
            launch(0, "GPT-6 Astra"), later(0, "GPT-6 Astra", event_type="open_weight_release"), True)
        cases["a price cut is its own development"] = (
            launch(1, "GPT-6 Sol"), later(1, "GPT-6 Sol", event_type="pricing_change"), False)
        cases["a feature post is its own development"] = (
            launch(2, "GPT-6 Luna"), later(2, "GPT-6 Luna", event_type="product_launch"), False)
        cases["the next model, naming the previous one"] = (
            launch(3, "GPT-6 Argon"), later(3, "GPT-6 Carbon", "GPT-6 Argon"), False)
        cases["no lab: a paper that merely used the model"] = (
            launch(4, "GPT-6 Terra"), later(4, "GPT-6 Terra", event_type="research_paper", source="press:a",
                                            lab_id=None), False)
        first = launch(5, "GPT-7 Luna")  # its same-day listing names Nova in passing
        add_insight(s, add_doc(s, "press:a", published=T0 + 300 * DAY + timedelta(hours=1)), e(0), lab_id="openai",
                    artifacts_=("GPT-7 Luna", "GPT-7 Nova"))
        cases["a name the launch only mentioned"] = (first, later(5, "GPT-7 Nova"), False)
        cases["no release in the development"] = (
            launch(6, "DeepJIT 2", event_type="technique"), later(6, "DeepJIT 2", event_type="open_weight_release"),
            False)
        cases["cosine alone keeps the 7-day window"] = (
            launch(7, vec=e(0)), later(7, vec=mix(e(0), 1, 0.95)), False)
        s.commit()
        ids = {name: (a.id, b.id, joins) for name, (a, b, joins) in cases.items()}
    cluster.run_clustering()
    with launch_world() as s:
        for name, (a, b, joins) in ids.items():
            same = s.get(Insight, a).development_id == s.get(Insight, b).development_id
            assert same is joins, name


def test_drafter_naming_the_model_family_joins_it(launch_world):
    """"Beam-1.0-DFlash2, Beam 1.0" names the family of the Beam-1.0 501B model released 6 days earlier.
    guess_kind files "Beam 1.0" under 'other' (it would call "Gemini 4" a model), so the drafter used to
    hit a subject conflict; a shared versioned name now counts as a shared subject. A sibling variant
    that does not name the family still conflicts."""
    with launch_world() as s:
        model = add_insight(s, add_doc(s, "reflection-ai:hf", published=T0), e(0), event_type="open_weight_release",
                            artifacts_=("Beam-1.0-501B-A23B",))
        sibling = add_insight(s, add_doc(s, "reflection-ai:hf", published=T0 + 5 * DAY), e(2),
                              event_type="open_weight_release", artifacts_=("Beam-1.0-Mini",))
        drafter = add_insight(s, add_doc(s, "reflection-ai:hf", published=T0 + 6 * DAY), mix(e(0), 1, 0.5),
                              event_type="open_weight_release", artifacts_=("Beam-1.0-DFlash2", "Beam 1.0"))
        s.commit()
        ids = model.id, sibling.id, drafter.id
    cluster.run_clustering()
    with launch_world() as s:
        model, sibling, drafter = (s.get(Insight, i) for i in ids)
        assert drafter.development_id == model.development_id
        assert sibling.development_id != model.development_id


def test_development_event_type_prefers_official_releases():
    rule = cluster.development_event_type
    # Claude Fable/Mythos 5.1: the release post, two same-day feature posts, an OpenRouter listing
    assert rule([("model_release", True), ("product_launch", True), ("model_release", False),
                 ("product_launch", True)]) == "model_release"
    assert rule([("technique", True), ("open_weight_release", True), ("technique", True)]) == "open_weight_release"
    assert rule([("open_weight_release", True), ("model_release", True), ("model_release", True)]) == "model_release"
    assert rule([("model_release", True), ("open_weight_release", True)]) == "model_release"  # tie -> earliest
    # no official release: official members first; a listing typed model_release does not win
    assert rule([("pricing_change", True), ("model_release", False), ("model_release", False)]) == "pricing_change"
    assert rule([("other", True), ("funding_ipo", False), ("funding_ipo", False)]) == "funding_ipo"
    assert rule([("product_launch", False), ("model_release", False), ("model_release", False)]) == "model_release"
    assert rule([("other", True), (None, False)]) == "other" and rule([]) == "other"


def test_recompute_types_a_launch_by_its_official_release(world):
    with world() as s:
        dev = add_dev(s, "anthropic")
        add_insight(s, add_doc(s, "anthropic:news", published=T0), e(0), title="Anthropic launches Claude Fable 5.1",
                    event_type="model_release", verified=True, dev=dev)
        for h in (1, 2):
            add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(hours=h)), e(h),
                        event_type="product_launch", dev=dev)
        add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(hours=3)), e(3), lab_id="anthropic",
                    event_type="product_launch", dev=dev)
        assert cluster.recompute_development(s, dev.id).event_type == "model_release"


def test_people_events_join_across_labs(world):
    with world() as s:
        a = add_insight(s, add_doc(s, "openai:news", published=T0), e(0), event_type="talent_move",
                        persons=("jane-doe",))
        b = add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(days=1)), e(1),
                        event_type="talent_move", persons=("jane-doe",))
        c = add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(days=1)), e(2),
                        event_type="research_paper", persons=("jane-doe",))
        s.commit()
        ids = a.id, b.id, c.id
    cluster.run_clustering()
    with world() as s:
        a, b, c = (s.get(Insight, i) for i in ids)
        assert a.development_id == b.development_id != c.development_id
        assert s.get(Development, a.development_id).lab_id == "openai"  # tie -> earliest official


def test_clustering_is_idempotent_and_recompute_is_stable(world):
    with world() as s:
        for h in range(3):
            doc = add_doc(s, "openai:news", published=T0 + timedelta(hours=h))
            add_insight(s, doc, mix(e(0), 1 + h, 0.95))
        s.commit()
    assert cluster.run_clustering()["assigned"] == 3
    assert cluster.run_clustering()["assigned"] == 0
    cols = ("title", "event_type", "lab_id", "person_ids", "artifacts", "first_seen_at", "last_seen_at",
            "insight_count", "source_count", "source_classes", "has_official", "has_verified", "centroid")
    with world() as s:
        dev = s.scalars(select(Development)).one()
        before = {c: getattr(dev, c) for c in cols}
        cluster.recompute_development(s, dev.id)
        assert {c: getattr(dev, c) for c in cols} == before
        assert cluster.recompute_development(s, 999) is None
        first = s.scalars(select(Insight).order_by(Insight.id)).first()
        s.delete(first)  # e.g. re-extraction replaced an insight
        s.commit()
    assert cluster.run_clustering()["repaired"] == 1
    with world() as s:
        assert s.scalars(select(Development)).one().insight_count == 2


def test_event_vote_and_family_rules():
    assert cluster._vote(["capability_claim", "model_release", "model_release", "other"], ignore="other") == \
        "model_release"
    assert cluster._vote(["b", "a"]) == "b"  # tie -> earliest
    assert cluster._vote(["other"], ignore="other") == "other"
    assert cluster.compatible_events("model_release", "pricing_change")
    assert cluster.compatible_events("other", "talent_move")
    assert not cluster.compatible_events("model_release", "talent_move")


# ------------------------------------------------------------------------------------ novelty
def test_novelty_entity_global_and_windows(world):
    with world() as s:
        va, vb = e(0), mix(e(0), 1, 0.6)
        vc = (0.9 * vb + np.sqrt(1 - 0.81) * e(3)).astype(np.float32)
        a = add_insight(s, add_doc(s, "openai:news", published=T0), va, dev=add_dev(s, "openai"))
        b = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=10)), vb,
                        dev=add_dev(s, "openai"))
        c = add_insight(s, add_doc(s, "anthropic:news", published=T0 + timedelta(days=11)), vc,
                        dev=add_dev(s, "anthropic"))
        d = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=101)), va,
                        dev=add_dev(s, "openai"))
        s.commit()
        ids = a.id, b.id, c.id, d.id
    assert novelty.run_novelty()["scored"] == 4
    with world() as s:
        a, b, c, d = (s.get(Insight, i) for i in ids)
        assert (a.novelty_entity, a.novelty_global, a.nearest_prior_insight_id) == (1.0, 1.0, None)
        assert (b.novelty_entity, b.novelty_global, b.nearest_prior_insight_id) == (0.4, 0.4, a.id)
        assert (c.novelty_entity, c.novelty_global, c.nearest_prior_insight_id) == (1.0, 0.1, b.id)
        assert (d.novelty_entity, d.novelty_global, d.nearest_prior_insight_id) == (1.0, 1.0, None)
    assert novelty.run_novelty()["scored"] == 0  # incremental


def test_novelty_excludes_own_development_and_breaks_time_ties_by_id(world):
    with world() as s:
        dev = add_dev(s, "openai")
        first = add_insight(s, add_doc(s, "openai:news", published=T0), e(0), dev=dev)
        corroboration = add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(hours=5)), e(0),
                                    lab_id="openai", dev=dev)
        same_time = add_insight(s, add_doc(s, "openai:news", published=T0), e(0), dev=add_dev(s, "openai"))
        s.commit()
        ids = first.id, corroboration.id, same_time.id
    novelty.run_novelty()
    with world() as s:
        first, corroboration, same_time = (s.get(Insight, i) for i in ids)
        assert first.novelty_entity == 1.0  # same timestamp, smaller id: nothing is prior
        assert corroboration.novelty_entity == 0.0 and corroboration.nearest_prior_insight_id == same_time.id
        assert same_time.novelty_entity == 0.0 and same_time.nearest_prior_insight_id == first.id


def test_novelty_uses_person_when_no_lab(world):
    with world() as s:
        g = add_insight(s, add_doc(s, "press:a", published=T0), e(0), lab_id=None, persons=("p1",),
                        dev=add_dev(s, None))
        h = add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(days=1)), mix(e(0), 1, 0.7),
                        lab_id=None, persons=("p1", "p2"), dev=add_dev(s, None))
        i = add_insight(s, add_doc(s, "press:b", published=T0 + timedelta(days=2)), e(0), lab_id=None,
                        persons=("p3",), dev=add_dev(s, None))
        s.commit()
        ids = g.id, h.id, i.id
    novelty.run_novelty()
    with world() as s:
        g, h, i = (s.get(Insight, x) for x in ids)
        assert (h.novelty_entity, h.nearest_prior_insight_id) == (0.3, g.id)
        assert (i.novelty_entity, i.novelty_global, i.nearest_prior_insight_id) == (1.0, 0.0, g.id)


# ------------------------------------------------------------------------------------ artifacts
@pytest.mark.parametrize("raw, expected", [
    ("OpenAI's GPT-5.1", "gpt 5.1"), ("GPT5.1", "gpt 5.1"), ("gpt-5.1.", "gpt 5.1"),
    ("claude-opus-4-5", "claude opus 4.5"), ("Anthropic Claude Opus 4.5", "claude opus 4.5"),
    ("deepseek-ai/DeepSeek-V3.2", "deepseek v 3.2"), ("https://huggingface.co/deepseek-ai/DeepSeek-V3.2",
                                                        "deepseek v 3.2"),
    ("Google DeepMind Gemini 3 Pro", "gemini 3 pro"), ("Llama-3-1-8B", "llama 3.1 8b"),
    ("gemma-3-27b", "gemma 3 27b"), ("the Claude 4 family", "claude 4"), ("OpenAI", "openai"), ("  ", ""),
    ("NVIDIA A100", "a 100"),
])
def test_normalize_artifact(raw, expected):
    assert artifacts.normalize_artifact(raw) == expected


@pytest.mark.parametrize("raw, kind", [
    ("Claude Opus 4.5", "model"), ("claude-opus-4-5", "api_model_id"),
    ("claude-3-5-sonnet-20241022", "api_model_id"), ("gpt-oss-120b", "model"), ("TPU v7 Ironwood", "chip"),
    ("MI450", "chip"), ("SWE-bench Verified", "dataset"),
    ("github.com/anthropics/claude-code", "repo"), ("Claude Code", "product"), ("ChatGPT Atlas", "product"),
    ("Stargate", "other"),
])
def test_guess_kind(raw, kind):
    assert artifacts.guess_kind(raw) == kind


def test_artifact_registry_first_seen_by_document(world):
    with world() as s:
        doc1 = add_doc(s, "openai:news", published=T0)
        i1 = add_insight(s, doc1, e(0), artifacts_=("GPT-6", "Codex CLI"), dev=add_dev(s, "openai"))
        i2 = add_insight(s, doc1, e(1), artifacts_=("gpt6",), dev=add_dev(s, "openai"))
        i3 = add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(days=1)), e(0), lab_id="openai",
                         artifacts_=("GPT-6",), dev=add_dev(s, "openai"))
        s.commit()
        ids, doc1_id = (i1.id, i2.id, i3.id), doc1.id
    assert novelty.run_novelty()["artifacts_new"] == 2
    assert novelty.run_novelty(recompute=True)["artifacts_new"] == 0  # idempotent
    with world() as s:
        i1, i2, i3 = (s.get(Insight, i) for i in ids)
        assert (i1.first_seen_artifacts, i2.first_seen_artifacts, i3.first_seen_artifacts) == \
            (["GPT-6", "Codex CLI"], ["gpt6"], [])
        rows = {a.name: a for a in s.scalars(select(Artifact))}
        assert set(rows) == {"gpt 6", "codex cli"} and rows["codex cli"].kind == "product"
        assert (rows["gpt 6"].first_document_id, rows["gpt 6"].first_seen_at) == (doc1_id, T0)
        # backfill: an earlier document turns up later and takes over the first sighting
        i0 = add_insight(s, add_doc(s, "press:b", published=T0 - timedelta(days=1)), e(2), lab_id="openai",
                         artifacts_=("GPT 6",), dev=add_dev(s, "openai"))
        s.commit()
        i0_id = i0.id
    assert novelty.run_novelty()["artifacts_moved"] == 1
    with world() as s:
        assert s.get(Insight, i0_id).first_seen_artifacts == ["GPT 6"]
        assert s.get(Insight, ids[0]).first_seen_artifacts == ["Codex CLI"]
        assert s.get(Insight, ids[1]).first_seen_artifacts == []
    rebuild = artifacts.run_artifacts()  # full chronological rebuild agrees with the incremental path
    assert (rebuild["insights"], rebuild["artifacts_new"], rebuild["artifacts_moved"]) == (4, 0, 0)


def test_stages_run_through_the_pipeline(world, monkeypatch):
    from fli.pipeline import run_stage

    monkeypatch.setattr(embed, "encode", lambda texts: embed.normalize_rows(np.ones((len(texts), DIM))))
    with world() as s:
        doc = add_doc(s, "openai:news", text=words(9))
        add_insight(s, doc, title="a")
        s.commit()
    assert run_stage("near_dedup")["checked"] == 1
    assert run_stage("embed")["embedded"] == 1
    assert run_stage("cluster")["created"] == 1
    assert run_stage("novelty")["scored"] == 1


# ------------------------------------------------------------------------------------ repair script
def _split_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fixes" / "split_insights.py"
    spec = importlib.util.spec_from_file_location("split_insights", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_split_insights_detaches_wrong_merges_idempotently(world, capsys):
    split = _split_script()
    with world() as s:
        dev = add_dev(s, "openai")
        oracle = add_insight(s, add_doc(s, "openai:news", published=T0), e(0), event_type="partnership",
                             title="OpenAI and Oracle bring apps into ChatGPT", artifacts_=("ChatGPT",), verified=True,
                             dev=dev)
        servicenow = add_insight(s, add_doc(s, "openai:news", published=T0 + timedelta(days=1)), e(1),
                                 event_type="partnership", title="OpenAI and ServiceNow expand partnership",
                                 artifacts_=("Codex",), dev=dev)
        safety = add_insight(s, add_doc(s, "press:a", published=T0 + timedelta(days=3)), e(2), lab_id="openai",
                             event_type="safety_policy", title="ChatGPT violence prevention safeguards",
                             artifacts_=("ChatGPT", "Safety Hub"), dev=dev)
        cluster.recompute_development(s, dev.id)
        alone = add_insight(s, add_doc(s, "anthropic:news"), e(3), dev=add_dev(s, "anthropic"))
        cluster.recompute_development(s, alone.development_id)
        loose = add_insight(s, add_doc(s, "anthropic:news"), e(4))
        s.commit()
        ids = dev.id, oracle.id, servicenow.id, safety.id, alone.id, loose.id
    dev_id, oracle_id, servicenow_id, safety_id, alone_id, loose_id = ids

    dry = split.split_insights([safety_id], dry_run=True)
    assert [o.action for o in dry] == ["would_split"]
    with world() as s:
        assert s.get(Insight, safety_id).development_id == dev_id  # a dry run writes nothing

    out = split.split_insights([safety_id, oracle_id, alone_id, loose_id, 999_999, safety_id])
    assert [o.action for o in out] == ["split", "split", "already_alone", "unclustered", "not_found"]
    with world() as s:
        old = s.get(Development, dev_id)
        assert (old.insight_count, old.title, old.artifacts) == (1, "OpenAI and ServiceNow expand partnership", ["Codex"])
        assert old.first_seen_at == old.last_seen_at == T0 + timedelta(days=1)
        new_safety = s.get(Development, s.get(Insight, safety_id).development_id)
        assert (new_safety.id, new_safety.insight_count, new_safety.event_type, new_safety.lab_id) == (
            out[0].new_development_id, 1, "safety_policy", "openai")
        assert (new_safety.title, new_safety.artifacts) == ("ChatGPT violence prevention safeguards",
                                                            ["ChatGPT", "Safety Hub"])
        assert new_safety.first_seen_at == T0 + timedelta(days=3) and new_safety.source_classes == ["press"]
        assert s.get(Development, s.get(Insight, oracle_id).development_id).has_verified
        n_devs = s.scalar(select(func.count()).select_from(Development))

    assert split.main([str(safety_id), str(oracle_id)]) == 0  # second run: nothing left to do
    printed = capsys.readouterr().out
    assert "already the only member" in printed and "split 0," in printed
    with world() as s:
        assert s.scalar(select(func.count()).select_from(Development)) == n_devs
        assert s.get(Insight, servicenow_id).development_id == dev_id
    assert split.main(["999999"]) == 1



def _merge_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fixes" / "merge_developments.py"
    spec = importlib.util.spec_from_file_location("merge_developments", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _old_development(s, *members) -> Development:
    """A development as the 7-day clusterer left it: these insights, aggregates recomputed."""
    dev = add_dev(s, None)
    for ins in members:
        ins.development_id = dev.id
    s.flush()
    return cluster.recompute_development(s, dev.id)


def _derived(s, dev: Development, audience: str, n: int) -> None:
    """Score, lens, exposure link and alert rows hanging off a development."""
    s.add_all([
        Score(development_id=dev.id, audience=audience, p_material=0.9, model_version="v1"),
        Lens(development_id=dev.id, audience=audience, headline=f"lens {n}", so_what="s", what_to_do="w",
             model="test", prompt_version="test"),
        ExposureLink(development_id=dev.id, ticker="NVDA"),
        Alert(development_id=dev.id, audience=audience, severity="P1", title=f"alert {n}", body_md="b",
              dedup_key=f"{audience}:{dev.id}"),
    ])


def _label(s, dev: Development, audience: str, annotator: str, value: int) -> None:
    s.add(Label(development_id=dev.id, audience=audience, annotator=annotator, label=value, rubric_version="r1"))


def test_merge_developments_joins_a_launch_split_by_the_old_window(launch_world, capsys):
    """Beam as the 7-day clusterer left it: the Hugging Face weights (A), the drafter that names the
    Beam 1.0 family 6 days later (B), the OpenRouter listing 17 days in (C). Not merged: a price cut
    in the late window (E), another lab naming the same model (X)."""
    merge = _merge_script()
    hf = lambda days: add_doc(s, "reflection-ai:hf", published=T0 + days * DAY)  # noqa: E731
    with launch_world() as s:
        s.add(Ticker(symbol="NVDA", company="NVIDIA"))
        a = _old_development(s, add_insight(s, hf(0), e(0), event_type="open_weight_release", verified=True,
                                            title="Reflection releases Beam-1.0", artifacts_=("Beam-1.0-501B-A23B",)))
        b = _old_development(s, add_insight(s, hf(6), mix(e(0), 1, 0.5), event_type="open_weight_release",
                                            artifacts_=("Beam-1.0-DFlash2", "Beam 1.0")),
                             add_insight(s, hf(8), e(1), event_type="open_weight_release", artifacts_=("Beam",)))
        price = _old_development(s, add_insight(s, hf(16.5), e(5), event_type="pricing_change",
                                                artifacts_=("Beam-1.0-501B-A23B",)))
        c = _old_development(s, add_insight(s, add_doc(s, "openrouter:models", published=T0 + 17 * DAY),
                                            mix(e(0), 2, 0.6), lab_id="reflection-ai", event_type="model_release",
                                            artifacts_=("Beam 1.0 501B A23B",)))
        other_lab = _old_development(s, add_insight(s, add_doc(s, "anthropic:news", published=T0 + DAY), e(6),
                                                    event_type="open_weight_release",
                                                    artifacts_=("Beam-1.0-501B-A23B",)))
        fable = _old_development(s, *(add_insight(s, add_doc(s, "anthropic:news", published=T0 - 30 * DAY + h * DAY),
                                                  e(h), event_type=t)
                                      for h, t in enumerate(("model_release", "product_launch", "product_launch"))))
        fable.event_type = "product_launch"  # what the majority vote stored
        for n, (dev, aud) in enumerate(((a, "ai_team"), (b, "ai_team"), (c, "investment"))):
            _derived(s, dev, aud, n)
        _label(s, a, "investment", "human:ana", 1)
        _label(s, a, "ai_team", "llm:judge", 2)
        _label(s, b, "investment", "human:ana", 2)  # the survivor has one: it stays on b
        _label(s, b, "ai_team", "human:ana", 2)  # moves
        _label(s, c, "ai_team", "human:ana", 0)  # b's moved first: it stays on c
        _label(s, c, "investment", "llm:judge", 1)  # moves
        s.commit()
        a_id, b_id, c_id, price_id, other_id, fable_id = a.id, b.id, c.id, price.id, other_lab.id, fable.id
        a_ins = s.scalars(select(Insight.id).where(Insight.development_id.in_([a_id, b_id, c_id]))).all()

    assert merge.main(["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert f"survivor {a_id} " in printed and f"absorb {b_id} " in printed and f"absorb {c_id} " in printed
    assert "within 7 days" in printed and "late launch join" in printed and "shared beam 1.0" in printed
    assert f"retype {fable_id} " in printed and "product_launch -> model_release" in printed
    assert f"would merge 2 developments into 1 (survivors: {a_id}); would re-type 1" in printed
    with launch_world() as s:  # a dry run writes nothing
        assert s.get(Development, b_id).insight_count == 2 and s.get(Development, b_id).status == "open"
        assert s.scalar(select(func.count()).select_from(Alert)) == 3
        assert s.get(Development, fable_id).event_type == "product_launch"

    reviewed, _, _ = merge.merge_developments(dry_run=True, exclude=[c_id])  # a reviewer keeps C apart
    assert [(g.survivor, g.absorbed) for g in reviewed] == [(a_id, [b_id])]

    groups, n_retyped, _ = merge.merge_developments()
    assert [(g.survivor, g.absorbed) for g in groups] == [(a_id, [b_id, c_id])] and n_retyped == 1
    with launch_world() as s:
        assert {s.get(Insight, i).development_id for i in a_ins} == {a_id}
        survivor = s.get(Development, a_id)
        assert (survivor.insight_count, survivor.title, survivor.event_type, survivor.status) == (
            4, "Reflection releases Beam-1.0", "open_weight_release", "open")
        assert survivor.source_classes == ["market_data", "official"] and survivor.last_seen_at == T0 + 17 * DAY
        for dev_id in (b_id, c_id):
            dev = s.get(Development, dev_id)
            assert (dev.status, dev.insight_count, dev.source_count) == ("merged", 0, 0)
            assert f"development {a_id}" in dev.summary
        for model in (Score, Lens, ExposureLink, Alert):
            rows = s.scalars(select(model.development_id)).all()
            assert sorted(rows) == [a_id], model.__tablename__  # absorbed rows deleted, the survivor's kept
        labels = {(lb.development_id, lb.audience, lb.annotator): lb.label for lb in s.scalars(select(Label))}
        assert labels == {(a_id, "investment", "human:ana"): 1, (a_id, "ai_team", "llm:judge"): 2,
                          (a_id, "ai_team", "human:ana"): 2, (a_id, "investment", "llm:judge"): 1,
                          (b_id, "investment", "human:ana"): 2, (c_id, "ai_team", "human:ana"): 0}
        assert s.get(Development, fable_id).event_type == "model_release"
        assert (s.get(Development, price_id).insight_count, s.get(Development, price_id).status) == (1, "open")
        assert (s.get(Development, other_id).insight_count, s.get(Development, other_id).status) == (1, "open")

    assert merge.main([]) == 0  # idempotent: the second run finds nothing
    assert "merged 0 developments into 0 (survivors: none); re-typed 0" in capsys.readouterr().out

    with launch_world() as s:  # tombstones are inert: new evidence joins the survivor
        new = add_insight(s, add_doc(s, "reflection-ai:hf", published=T0 + 18 * DAY), e(0),
                          event_type="open_weight_release", artifacts_=("Beam 1.0 501B A23B",))
        s.commit()
        new_id = new.id
    assert cluster.run_clustering()["repaired"] == 0
    with launch_world() as s:
        assert s.get(Insight, new_id).development_id == a_id
        assert s.get(Development, b_id).status == "merged"
