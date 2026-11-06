"""Exposure mapping, features, labels and materiality scoring (no network, judge mocked)."""

from __future__ import annotations

import asyncio
import csv
import itertools
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pytest
import yaml
from sqlalchemy import func, select

from fli.analyze import exposure, features, labeling, scoring
from fli.analyze.features import FEATURE_NAMES, development_features
from fli.config import CONFIG_DIR, get_settings
from fli.db.models import (
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Label,
    OpsEvent,
    Person,
    Score,
    Source,
    Ticker,
)
from fli.llm.client import BudgetExceeded, LLMResult
from fli.timeutil import utcnow

T0 = datetime(2026, 8, 3)
_seq = itertools.count(1)
LABS = [
    Lab(id="anthropic", name="Anthropic", kind="frontier_lab", tier=1, public_exposure=[
        {"ticker": "AMZN", "relation": "investor_and_cloud_partner", "note": "Trainium / AWS"},
        {"ticker": "GOOGL", "relation": "investor_and_cloud_partner", "note": "TPU capacity"},
        {"ticker": "NVDA", "relation": "chip_supplier_and_investor"},
        {"ticker": "AKAM", "relation": "compute_supplier", "note": "compute agreement"},
        {"ticker": "TSLA", "relation": "affiliated", "note": "reported tie (verify)"}]),
    Lab(id="qwen", name="Qwen", kind="open_weight_lab", tier=2, public_exposure=[{"ticker": "BABA", "relation": "parent"}]),
    Lab(id="tml", name="Thinking Machines", kind="alumni_startup", tier=3, public_exposure=[]),
]
SOURCES = {"anthropic:news": "official", "qwen:hf": "official", "press:wire": "press", "x:jane": "personal",
           "arxiv:cs": "research"}


@pytest.fixture()
def world(db, tmp_path, monkeypatch):
    """Register rows (real exposure.yaml tickers/themes), and model/eval dirs redirected to tmp."""
    monkeypatch.setattr(scoring, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(scoring, "EVAL_DIR", tmp_path / "eval")
    cfg = yaml.safe_load((CONFIG_DIR / "exposure.yaml").read_text(encoding="utf-8"))
    weights = {h["symbol"]: h["weight"] for h in cfg["bit_holdings"]}
    members = defaultdict(list)
    for th in cfg["themes"]:
        for sym in th["tickers"]:
            members[sym].append(th["id"])
    with db() as s:
        for t in cfg["tickers"]:
            s.add(Ticker(symbol=t["symbol"], company=t["company"], themes=sorted(members[t["symbol"]]),
                         bit_weight=weights.get(t["symbol"])))
        s.add_all([Lab(id=lab.id, name=lab.name, kind=lab.kind, tier=lab.tier, public_exposure=lab.public_exposure)
                   for lab in LABS])
        s.add_all([Source(id=sid, adapter="test", kind="official_blog", source_class=c) for sid, c in SOURCES.items()])
        s.add(Person(id="jane", name="Jane Doe", tier="A", lab_id="anthropic", title="Chief Scientist"))
        s.commit()
    return db


def make_dev(s, title, event_type="model_release", lab_id="anthropic", docs=(("anthropic:news", {}),),
             claims=(), magnitude=3, confidence=0.8, rumor=False, verified=True, novelty=None, first_seen=(),
             persons=(), when=T0) -> Development:
    """A development with one insight per (source, meta) document; per-insight values may be lists."""
    pick = lambda v, k: v[k] if isinstance(v, list | tuple) else v
    dev = Development(title=title, event_type=event_type, lab_id=lab_id, first_seen_at=when, last_seen_at=when,
                      person_ids=list(persons))
    s.add(dev)
    s.flush()
    for k, (source_id, meta) in enumerate(docs):
        n = next(_seq)
        doc = Document(source_id=source_id, natural_key=f"k{n}", url=f"https://example.com/{n}",
                       canonical_url=f"https://example.com/{n}", content_hash=f"h{n}", lab_id=lab_id,
                       source_class=SOURCES[source_id], meta=meta, published_at=when, title=title)
        s.add(doc)
        s.flush()
        s.add(Insight(document_id=doc.id, development_id=dev.id, lab_id=lab_id, event_type=event_type, title=title,
                      claim=claims[k] if k < len(claims) else title, magnitude=pick(magnitude, k),
                      confidence=pick(confidence, k), is_rumor=pick(rumor, k), quotes_verified=pick(verified, k),
                      is_official=SOURCES[source_id] == "official", novelty_entity=pick(novelty, k),
                      first_seen_artifacts=list(pick(first_seen, k)) if first_seen else [],
                      evidence_quotes=[{"quote": f"quote {n}"}], person_ids=list(persons), prompt_version="t", model="t"))
    dev.source_count = len({d for d, _ in docs})
    dev.source_classes = sorted({SOURCES[d] for d, _ in docs})
    dev.has_official = any(SOURCES[d] == "official" for d, _ in docs)
    dev.has_verified = any(pick(verified, k) for k in range(len(docs)))
    s.flush()
    return dev


def links_of(s, dev_id) -> dict[str, ExposureLink]:
    return {lk.ticker: lk for lk in s.scalars(select(ExposureLink).where(ExposureLink.development_id == dev_id))}


# ------------------------------------------------------------------------------------ exposure


def test_exposure_rules_cap_rank_and_method(world):
    with world() as s:
        deal = make_dev(s, "Anthropic signs multi-gigawatt datacenter capacity agreement with Oracle",
                        event_type="compute_deal", claims=["New campus adds power and capacity for training."]).id
        paper = make_dev(s, "Anthropic interpretability paper", event_type="research_paper",
                         claims=["Method runs on existing datacenter capacity."]).id
        s.commit()
    out = exposure.run_exposure()
    assert out["developments_linked"] == 2
    assert out["links_by_method"] == {"rule": 8, "lab": 5}
    with world() as s:
        d = links_of(s, deal)
        assert len(d) == exposure.MAX_LINKS and {lk.method for lk in d.values()} == {"rule"}
        # the deal's own evidence fills the cap, strongest first: named ORCL (0.75 + 0.15), the corroborated
        # AKAM (relation + keywords), then the neocloud keyword theme (0.55 + 0.15) ...
        assert set(d) == {"ORCL", "AKAM", "IREN", "CRWV", "NBIS", "CIFR", "WULF", "HUT"}
        assert "CORZ" not in d  # ... where, at equal strength, the BIT holding IREN wins the tie, nothing more
        # lab-only relations (AMZN, NVDA, GOOGL, TSLA) describe Anthropic, not this deal: no slot taken from it
        assert not {"AMZN", "NVDA", "GOOGL", "TSLA"} & set(d)
        assert d["ORCL"].confidence == pytest.approx(0.75)  # named + theme keywords: corroborated
        assert d["AKAM"].confidence == pytest.approx(0.7) and "public exposure" in d["AKAM"].rationale
        assert d["IREN"].theme == "neocloud_capacity" and "keywords" in d["IREN"].rationale
        assert {lk.direction for lk in d.values()} == {"unclear"}
        p = links_of(s, paper)  # research_paper moves no keyword theme: lab context fills the free slots
        assert set(p) == {"AMZN", "GOOGL", "NVDA", "AKAM", "TSLA"} and {lk.method for lk in p.values()} == {"lab"}
        assert p["TSLA"].confidence == pytest.approx(0.4) and p["NVDA"].theme == "accelerators"
        assert p["AMZN"].confidence == pytest.approx(0.6) and p["AMZN"].theme == "hyperscaler_partners"
    assert exposure.run_exposure()["links_created"] == 0  # idempotent


def _tinfo(sym, weight=0.0, themes=("t",)):
    return exposure.TickerInfo(sym, f"{sym} Corp", tuple(themes), weight)


def test_propose_links_event_evidence_first_weight_breaks_ties():
    themes = [exposure.Theme("t", "Theme", ("OTHER", "HELD"), frozenset({"model_release"}), ("widget",))]
    tickers = {"OTHER": _tinfo("OTHER"), "HELD": _tinfo("HELD", 9.0), "NAMED": _tinfo("NAMED", themes=()),
               "LABHELD": _tinfo("LABHELD", 8.0, themes=())}
    lab = Lab(id="x", name="X Lab", public_exposure=[{"ticker": "LABHELD", "relation": "investor"}])
    ranked = exposure.propose_links("model_release", lab, "X Lab ships a widget built with NAMED Corp", themes, tickers)
    # holdings no longer come first: the named company (0.65 + bonus) leads, the one-keyword theme links
    # (0.35 + bonus) follow with the heavier holding first on the tie, and the lab's investor relation
    # (an 8% holding, 0.6) only takes what the development's own evidence leaves
    assert [(c.ticker, c.method, round(c.strength, 2)) for c in ranked] == [
        ("NAMED", "rule", 0.8), ("HELD", "rule", 0.5), ("OTHER", "rule", 0.5), ("LABHELD", "lab", 0.6)]


def test_exposure_refresh_upgrades_and_replaces_lab_links(world, monkeypatch):
    monkeypatch.setattr(exposure, "MAX_LINKS", 4)
    with world() as s:
        dev = make_dev(s, "Anthropic paper on sparse attention", event_type="research_paper")
        s.commit()
        dev_id = dev.id
    exposure.run_exposure()
    with world() as s:
        before = links_of(s, dev_id)
        assert set(before) == {"AMZN", "GOOGL", "NVDA", "AKAM"}  # cap reached: TSLA (0.4) left out
        assert {lk.method for lk in before.values()} == {"lab"}
        s.add(ExposureLink(development_id=dev_id, ticker="SMCI", method="llm", confidence=0.3,
                           rationale="lens"))  # another stage's row: never touched, even though it is weakest
        s.commit()
    with world() as s:
        doc = Document(source_id="press:wire", natural_key="extra", url="https://e.com/x", canonical_url="https://e.com/x",
                       content_hash="x", source_class="press")
        s.add(doc)
        s.flush()
        s.add(Insight(document_id=doc.id, development_id=dev_id, event_type="research_paper", title="Coverage",
                      claim="Trained on AWS Trainium; Micron HBM did the rest.", prompt_version="t", model="t"))
        s.get(Development, dev_id).insight_count = 2
        s.commit()
    assert exposure.run_exposure()["links_created"] == 1
    with world() as s:
        after = links_of(s, dev_id)
        assert set(after) == {"AMZN", "NVDA", "AKAM", "SMCI", "MU"}  # MU (named) replaced the weakest lab link
        assert after["AMZN"].method == "rule" and after["AMZN"].confidence == pytest.approx(0.75)  # now named
        assert "named in development text" in after["AMZN"].rationale
        assert after["MU"].method == "rule" and after["SMCI"].method == "llm" and after["NVDA"].method == "lab"
    assert exposure.run_exposure()["links_created"] == 0


def test_migration_relabels_exactly_the_lab_only_links(world):
    """The one-off SQL agrees with the stage: rows written before the change (all 'rule') get the method
    the stage now writes; hand-made or LLM rows are left alone; a re-run changes nothing."""
    with world() as s:
        make_dev(s, "Anthropic signs multi-gigawatt datacenter capacity agreement with Oracle",
                 event_type="compute_deal", claims=["New campus adds power and capacity for training."])
        paper = make_dev(s, "Anthropic interpretability paper on AWS Trainium", event_type="research_paper")
        make_dev(s, "Qwen releases Qwen4", lab_id="qwen")
        s.commit()
    exposure.run_exposure()
    with world() as s:
        expected = {(lk.development_id, lk.ticker): lk.method for lk in s.scalars(select(ExposureLink))}
        assert set(expected.values()) == {"rule", "lab"}
        assert expected[(paper.id, "AMZN")] == "rule" and expected[(paper.id, "NVDA")] == "lab"
        s.add(ExposureLink(development_id=paper.id, ticker="ORCL", method="llm", rationale="Anthropic public exposure: x"))
        s.execute(ExposureLink.__table__.update().where(ExposureLink.__table__.c.method == "lab").values(method="rule"))
        s.commit()
    sql = (CONFIG_DIR.parent / "scripts" / "migrations" / "20261104_exposure_method.sql").read_text(encoding="utf-8")
    for _ in range(2):
        with world() as s:
            s.connection().exec_driver_sql(sql)
            s.commit()
    with world() as s:
        got = {(lk.development_id, lk.ticker): lk.method for lk in s.scalars(select(ExposureLink))}
    assert got.pop((paper.id, "ORCL")) == "llm"
    assert got == expected


def test_exposure_refreshes_changed_development(world):
    with world() as s:
        dev = make_dev(s, "Anthropic paper on sparse attention", event_type="research_paper")
        s.commit()
        dev_id = dev.id
    exposure.run_exposure()
    with world() as s:
        dev = s.get(Development, dev_id)
        doc = Document(source_id="press:wire", natural_key="extra", url="https://e.com/x", canonical_url="https://e.com/x",
                       content_hash="x", source_class="press")
        s.add(doc)
        s.flush()
        s.add(Insight(document_id=doc.id, development_id=dev_id, event_type="research_paper", title="Coverage",
                      claim="Experiments ran on Micron HBM.", prompt_version="t", model="t"))
        dev.insight_count = 2  # cluster stage touches the row -> updated_at moves
        s.commit()
    assert exposure.run_exposure()["links_created"] >= 1  # mention link (+ memory theme links on "HBM")
    with world() as s:
        assert "MU" in links_of(s, dev_id)


# ------------------------------------------------------------------------------------ features


def test_development_features(world):
    with world() as s:
        dev = make_dev(s, "Claude 5 released",
                       docs=[("anthropic:news", {"kind": "release", "repo": "anthropics/sdk", "stars": 30000}),
                             ("press:wire", {"public_metrics": {"like_count": 50}})],
                       magnitude=[3, 5], confidence=[0.8, 0.6], rumor=[False, True], novelty=[0.2, 0.9],
                       first_seen=[[], ["claude-5"]], persons=["jane"])
        s.add_all([ExposureLink(development_id=dev.id, ticker="AMZN", method="rule", confidence=0.75),
                   ExposureLink(development_id=dev.id, ticker="MU", method="rule"),  # column default 0.5
                   ExposureLink(development_id=dev.id, ticker="NVDA", method="llm", confidence=0.9),  # lens: ignored
                   ExposureLink(development_id=dev.id, ticker="TSM", method="lab", confidence=0.6)])  # lab-level: ignored
        press = make_dev(s, "Report: raise talks", event_type="funding_ipo", lab_id="tml",
                         docs=[("press:wire", {})], rumor=True, verified=False, magnitude=None)
        s.commit()
        f, g = development_features(s, dev), development_features(s, press)
        weight = {t.symbol: t.bit_weight for t in s.scalars(select(Ticker))}
    assert list(f) == FEATURE_NAMES
    assert f["has_official"] == 1 and f["log_sources"] == pytest.approx(math.log1p(2)) and f["n_source_classes"] == 2
    assert (f["max_magnitude"], f["mean_confidence"], f["rumor_share"], f["max_novelty"]) == (5, 0.7, 0.5, 0.9)
    assert f["first_seen_artifact"] == 1 and f["lab_tier_1"] == 1 and f["person_tier"] == 1.0
    assert f["grp_release"] == 1 and sum(f[n] for n in FEATURE_NAMES if n.startswith("grp_")) == 1
    # event-specific links only, each weighted by its confidence
    assert f["exposure_bit"] == pytest.approx((weight["AMZN"] * 0.75 + weight["MU"] * 0.5) / 10)
    assert f["n_tickers"] == pytest.approx(1.25)
    # the release's 30k repository stars are not attention to this item; the post's likes are
    assert f["attention"] == pytest.approx(math.log1p(50)) and f["press_only"] == 0
    assert g["press_only"] == 1 and g["has_official"] == 0 and g["lab_tier_3"] == 1 and g["grp_capital"] == 1
    assert g["max_magnitude"] == 2.0 and g["max_novelty"] == 0.5  # documented imputations
    assert g["exposure_bit"] == 0 and g["n_tickers"] == 0 and g["attention"] == 0


def test_attention_counts_only_per_item_engagement():
    eng = features.engagement
    assert eng({"kind": "release", "repo": "openai/openai-python", "stars": 32040, "forks": 4000}) == 0
    assert eng({"kind": "repo", "repo": "org/new", "stars": 24, "stargazers_count": 24}) == 0
    assert eng({"upvotes": 15, "num_comments": 2}) == 15  # HF daily paper
    assert eng({"likes": 5220, "downloads": 221621}) == 221621  # HF model family: that artifact's own counters
    assert eng({"metrics": {"like_count": 40, "impression_count": 90000}}) == 40  # X public_metrics
    assert eng({"metrics": {"likeCount": 12, "repostCount": 3}}) == 12  # Bluesky
    assert eng({"likes": "n/a", "upvotes": None}) == 0 and eng(None) == 0 and eng(["x"]) == 0


# ------------------------------------------------------------------------------------ scoring


def test_prior_scores_contributions_and_idempotency(world):
    with world() as s:
        ids = [make_dev(s, f"dev {i}", magnitude=1 + i % 5).id for i in range(5)]
        s.commit()
    first = scoring.run_scoring()
    assert first["investment"]["new"] == 5 and first["investment"]["version"] == scoring.PRIOR_VERSION
    assert first["investment"]["validated"] is False
    assert scoring.run_scoring()["developments"] == 0  # nothing changed: nothing to do
    model = scoring.load_model("investment")
    with world() as s:
        for sc in s.scalars(select(Score).where(Score.audience == "investment")):
            full = model.contributions(sc.features)
            assert scoring.sigmoid(model.intercept + sum(full.values())) == pytest.approx(sc.p_material, abs=1e-6)
            top = sorted(full.items(), key=lambda kv: -abs(kv[1]))[:scoring.TOP_CONTRIBUTIONS]
            assert list(sc.contributions) == [k for k, _ in top]
        dev = s.get(Development, ids[0])
        dev.source_count, dev.source_classes = 3, ["official", "press"]
        s.add(Score(development_id=ids[1], audience="ai_team", p_material=0.9, model_version="lr-old",
                    created_at=utcnow() + timedelta(seconds=1)))  # a newer row from another version
        s.commit()
    again = scoring.run_scoring()
    assert again["investment"]["updated"] == 1 and again["ai_team"]["updated"] == 1
    with world() as s:
        per_key = s.execute(select(func.count()).select_from(Score).group_by(
            Score.development_id, Score.audience, Score.model_version)).scalars().all()
        assert set(per_key) == {1}
        newest = s.scalars(select(Score).where(Score.development_id == ids[1], Score.audience == "ai_team")
                           .order_by(Score.created_at.desc())).first()
        assert newest.model_version == scoring.PRIOR_VERSION  # consumers read the newest row


def _synthetic_labeled(s, n=160, seed=7):
    """Linearly separable: worth knowing iff magnitude + 2 * official >= 5. Lab tier and event type are
    random nuisance features (the hand-set prior weights them, so it cannot be perfect here)."""
    rng = random.Random(seed)
    for i in range(n):
        official, magnitude = rng.random() < 0.5, rng.randint(1, 5)
        dev = make_dev(s, f"synthetic {i}", lab_id=rng.choice(["anthropic", "qwen"]),
                       event_type=rng.choice(["model_release", "research_paper"]),
                       docs=[("anthropic:news" if official else "press:wire", {})], magnitude=magnitude,
                       confidence=rng.uniform(0.6, 0.95), when=T0 + timedelta(hours=12 * i))
        label = 2 if magnitude + 2 * official >= 6 else 1 if magnitude + 2 * official >= 5 else 0
        for aud in ("investment", "ai_team"):
            s.add(Label(development_id=dev.id, audience=aud, label=label, annotator="llm:test", rubric_version="r1"))
    s.commit()


def test_training_on_separable_data(world, tmp_path):
    with world() as s:
        _synthetic_labeled(s)
    out = scoring.train_models()
    inv = out["investment"]
    assert inv["status"] == "trained" and inv["deployed"] and inv["test_roc_auc"] > 0.9
    assert inv["version"] == f"lr-investment-{utcnow():%Y%m%d}-llm"
    saved = json.loads((tmp_path / "models" / "scoring_investment.json").read_text())
    assert {"feature_names", "scaler_mean", "scaler_scale", "coefficients", "intercept", "metrics"} <= set(saved)
    assert saved["validated"] is True and saved["label_source"] == "llm"
    report = json.loads((tmp_path / "eval" / "scoring_metrics.json").read_text())["audiences"]["investment"]
    assert set(report["baselines"]) == {"official_only", "magnitude_only", "n_sources_only", scoring.PRIOR_VERSION}
    assert report["ablation_without_llm_features"]["dropped"] == ["max_magnitude", "mean_confidence"]
    assert len(report["model"]["calibration"]) == 5 and report["model"]["brier"] < 0.25
    assert report["split"]["train"] == 112 and report["split"]["test"] == 48
    model = scoring.load_model("investment")
    assert model.version == inv["version"] and out["rescored"]["investment"]["new"] == 160
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["scoring_ai_team.json", "scoring_investment.json"]
    assert scoring.current_versions() == {"investment": inv["version"], "ai_team": out["ai_team"]["version"]}
    with world() as s:
        sc = s.scalars(select(Score).where(Score.audience == "investment", Score.model_version == model.version)).first()
        assert model.predict(sc.features) == pytest.approx(sc.p_material, abs=1e-6)


def test_training_skips_with_too_few_labels(world, tmp_path):
    with world() as s:
        _synthetic_labeled(s, n=20)
    out = scoring.train_models()
    assert out["investment"]["status"] == "skipped" and not out["investment"]["deployed"]
    assert not (tmp_path / "models" / "scoring_investment.json").exists()
    assert scoring.load_model("investment").version == scoring.PRIOR_VERSION


def _deployable(audience: str, version: str) -> scoring.ScoringModel:
    model = scoring.prior_model(audience)
    model.version, model.validated = version, True
    return model


def test_save_model_is_atomic(world, tmp_path, monkeypatch):
    path = scoring.save_model(_deployable("investment", "lr-a"))
    assert scoring.load_model("investment").version == "lr-a"

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(scoring.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        scoring.save_model(_deployable("investment", "lr-b"))
    assert json.loads(path.read_text())["version"] == "lr-a"  # the deployed file is untouched
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # and no temp file is left behind


def test_unreadable_model_halts_scoring_and_opens_ops_event(world, tmp_path):
    with world() as s:
        first = make_dev(s, "first").id
        s.commit()
    path = scoring.save_model(_deployable("investment", "lr-good"))
    scoring.run_scoring()
    path.write_text('{"audience": "investment", "version": "lr-good", "coefficients": [0.1, ')  # half-written
    with pytest.raises(scoring.ModelFileError, match="unreadable"):
        scoring.load_model("investment")
    with world() as s:
        second = make_dev(s, "second").id
        s.commit()
    with pytest.raises(scoring.ModelFileError):
        scoring.run_scoring()
    with world() as s:
        rows = {(sc.development_id, sc.audience): sc.model_version for sc in s.scalars(select(Score))}
        events = list(s.scalars(select(OpsEvent)))
    # no silent fallback: the prior never scored investment, the last good rows stay the live ones ...
    assert rows == {(first, "investment"): "lr-good", (first, "ai_team"): scoring.PRIOR_VERSION,
                    (second, "ai_team"): scoring.PRIOR_VERSION}  # ... while ai_team still got scored
    assert [(e.kind, e.severity, e.subject, e.resolved_at) for e in events] == [
        (scoring.OPS_KIND, "crit", "scoring_model:investment", None)]  # recorded once, though it failed twice
    assert scoring.current_versions() == {"investment": "lr-good", "ai_team": scoring.PRIOR_VERSION}  # no raise
    assert [p.audience for p in scoring.model_file_problems()] == ["investment"]
    path.write_text(json.dumps(_deployable("ai_team", "lr-x").to_dict()))  # valid JSON, wrong audience
    with pytest.raises(scoring.ModelFileError, match="ai_team"):
        scoring.load_model("investment")
    path.unlink()  # nothing deployed: the prior is the live model again, by choice
    assert scoring.run_scoring()["investment"]["version"] == scoring.PRIOR_VERSION
    assert scoring.model_file_problems() == []


def test_current_versions_waits_for_rows_of_a_new_model(world):
    with world() as s:
        make_dev(s, "only")
        s.commit()
    assert scoring.current_versions() == {"investment": scoring.PRIOR_VERSION, "ai_team": scoring.PRIOR_VERSION}
    scoring.run_scoring()
    scoring.save_model(_deployable("investment", "lr-new"))  # deployed, not re-scored yet
    with world() as s:
        assert scoring.current_versions(s)["investment"] == scoring.PRIOR_VERSION  # pages keep showing rows
        assert scoring.current_scores(s, "investment") != {}
    scoring.run_scoring()
    assert scoring.current_versions()["investment"] == "lr-new"


def test_choose_c_uses_a_tolerance_band(monkeypatch):
    """CV numbers of the investment model after the exposure fix: rounding to 0.01 picked C=3.0 (0.92
    vs 0.91); within 0.01 of the best AUC, C=1.0 has the best log-loss."""
    auc = {0.01: 0.8993, 0.03: 0.9065, 0.1: 0.9072, 0.3: 0.9099, 1.0: 0.9144, 3.0: 0.9194}
    log_loss = {0.01: 0.4853, 0.03: 0.4163, 0.1: 0.3657, 0.3: 0.3421, 1.0: 0.3351, 3.0: 0.3427}

    def fake_cv(est, X, y, cv, scoring):
        c = est[-1].C
        return {"test_auc": np.array([auc[c]]), "test_ll": np.array([-log_loss[c]])}

    monkeypatch.setattr(scoring, "cross_validate", fake_cv)
    best, table = scoring.choose_c(np.zeros((20, 2)), np.array([0, 1] * 10))
    assert best == 1.0 and table["3.0"]["roc_auc"] == 0.9194


def test_human_labels_preferred_over_llm(world):
    with world() as s:
        a, b = make_dev(s, "a"), make_dev(s, "b")
        s.add_all([Label(development_id=a.id, audience="investment", label=0, annotator="llm:x", rubric_version="r1"),
                   Label(development_id=a.id, audience="investment", label=2, annotator="human:ann", rubric_version="r1"),
                   Label(development_id=b.id, audience="investment", label=2, annotator="llm:x", rubric_version="r1")])
        s.commit()
        got = {r.dev.id: r for r in labeling.training_labels(s, "investment")}
    assert (got[a.id].y, got[a.id].source) == (1, "human")
    assert (got[b.id].y, got[b.id].y2, got[b.id].source) == (1, 1, "llm")


def test_metric_helpers():
    y, s = np.array([1, 0, 1, 0]), np.array([1.0, 1.0, 0.0, 0.0])
    assert scoring.precision_at_k(y, s, 1) == 0.5  # tie: expected precision
    assert scoring.roc_auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0 and scoring.roc_auc(y, s) == 0.5
    table = scoring.calibration_table(np.array([0, 1, 1]), np.array([0.05, 0.5, 0.95]))
    assert [r["n"] for r in table] == [1, 0, 1, 0, 1]


# ------------------------------------------------------------------------------------ labels


def test_allocation_and_systematic_sampling():
    alloc = labeling.allocate({"big": 100, "mid": 9, "tiny": 1}, 10)
    assert sum(alloc.values()) == 10 and min(alloc.values()) >= 1 and alloc["big"] > alloc["mid"] > alloc["tiny"]
    assert labeling.allocate({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}
    assert labeling.allocate({"a": 5, "b": 3, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}  # fewer seats than strata
    assert labeling.systematic([1, 2, 3], 0, random.Random(0)) == []
    picked = labeling.systematic(list(range(100)), 10, random.Random(0))
    assert len(set(picked)) == 10 and min(picked) < 10 and max(picked) >= 90  # low- and high-prior ends


def _strata_world(s):
    """4 strata of very different sizes; magnitudes spread so the prior ranks items within a stratum."""
    groups = {"release": [], "research": [], "capital": [], "hiring": []}
    for i in range(30):
        groups["release"].append(make_dev(s, f"rel {i}", magnitude=1 + i % 5).id)
    for i in range(5):
        groups["research"].append(make_dev(s, f"res {i}", "research_paper", "qwen", docs=[("x:jane", {})]).id)
    for i in range(3):
        groups["capital"].append(make_dev(s, f"cap {i}", "funding_ipo", docs=[("press:wire", {})]).id)
    for i in range(2):
        groups["hiring"].append(make_dev(s, f"hir {i}", "hiring_signal").id)
    done = make_dev(s, "already labelled").id
    s.add_all([Label(development_id=done, audience=a, label=1, annotator="llm:x", rubric_version="r1")
               for a in ("investment", "ai_team")])
    s.commit()
    return groups, done


def test_stratified_sample_covers_strata_and_low_prior(world):
    with world() as s:
        groups, done = _strata_world(s)
        sample = labeling.sample_for_labeling(s, 12)
        ids = {d.id for d in sample}
        mags = {d.id: max(i.magnitude for i in s.scalars(select(Insight).where(Insight.development_id == d.id)))
                for d in sample}
    assert len(sample) == 12 and done not in ids
    assert all(ids & set(members) for members in groups.values())  # every stratum represented
    rel = [mags[i] for i in ids & set(groups["release"])]
    assert min(rel) <= 2 and max(rel) >= 4  # not only top-ranked items


class FakeJudge:
    """Stands in for LLMGateway, including its concurrency semaphore (bound to the first event loop
    that waits on it), so a chunked run that used a new loop per chunk would fail here too."""

    def __init__(self, fail_ids=(), budget_after: int | None = None):
        self.calls, self.fail_ids, self.budget_after = [], set(fail_ids), budget_after
        self._sem = asyncio.Semaphore(1)

    async def _one(self, kw):
        async with self._sem:
            await asyncio.sleep(0)
        good = {"investment": {"reason": "word " * 40, "label": 2}, "ai_team": {"reason": "no tech", "label": 0}}
        if kw["ref"]["development_id"] in self.fail_ids:
            return LLMResult(None, "error", 0.0, kw["model"], "v", "boom")
        return LLMResult(good, "ok", 0.01, kw["model"], "v")

    async def structured_many(self, calls):
        if self.budget_after is not None and len(self.calls) >= self.budget_after:
            raise BudgetExceeded("daily budget reached")
        self.calls += calls
        return list(await asyncio.gather(*(self._one(kw) for kw in calls)))


def test_silver_labels_with_mocked_judge(world, monkeypatch):
    with world() as s:
        ids = [make_dev(s, f"d{i}", magnitude=3).id for i in range(6)]
        s.commit()
    judge = FakeJudge(fail_ids={ids[0]})
    monkeypatch.setattr(labeling, "_gateway", lambda: judge)
    out = labeling.run_labels("silver", n=4)
    model = get_settings().model_judge
    assert out["sampled"] == 4 and out["annotator"] == f"llm:{model}"
    assert {(c["purpose"], c["effort"], c["model"]) for c in judge.calls} == {("judge", "medium", model)}
    leaks = re.compile(r"\b(magnitude|confidence|novelty|tier|p_material)\b", re.IGNORECASE)  # no scoring features
    assert not any(leaks.search(c["user"]) for c in judge.calls)
    prompt = judge.calls[0]["prompt"]
    assert prompt.version_for(labeling.JUDGE_SCHEMA).startswith("judge_materiality@v1+")
    assert "IREN" in prompt.system and "{{" not in prompt.system  # holdings filled into the rubric
    with world() as s:
        rows = list(s.scalars(select(Label)))
        assert {(r.annotator, r.rubric_version) for r in rows} == {(f"llm:{model}", "r1")}
        assert all(len(r.reason.split()) <= labeling.REASON_MAX_WORDS for r in rows)
    expected_failed = int(ids[0] in {c["ref"]["development_id"] for c in judge.calls})
    assert out["failed"] == expected_failed and out["labels_written"] == 2 * (4 - expected_failed)
    monkeypatch.setattr(labeling, "_gateway", FakeJudge)  # a fresh gateway per run, as in production
    again = labeling.run_labels("silver", n=10)  # only developments still missing an LLM label
    assert again["sampled"] == 6 - 4 + expected_failed


def test_silver_budget_stop_keeps_finished_chunks(world, monkeypatch):
    with world() as s:
        for i in range(5):
            make_dev(s, f"d{i}")
        s.commit()
    monkeypatch.setattr(labeling, "_gateway", lambda: FakeJudge(budget_after=4))
    out = labeling.run_silver(5, chunk=2)
    assert out["stopped"] and out["labels_written"] == 8  # two chunks (2 developments x 2 audiences each)


def test_export_import_and_agreement(world, monkeypatch, tmp_path):
    with world() as s:
        ids = [make_dev(s, f"d{i}").id for i in range(4)]
        s.commit()
    monkeypatch.setattr(labeling, "_gateway", lambda: FakeJudge())
    labeling.run_labels("silver", n=4)
    path = tmp_path / "labels.csv"
    assert labeling.run_labels("export", path=str(path))["rows"] == 4
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["llm_investment"] == "2" and rows[0]["human_investment"] == ""
    for r, (inv, ai) in zip(rows, [("2", "0"), ("2", "1"), ("0", "0"), ("1", "0")], strict=True):
        r.update(human_name="Alice B", human_investment=inv, human_ai_team=ai)
    rows[3]["human_ai_team"] = "7"  # invalid value is reported, not imported
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    imported = labeling.run_labels("import", path=str(path))
    assert imported["created"] == 7 and imported["invalid"] == 1
    assert labeling.run_labels("import", path=str(path))["unchanged"] == 7  # idempotent
    stats = labeling.run_labels("stats")
    assert stats["counts"]["human:alice-b"]["investment"]["n"] == 4
    inv = stats["human_vs_llm"]["investment:human:alice-b"]
    assert inv["n"] == 4 and inv["exact_agreement"] == 0.5
    labeling.run_labels("export", path=str(path), blind=True)  # re-export keeps reviewer columns
    again = {r["development_id"]: r for r in csv.DictReader(path.open())}
    assert again[str(ids[0])]["human_name"] == "Alice B" and again[str(ids[0])]["llm_investment"] == ""
    with world() as s:
        got = {r.dev.id: r.source for r in labeling.training_labels(s, "investment")}
    assert set(got.values()) == {"human"}


def test_parse_judgement_rejects_bad_output():
    ok = {"investment": {"reason": "x", "label": 1}, "ai_team": {"reason": "y", "label": 0}}
    assert labeling.parse_judgement(ok) == {"investment": (1, "x"), "ai_team": (0, "y")}
    assert labeling.parse_judgement({**ok, "ai_team": {"reason": "y", "label": 3}}) is None
    assert labeling.parse_judgement({**ok, "ai_team": {"reason": "y", "label": True}}) is None
    assert labeling.parse_judgement({"investment": ok["investment"]}) is None
    with pytest.raises(ValueError):
        labeling.run_labels("nope")
