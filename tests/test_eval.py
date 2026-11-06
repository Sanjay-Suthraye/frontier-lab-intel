"""Evaluation harness: known-event matching (whole words, best candidate rather than the earliest,
delay from the reporting source) and alert metrics (pushed P1/P2 only, precision against labels,
label-2 recall, the scorer's hold-out window, in-sample marking), plus report merging; and the
signal-quality suites: lens faithfulness (Lens.checks), coverage by BIT holding, digest quality
(Report.model and item flags) and source coverage."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import yaml

from fli import config as fli_config
from fli.db.models import (
    Alert,
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Label,
    Lens,
    Report,
    Score,
    Source,
    Ticker,
)
from fli.eval import run as ev

_seq = itertools.count()


@pytest.fixture()
def world(db, tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    monkeypatch.setattr(ev, "EVAL_DIR", eval_dir)
    monkeypatch.delenv(ev.OUT_DIR_ENV, raising=False)  # a developer's override must not redirect test reports
    with db() as s:
        s.add_all([Lab(id="openai", name="OpenAI", kind="frontier_lab"),
                   Lab(id="anthropic", name="Anthropic", kind="frontier_lab")])
        s.add_all([Source(id="openai:news", lab_id="openai", adapter="rss", kind="official_blog",
                          source_class="official"),
                   Source(id="anthropic:news", lab_id="anthropic", adapter="rss", kind="official_blog",
                          source_class="official")])
        s.commit()
    return db


def make_dev(s, lab: str, event_type: str, members: list[tuple[str, str, datetime]],
             insight_published: dict[int, datetime] | None = None) -> tuple[Development, list[Insight]]:
    """A development with one insight per (title, claim, document published_at) member."""
    times = [t for _, _, t in members]
    dev = Development(title=members[0][0], event_type=event_type, lab_id=lab, first_seen_at=min(times),
                      last_seen_at=max(times), insight_count=len(members))
    s.add(dev)
    s.flush()
    out = []
    for k, (title, claim, published) in enumerate(members):
        n = next(_seq)
        doc = Document(source_id=f"{lab}:news", natural_key=f"k{n}", url=f"https://example.com/{n}",
                       canonical_url=f"https://example.com/{n}", title=title, published_at=published,
                       content_text=claim, content_hash=f"h{n}", lab_id=lab, source_class="official")
        s.add(doc)
        s.flush()
        ins = Insight(document_id=doc.id, development_id=dev.id, lab_id=lab, event_type=event_type, title=title,
                      claim=claim, published_at=(insight_published or {}).get(k, published), prompt_version="t",
                      model="t")
        s.add(ins)
        out.append(ins)
    s.flush()
    return dev, out


def add_alert(s, dev: Development, audience: str, severity: str, at: datetime | None = None) -> Alert:
    a = Alert(development_id=dev.id, audience=audience, severity=severity, title=dev.title, body_md="-",
              dedup_key=f"{dev.id}:{audience}", created_at=at or dev.first_seen_at)
    s.add(a)
    return a


def add_label(s, dev: Development, audience: str, label: int, annotator: str = "llm:judge") -> None:
    s.add(Label(development_id=dev.id, audience=audience, label=label, annotator=annotator, rubric_version="t"))


def write_events(tmp_dir, events: list[dict]):
    path = tmp_dir / "known_events.yaml"
    path.write_text(yaml.safe_dump({"events": events}), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------ known events
def test_terms_match_whole_words_only():
    assert not ev._has_term("according to its openrouter listing", "accord")
    assert ev._has_term("a voluntary ai accord, signed", "accord")
    assert not ev._has_term("the cs-1 rocket", "s-1")
    assert ev._has_term("anthropic files s-1 with sec", "s-1")
    assert ev._has_term("gpt-6.1 sol launches", "6.1")
    assert not ev._has_term("version 16.1", "6.1")
    assert ev._groups_hit("white house ai accord", [["white house"], ["accord", "pledge"], ["astra"]]) == 2


@pytest.mark.parametrize("event_type", ["funding_ipo", None])
def test_known_event_credits_best_candidate_not_earliest(world, tmp_path, event_type):
    """The harness credited the Anthropic S-1 to an earlier API changelog that said 'listing'."""
    with world() as s:
        changelog, _ = make_dev(s, "anthropic", "product_launch", [
            ("Anthropic releases API version 2026-10-21 and new SDKs",
             "Adds a usage_tier scope for listing spend limits.", datetime(2026, 10, 21))])
        s1, _ = make_dev(s, "anthropic", "funding_ipo", [
            ("Anthropic files S-1 with SEC for proposed Nasdaq IPO", "Public S-1 filed.", datetime(2026, 10, 26, 6))])
        emptied = Development(title="Anthropic files S-1 (IPO)", event_type="funding_ipo", lab_id="anthropic",
                              first_seen_at=datetime(2026, 10, 26), last_seen_at=datetime(2026, 10, 26), status="empty")
        s.add(emptied)
        s.flush()
        for dev, p in ((changelog, 0.04), (s1, 0.99), (emptied, 1.0)):
            s.add(Score(development_id=dev.id, audience="investment", p_material=p, model_version="m"))
        add_alert(s, s1, "investment", "P1")
        s.commit()
        changelog_id, s1_id = changelog.id, s1.id
    spec = {"id": "anthropic-s1", "date": "2026-10-26", "labs": ["anthropic"], "title": "Anthropic files S-1",
            "audiences": ["investment"], "match_all": [["s-1", "ipo", "listing"]], "window_days": 7,
            "confidence": "high"}
    if event_type:
        spec["event_type"] = event_type
    res = ev.eval_known_events(write_events(tmp_path, [spec]))
    row = res["details"][0]
    assert row["matched_developments"] == 2  # the changelog still matches 'listing' as a whole word
    assert row["development_id"] == s1_id != changelog_id
    assert (row["rank_investment"], row["alert_investment"], row["delay_days"]) == (1, "P1", 0.25)
    assert (res["recall"], res["alerted_primary_audience"], res["pushed_primary_audience"]) == (1.0, 1.0, 1.0)


def test_known_event_delay_from_reporting_insight_and_stale_alert_flagged(world, tmp_path):
    """A product launch folded into an older model-release development: the delay is measured from the
    launch post's source date, and the development's earlier alert is flagged as not about it."""
    t0 = datetime(2026, 9, 1, 13)
    with world() as s:
        astra, (intro, finance) = make_dev(s, "openai", "model_release", [
            ("OpenAI introduces GPT-6 Astra", "GPT-6 Astra, a new flagship generation.", t0),
            ("OpenAI launches ChatGPT for Financial Services with GPT-6 Astra",
             "Financial services edition with S&P Global data.", datetime(2026, 9, 10, 7))],
            insight_published={1: datetime(2026, 9, 10, 12)})  # the document date wins
        s.add(Score(development_id=astra.id, audience="investment", p_material=0.9, model_version="m"))
        add_alert(s, astra, "investment", "P1", at=t0)
        s.commit()
        astra_id, intro_id, finance_id = astra.id, intro.id, finance.id
    events = [
        {"id": "fin", "date": "2026-09-10", "labs": ["openai"], "event_type": "product_launch", "title": "ChatGPT fin",
         "audiences": ["investment"], "match_all": [["financial"], ["chatgpt", "openai"]], "window_days": 10},
        {"id": "astra", "date": "2026-09-03", "labs": ["openai"], "event_type": "model_release", "title": "Astra",
         "audiences": ["investment"], "match_all": [["gpt-6", "astra"]], "window_days": 10},
        {"id": "absent", "date": "2026-09-03", "labs": ["openai"], "title": "Not in the corpus",
         "audiences": ["investment"], "match_all": [["white house"], ["accord"]]},
    ]
    res = ev.eval_known_events(write_events(tmp_path, events))
    fin, astra_row, absent = res["details"]
    assert (fin["development_id"], fin["evidence_insight_id"], fin["delay_days"]) == (astra_id, finance_id, 0.29)
    assert fin["evidence_published_at"] == "2026-09-10T07:00:00"
    assert fin["alert_investment_predates_evidence"] is True
    assert (astra_row["evidence_insight_id"], astra_row["delay_days"]) == (intro_id, -1.46)
    assert "alert_investment_predates_evidence" not in astra_row
    assert absent["found"] is False
    assert res["alert_predates_evidence"] == ["fin"]
    assert (res["recall"], res["alerted_primary_audience"], res["median_delay_days"]) == (0.667, 0.667, -0.585)


def test_shipped_known_events_keep_events_and_tighten_only_audited_terms():
    spec = yaml.safe_load((ev.DATA_DIR / "eval" / "known_events.yaml").read_text(encoding="utf-8"))
    events = {e["id"]: e for e in spec["events"]}
    assert len(events) == 18
    terms = {t for e in events.values() for g in e["match_all"] for t in g}
    assert not terms & {"listing", "shelv", "withh"}  # a false match, and prefixes whole words never match
    s1 = [[t.lower() for t in g] for g in events["anthropic-s1"]["match_all"]]
    assert ev._groups_hit("adds a usage_tier scope for listing spend limits", s1) == 0
    assert ev._groups_hit("anthropic files s-1 with sec for proposed nasdaq ipo", s1) == len(s1)
    accord = [[t.lower() for t in g] for g in events["white-house-accord"]["match_all"]]
    assert ev._groups_hit("according to its openrouter listing", accord) == 0
    assert ev._groups_hit("white house voluntary ai accord signed", accord) == len(accord)
    shelved = [[t.lower() for t in g] for g in events["openai-gpt61-shelved"]["match_all"]]
    assert ev._groups_hit("openai launches gpt-6.1 sol with astra-class safety", shelved) < len(shelved)
    assert ev._groups_hit("openai shelved gpt-6.1 astra after testing", shelved) == len(shelved)


# ------------------------------------------------------------------------------------ alerts
def test_alert_metrics_count_pushed_alerts_only(world, tmp_path):
    (tmp_path / "eval" / "scoring_metrics.json").write_text(json.dumps({"audiences": {
        aud: {"split": {"test_from": "2026-10-15 00:00:00"}} for aud in ("investment", "ai_team")}}))
    day = lambda d: datetime(2026, 10, d, 9)  # noqa: E731
    with world() as s:
        d = {k: make_dev(s, "openai", "model_release", [(f"dev {k}", "c", day(t))])[0]
             for k, t in ((1, 1), (2, 1), (3, 2), (4, 20), (5, 21), (6, 22), (7, 29))}
        for k, sev in ((1, "P1"), (2, "P2"), (3, "P3"), (4, "P2"), (6, "P1"), (7, "P2")):
            add_alert(s, d[k], "investment", sev)
        add_alert(s, d[7], "ai_team", "P1")
        for k, lab in ((1, 2), (2, 0), (3, 2), (4, 1), (5, 2), (6, 0)):
            add_label(s, d[k], "investment", lab)
        add_label(s, d[6], "investment", 2, annotator="human:alice")  # human beats the LLM judge
        s.add(Lens(development_id=d[1].id, audience="investment", headline="h", so_what="s", what_to_do="watch",
                   tickers=[{"symbol": "MSFT"}], action="watch", model="m", prompt_version="v"))
        for k, aud, action in ((6, "investment", "no_action"), (7, "ai_team", "hold"), (7, "investment", "watch")):
            s.add(Lens(development_id=d[k].id, audience=aud, headline="h", so_what="s", what_to_do="", action=action,
                       model="m", prompt_version="v"))
        s.commit()
    res = ev.eval_alerts()
    inv = res["investment"]
    assert (inv["count"], inv["by_severity"]) == (6, {"P1": 2, "P2": 3, "P3": 1})
    p = inv["pushed"]
    assert (p["count"], p["by_severity"]) == (5, {"P1": 2, "P2": 3})  # the digest-only P3 is not a push
    assert (p["per_week"], p["per_active_day"], p["per_push_day"], p["max_per_day"]) == (1.25, 0.83, 1.25, 2)
    assert p["precision"]["P1+P2"] == {"n": 5, "labelled": 4, "label_ge1": 0.75, "label_eq2": 0.5, "label_0": 1}
    assert p["precision"]["P1"] == {"n": 2, "labelled": 2, "label_ge1": 1.0, "label_eq2": 1.0, "label_0": 0}
    assert p["label2_recall"] == {"label2_developments": 4, "pushed": 2, "share": 0.5}
    assert (p["label_sources"], p["in_sample"]) == ({"llm": 3, "human": 1}, True)
    assert p["complete_actionable_share"] == 0.2  # only d1 has a what-to-do and tickers
    assert p["lens_dismissed"] == {"n": 1, "share": 0.2, "P1": 1}  # d6: pushed P1, own lens says no_action
    h = inv["holdout"]
    assert (h["from"], h["count"], h["per_week"], h["per_active_day"]) == ("2026-10-15 00:00:00", 3, 1.46, 0.75)
    assert h["precision"]["P1+P2"]["labelled"] == 2 and h["precision"]["P1+P2"]["label_eq2"] == 0.5
    assert h["label2_recall"] == {"label2_developments": 2, "pushed": 1, "share": 0.5}
    assert h["lens_dismissed"]["n"] == 1
    ai = res["ai_team"]["pushed"]
    assert (ai["count"], ai["precision"]["P1+P2"]["label_eq2"], ai["in_sample"]) == (1, None, False)
    assert ai["lens_dismissed"] == {"n": 1, "share": 1.0, "P1": 1}  # 'hold' is the AI team's dismissal
    assert "in-sample" in res["label_note"]
    headline = ev._headline("alerts", res)
    assert headline["investment"]["pushed"]["precision_label_eq2"] == 0.5
    assert headline["investment"]["holdout"]["count"] == 3


def test_run_eval_merges_subset_into_existing_report(world, tmp_path):
    out = tmp_path / "eval"
    (out / "report.json").write_text(json.dumps({"generated_at": "2026-11-01T00:00:00", "costs": {"llm_total_usd": 1.5}}))
    with world() as s:
        dev, _ = make_dev(s, "openai", "model_release", [("OpenAI ships a model", "c", datetime(2026, 10, 1))])
        add_alert(s, dev, "investment", "P1")
        s.commit()
    headline = ev.run_eval("alerts")
    report = json.loads((out / "report.json").read_text())
    assert report["costs"] == {"llm_total_usd": 1.5}  # kept from the earlier run
    assert report["suite_generated_at"]["costs"] == "2026-11-01T00:00:00"
    assert report["suite_generated_at"]["alerts"] == report["generated_at"]
    assert report["alerts"]["investment"]["holdout"] == {"status": "no scorer split yet (run fli train)"}
    assert headline["alerts"]["investment"]["pushed"]["count"] == 1
    md = (out / "report.md").read_text()
    assert "## Alerts" in md and "pushed (P1/P2)" in md and "## costs" in md
    write_events(out, [{"id": "e", "date": "2026-10-01", "title": "OpenAI ships", "audiences": ["investment"],
                        "match_all": [["model"]]}])
    full = ev.run_eval("all")
    report = json.loads((out / "report.json").read_text())
    assert set(ev.SUITES) <= set(report) and not [k for k in ev.SUITES if "error" in (report[k] or {})]
    assert report["known_events"]["recall"] == 1.0 and full["known_events"]["pushed_primary_audience"] == 1.0


# ------------------------------------------------------------------------------------ lens faithfulness
def checks(grounded: bool, unsupported: tuple = (), official: tuple = (), **extra) -> dict:
    """Lens.checks as fli.deliver.lens.faithfulness_checks stores them."""
    return {"version": 1, "checked_at": "2026-11-04T00:00:00", "unsupported_numbers": list(unsupported),
            "official_wording": list(official), "grounded": grounded, **extra}


def add_lens(s, dev: Development, audience: str, lens_checks: dict, headline: str = "h") -> Lens:
    lz = Lens(development_id=dev.id, audience=audience, headline=headline, so_what="s", what_to_do="w",
              action="watch" if audience == "investment" else "assess", checks=lens_checks, model="m", prompt_version="v")
    s.add(lz)
    return lz


def test_lens_faithfulness_per_audience_pushed_and_examples(world):
    with world() as s:
        d = {k: make_dev(s, "openai", "model_release", [(f"dev {k}", "c", datetime(2026, 9, k, 9))])[0]
             for k in range(1, 7)}
        d[6].status = "merged"  # a dead development's lens is not counted
        add_lens(s, d[1], "investment", checks(True))
        add_lens(s, d[2], "investment", checks(False, ({"field": "so_what", "text": "80%"},
                                                         {"field": "tickers", "symbol": "NVDA", "text": "1M"})))
        add_lens(s, d[3], "investment", checks(True, ({"field": "what_to_do", "text": "27"},),
                                                headline_replaced={"original": "Prices cut 80%", "unsupported_numbers": ["80%"]},
                                                tickers_dropped=["MSFT"]), headline="dev 3")
        add_lens(s, d[4], "investment", checks(False, official=({"field": "so_what", "issue": "no_official_source",
                                                                  "text": "Confirmed by official sources."},)))
        add_lens(s, d[5], "investment", {})  # never checked
        add_lens(s, d[6], "investment", checks(False, ({"field": "so_what", "text": "9"},)))
        add_lens(s, d[1], "ai_team", checks(True, ({"field": "what_to_do", "text": "100k"},)))
        add_lens(s, d[2], "ai_team", checks(False, ({"field": "so_what", "text": "165"},)))
        add_alert(s, d[1], "investment", "P1")
        add_alert(s, d[2], "investment", "P2")
        add_alert(s, d[3], "investment", "P3")  # digest-only: not behind a push
        s.add(Alert(development_id=d[2].id, audience="ai_team", severity="P2", title="t", body_md="-",
                    dedup_key=f"ai_team:{d[2].id}"))
        s.commit()
        ids = {k: v.id for k, v in d.items()}
    res = ev.eval_lens()
    inv = res["investment"]
    assert (inv["lenses"], inv["checked"], inv["grounded"], inv["grounded_share"]) == (5, 4, 2, 0.5)
    assert (inv["with_unsupported_numbers"], inv["unsupported_numbers"]) == (2, 3)
    assert inv["unsupported_by_field"] == {"so_what": 1, "tickers": 1, "what_to_do": 1}
    assert (inv["headlines_replaced"], inv["with_official_wording_flags"], inv["tickers_dropped"]) == (1, 1, 1)
    assert inv["official_wording_by_issue"] == {"no_official_source": 1}
    assert {k: inv["pushed"][k] for k in ("lenses", "checked", "grounded", "grounded_share")} == {
        "lenses": 2, "checked": 2, "grounded": 1, "grounded_share": 0.5}  # d1 (P1) and d2 (P2), not the P3
    ai = res["ai_team"]
    assert (ai["lenses"], ai["grounded_share"], ai["unsupported_numbers"]) == (2, 0.5, 2)
    assert (ai["pushed"]["checked"], ai["pushed"]["grounded_share"]) == (1, 0.0)
    assert (res["all"]["lenses"], res["all"]["checked"], res["all"]["grounded"]) == (7, 6, 3)
    assert res["checks_versions"] == {"1": 6}
    assert res["flagged"] == 4
    order = [(e["development_id"], e["audience"], e["alert"]) for e in res["examples"]]
    assert order == [(ids[2], "ai_team", "P2"), (ids[2], "investment", "P2"), (ids[3], "investment", None),
                     (ids[4], "investment", None)]  # pushed first
    assert res["examples"][1]["unsupported_numbers"] == ["so_what: 80%", "tickers: 1M"]
    assert res["examples"][2]["replaced_headline"] == "Prices cut 80%"
    assert res["examples"][3]["official_wording"] == ["no_official_source"]
    md = "\n".join(ev._lens_markdown(res))
    assert ("| investment | 5 (4) | 2 (50.0%) | 2 | so_what 1, tickers 1, what_to_do 1 | 1 | 1 (no_official_source 1) "
            "| 1 | 1/2 (50.0%) |") in md
    assert 'headline replaced (was "Prices cut 80%")' in md
    assert ev._headline("lens", res)["investment"]["pushed_grounded_share"] == 0.5


def test_lens_suite_reads_the_checks_the_lens_stage_writes(world):
    """Contract with fli.deliver.lens: a headline with an invented figure is replaced and recorded."""
    from fli.deliver.lens import LensJob, apply_checks

    job = LensJob(dev_id=1, audience="investment", user="GPT-6.1 Sol launched at $2 input and $10 output.",
                  symbols=[], title="OpenAI launches GPT-6.1 Sol", has_official=True,
                  official_text="GPT-6.1 Sol launched at $2 input and $10 output.", lab_names=("OpenAI",))
    data, stored = apply_checks({"headline": "OpenAI cuts token prices 80%", "so_what": "Priced at $2 input.",
                                 "what_to_do": "Re-run the cost model.", "tickers": []}, job)
    assert data["headline"] == "OpenAI launches GPT-6.1 Sol"
    with world() as s:
        dev, _ = make_dev(s, "openai", "model_release", [("OpenAI launches GPT-6.1 Sol", "c", datetime(2026, 9, 1))])
        add_lens(s, dev, "investment", stored, headline=data["headline"])
        s.commit()
    inv = ev.eval_lens()["investment"]
    assert (inv["checked"], inv["grounded"], inv["headlines_replaced"]) == (1, 1, 1)


# ------------------------------------------------------------------------------------ coverage by BIT holding
def add_doc(s, title: str, text: str, published: datetime, status: str = "extracted", source_id: str = "openai:news",
            lab: str | None = "openai", source_class: str = "official") -> Document:
    n = next(_seq)
    doc = Document(source_id=source_id, natural_key=f"d{n}", url=f"https://example.com/d{n}",
                   canonical_url=f"https://example.com/d{n}", title=title, published_at=published, content_text=text,
                   content_hash=f"dh{n}", lab_id=lab, source_class=source_class, status=status)
    s.add(doc)
    return doc


def add_link(s, dev: Development, ticker: str, method: str, confidence: float, rationale: str) -> None:
    s.add(ExposureLink(development_id=dev.id, ticker=ticker, method=method, confidence=confidence, rationale=rationale))


def test_listed_tickers_reads_the_alert_renderer():
    """Contract with fli.deliver.alerts: the Tickers line of a real alert body (lens tickers, else exposure
    links with lab-level relations marked)."""
    from fli.deliver.alerts import render_alert
    from fli.deliver.materiality import DevContext, Evaluation, load_policy

    dev = Development(id=1, title="OpenAI signs a compute deal", event_type="compute_deal", lab_id="openai",
                      first_seen_at=datetime(2026, 9, 1), last_seen_at=datetime(2026, 9, 1))
    lens = Lens(development_id=1, audience="investment", headline="h", so_what="s", what_to_do="w", action="watch",
                tickers=[{"symbol": "AMZN", "direction": "positive", "why": "Bedrock demand; NVDA (supplier) gains"},
                         {"symbol": "BABA", "direction": "mixed", "why": "competitor"}], model="m", prompt_version="v")
    ev_ = Evaluation(audience="investment", severity="P2", rule_severity="P2", fired=[], guards=[], unconfirmed=False,
                     facts={})
    policy = load_policy()
    _, body = render_alert(DevContext(dev=dev, lenses={"investment": lens}, holdings={"AMZN": 9.6, "NVDA": 8.0}),
                           ev_, "P2", [], policy)
    assert ev.listed_tickers(body) == {"AMZN": "event", "BABA": "event"}  # 'NVDA (supplier)' is inside a why
    links = [ExposureLink(development_id=1, ticker="NVDA", method="lab", confidence=0.6),
             ExposureLink(development_id=1, ticker="MU", method="rule", direction="unclear", confidence=0.35)]
    _, body = render_alert(DevContext(dev=dev, exposures=links, holdings={"NVDA": 8.0, "MU": 6.5}), ev_, "P2", [], policy)
    assert ev.listed_tickers(body) == {"NVDA": "lab", "MU": "event"}
    assert ev.listed_tickers("- **What to do:** watch") == {} and ev.listed_tickers(None) == {}


def test_named_link_marker_matches_the_exposure_rules():
    from fli.analyze.exposure import TickerInfo, load_themes, propose_links

    cands = propose_links("compute_deal", None, "A capacity deal with Micron Technology",
                          load_themes(), {"MU": TickerInfo("MU", "Micron Technology", ("memory_storage",), 6.5)})
    assert any(ev.NAMED_LINK in why for c in cands for why in c.rationale)


def test_holdings_coverage_documents_links_and_pushed_alerts(world, tmp_path, monkeypatch):
    import fli.deliver.materiality as materiality

    monkeypatch.setattr(materiality, "load_policy", lambda: SimpleNamespace(exposure_min_confidence=0.6))
    cfg = tmp_path / "exposure.yaml"
    cfg.write_text(yaml.safe_dump({
        "bit_holdings": [{"symbol": s, "weight": w} for s, w in (("AMZN", 9.6), ("MU", 6.5), ("IREN", 9.2), ("NVDA", 8.0))],
        "tickers": [{"symbol": "AMZN", "company": "Amazon"}, {"symbol": "IREN", "company": "IREN"},
                    {"symbol": "MU", "company": "Micron Technology"}, {"symbol": "NVDA", "company": "NVIDIA"}]}))
    with world() as s:
        s.add_all([Ticker(symbol=sym, company=sym) for sym in ("AMZN", "IREN", "MU", "NVDA")])  # exposure_link FK
        add_doc(s, "Anthropic and Amazon expand their deal", "Capacity.", datetime(2026, 9, 2))
        add_doc(s, "Amazon invests again", "More.", datetime(2026, 9, 3), status="skipped")
        add_doc(s, "Amazon invests again", "More.", datetime(2026, 9, 3), status="duplicate")  # not counted twice
        add_doc(s, "Micron Technology HBM supply", "Old news.", datetime(2026, 7, 1))  # before the window
        add_doc(s, "IRENE and the amazon rainforest", "Micronesia; feature sizes in microns.", datetime(2026, 9, 4))
        add_doc(s, "Runs on NVIDIA GB300", "Press story.", datetime(2026, 9, 5), status="relevant", source_class="press")
        add_doc(s, "Benchmarks", "Trained on NVIDIA hardware.", datetime(2026, 9, 6), status="skipped")
        t = datetime(2026, 9, 2)
        a, _ = make_dev(s, "openai", "compute_deal", [("Deal A", "c", t)])
        b, _ = make_dev(s, "openai", "model_release", [("Model B", "c", t)])
        dead, _ = make_dev(s, "openai", "compute_deal", [("Dead", "c", t)])
        dead.status = "merged"
        old, _ = make_dev(s, "openai", "model_release", [("Old", "c", datetime(2026, 7, 15))])
        add_link(s, a, "AMZN", "rule", 0.75, "named in development text: 'Amazon'")
        add_link(s, a, "NVDA", "lab", 0.6, "OpenAI public exposure: chip_supplier")
        add_link(s, b, "MU", "rule", 0.35, "theme 'Memory (HBM) and storage': model_release + keywords [hbm]")
        add_link(s, b, "AMZN", "rule", 0.65, "OpenAI public exposure: compute_supplier; theme 'Hyperscalers': ...")
        add_link(s, dead, "AMZN", "rule", 0.75, "named in development text: 'Amazon'")
        add_link(s, old, "MU", "rule", 0.65, "named in development text: 'Micron'")
        alert = add_alert(s, a, "investment", "P1")
        alert.body_md = ("**h**\n\n- **So what:** s\n- **Tickers:** AMZN (positive, BIT 9.6%): Bedrock; distribution; "
                         "NVDA (lab-level link, BIT 8.0%)\n\n**Why flagged**")
        add_alert(s, b, "investment", "P2").body_md = "- **Tickers:** MU (unclear, BIT 6.5%): thematic; AMZN (mixed, BIT 9.6%)"
        c, _ = make_dev(s, "openai", "compute_deal", [("Deal C", "c", t)])
        add_alert(s, c, "investment", "P3").body_md = "- **Tickers:** IREN (positive, BIT 9.2%)"  # digest-only
        add_alert(s, dead, "investment", "P2").body_md = "- **Tickers:** AMZN (positive, BIT 9.6%)"
        s.add(Alert(development_id=a.id, audience="ai_team", severity="P1", title="t", body_md="-", dedup_key="ai:a"))
        s.commit()
    res = ev.eval_holdings(cfg)
    rows = {r["symbol"]: r for r in res["holdings"]}
    assert [r["symbol"] for r in res["holdings"]] == ["AMZN", "IREN", "NVDA", "MU"]  # by fund weight
    assert "Micron Technology" in rows["MU"]["names"]
    pick = ("documents", "documents_relevant", "developments_event_specific", "developments_named",
            "developments_at_policy_floor", "developments_lab_only", "pushed_alerts", "pushed_p1", "pushed_lab_level",
            "coverage")
    assert {k: rows["AMZN"][k] for k in pick} == dict(zip(pick, (2, 1, 2, 1, 2, 0, 2, 1, 0, "direct"), strict=True))
    assert {k: rows["IREN"][k] for k in pick} == dict(zip(pick, (0, 0, 0, 0, 0, 0, 0, 0, 0, "none"), strict=True))
    assert {k: rows["NVDA"][k] for k in pick} == dict(zip(pick, (2, 1, 0, 0, 0, 1, 1, 1, 1, "direct"), strict=True))
    assert {k: rows["MU"][k] for k in pick} == dict(zip(pick, (0, 0, 1, 0, 0, 0, 1, 0, 0, "indirect"), strict=True))
    assert rows["NVDA"]["documents_by_class"] == {"press": 1, "official": 1}
    # 5 added documents in the window (one duplicate, one older left out) + the 4 make_dev documents in it
    assert (res["window"], res["weight_total"]) == ({"from": "2026-08-01", "documents": 9}, 33.3)
    assert (res["no_direct_coverage"], res["weight_no_direct_coverage"]) == (["IREN", "MU"], 15.7)
    assert (res["never_pushed"], res["pushed_without_naming_document"]) == (["IREN"], {"MU": 1})
    md = "\n".join(ev._holdings_markdown(res))
    assert "| MU (Micron Technology) | 6.5% | 0 (0) | 1 (0 / 0) | 0 | 1 (0) | indirect |" in md
    assert "No collected document names IREN (9.2%), MU (6.5%): 15.7 of the 33.3 percentage points" in md
    assert ev._headline("holdings", res)["by_holding"]["MU"] == {"documents": 0, "event_specific": 1, "named": 0,
                                                                 "pushed": 1}


def test_shipped_holdings_are_the_exposure_config_holdings():
    syms = [sym for sym, _, _ in ev.bit_holdings()]
    cfg = yaml.safe_load((ev.CONFIG_DIR / "exposure.yaml").read_text(encoding="utf-8"))
    assert sorted(syms) == sorted(h["symbol"] for h in cfg["bit_holdings"]) and {"IREN", "MU"} <= set(syms)


# ------------------------------------------------------------------------------------ digest quality
def add_report(s, audience: str, week: datetime, model: str | None, items: list[dict],
               created: datetime | None = None) -> None:
    s.add(Report(audience=audience, period_start=week, period_end=week + timedelta(days=7), title="t", body_md="b",
                 items=items, model=model, created_at=created or datetime(2026, 10, 5)))


def item(n: int, cited: bool = True, also: bool = False) -> dict:
    return {"development_id": n, "rank": n, "p_material": 0.5, "cited": cited, "also_this_week": also}


def test_digest_quality_writer_shares_coverage_and_missing_weeks(world, monkeypatch):
    monkeypatch.setenv("BACKFILL_START", "2026-09-07")
    fli_config.get_settings.cache_clear()
    w = {d: datetime(2026, 9, d) for d in (7, 14, 21, 28)}
    with world() as s:
        add_report(s, "investment", w[7], "claude-opus-5-5", [item(1), item(2), item(3), item(4, also=True)])
        add_report(s, "investment", w[14], "fallback", [item(5), item(6)])
        add_report(s, "investment", w[28], None, [])  # nothing reached the bar
        add_report(s, "ai_team", w[7], "claude-opus-5-5", [item(1), item(2), *(item(n, also=True) for n in (3, 4, 5))],
                   created=datetime(2026, 9, 10))  # a mid-week preview
        add_report(s, "ai_team", w[14], "claude-opus-5-5", [{"development_id": 9, "rank": 1}])  # legacy row
        s.commit()
    res = ev.eval_digests(now=datetime(2026, 10, 6))
    inv = res["investment"]
    assert inv["by_writer"] == {"model": 1, "fallback": 1, "empty": 1, "unknown": 0}
    assert (inv["model_share"], inv["models"], inv["fallback_weeks"]) == (0.5, {"claude-opus-5-5": 1}, ["2026-09-14"])
    assert (inv["items"], inv["cited"], inv["cited_share"], inv["written"], inv["written_share"]) == (6, 6, 1.0, 5, 0.833)
    assert (inv["also_this_week"], inv["reports_with_also_this_week"], inv["missing_weeks"]) == (1, 1, ["2026-09-21"])
    assert (inv["model_written"]["written_share"], inv["model_written"]["min_report_written_share"],
            inv["model_written"]["reports_below_min_coverage"]) == (0.75, 0.75, 0)
    ai = res["ai_team"]
    assert (ai["by_writer"]["model"], ai["model_share"], ai["preview_reports"]) == (2, 1.0, 1)
    assert (ai["items"], ai["flags_missing"], ai["cited_share"], ai["written_share"]) == (6, 1, 1.0, 0.4)
    assert ai["missing_weeks"] == ["2026-09-21", "2026-09-28"]
    assert (ai["model_written"]["min_report_written_share"], ai["model_written"]["reports_below_min_coverage"]) == (0.4, 1)
    assert [r["writer"] for r in ai["weeks"]] == ["model", "model"]
    md = "\n".join(ev._digests_markdown(res))
    assert "| investment | 3 | 1 / 1 / 1 | 6 | 6 (100.0%) | 5 (83.3%) | 1 (1) | 75.0% | 2026-09-14 | 2026-09-21 |" in md
    assert "ai_team: 1 items without flags" in md and "generated before their week ended" in md
    assert ev._headline("digests", res)["ai_team"]["model_written_share"] == 0.4


# ------------------------------------------------------------------------------------ source coverage
def test_source_coverage_classes_labs_and_silent_sources(world):
    ran = datetime(2026, 9, 20)
    with world() as s:
        s.add(Lab(id="ssi", name="Safe Superintelligence", kind="alumni_startup", tier=3))
        s.add_all([
            Source(id="openai:x", lab_id="openai", adapter="x", kind="x_account", source_class="official"),
            Source(id="x:person", adapter="x", kind="x_account", source_class="personal"),
            Source(id="anthropic:jobs", lab_id="anthropic", adapter="jobs", kind="job_board", source_class="official",
                   enabled=False),
            Source(id="press:gdelt", adapter="gdelt", kind="press", source_class="press", last_run_at=ran,
                   consecutive_failures=2, last_error="HTTP 429"),
            Source(id="sec:edgar", adapter="edgar", kind="sec", source_class="filing", last_run_at=ran,
                   last_success_at=ran),
            Source(id="press:wire", adapter="rss", kind="press", source_class="press", last_run_at=ran,
                   last_success_at=ran),
        ])
        s.flush()
        add_doc(s, "a", "x", datetime(2026, 9, 1))
        add_doc(s, "b", "x", datetime(2026, 9, 9), status="skipped", lab=None)  # the source's lab
        add_doc(s, "c", "x", datetime(2026, 7, 20))  # before the window
        add_doc(s, "d", "x", datetime(2026, 9, 2), status="duplicate", source_id="anthropic:news", lab="anthropic")
        add_doc(s, "e", "x", datetime(2026, 9, 16), status="relevant", source_id="press:wire", lab=None,
                source_class="press")
        s.commit()
    res = ev.eval_sources()
    assert (res["documents"], res["before_window"], res["duplicates"]) == (5, 1, 1)
    assert (res["sources"], res["enabled"], res["with_documents"], res["silent"]) == (8, 7, 3, 5)
    assert res["silent_by_reason"] == {"never_run": 2, "disabled": 1, "failing": 1, "no_items": 1}
    assert res["window"] == {"from": "2026-08-01", "to": "2026-09-16 00:00:00", "weeks": 8}
    assert res["by_source_class"] == {
        "official": {"sources": 4, "with_documents": 2, "documents": 4, "relevant": 2},
        "press": {"sources": 2, "with_documents": 1, "documents": 1, "relevant": 1},
        "filing": {"sources": 1, "with_documents": 0, "documents": 0, "relevant": 0},
        "personal": {"sources": 1, "with_documents": 0, "documents": 0, "relevant": 0}}
    assert res["by_kind"]["x_account"] == {"sources": 2, "with_documents": 0, "documents": 0, "relevant": 0}
    assert list(res["by_lab"]) == ["anthropic", "openai", "ssi", "unattributed"]  # tier, then id
    pick = ("sources", "silent_sources", "documents", "relevant", "active_weeks", "last_document")
    assert {k: res["by_lab"]["openai"][k] for k in pick} == dict(zip(pick, (2, 1, 3, 2, 2, "2026-09-09 00:00:00"),
                                                                     strict=True))
    assert {k: res["by_lab"]["unattributed"][k] for k in pick} == dict(zip(pick, (4, 3, 1, 1, 1, "2026-09-16 00:00:00"),
                                                                           strict=True))
    assert res["labs_without_documents"] == ["ssi"]
    reasons = {r["id"]: r["reason"] for r in res["silent_sources"]}
    assert reasons == {"anthropic:jobs": "disabled", "openai:x": "never_run", "press:gdelt": "failing",
                       "sec:edgar": "no_items", "x:person": "never_run"}
    md = "\n".join(ev._sources_markdown(res))
    assert "8 registered sources (7 enabled): 3 produced at least one document, 5 never did" in md
    assert "| x_account | official | never_run | 1 | openai:x |" in md
    assert "Tracked labs without a single document: ssi." in md
    assert ev._headline("sources", res)["documents_by_class"] == {"official": 4, "press": 1, "filing": 0, "personal": 0}


# ------------------------------------------------------------------------------------ report writing
def test_run_eval_writes_to_out_dir_override_and_renders_new_sections(world, tmp_path, monkeypatch):
    copy_dir = tmp_path / "copy"
    monkeypatch.setenv(ev.OUT_DIR_ENV, str(copy_dir))
    with world() as s:
        dev, _ = make_dev(s, "openai", "model_release", [("OpenAI ships a model", "c", datetime(2026, 10, 1))])
        add_lens(s, dev, "investment", checks(True))
        add_doc(s, "Amazon partnership", "x", datetime(2026, 10, 1), status="skipped")
        s.commit()
    headline = ev.run_eval("noise,lens,holdings,digests,sources")
    assert (copy_dir / "report.json").exists() and (copy_dir / "noise_audit_sample.csv").exists()
    assert not (ev.EVAL_DIR / "report.json").exists() and not (ev.EVAL_DIR / "noise_audit_sample.csv").exists()
    md = (copy_dir / "report.md").read_text()
    for section in ("## Lens faithfulness", "## Coverage by BIT holding", "## Digest quality", "## Source coverage"):
        assert section in md
    assert headline["lens"]["investment"]["grounded_share"] == 1.0
    assert headline["holdings"]["by_holding"]["AMZN"]["documents"] == 1  # the shipped exposure config
    assert headline["noise"]["audit_sample"] == str(copy_dir / "noise_audit_sample.csv")


def test_markdown_survives_a_stored_section_of_another_layout():
    md = ev._markdown({"generated_at": "2026-11-05T00:00:00", "holdings": {"holdings": [{"symbol": "AMZN"}], "n": 1},
                       "lens": {"error": "RuntimeError: boom"}})
    assert '## holdings\n\n```json\n{\n  "n": 1\n}' in md and "RuntimeError: boom" in md
