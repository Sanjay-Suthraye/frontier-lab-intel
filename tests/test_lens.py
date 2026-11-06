"""Audience lenses: selection, candidate-restricted tickers, storage and idempotency (fake gateway)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from fli.db.models import (
    Alert,
    Development,
    Document,
    ExposureLink,
    Insight,
    Lab,
    Lens,
    LLMCall,
    Score,
    Source,
    Ticker,
)
from fli.deliver import lens as lens_mod
from fli.deliver.lens import (
    LensJob,
    apply_checks,
    ensure_lens,
    faithfulness_checks,
    figures,
    is_grounded,
    official_assertions,
    recheck_lenses,
    run_lens,
    select_for_lens,
    support_values,
)
from fli.extract.schemas import clean_lens
from fli.extract.verify import verify_quotes
from fli.llm.client import BudgetExceeded, LLMResult

WEEK_A = datetime(2026, 10, 5)  # a Monday
WEEK_B = datetime(2026, 10, 12)
QUOTE = "Anthropic will deploy one million Trainium chips on AWS"
TEXT = f"{QUOTE} over the next two years, the company said."


class FakeGateway:
    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    async def structured(self, **kw) -> LLMResult:
        self.calls.append(kw)
        ans = self.responder(kw)
        return ans if isinstance(ans, LLMResult) else LLMResult(ans, "ok", 0.02, kw["model"],
                                                                 kw["prompt"].version_for(kw["schema"]))

    async def structured_many(self, calls: list[dict]) -> list[LLMResult]:
        return [await self.structured(**kw) for kw in calls]


INVESTMENT = {"headline": "AWS Trainium win deepens Amazon's Anthropic tie",
              "so_what": "Confirms AMZN's Trainium ramp; a large Anthropic commitment supports AWS growth.",
              "tickers": [{"symbol": "AMZN", "direction": "positive", "why": "Trainium and AWS capacity demand"},
                          {"symbol": "MU", "direction": "positive", "why": "not a candidate"},
                          {"symbol": "NVDA", "direction": "negative", "why": "share shift to custom silicon"}],
              "direction": "positive", "horizon": "quarters", "action": "update_thesis",
              "what_to_do": "Ask AMZN IR about Trainium 3 capacity for 2027.", "confidence": 0.7}
AI_TEAM = {"headline": "Trainium capacity matters little for us", "so_what": "No change to our stack.",
           "action": "hold", "artifacts_to_try": ["Claude on Bedrock", "Neuron SDK"],
           "what_to_do": "Nothing now.", "confidence": 0.6}


def _answer(kw: dict) -> dict:
    return INVESTMENT if kw["prompt"].name == "lens_investment" else AI_TEAM


def seed(db) -> None:
    with db() as s:
        s.add(Lab(id="anthropic", name="Anthropic", kind="frontier_lab", tier=1,
                  public_exposure=[{"ticker": "AMZN", "relation": "investor_and_cloud_partner", "note": "Trainium"},
                                   {"ticker": "ZZZZ", "relation": "not in the ticker table"}]))
        s.add_all([Ticker(symbol="AMZN", company="Amazon", sector="Internet & cloud", bit_weight=9.6,
                          themes=["hyperscaler_partners"]),
                   Ticker(symbol="NVDA", company="NVIDIA", sector="Semiconductors", bit_weight=8.0,
                          themes=["accelerators"]),
                   Ticker(symbol="MU", company="Micron Technology", sector="Memory", bit_weight=6.5)])
        s.add(Source(id="anthropic:news", lab_id="anthropic", adapter="rss", kind="official_blog",
                     source_class="official"))
        s.commit()


def add_dev(db, first_seen: datetime, scores: list[float], audience: str = "investment") -> int:
    """A development with one insight and `scores` written oldest first (the last one is latest)."""
    with db() as s:
        dev = Development(title="Anthropic expands AWS Trainium deal", event_type="compute_deal",
                          lab_id="anthropic", first_seen_at=first_seen, last_seen_at=first_seen,
                          insight_count=1, source_count=1, source_classes=["official"], has_official=True,
                          has_verified=True)
        s.add(dev)
        s.flush()
        doc = Document(source_id="anthropic:news", natural_key=f"d{dev.id}", url=f"https://anthropic.com/n/{dev.id}",
                       canonical_url=f"https://anthropic.com/n/{dev.id}", title="Expanding with AWS", content_text=TEXT,
                       content_hash=f"h{dev.id}", source_class="official", lab_id="anthropic", status="extracted")
        s.add(doc)
        s.flush()
        quotes = [QUOTE, "a fabricated sentence about one gigawatt in Texas"]
        s.add(Insight(document_id=doc.id, development_id=dev.id, lab_id="anthropic", event_type="compute_deal",
                      title="Anthropic to deploy 1M Trainium chips", claim="Anthropic says it will deploy 1M Trainium chips.",
                      evidence_quotes=quotes, verification=verify_quotes(quotes, TEXT), quotes_verified=False,
                      artifacts=["Trainium"], is_official=True, published_at=first_seen, prompt_version="t", model="m"))
        for i, p in enumerate(scores):
            s.add(Score(development_id=dev.id, audience=audience, p_material=p, model_version=f"v{i}",
                        created_at=first_seen + timedelta(hours=i + 1)))
        s.commit()
        return dev.id


def test_selection_uses_latest_score_threshold_and_weekly_top(db):
    seed(db)
    high = add_dev(db, WEEK_A, [0.8])
    demoted = add_dev(db, WEEK_A, [0.9, 0.1])  # newest score wins
    low = add_dev(db, WEEK_A + timedelta(days=2), [0.2])
    alone = add_dev(db, WEEK_B, [0.05])  # tops its own week
    done = add_dev(db, WEEK_A, [0.6])  # ranks (rank ignores lenses) but already has a lens
    with db() as s:
        s.add(Lens(development_id=done, audience="investment", headline="h", so_what="s", what_to_do="w",
                   model="m", prompt_version="p"))
        s.commit()
        assert select_for_lens(s, "investment", top_n=1) == [high, alone]
        assert select_for_lens(s, "investment") == [high, low, demoted, alone]
        assert select_for_lens(s, "ai_team") == []  # no ai_team scores


def test_investment_lens_restricts_tickers_to_candidates(db):
    seed(db)
    dev_id = add_dev(db, WEEK_A, [0.7])
    with db() as s:
        s.add(ExposureLink(development_id=dev_id, ticker="NVDA", theme="accelerators", direction="negative",
                           rationale="custom silicon competes with GPUs", method="rule", confidence=0.6))
        s.commit()
    gw = FakeGateway(_answer)
    out = run_lens(gateway=gw)
    assert out["investment"]["created"] == 1 and out["ai_team"]["todo"] == 0
    call = gw.calls[0]
    enum = call["schema"]["properties"]["tickers"]["items"]["properties"]["symbol"]["enum"]
    assert enum == ["AMZN", "NVDA"]  # MU is a holding but not a candidate; ZZZZ is not a ticker
    assert "allowed symbols: AMZN, NVDA" in call["user"] and "lab relation: investor_and_cloud_partner" in call["user"]
    assert QUOTE in call["user"] and "fabricated sentence" not in call["user"]  # only verified quotes are cited
    assert "AMZN (Amazon): 9.6%" in call["prompt"].system and "{{" not in call["prompt"].system
    assert call["model"] == "claude-sonnet-5-5" and call["effort"] == "medium"
    with db() as s:
        row = s.query(Lens).one()
    assert [t["symbol"] for t in row.tickers] == ["AMZN", "NVDA"]
    assert (row.action, row.direction, row.horizon) == ("update_thesis", "positive", "quarters")
    assert row.prompt_version.startswith("lens_investment@v2+") and row.model == "claude-sonnet-5-5"
    assert row.checks["grounded"] and row.checks["version"] == lens_mod.CHECKS_VERSION
    assert run_lens(gateway=FakeGateway(_answer))["investment"]["todo"] == 0  # idempotent


def test_ai_team_lens_stores_artifacts_and_has_no_tickers(db):
    seed(db)
    dev_id = add_dev(db, WEEK_A, [0.5], audience="ai_team")
    assert ensure_lens([dev_id, dev_id, 999], "ai_team", gateway=FakeGateway(_answer)) == 1
    with db() as s:
        row = s.query(Lens).one()
    assert (row.audience, row.tickers, row.direction, row.action) == ("ai_team", [], None, "hold")
    assert row.what_to_do == "Nothing now."
    assert row.artifacts == ["Claude on Bedrock", "Neuron SDK"]
    assert row.prompt_version.startswith("lens_ai_team@v2+") and row.checks["grounded"] is True
    again = FakeGateway(_answer)
    assert ensure_lens([dev_id], "ai_team", gateway=again) == 0 and again.calls == []


def test_ensure_lens_survives_budget_exhaustion(db):
    seed(db)
    dev_id = add_dev(db, WEEK_A, [0.9])

    class Broke(FakeGateway):
        async def structured_many(self, calls):
            raise BudgetExceeded("LLM daily budget reached")

    assert ensure_lens([dev_id], "investment", gateway=Broke(_answer)) == 0


def test_repeatedly_failing_lens_is_dead_lettered(db):
    seed(db)
    dev_id = add_dev(db, WEEK_A, [0.9])
    with db() as s:
        for _ in range(lens_mod.MAX_FAILURES):
            s.add(LLMCall(purpose="lens", model="claude-sonnet-5-5", prompt_version="p", status="refusal",
                          ref={"development_id": dev_id, "audience": "investment"}))
        s.commit()
        assert select_for_lens(s, "investment") == []
    failing = FakeGateway(lambda kw: LLMResult(None, "error", 0.0, kw["model"], "v", "boom"))
    assert ensure_lens([dev_id], "investment", gateway=failing) == 0  # on-demand still tries; nothing stored


def test_union_symbol_enum_is_shared_but_tickers_are_filtered_per_development(db):
    seed(db)
    with db() as s:
        s.add(Lab(id="openai", name="OpenAI", kind="frontier_lab", tier=1,
                  public_exposure=[{"ticker": "MU", "relation": "memory supplier (illustrative)"}]))
        s.commit()
    first = add_dev(db, WEEK_A, [0.9])  # anthropic: candidates AMZN
    second = add_dev(db, WEEK_A, [0.8])
    with db() as s:
        s.get(Development, second).lab_id = "openai"  # openai: candidates MU
        s.commit()
    gw = FakeGateway(_answer)  # always answers AMZN, MU and NVDA
    run_lens(gateway=gw)
    enums = {tuple(c["schema"]["properties"]["tickers"]["items"]["properties"]["symbol"]["enum"]) for c in gw.calls}
    assert enums == {("AMZN", "MU")}
    with db() as s:
        stored = {row.development_id: [t["symbol"] for t in row.tickers] for row in s.query(Lens)}
    assert stored == {first: ["AMZN"], second: ["MU"]}


# ------------------------------------------------------------------------------- abstention
def test_no_action_lens_keeps_no_tickers_or_direction():
    raw = {**INVESTMENT, "action": "no_action"}
    out = clean_lens(raw, "investment", ["AMZN", "NVDA"])
    assert (out["tickers"], out["direction"], out["dropped_tickers"]) == ([], "unclear", ["AMZN", "NVDA"])
    kept = clean_lens(INVESTMENT, "investment", ["AMZN", "NVDA"])
    assert [t["symbol"] for t in kept["tickers"]] == ["AMZN", "NVDA"] and kept["dropped_tickers"] == []


def test_stored_no_action_lens_records_dropped_tickers(db):
    seed(db)
    add_dev(db, WEEK_A, [0.7])
    out = run_lens(gateway=FakeGateway(lambda kw: {**_answer(kw), "action": "no_action"}))
    assert out["investment"]["created"] == 1
    with db() as s:
        row = s.query(Lens).one()
    assert (row.action, row.tickers, row.direction) == ("no_action", [], "unclear")
    assert row.checks["tickers_dropped"] == ["AMZN"]  # the only candidate; MU and NVDA were never allowed


# ------------------------------------------------------------------------------- figures and grounding
def test_figures_parse_units_and_skip_identifiers():
    text = ("GPT-6.1 Sol at $2/$10 per 1M tokens is 80% cheaper; p95 latency, Q3 results, 3D, H100, 4o, "
            "a 1.05M-token window, $11.6 billion, 3x faster, ~50B active, a 6GW deal")
    got = [(f.text, f.value, f.approx) for f in figures(text)]
    assert got == [("$2", Decimal(2), False), ("$10", Decimal(10), False), ("80%", Decimal(80), False),
                   ("1.05M", Decimal(1_050_000), False), ("$11.6 billion", Decimal(11_600_000_000), False),
                   ("3x", Decimal(3), False), ("50B", Decimal(50_000_000_000), True), ("6", Decimal(6), False)]


def test_grounding_allows_rounding_multipliers_and_number_words():
    support = support_values("a 1,050,000-token context window; 16.7x faster; trailing average 10.2; "
                             "best on five of 12 benchmarks; 49 billion active parameters; one million chips")
    for text in ("1.05M", "1M", "16x", "17x", "10", "5", "~50B", "1 million", "12"):
        assert is_grounded(figures(text)[0], support), text
    for text in ("80%", "2M", "15x", "50B", "11", "$3"):
        assert not is_grounded(figures(text)[0], support), text


def test_official_assertions_skip_negated_and_conditional_uses():
    text = ("Confirmed by official sources: a 1M context. No official source confirms pricing. "
            "Single listing, not an official release. Revisit if OpenAI confirms the price. "
            "OpenAI confirmed the deal. Confirmed by official release notes. The cap rises to 128 MB.")
    got = official_assertions(text, ("OpenAI",))
    assert [s for s, _ in got] == ["Confirmed by official sources: a 1M context.", "OpenAI confirmed the deal.",
                                   "Confirmed by official release notes."]
    assert got[2][1].endswith("The cap rises to 128 MB.")  # a bare lead-in covers the next sentence


EVIDENCE = ("<evidence>\n[1] official | quotes verified | 2026-10-13 | https://blog.example/argon\n"
            "claim: Google says Gemini 4 Argon is generally available.\n\n"
            "[2] market_data | quotes verified | 2026-10-13 | https://openrouter.ai/google/gemini-4-argon\n"
            "claim: OpenRouter lists Gemini 4 Argon with a 1,048,576-token context window at $2.00 input.\n</evidence>")
OFFICIAL_TEXT = "2026-10-13\nclaim: Google says Gemini 4 Argon is generally available."


def _job(has_official: bool = True, audience: str = "ai_team") -> LensJob:
    return LensJob(1, audience, "title: Gemini 4 Argon is generally available\n" + EVIDENCE, [],
                   title="Gemini 4 Argon is generally available", has_official=has_official,
                   official_text=OFFICIAL_TEXT if has_official else "", lab_names=("Google DeepMind", "Google"))


def test_official_wording_needs_official_evidence_for_its_figures():
    fields = {"headline": "Gemini 4 Argon GA at $2.00 input", "what_to_do": "Trial it on one pipeline.",
              "so_what": "Confirmed by official sources: Argon has a 1M-token context."}
    checks = faithfulness_checks(fields, _job())
    assert checks["official_wording"] == [{
        "field": "so_what", "issue": "figures_not_in_official_evidence", "figures": ["1M"],
        "text": "Confirmed by official sources: Argon has a 1M-token context."}]
    assert checks["unsupported_numbers"] == [] and checks["grounded"] is False
    joint = faithfulness_checks({**fields, "so_what": "Official notes and the OpenRouter listing give a 1M context."},
                                _job())
    assert joint["official_wording"] == [] and joint["grounded"] is True
    no_official = faithfulness_checks({**fields, "so_what": "Confirmed by official sources: it is GA."},
                                      _job(has_official=False))
    assert [o["issue"] for o in no_official["official_wording"]] == ["no_official_source"]
    honest = faithfulness_checks({**fields, "so_what": "A single listing, not an official release."},
                                 _job(has_official=False))
    assert honest["official_wording"] == [] and honest["grounded"] is True


def test_unsupported_headline_figure_is_replaced_and_recorded():
    data = {"headline": "Google cuts Argon prices 80%", "so_what": "Listed at $2.00 input.",
            "what_to_do": "Benchmark on 200 filings.",
            "tickers": [{"symbol": "GOOGL", "direction": "positive", "why": "Gains 30% share"}]}
    out, checks = apply_checks(data, _job(audience="investment"))
    assert out["headline"] == "Gemini 4 Argon is generally available"
    assert checks["headline_replaced"] == {"original": "Google cuts Argon prices 80%", "unsupported_numbers": ["80%"]}
    assert checks["unsupported_numbers"] == [{"field": "what_to_do", "text": "200"},
                                             {"field": "tickers", "symbol": "GOOGL", "text": "30%"}]
    assert checks["grounded"] is False  # the ticker rationale invents a figure
    ok, ok_checks = apply_checks({**data, "tickers": []}, _job(audience="investment"))
    assert ok_checks["grounded"] is True  # a figure in what_to_do is reported but does not gate
    official_headline, c2 = apply_checks({**data, "headline": "Officially confirmed: Argon GA"}, _job(False))
    assert official_headline["headline"] == "Gemini 4 Argon is generally available"
    assert c2["headline_replaced"]["official_wording"] == ["no_official_source"]


def test_lens_with_invented_headline_figure_is_stored_with_safe_headline(db):
    seed(db)
    add_dev(db, WEEK_A, [0.7])
    invented = {**INVESTMENT, "headline": "Anthropic cuts Trainium prices 80%"}
    out = run_lens(gateway=FakeGateway(lambda kw: invented if kw["prompt"].name == "lens_investment" else AI_TEAM))
    assert out["investment"]["created"] == 1 and out["investment"]["headline_replaced"] == 1
    with db() as s:
        row = s.query(Lens).one()
    assert row.headline == "Anthropic expands AWS Trainium deal"  # the development title
    assert row.checks["headline_replaced"]["original"] == "Anthropic cuts Trainium prices 80%"
    assert row.checks["grounded"] is True


# ------------------------------------------------------------------------------- recheck (no LLM)
def test_recheck_lenses_fixes_stored_rows_and_alerts_idempotently(db):
    seed(db)
    quiet, loud = add_dev(db, WEEK_A, [0.4]), add_dev(db, WEEK_A, [0.9, 0.8])
    holding = [{"symbol": "AMZN", "direction": "positive", "why": "Trainium demand"}]
    with db() as s:
        s.add_all([
            Lens(development_id=quiet, audience="investment", headline="No listed read-through", so_what="Abstain.",
                 what_to_do="Nothing.", tickers=holding, direction="positive", action="no_action",
                 model="m", prompt_version="lens_investment@v1+x"),
            Lens(development_id=loud, audience="investment", headline="Anthropic cuts Trainium prices 80%",
                 so_what="The system's 0.90 estimate looks right.", what_to_do="Watch AWS.", tickers=holding,
                 direction="positive", action="watch", model="m", prompt_version="lens_investment@v1+x"),
            Lens(development_id=loud, audience="ai_team", headline="Trainium matters little for us",
                 so_what="No change.", what_to_do="Nothing.", action="hold", model="m",
                 prompt_version="lens_ai_team@v1+x"),
            Alert(development_id=loud, audience="investment", severity="P1", rules=["r"], dedup_key="k",
                  title="Anthropic cuts Trainium prices 80%",
                  body_md="**Anthropic cuts Trainium prices 80%**\n\n- **So what:** ..."),
        ])
        s.commit()
    out = recheck_lenses()
    assert {k: out[k] for k in ("checked", "updated", "tickers_cleared", "headlines_replaced", "alerts_retitled",
                                "ungrounded")} == {"checked": 3, "updated": 3, "tickers_cleared": 1,
                                                   "headlines_replaced": 1, "alerts_retitled": 1, "ungrounded": 0}
    assert out["by_audience"]["investment"]["tickers_cleared"] == 1
    with db() as s:
        rows = {(r.development_id, r.audience): r for r in s.query(Lens)}
        alert = s.query(Alert).one()
    abstained, fixed = rows[(quiet, "investment")], rows[(loud, "investment")]
    assert (abstained.tickers, abstained.direction, abstained.checks["tickers_dropped"]) == ([], "unclear", ["AMZN"])
    assert fixed.headline == "Anthropic expands AWS Trainium deal" and fixed.tickers == holding
    assert fixed.checks["headline_replaced"]["unsupported_numbers"] == ["80%"]
    assert fixed.checks["unsupported_numbers"] == []  # 0.90 was an earlier materiality estimate of this development
    assert alert.title == fixed.headline and alert.body_md.startswith(f"**{fixed.headline}**")
    again = recheck_lenses()
    assert (again["updated"], again["unchanged"], again["headlines_replaced"]) == (0, 3, 0)
    with db() as s:
        assert s.get(Lens, fixed.id).checks["headline_replaced"]["original"] == "Anthropic cuts Trainium prices 80%"
        assert s.get(Lens, abstained.id).checks["tickers_dropped"] == ["AMZN"]
    assert recheck_lenses(limit=1)["checked"] == 1


# ------------------------------------------------------------------------------- basis and refresh
PRESS_TEXT = "Anthropic is said to be close to a deal for one million Trainium chips, people familiar said."
PRESS_QUOTE = "Anthropic is said to be close to a deal for one million Trainium chips"
RUMOR = {**INVESTMENT, "headline": "Reported Trainium deal would deepen Amazon's Anthropic tie",
         "so_what": "Unconfirmed press report of a large Anthropic commitment to AWS.", "action": "watch"}
CONFIRMED = {**INVESTMENT, "headline": "Anthropic and AWS sign the Trainium deal",
             "so_what": "The lab's own post backs the report: a large Anthropic commitment to AWS.",
             "action": "update_thesis"}


def _stage_answer(kw: dict) -> dict:
    """Rumor-stage analysis until the input shows an official source, confirmed-stage after."""
    if kw["prompt"].name != "lens_investment":
        return AI_TEAM
    return CONFIRMED if "official source: yes" in kw["user"] else RUMOR


def add_press_dev(db, first_seen: datetime, p: float = 0.7) -> int:
    """A rumor-stage development: one press insight, no official source."""
    with db() as s:
        if s.get(Source, "press:wire") is None:
            s.add(Source(id="press:wire", adapter="rss", kind="press", source_class="press"))
        dev = Development(title="Anthropic nears AWS Trainium deal, report says", event_type="compute_deal",
                          lab_id="anthropic", first_seen_at=first_seen, last_seen_at=first_seen, insight_count=1,
                          source_count=1, source_classes=["press"], has_official=False, has_verified=True)
        s.add(dev)
        s.flush()
        doc = Document(source_id="press:wire", natural_key=f"p{dev.id}", url=f"https://press.example/{dev.id}",
                       canonical_url=f"https://press.example/{dev.id}", title="Report: Anthropic nears AWS deal",
                       content_text=PRESS_TEXT, content_hash=f"p{dev.id}", source_class="press", status="extracted")
        s.add(doc)
        s.flush()
        s.add(Insight(document_id=doc.id, development_id=dev.id, lab_id="anthropic", event_type="compute_deal",
                      title="Anthropic reportedly nears Trainium deal", claim="A report says Anthropic nears a deal.",
                      evidence_quotes=[PRESS_QUOTE], verification=verify_quotes([PRESS_QUOTE], PRESS_TEXT),
                      quotes_verified=True, is_rumor=True, published_at=first_seen, prompt_version="t", model="t"))
        s.add(Score(development_id=dev.id, audience="investment", p_material=p, model_version="v0",
                    created_at=first_seen + timedelta(hours=1)))
        s.commit()
        return dev.id


def add_member(db, dev_id: int, *, official: bool = True, n: int = 1, when: datetime | None = None) -> list[int]:
    """New insights joining a development (official lab post or press), aggregates updated as clustering would."""
    ids = []
    with db() as s:
        dev = s.get(Development, dev_id)
        for _ in range(n):
            k = s.query(Document).count() + 1
            source, cls = ("anthropic:news", "official") if official else ("press:wire", "press")
            doc = Document(source_id=source, natural_key=f"m{k}", url=f"https://{cls}.example/m{k}",
                           canonical_url=f"https://{cls}.example/m{k}", title="Expanding with AWS", content_text=TEXT,
                           content_hash=f"m{k}", source_class=cls, lab_id="anthropic", status="extracted")
            s.add(doc)
            s.flush()
            ins = Insight(document_id=doc.id, development_id=dev_id, lab_id="anthropic", event_type="compute_deal",
                          title="Anthropic to deploy 1M Trainium chips", claim="Anthropic says it will deploy 1M chips.",
                          evidence_quotes=[QUOTE], verification=verify_quotes([QUOTE], TEXT), quotes_verified=True,
                          is_official=official, published_at=when or dev.last_seen_at, prompt_version="t", model="t")
            s.add(ins)
            s.flush()
            ids.append(ins.id)
            dev.insight_count += 1
            dev.source_classes = sorted({*dev.source_classes, cls})
            dev.has_official = dev.has_official or official
        s.commit()
    return ids


def test_lens_records_its_basis_and_is_refreshed_in_place_when_the_lab_confirms(db):
    seed(db)
    dev_id = add_press_dev(db, WEEK_A)
    gw = FakeGateway(_stage_answer)
    assert run_lens(gateway=gw)["investment"]["created"] == 1
    with db() as s:
        row = s.query(Lens).filter_by(audience="investment").one()
        lens_id, first_basis = row.id, row.checks["basis"]
        ins_id = s.query(Insight.id).filter_by(development_id=dev_id).scalar()
    assert first_basis == {"insight_count": 1, "source_classes": ["press"], "has_official": False,
                           "newest_insight_id": ins_id}
    assert row.headline == RUMOR["headline"] and row.action == "watch"
    assert run_lens(gateway=gw)["investment"]["refresh_todo"] == 0  # nothing changed

    (official_id,) = add_member(db, dev_id, official=True)  # the lab's own post joins
    calls_before = len(gw.calls)
    out = run_lens(gateway=gw)
    assert out["investment"]["refresh_todo"] == 1 and out["investment"]["refreshed"] == 1
    assert out["investment"]["todo"] == 0 and out["investment"].get("created", 0) == 0
    call = gw.calls[calls_before]
    assert call["ref"] == {"development_id": dev_id, "audience": "investment", "refresh": True}
    assert "official source: yes" in call["user"]
    with db() as s:
        row = s.query(Lens).filter_by(audience="investment").one()
    assert row.id == lens_id  # regenerated in place
    assert (row.headline, row.action) == (CONFIRMED["headline"], "update_thesis")
    assert row.checks["basis"] == {"insight_count": 2, "source_classes": ["official", "press"], "has_official": True,
                                   "newest_insight_id": official_id}
    assert row.checks["refresh"]["count"] == 1
    assert row.checks["refresh"]["reasons"] == ["official_source", "new_source_class:official", "new_insights"]
    assert row.checks["refresh"]["previous"] == {"headline": RUMOR["headline"], "action": "watch",
                                                 "basis": first_basis}
    assert row.checks["version"] == lens_mod.CHECKS_VERSION and "unsupported_numbers" in row.checks  # checks redone
    assert run_lens(gateway=gw)["investment"]["refresh_todo"] == 0  # idempotent once current


def test_refresh_needs_an_official_source_a_new_class_or_two_new_insights(db):
    seed(db)
    dev_id = add_press_dev(db, WEEK_A)
    run_lens(gateway=FakeGateway(_stage_answer))
    add_member(db, dev_id, official=False)  # one more press report: not material
    with db() as s:
        assert lens_mod.select_for_refresh(s, "investment") == []
    add_member(db, dev_id, official=False)  # a second one is
    with db() as s:
        (r,) = lens_mod.select_for_refresh(s, "investment")
    assert (r.development_id, r.reasons) == (dev_id, ("new_insights:2",))
    basis = {"insight_count": 1, "source_classes": ["press"], "has_official": False, "newest_insight_id": 5}
    members = [lens_mod.Member(5, None, "press", False), lens_mod.Member(9, None, "research", False)]
    assert lens_mod.refresh_reasons(basis, members) == ["new_source_class:research"]
    assert lens_mod.refresh_reasons(basis, members, strict=True) == ["new_source_class:research", "new_insights:1"]
    assert lens_mod.refresh_reasons(basis, members[:1], strict=True) == []


def test_refresh_is_capped_per_run_official_confirmations_first(db):
    seed(db)
    grown = add_press_dev(db, WEEK_A, p=0.9)  # two more press reports
    confirmed = add_press_dev(db, WEEK_A, p=0.5)  # the lab confirms (lower p, but first)
    run_lens(gateway=FakeGateway(_stage_answer))
    add_member(db, grown, official=False, n=2)
    add_member(db, confirmed, official=True)
    gw = FakeGateway(_stage_answer)
    out = run_lens(gateway=gw, max_refresh=1)
    assert out["investment"]["refresh_todo"] == 1 and out["investment"]["refresh_deferred"] == 1
    assert [c["ref"]["development_id"] for c in gw.calls] == [confirmed]
    assert lens_mod.REFRESH_PER_RUN == 30
    out = run_lens(gateway=gw)  # the next run takes the deferred one
    assert out["investment"]["refreshed"] == 1 and gw.calls[-1]["ref"]["development_id"] == grown


def test_refresh_skips_developments_that_no_longer_warrant_a_lens(db):
    seed(db)
    faded = add_press_dev(db, WEEK_A, p=0.6)
    alerted = add_press_dev(db, WEEK_A, p=0.6)
    gone = add_press_dev(db, WEEK_A, p=0.6)
    run_lens(gateway=FakeGateway(_stage_answer))
    with db() as s:
        for dev_id in (faded, alerted):  # re-scored far below LENS_MIN_P (weekly top-N off below)
            s.add(Score(development_id=dev_id, audience="investment", p_material=0.05, model_version="v1",
                        created_at=WEEK_A + timedelta(days=1)))
        s.add(Alert(development_id=alerted, audience="investment", severity="P2", title="t", body_md="b",
                    rules=[], dedup_key=f"investment:{alerted}"))
        s.get(Development, gone).status = "merged"
        s.commit()
    for dev_id in (faded, alerted, gone):
        add_member(db, dev_id, official=True)
    with db() as s:
        picked = [r.development_id for r in lens_mod.select_for_refresh(s, "investment", top_n=0)]
    assert picked == [alerted]  # an alert keeps a lens worth refreshing; a merged development is skipped


def test_backfill_infers_the_basis_from_timestamps_without_llm(db):
    seed(db)
    stale_dev = add_press_dev(db, WEEK_A)
    current_dev = add_press_dev(db, WEEK_A)
    t_lens = datetime(2026, 10, 6, 12)
    with db() as s:
        for dev_id in (stale_dev, current_dev):  # legacy rows: no basis in checks
            s.add(Lens(development_id=dev_id, audience="investment", headline="h", so_what="s", what_to_do="w",
                       action="watch", checks={"grounded": True}, model="m", prompt_version="p", created_at=t_lens))
        for ins in s.query(Insight):
            ins.created_at = t_lens - timedelta(hours=1)
        s.commit()
    new_ids = add_member(db, stale_dev, official=True)  # created now: after the lens
    with db() as s:  # an old insight the lens could have seen does not count as new
        old = add_member(db, current_dev, official=False)
        s.get(Insight, old[0]).created_at = t_lens - timedelta(minutes=5)
        s.commit()
    preview = lens_mod.backfill_lens_basis(dry_run=True)
    assert (preview["backfilled"], preview["would_refresh"]) == (2, 1)
    with db() as s:
        assert all("basis" not in (r.checks or {}) for r in s.query(Lens))  # dry run writes nothing
    out = lens_mod.backfill_lens_basis()
    assert (out["checked"], out["backfilled"], out["had_basis"], out["would_refresh"], out["next_run"]) == (2, 2, 0, 1, 1)
    assert out["refresh"] == [{"development_id": stale_dev, "audience": "investment",
                               "reasons": ["official_source", "new_source_class:official"]}]
    with db() as s:
        rows = {r.development_id: r.checks for r in s.query(Lens)}
    assert rows[stale_dev]["basis"]["has_official"] is False and rows[stale_dev]["basis"]["backfilled"] is True
    assert rows[stale_dev]["basis"]["newest_insight_id"] < new_ids[0]
    assert rows[current_dev]["basis"]["insight_count"] == 2 and rows[current_dev]["grounded"] is True  # checks kept
    again = lens_mod.backfill_lens_basis()
    assert (again["backfilled"], again["had_basis"]) == (0, 2)


def test_recheck_keeps_the_basis_the_text_was_written_from(db):
    seed(db)
    dev_id = add_press_dev(db, WEEK_A)
    run_lens(gateway=FakeGateway(_stage_answer))
    add_member(db, dev_id, official=True)
    with db() as s:
        before = s.query(Lens).filter_by(audience="investment").one().checks["basis"]
    recheck_lenses()
    with db() as s:
        assert s.query(Lens).filter_by(audience="investment").one().checks["basis"] == before
        assert [r.development_id for r in lens_mod.select_for_refresh(s, "investment")] == [dev_id]  # still stale


def test_ensure_lens_refresh_regenerates_only_lenses_older_than_the_evidence(db):
    seed(db)
    stale, current = add_press_dev(db, WEEK_A), add_press_dev(db, WEEK_A)
    run_lens(gateway=FakeGateway(_stage_answer))
    add_member(db, stale, official=False)  # one new same-class insight: stale for a push, not for run_lens
    gw = FakeGateway(_stage_answer)
    assert ensure_lens([stale, current], "investment", gateway=gw) == 0 and gw.calls == []  # no refresh asked
    assert ensure_lens([stale, current], "investment", gateway=gw, refresh=True) == 1
    assert [c["ref"] for c in gw.calls] == [{"development_id": stale, "audience": "investment", "refresh": True}]
    assert ensure_lens([stale, current], "investment", gateway=gw, refresh=True) == 0  # now current
    assert lens_mod.regenerate_lenses([current], "investment", gateway=gw) == 1  # forced, e.g. after a data repair
    with db() as s:
        assert s.query(Lens).filter_by(development_id=current).one().checks["refresh"]["reasons"] == ["regenerated"]
