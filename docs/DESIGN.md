# Frontier Lab Intelligence - system design

Status: working system on real data (window 2026-08-01 to 2026-11-04). Numbers in this document come
from `fli eval` (data/eval/report.json), `fli costs` and data/eval/scoring_metrics.json.

## 1. Who this is for and why

BIT Capital runs about EUR 3bn with a ~10-person tech team and an explicit "Co-Intelligence" process
(models find patterns, humans decide); its platform already has a *Signals & Alerts* layer. The
flagship fund's top holdings (factsheet, 30 Sep 2026) are almost all downstream of frontier labs:
AMZN 9.6% (Anthropic investor, Trainium), IREN 9.2% (AI capacity), NVDA 8.0%, MU 6.5% (HBM), TSM 5.1%,
META 4.6%, MSFT 4.4% (OpenAI). So the useful question is never "what did a lab publish" but "what does
it change for a position or a thesis" (PMs) and "what should we build with or migrate off" (AI team).
The system is built to feed that layer: a JSON API (`/api/developments`, `/api/alerts`) next to a UI.

## 2. Architecture and data flow

```
config/*.yaml (register, sources, exposure map, materiality policy)
      |
collectors (official APIs, SDKs, published feeds) --> runner: idempotent upsert, exact dedup, job ledger, source health
      |                                                                      |
      v                                                                      v
document --> near-dup (MinHash) --> triage (Haiku) --> extraction (Haiku) --> quote verification (code)
                                                                                   |
insight --> embed (bge-small, local) --> cluster into developments --> novelty / first-seen artifacts
                                                                                   |
development --> exposure links (closed ticker list) --> features --> score per audience (logistic regression)
                                                                                   |
                         lens per audience (Sonnet) --> faithfulness checks (code) --> materiality rules --> alerts
                                                                                   +--> weekly digests (Opus)
```

- **Unit of value = development**, not document: a launch seen on a blog, a changelog, Hugging Face and
  OpenRouter is one development with one score, one lens and at most one alert per audience.
- **Every stage is an idempotent function over pending rows** (`status` columns, NULL embeddings, missing
  scores). The scheduler re-runs them every 30 minutes; a crash loses nothing.
- **Storage**: one SQLAlchemy schema (23 tables, `docs/schema.sql`) that runs on Postgres in production and
  SQLite for tests and the committed snapshot. Embeddings are float32 bytes and similarity is numpy
  brute force: at ~700 insights a query is milliseconds, so pgvector was rejected as an extra moving part.

## 3. Key decisions

| Decision | Why | Consequence / rejected alternative |
|---|---|---|
| Official APIs, SDKs and published feeds only; respect robots.txt; never bypass bot protection | Defensible sourcing for a regulated fund; fewer silent breakages | OpenAI article pages and x.ai/news sit behind bot challenges and Meta's AI blog forbids automated collection, so those labs come in via RSS summaries, API changelogs, newsroom/engineering blogs, GitHub and Hugging Face. Rejected: headless-browser scraping. |
| Deterministic spine, LLMs only at the leaves (triage, extraction, lens, digest, judge, entity-resolution gray zone) | Each LLM step is a single schema-constrained call that can be cached, costed and evaluated | No autonomous browsing agent in the production path: fetched pages are data, never instructions (prompt-injection), and agent loops are hard to cost-cap and evaluate. Rejected: per-entity research agents. |
| Citations are owned by code: URL, date, author and source class come from source metadata; the model only supplies verbatim quotes, which code verifies | Makes hallucinated citations structurally impossible | Insights whose quotes fail verification are kept but cannot alert. Rejected: the native citations feature (it cannot be combined with structured outputs in one call). |
| Closed-world tickers from `config/exposure.yaml` (77 tickers, 12 themes, BIT weights) | The LLM cannot invent an exposure | Lab-level relations (e.g. OpenAI -> MSFT) are stored as `method='lab'` context and never count as event exposure (see section 5 - this was a measured bug). |
| One scorer per audience: logistic regression on ~20 interpretable features, trained on labels, validated on a time split | "A simple model you can defend and have tested"; coefficients double as the *why flagged* panel | Rejected: weighted sums (unvalidated), LambdaMART (needs thousands of labels). |
| Alerts are rules over validated signals; the LLM lens may **veto** an alert but never promote one | The score measures "worth knowing", not "material" (it saturates); the lens is an unvalidated opinion | Policy is versioned YAML (`config/materiality.yaml`) with a structural test enforcing the veto-only invariant. |
| Reproducibility via a response cache keyed by (model, prompt version, input) | The current API rejects sampling parameters, so temperature=0 is not an option | Re-runs are free and identical; prompt edits change the version and invalidate cleanly. |
| Hard budget stops in code (total and daily), reserved per in-flight call | A fan-out of concurrent calls must not overshoot; a reviewer reproduced 4x overshoot before the fix | Budget exhaustion skips only LLM stages; clustering, scoring and rule-based alerts keep running. |

**Register.** 14 labs (tier 1: OpenAI, Anthropic, Google DeepMind, Meta, xAI; tier 2: open-weight leaders
Mistral, DeepSeek, Qwen, Moonshot; tier 3: alumni startups Thinking Machines, SSI, Reflection, Periodic,
Discovery Loop) and 103 people. "Key" is defined in `config/people.yaml`: tier A = founders, chief
scientists and research heads; tier B = named leads of flagship model work; tier C = high-signal
contributors (lab-repo maintainers, prolific technical authors). Affiliations are dated rows with
sources, so a move is data, not prose. Entity resolution goes deterministic first (exact handles and
arXiv names, then GitHub self-asserted links: 10 X handles confirmed from users' own profiles), fuzzy
second (token-sort ratio >= 93 plus lab agreement), and an LLM only for the 85-93 gray zone, where it must
quote a fact present in both records.

## 4. Model choices

| Task | Model | Why | Fallback |
|---|---|---|---|
| Triage (relevant? type, audiences, noise reason) | Claude Haiku 5.5, effort low | 1,030 documents for $0.18; a classification with a fixed schema | 3 attempts, then dead-letter (`status=failed`) |
| Extraction (0-3 insights, verbatim quotes, magnitude rubric) | Claude Haiku 5.5, effort medium | Quote verification is done in code, so a cheap extractor is safe: 99.8% of quotes verify | Same |
| Entity-resolution gray zone | Claude Haiku 5.5 | Rare (3 pairs in the labelled set); must quote shared facts | Leave as `candidate` |
| Audience lens (so-what, what-to-do, tickers from a closed list) | Claude Sonnet 5.5, effort medium | The text a PM reads; needs judgment. Only for developments above relevance 0.35 or top 15/week | Server-side refusal fallback; deterministic summary if absent |
| Weekly digest | Claude Opus 5.5, effort high | 28 calls total; writing quality matters; output validated (coverage >= 60% of selected items) | One uncached retry on Claude Sonnet 5.5 (a resend to Opus repeats the failure), then a deterministic digest |
| Silver labels (judge) | Claude Opus 5.5 | Strongest available rubric-follower for ground truth | Human labels override (see below) |
| Embeddings | BAAI/bge-small-en-v1.5 via fastembed (local, 64 MB) | No API key needed to run the repo; ~120 texts/s on CPU | - |

Model routing is configuration (`MODEL_*` settings). The extractor and the judge are the same model
family, which can correlate errors; the human-labelling queue (`/label`) exists to break that loop.

## 5. Evaluation

**Ground truth, built three ways.**
1. *Known events.* Before looking at any output I listed 18 real developments of the window from
   independent sources (`data/eval/known_events.yaml`): releases, the Anthropic S-1, the Akamai-Anthropic
   $11.6B compute deal, Jeff Dean leaving Google to co-found Discovery Loop, etc.
2. *Silver labels.* Opus 5.5 labelled a stratified sample of 300 developments for both audiences
   (0 noise / 1 worth knowing / 2 material) against a written rubric. The sample deliberately includes
   low-prior items so labels are not only top-ranked items. Investment: 209/80/11; AI team: 105/172/23.
3. *Hand-labelled pairs* for entity resolution (58 pairs, including same-name hard negatives).

**Results.**

| Check | Result |
|---|---|
| Quote verification (hallucination control at extraction) | 99.8% of 1,231 quotes verbatim in the stored source; 99.7% of insights fully verified; citation validity 100% |
| Independent claim audit (top-20 developments per audience + all investment P1s) | 139/139 insight claims supported by stored text |
| Lens faithfulness | 12/669 lens statements unsupported in the audit (one P1 headline invented an "80% price cut"). Fix: code now checks every number in lens text against the lens input; 1 headline replaced, flags shown in the UI |
| Scorer, investment (time split: train 210 before 13 Oct, test 90 after) | AUC 0.87 (prior policy 0.80, magnitude-only 0.72, official-source-only 0.53); P@10 0.90, P@20 0.75 |
| Scorer, AI team | AUC 0.82 (prior 0.75); P@10 1.00, P@20 0.95. Ablation: dropping the LLM-derived features does not hurt (0.84), so for this audience they add nothing |
| Alerts after policy recalibration (precision vs judge labels, in-sample) | Investment: 25 pushes in 14 weeks (1.8/week), 100% worth knowing, 42% material, P1 3/3 material. AI team: 55 pushes (4.0/week), 97% / 53%, P1 79% material |
| Known-event recall | 12/18 found, median delay 0 days; 11 reached their primary audience as P1/P2. The Discovery Loop founding and the DeepMind reshuffle were found only after curated register moves became a source (round 2). The 6 misses are press and filing stories absent from the corpus |
| Entity resolution | Precision 0.97, recall 0.93 on 58 pairs |

**What the evaluation changed.** Five independent reviewers (claim grounding, PM persona, AI-engineer
persona, alert policy, code) audited the first full run. Three findings changed the design:
- *Exposure was attached by lab, not by event.* 93% of developments "touched" a BIT holding because every
  OpenAI item inherited MSFT/NVDA/TSM. Lab relations are now `method='lab'` context; event-specific
  exposure became the investment model's second-strongest feature (+1.16 per SD).
- *The score saturates.* It predicts "worth knowing" (65% base rate for the AI team), so no threshold
  isolates "material". Alerts are now gated on the extractor's magnitude rubric plus source and tier,
  with the lens as a veto. Pushes fell from 94 to 25 (investment) and 208 to 55 (AI team) while the
  material share rose from 14% to 42% and from 9% to 53%; recall of material items moved from 9/11 to
  8/11 and 20/23 to 18/23. The daily cap no longer demotes P1s.
- *My own harness had a bug.* Substring matching credited events to the wrong developments ("accord" in
  "according"). Word-boundary matching lowered reported recall from 12/18 to 10/18.

**Honest limits.** Alert precision is measured against the same LLM labels the scorer was trained on
(in-sample); only 34 developments are labelled "material", so the label-2 numbers have wide error bars.
The next step is a blind human pass over the ~80 pushed alerts via `/label` (`fli labels export --blind`).

## 6. Cost

From the `llm_call` ledger (`fli costs`), for the full 3-month window including every re-run:

| Workflow | Model | Calls | Spend | Unit cost |
|---|---|---|---|---|
| Triage | Haiku 5.5 | 1,036 | $0.18 | $0.00017 / document |
| Extraction | Haiku 5.5 | 719 | $0.34 | $0.0005 / extracted document |
| Lens (two audiences) | Sonnet 5.5 | 715 | $4.16 | ~$0.006 / lens |
| Silver labels | Opus 5.5 | 301 | $2.08 | $0.007 / development (both audiences) |
| Weekly digests incl. re-runs | Opus 5.5 (1 Sonnet 5.5 retry) | 132 | $7.14 | ~$0.11 / digest at effort high |
| Debugging the digest failure | Opus/Sonnet 5.5 | 11 | $0.49 | |
| **Total** | | | **$14.40** | |

Running cost for a normal week: about 70 new documents (~$0.05 triage and extraction), ~50 lenses
(~$0.30), 2 digests (~$0.22): roughly **$0.60 per week**. X reads would be the largest variable cost at
$0.005 per post; the budget guard caps them at $10 total.

How cost shaped decisions: Haiku for everything high-volume (quote verification in code makes the cheap
model safe); Sonnet only above a relevance threshold; Opus only where it is called a few dozen times;
local embeddings; a response cache so re-runs are free. A Batch API path (50% off) exists for backfills
but was not needed at this volume. Effort was tuned per route: digests at medium under-generated in 7 of
14 AI-team weeks, while high produced complete, fully cited output for about $0.04 more per digest.
The residual failure (2 AI-team weeks): the model drafts the whole digest in its thinking, then emits a
single empty entry and ends the turn. A resend to the same model repeats it, so the retry goes to
Sonnet 5.5. One week (2026-08-31) fails on both models and ships the deterministic fallback, so 27 of
28 digests are model-written.

## 7. How I worked

I built this with Claude Code acting as an orchestrator over parallel sub-agents, with files as the
coordination boundary:

1. **Research** (4 agents): BIT Capital (factsheets, process, holdings), source access and pricing,
   techniques and prior art, ground-truth events. Three of the four timed out after accumulating very large
   web contexts; I recovered their evidence from the transcripts and re-ran synthesis with small outputs.
2. **Foundation** written by hand: schema, collector contract, LLM gateway, pipeline contract, CLI,
   register loader. These are the integration contracts.
3. **Build** (9 agents, each owning distinct files): feeds, research and code sources, social/jobs/market,
   LLM stages, events, scoring, delivery and ops, web, register. Each wrote tests (no network, no LLM)
   and ran small live checks on its own database.
4. **Review** (5 independent agents, read-only on database copies): claim grounding, PM persona,
   AI-engineer persona, alert policy (with a simulator that reproduced the live alert table exactly
   before proposing changes), code review. **Fix** (6 agents), then re-run and re-evaluate.

What the agents verified themselves: 400 unit/integration tests, ruff, live endpoint checks, policy
simulations, and a deploy preflight. Where the loop broke down:
- research agents timing out on huge contexts (fix: narrow scope, small structured outputs);
- a safety filter stopped my attempts to write a generic crawler and to start bulk collection runs, so
  ingestion uses official APIs and feeds only, and the corpus is the data the build agents collected in
  capped live checks (1,002 documents) - the main coverage limitation;
- a hung API connection stalled extraction for 10 minutes (fix: explicit timeouts);
- my own evaluation harness and a hand-written budget guard each had a bug that only an independent
  reviewer caught. The lesson I took: the reviewer must never be the author.

## 8. What works, what's next

Works: end-to-end pipeline on real data with code-owned citations; per-audience scoring that beats the
policy prior on later data; alert volume of ~2/week (investment) and ~4/week (AI team) with P1 reserved
for material items; weekly digests with a holdings check; monitored ops (job ledger, freshness SLOs,
hard budgets, degrade-not-stop) and a one-command AWS deployment.

Next, in order:
1. **Coverage where holdings move**: SEC EDGAR full-text search (needs a contact user agent), press
   counterparty queries (IREN, CoreWeave, Nebius, Akamai, Micron x lab names), and an X token enrolled in
   a project. All 8 missed known events are press, filing or people-move stories; zero documents mention
   IREN (9.2%) or MU (6.5%) directly.
2. **Human ground truth**: a blind pass over the ~80 pushed alerts and a stratified label batch (at least
   15 each for capital, infra and people events) before retraining.
3. **Talent moves**: done for curated moves - dated, cited register affiliations now enter the pipeline as
   `curated` documents (the Discovery Loop founding alerts the investment team). Next: detect moves
   automatically from X/Bluesky bio changes and affiliation changes on new papers.
4. Refresh lenses when a development gains an official source; re-alert on material sub-events inside an
   already-alerted development.

## 9. What the system surfaced (real, cited)

1. **DeepSeek V4.1-Flash cuts KV-cache memory per token to ~1/4 (890 bytes/token in HBM) and SSD to
   ~1/8** (official news, 2026-09-10; paper 2609.19969). For BIT this directly tests the memory-content
   thesis behind MU (6.5%) and SIMO (3.9%): efficiency per token versus volume growth. For the AI team it
   makes long-context agent loops much cheaper, with open weights.
2. **DeepSeek introduced peak/off-peak pricing "to allocate resources more reasonably"** (2026-08-13) and
   kept it with V4.1. China's leading open-weight lab is rationing compute, which cuts against the
   "efficiency shock means fewer GPUs" reading for NVDA (8.0%). AI team: schedule batch extraction
   off-peak at half price.
3. **Meta disclosed Tahan, an Ethernet scale-up switch for its in-house MTIA 300 training chip built on
   Broadcom Tomahawk5** (OCP, 2026-10-26), two months after describing MTIA 300; a week earlier it said
   Prometheus is being scaled to ~1M GPUs. A custom-ASIC path at an NVDA customer, upside for AVGO switch
   silicon, and a large dated accelerator-demand datapoint - all relevant to META (4.6%) and NVDA.
4. **Anthropic filed its S-1 for a Nasdaq listing under ANT** (2026-10-26, official): ranked first for
   the investment team that week and pushed as P1, because it will mark AMZN's (9.6%) Anthropic stake.
5. **For the AI team, deadlines rather than launches**: Google will route every `gemini-2.5-pro` call to
   `gemini-3.8-flash` from 2026-12-01 at the new rates; Anthropic removes Opus 4.8 fast mode on
   2026-11-30; and new paid Gemini API projects now store requests by default - a data-control default a
   regulated fund must override. These came from changelogs that the first triage version dismissed as
   routine; fixing that was one of the review-driven changes.
