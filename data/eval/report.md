# Evaluation report

Generated 2026-11-06T07:38:10 by `fli eval`.

## Known events

Recall 0.667 (18 events); in weekly top-10 for the primary audience 0.611; alerted 0.611; pushed (P1/P2) 0.611; median detection delay 0.0 days, measured from the source date of the insight that reported the event. Alert raised for an earlier event of the same development: openai-chatgpt-financial-services.

| event | found | development | rank inv | rank ai | alert inv | alert ai | delay (d) |
|---|---|---|---|---|---|---|---|
| Jeff Dean leaves Google after 27 years to co-found Discovery Loop; Dee | yes | 478 | 1 | 25 | P2 |  | 0.0 |
| Hassabis becomes Google DeepMind chair and Alphabet chief scientist; K | yes | 479 | 2 |  | P2 |  | 0.0 |
| Discovery Loop reportedly raising at ~$50B, weeks after talks at ~$10B | no |  |  |  |  |  |  |
| OpenAI's Broadcom-built Jalapeno inference chip claims 1.5-1.9x work p | yes | 89 | 2 | 34 | P2 |  | 0.29 |
| Anthropic releases Claude Fable 5.1 (general) and Claude Mythos 5.1 (v | yes | 120 | 1 | 10 | P2 | P1 | 0.0 |
| SB Energy files S-1 (Nasdaq SBE); prospectus says it is 'substantially | no |  |  |  |  |  |  |
| OpenAI GPT-6 'Astra' staged rollout | yes | 123 | 2 | 1 | P1 | P1 | -1.46 |
| OpenAI launches ChatGPT for financial services with S&P Global / FactS | yes | 123 | 2 |  | P1 |  | 0.29 |
| OpenAI says an internal agent swarm produced a proof of finite-time bl | yes | 149 | 9 | 7 |  | P2 | -6.58 |
| Anthropic releases Claude Opus 5.5 (lower price than Opus 5) | yes | 209 | 1 | 11 |  | P2 | 0.0 |
| Alibaba Apsara: Qwen 4 in training; plans a 5-10T-parameter model | no |  |  |  |  |  |  |
| Anthropic signs $11.6B, 7-year compute agreement with Akamai; AKAM +20 | no |  |  |  |  |  |  |
| Reuters: OpenAI decided not to release GPT-6.1 Astra after internal te | no |  |  |  |  |  |  |
| White House voluntary AI accord signed by lab and chip CEOs | no |  |  |  |  |  |  |
| Google unveils Gemini 4 Argon | yes | 270 | 8 | 2 | P2 |  | 2.62 |
| OpenAI frontier models and Codex come to Google Cloud Model Garden (pr | yes | 375 | 1 | 54 | P2 |  | 0.5 |
| Anthropic files public S-1 for a Nasdaq listing (ANT); H1-2026 revenue | yes | 405 | 1 |  | P1 |  | 0.0 |
| Google announces Gemini 4 Carbon (frontier, cyber focus, restricted ac | yes | 425 | 10 | 2 | P2 | P1 | 0.71 |

## Alerts

89 alert rows. Only pushed alerts (P1/P2, they reach a person) are measured below; digest-only P3 rows are not. Precision and recall use labels: One label per development: human annotators (majority) when present, else the latest LLM judge. LLM-judge precision is in-sample: `fli train` refits the deployed scorers on all labels after the time-split evaluation (hold-out window included), so treat it as optimistic until human labels exist.

| audience | window | pushed (P1/P2) | per week | per active day | max/day | labelled (source) | P(label>=1) | P(label=2) | P1: P(label=2) (labelled) | label-0 pushes | label-2 recall | own lens says no_action / hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| investment | all from 2026-08-01 | 29 (3/26) | 2.13 | 0.33 | 2 | 19 (llm in-sample) | 1.000 | 0.368 | 1.000 (3) | 0 | 7/10 | 0 (0 P1) |
| investment | hold-out from 2026-10-13 | 7 (1/6) | 2.29 | 0.30 | 2 | 6 (llm in-sample) | 1.000 | 0.333 | 1.000 (1) | 0 | 2/2 | 0 (0 P1) |
| ai_team | all from 2026-08-01 | 56 (19/37) | 4.12 | 0.64 | 2 | 34 (llm in-sample) | 0.971 | 0.529 | 0.857 (14) | 1 | 18/22 | 0 (0 P1) |
| ai_team | hold-out from 2026-10-13 | 14 (2/12) | 4.57 | 0.61 | 2 | 8 (llm in-sample) | 0.875 | 0.500 | 0.500 (2) | 1 | 4/5 | 0 (0 P1) |

## Lens faithfulness

Deterministic checks stored on every lens (`Lens.checks`, refreshed by `fli lens-recheck`): every figure in the headline, so-what, what-to-do and ticker rationales must be written in the lens input, and wording that presents facts as official needs an official source. Grounded = no unsupported figure in the headline, so-what or a ticker rationale and no official-wording flag (figures in what-to-do are reported, not gating). A code check of figures and provenance wording, not a semantic entailment judge.

| audience | lenses (checked) | grounded | lenses with unsupported figures | unsupported figures by field | headlines replaced | official-wording flags | tickers dropped (no_action) | behind P1/P2 alerts: grounded |
|---|---|---|---|---|---|---|---|---|
| investment | 253 (253) | 245 (96.8%) | 8 | so_what 6, what_to_do 3, tickers 1 | 1 | 2 (figures_not_in_official_evidence 2) | 64 | 27/29 (93.1%) |
| ai_team | 451 (451) | 448 (99.3%) | 5 | so_what 4, what_to_do 2 | 0 | 0 | 0 | 56/56 (100.0%) |
| all | 704 (704) | 693 (98.4%) | 13 | so_what 10, what_to_do 5, tickers 1 | 1 | 2 (figures_not_in_official_evidence 2) | 64 |  |

Flagged lenses (12, ungrounded or headline replaced; pushed first, 12 shown):

- development 123 (investment, P1 alert): official wording: figures_not_in_official_evidence
- development 242 (investment, P2 alert): headline replaced (was "OpenAI cuts frontier-class token prices 80%: bullish volume, bearish pricing for compute resellers")
- development 312 (investment, P2 alert): unsupported figures: so_what: 1M, what_to_do: 27, tickers: 1M
- development 34 (investment): unsupported figures: so_what: 50%
- development 76 (investment): official wording: figures_not_in_official_evidence
- development 78 (investment): unsupported figures: so_what: 0.14
- development 112 (investment): unsupported figures: so_what: 0.71
- development 156 (investment): unsupported figures: so_what: 0.94
- development 307 (ai_team): unsupported figures: so_what: 0.87
- development 327 (ai_team): unsupported figures: so_what: 0.76
- development 377 (ai_team): unsupported figures: so_what: 165, so_what: 100
- development 451 (investment): unsupported figures: so_what: 0.05

## Coverage by BIT holding

Window from 2026-08-01: 1002 documents, duplicates excluded. Documents naming it: title or text names the company or a brand it owns (the exposure rules' whole-word names). Event-specific developments: an exposure link that is not lab-level; named = the development's own text names the company, the rest are theme-keyword links; >= floor = confidence >= 0.6, what the alert rules count as exposure. Lab-level only: linked through the lab's curated relations alone (context, not exposure). Pushed alerts: investment P1/P2 alerts whose Tickers line lists the holding.

| holding | BIT weight | documents naming it (triaged relevant) | event-specific developments (named / >= floor) | lab-level only | pushed alerts listing it (P1) | coverage |
|---|---|---|---|---|---|---|
| AMZN (Amazon) | 9.6% | 14 (12) | 23 (7 / 18) | 148 | 10 (3) | direct |
| IREN | 9.2% | 0 (0) | 1 (0 / 0) | 0 | 0 (0) | indirect |
| NVDA (NVIDIA) | 8.0% | 19 (16) | 38 (10 / 22) | 267 | 22 (3) | direct |
| MU (Micron Technology) | 6.5% | 0 (0) | 10 (0 / 0) | 0 | 7 (2) | indirect |
| TSM (TSMC) | 5.1% | 0 (0) | 0 (0 / 0) | 163 | 8 (1) | none |
| META (Meta Platforms) | 4.6% | 85 (50) | 110 (78 / 78) | 12 | 5 (0) | direct |
| MSFT (Microsoft) | 4.4% | 15 (11) | 26 (9 / 23) | 144 | 14 (3) | direct |
| HNGE (Hinge Health) | 4.0% | 0 (0) | 17 (0 / 0) | 0 | 0 (0) | indirect |
| SIMO (Silicon Motion) | 3.9% | 0 (0) | 7 (0 / 0) | 0 | 6 (1) | indirect |

No collected document names IREN (9.2%), MU (6.5%), TSM (5.1%), HNGE (4.0%), SIMO (3.9%): 28.7 of the 55.3 percentage points of fund weight in the tracked holdings sit in positions without direct source coverage, so silence on them is not evidence that nothing happened.
Listed by pushed alerts although no document names them (theme-keyword or lab-level links only): MU 7, TSM 8, SIMO 6.
Never listed by a pushed investment alert: IREN, HNGE.

## Digest quality

Stored weekly reports. Cited: selected items the report cites anywhere; in written sections: cited by the model's (or the fallback's) sections rather than appended under 'Also this week'. The digest rejects model output citing fewer than 60.0% of the selected items, retries once on a different model, then renders a deterministic fallback.

| audience | reports | model / fallback / empty | items | cited | in written sections | 'Also this week' items (reports) | model-written: lowest share in written sections | fallback weeks | missing weeks |
|---|---|---|---|---|---|---|---|---|---|
| investment | 14 | 14 / 0 / 0 | 115 | 115 (100.0%) | 115 (100.0%) | 0 (0) | 100.0% | none | none |
| ai_team | 14 | 13 / 1 / 0 | 159 | 159 (100.0%) | 159 (100.0%) | 0 (0) | 100.0% | 2026-08-31 | none |

## Source coverage

98 registered sources (95 enabled): 49 produced at least one document, 49 never did (never_run 46, disabled 3). 1008 documents (3 duplicates, 3 dated before 2026-08-01); newest 2026-11-04.

| source class | sources (with documents) | documents | triaged relevant |
|---|---|---|---|
| official | 64 (44) | 826 | 532 |
| research | 2 (2) | 123 | 111 |
| market_data | 1 (1) | 49 | 47 |
| curated | 1 (1) | 6 | 6 |
| press | 4 (1) | 4 | 0 |
| filing | 1 (0) | 0 | 0 |
| personal | 25 (0) | 0 | 0 |

| lab | tier | sources (never produced) | documents | triaged relevant | weeks with documents (of 15) | newest document |
|---|---|---|---|---|---|---|
| Anthropic | 1 | 9 (2) | 146 | 93 | 15 | 2026-11-04 |
| Google DeepMind | 1 | 7 (1) | 127 | 106 | 14 | 2026-11-04 |
| Meta AI (Superintelligence Labs / FAIR) | 1 | 7 (3) | 169 | 116 | 14 | 2026-11-03 |
| OpenAI | 1 | 7 (2) | 310 | 185 | 15 | 2026-11-04 |
| xAI | 1 | 5 (2) | 39 | 33 | 15 | 2026-11-03 |
| DeepSeek | 2 | 5 (1) | 28 | 26 | 9 | 2026-10-31 |
| Mistral AI | 2 | 6 (2) | 49 | 26 | 11 | 2026-11-04 |
| Moonshot AI (Kimi) | 2 | 4 (2) | 12 | 5 | 7 | 2026-10-30 |
| Qwen (Alibaba) | 2 | 3 (1) | 55 | 46 | 13 | 2026-11-04 |
| Discovery Loop | 3 | 0 (0) | 1 | 1 | 1 | 2026-08-05 |
| Periodic Labs | 3 | 2 (0) | 11 | 11 | 8 | 2026-10-11 |
| Reflection AI | 3 | 4 (1) | 15 | 15 | 10 | 2026-11-01 |
| Safe Superintelligence (SSI) | 3 | 0 (0) | 0 | 0 | 0 |  |
| Thinking Machines Lab | 3 | 6 (3) | 22 | 16 | 12 | 2026-11-01 |
| unattributed |  | 33 (29) | 24 | 17 | 13 | 2026-11-04 |

Tracked labs without a single document: ssi.

Sources that never produced a document:

| kind | class | reason | sources | examples |
|---|---|---|---|---|
| x_account | personal | never_run | 25 | x:alexandr_wang, x:barret_zoph, x:bcherny, x:ch402 (+21 more) |
| x_account | official | never_run | 9 | anthropic:x, deepseek:x, google-deepmind:x, meta-ai:x (+5 more) |
| hf_org | official | never_run | 5 | meta-ai:hf-llama, moonshot:hf, openai:hf, thinking-machines:hf (+1 more) |
| press | press | never_run | 3 | press:gdelt-alumni, press:gdelt-frontier-deals, press:gdelt-talent |
| x_account | official | disabled | 2 | moonshot:x, reflection-ai:x |
| bluesky_account | official | never_run | 1 | anthropic:bluesky |
| github_org | official | never_run | 1 | meta-ai:github-llama |
| job_board | official | disabled | 1 | mistral:jobs |
| research_blog | official | never_run | 1 | thinking-machines:blog |
| sec | filing | never_run | 1 | sec:edgar-fts |

## extraction

```json
{
  "insights": 689,
  "quotes": 1267,
  "quote_verification_rate": 0.9984,
  "insights_all_quotes_verified": 0.9971,
  "insights_at_least_one_verified": 1.0,
  "citation_validity": 1.0
}
```

## noise

```json
{
  "suppressed_share": 0.307,
  "noise_reasons": {
    "minor_update": 141,
    "customer_story": 80,
    "marketing": 48,
    "off_topic": 25,
    "opinion_chatter": 8,
    "event_promo": 5,
    "repost": 2
  },
  "audit_sample": "/Users/sanjaysuthraye/anaconda_projects/1. Jobs Take home Test/Bit-capital/frontier-lab-intel/data/eval/noise_audit_sample.csv"
}
```

## scoring

```json
{
  "generated_at": "2026-11-04T15:52:35",
  "target": "label >= 1 (worth knowing or material); label == 2 reported as roc_auc_material",
  "investment": {
    "roc_auc": 0.8694,
    "roc_auc_material": 0.9716,
    "material_n": 2,
    "precision_at_10": 0.9,
    "precision_at_20": 0.75,
    "label_source": "llm",
    "deployed": true
  },
  "ai_team": {
    "roc_auc": 0.8228,
    "roc_auc_material": 0.88,
    "material_n": 5,
    "precision_at_10": 1.0,
    "precision_at_20": 0.95,
    "label_source": "llm",
    "deployed": true
  }
}
```

## er

```json
{
  "n": 58,
  "tp": 37,
  "fp": 1,
  "fn": 3,
  "tn": 17,
  "precision": 0.974,
  "recall": 0.925,
  "f1": 0.949,
  "gray_zone": 3,
  "recall_if_gray_accepted": 1.0
}
```

## costs

```json
{
  "llm_total_usd": 14.4002,
  "x_api_usd": 0.0,
  "usd_per_triaged_document": 0.00018,
  "usd_per_extracted_document": 0.0005
}
```
