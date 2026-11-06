# Frontier Lab Intelligence

Tracks what the frontier AI labs and their key people publish, and turns it into cited, scored
developments, alerts and weekly digests for two audiences at BIT Capital:

- **Investment team**: what a development means for positions and theses, mapped to a closed list of
  public tickers, with BIT's top holdings flagged by fund weight.
- **AI team**: what to adopt, trial, assess or hold, framed technically.

One shared core (register → ingestion → extraction → developments → scoring) feeds two audience
lenses, rather than two separate systems. Built for the BIT Capital AI Engineer case study.

**Documents:** [System design](docs/DESIGN.md) · [Evaluation report](data/eval/report.md) ·
[Database schema](docs/schema.sql)

## Live demo

**https://63-186-176-99.sslip.io** (the login is in the submission email): the production deployment
on AWS (EC2, Postgres, Docker Compose, Caddy TLS), where labels and alert acks are saved. It serves the
real data collected from 1 Aug to 4 Nov 2026. Collection is paused, so the line under the header dates
the snapshot and the "last 7 days" views count back from the newest data.

## A five-minute tour

1. **Overview** (`/`): the latest week for each audience, open P1/P2 alerts, a radar of BIT's
   holdings, the latest digests and system health.
2. **Investment feed** (`/feed/investment`) and **AI-team feed** (`/feed/ai-team`): developments
   ranked for that audience (pushed alerts first, then the lens's recommended action, then relevance).
   Each card has the audience's *so what*, *what to do* and tickers or radar ring. Filter by week, lab,
   event type, ticker or relevance.
3. **A development page**, for example `/developments/405` (Anthropic files its S-1, the investment
   team's P1 alert):
   - every insight with its verbatim evidence quote, marked verified or not against the stored source;
   - the sources;
   - the score with its top contributing features;
   - **why it was flagged**: which materiality rules fired on which facts, and which did not;
   - the ticker exposure and both audience lenses.
4. **Alerts** (`/alerts`): P1/P2 pushes and P3 digest-only items per audience, each with the rule
   that fired.
5. **Reports** (`/reports`): the 28 weekly digests (14 per audience), every claim cited to a
   development. Start with the week of 26 Oct 2026.
6. **Register** (`/register/labs`, `/register/people`): the 14 labs and 103 people tracked and why,
   with identities across platforms and how each link was established.
7. **Status** (`/status`): source freshness against each source's SLO, failures, job runs, and LLM
   spend against the hard budgets.
8. **Label** (`/label`): the labelling queue that produces human ground truth for the scorer (AWS
   or local only).

Developments worth opening, all from the real data:

| ID | Development | Why it matters |
|---|---|---|
| 405 | Anthropic files S-1 for a Nasdaq IPO | Pushed as the investment team's P1: it will mark AMZN's Anthropic stake |
| 160 | DeepSeek V4.1-Flash cuts KV-cache memory per token | Tests the memory thesis behind MU and SIMO; cheaper long-context agents |
| 408 | Meta's Tahan scale-up switch for its MTIA chip | Custom-ASIC path at an NVDA customer; AVGO switch silicon |
| 46 | DeepSeek peak/off-peak API pricing | A compute-rationing signal; schedule batch work off-peak |
| 443 | Google re-routes Gemini 2.5 Pro to Gemini 3.x | Migration deadline for the AI team, found in a changelog |
| 478 | Jeff Dean and others co-found Discovery Loop | Tier-A talent move, through the curated register |

The JSON API (`/api/developments`, `/api/alerts`, `/api/health`; reference page at `/api`) is meant
for integration with an existing signals layer.

## Run it locally (no API keys needed)

The repo ships a snapshot of the real database (`data/snapshot/fli_snapshot.sqlite.gz`). With
[uv](https://docs.astral.sh/uv/) installed (it also installs Python 3.12):

```bash
git clone https://github.com/Sanjay-Suthraye/frontier-lab-intel.git && cd frontier-lab-intel
uv sync --python 3.12
uv run fli db init && uv run fli snapshot import && uv run fli serve
```

Open http://127.0.0.1:8000. `make demo` does the same. From your own machine, labelling and alert
acks work without a login.

A production-like stack (Postgres, web, and an optional scheduler worker), seeded from the same snapshot:

```bash
docker compose up -d --build   # web on http://localhost:8000
```

## How it works

```
config/*.yaml: labs, people, sources, ticker exposure map, materiality policy
      |
collectors (official APIs and published feeds) -> idempotent upsert, dedup, job ledger, source health
      |
document -> near-dup (MinHash) -> triage (Haiku) -> extraction (Haiku) -> quote verification (code)
      |
insight -> embeddings (local bge-small) -> clustered into developments -> novelty
      |
development -> ticker exposure -> features -> score per audience (logistic regression)
      |
lens per audience (Sonnet) -> faithfulness checks (code) -> materiality rules (YAML) -> alerts
                                                                                      -> weekly digests (Opus)
```

- **The development is the unit of value.** A launch seen on a blog, a changelog, Hugging Face and
  OpenRouter is one development, with one score, one lens and at most one alert per audience.
- **Deterministic spine, models at the leaves.** Each model call is a single schema-constrained step
  that can be cached, costed and evaluated. Code verifies every evidence quote against the stored
  source.
- **Alerting is a written policy, not a model.** [`config/materiality.yaml`](config/materiality.yaml)
  says what is material and for whom. Every alert records the rule and facts that fired it. The
  audience lens can veto an alert but never promote one.
- **Scoring is validated, not hand-weighted.** One logistic regression per audience, trained on labels
  and checked on a time-split hold-out. It is deployed only if it beats the policy prior.
- **Built to be run.** Every stage is idempotent and the scheduler re-runs it. There are freshness
  SLOs per source, hard LLM and API budgets enforced in code, a cost ledger per call, and a status
  page with a JSON health probe.
- **Sources** are official APIs and published feeds only, with robots.txt respected.

| Task | Model | Fallback |
|---|---|---|
| Triage, extraction, entity-resolution gray zone | Claude Haiku 5.5 | 3 attempts, then dead-letter (`status=failed`); gray-zone pairs stay `candidate` |
| Audience lens | Claude Sonnet 5.5 | Server-side fallback; deterministic summary |
| Weekly digest | Claude Opus 5.5 | Retry on Sonnet 5.5, then a deterministic, cited digest |
| Silver labels (LLM judge) | Claude Opus 5.5 | Kept separate; human labels override |
| Embeddings | bge-small (local, fastembed) | n/a |

The reasoning behind each choice, and the alternatives rejected, are in [docs/DESIGN.md](docs/DESIGN.md).

## Results

Full detail is in [data/eval/report.md](data/eval/report.md) (generated by `fli eval`).

| | |
|---|---|
| Corpus | 1,008 documents (incl. 6 curated talent moves), 1 Aug - 4 Nov 2026 → 689 insights → 476 developments |
| Citations | 99.8% of 1,255 evidence quotes verified verbatim in the stored source |
| Scoring (time-split hold-out) | AUC 0.87 investment / 0.82 AI team, vs 0.80 / 0.75 for the policy prior |
| Alerts | 29 investment pushes (2.1 a week), 56 AI-team pushes (4.1 a week); daily cap of 2 pushes per audience (P1 is never demoted) |
| Known events | 12 of 18 recalled, from a list written before seeing any output; 11 alerted their primary audience |
| Entity resolution | precision 0.974, recall 0.925 |
| Digests | 27 of 28 written by the model; 1 deterministic fallback |
| Cost | $14.40 of LLM spend for the whole window incl. all re-runs and debugging; about $0.60 a week steady state |

## Running the pipeline yourself

Copy `.env.example` to `.env` and add keys. Only `ANTHROPIC_API_KEY` is needed for the model stages;
every other key is optional, and sources without one are skipped cleanly.

```bash
uv run fli db init && uv run fli register     # load config/*.yaml (labs, people, sources, exposure map)
uv run fli ingest --since 2026-08-01          # run collectors (official APIs and published feeds only)
uv run fli pipeline                           # resolve -> dedup -> triage -> extract -> embed -> cluster
                                              #   -> novelty -> exposure -> score -> lens -> alerts
uv run fli labels silver --n 300              # LLM-judge labels (kept separate from human labels)
uv run fli train                              # time-split validated scoring model per audience
uv run fli digest --backfill                  # weekly reports
uv run fli eval                               # writes data/eval/report.{json,md}
uv run fli costs                              # spend per workflow and model, from the ledger
uv run fli worker                             # scheduler: cadenced ingest, pipeline, health, digests
```

Hard budget stops are enforced in code (`LLM_BUDGET_TOTAL_USD`, `LLM_BUDGET_DAILY_USD`,
`X_BUDGET_TOTAL_USD`).

## Repository layout

```
config/            what is tracked (files, not a UI): labs, people, sources_*, exposure map, materiality rules
src/fli/
  db/              SQLAlchemy schema (portable SQLite/Postgres); DDL in docs/schema.sql
  ingest/          collector contract, runner (idempotent upserts, dedup, job ledger), collectors/
  register/        YAML loader, entity resolution, contributor score
  extract/         triage and extraction (LLM), quote verification (code)
  analyze/         near-dedup, embeddings, clustering into developments, novelty, exposure, features, scoring, labels
  deliver/         audience lenses, materiality rules, alerts, digests, notifications
  llm/             the only module that calls the model API: cost ledger, budgets, cache, fallbacks, batches
  prompts/         every prompt, versioned (the version is stored on every output row)
  ops/             health checks, scheduler, snapshot export/import
  web/             FastAPI + server-rendered pages + JSON API
  eval/            evaluation harness
data/eval/         ground truth (known events, entity-resolution pairs), labels, metrics, the evaluation report
data/snapshot/     the committed real-data snapshot
deploy/aws/        one-command EC2 deployment (Docker Compose + Caddy TLS) and teardown
deploy/vercel/     read-only web deployment on Vercel's free tier
tests/             482 tests, no network and no LLM calls
```

## Deployment

- **AWS** (`deploy/aws/deploy.sh`):
  - provisions one EC2 instance (eu-central-1, t4g.small, encrypted disk, IMDSv2, SSH only from your
    current IP);
  - ships the code and `.env`, and starts Postgres and the web app behind Caddy, with an automatic
    TLS certificate on `<ip>.sslip.io`;
  - refuses to run unless `.env` has `FLI_BASIC_AUTH` with a password of at least 12 characters, and
    puts every page behind that login (`FLI_AUTH_ALL=1`);
  - starts the scheduler worker only with `FLI_WORKER=1`;
  - `FLI_RESET_DB=1` re-seeds the database from the snapshot, and `deploy/aws/teardown.sh` removes
    everything it created.
- **Vercel** (`deploy/vercel/deploy.sh`, needs `VERCEL_TOKEN`):
  - publishes a 6 MB bundle with only the web dependencies, serving the snapshot from `/tmp`;
  - uses the same login;
  - sets `FLI_READ_ONLY=1`, because that database does not survive a cold start.

## Access control

- **Writes** (labels, alert acks) need `FLI_BASIC_AUTH=user:password` unless the request comes from
  a loopback client. Labels are then recorded as `human:<user>`.
- **`FLI_AUTH_ALL=1`** puts every page and API read behind the same login; `/healthz` stays public.
- **`FLI_READ_ONLY=1`** refuses all writes and hides the forms.

## Tests

```bash
uv run pytest -q        # unit + integration tests; no network, no LLM
uv run ruff check src tests
```
