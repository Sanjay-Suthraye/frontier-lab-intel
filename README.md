# Frontier Lab Intelligence

Tracks what frontier AI labs and their key people publish, turns it into cited, scored insights, and
delivers alerts and weekly digests to two audiences at BIT Capital:

- **Investment team** - what a lab development means for positions and theses, mapped to a closed
  list of public tickers (BIT's top holdings flagged with their fund weights).
- **AI team** - what to adopt, trial, assess or ignore, framed technically.

One shared core (register -> ingestion -> extraction -> events -> scoring) feeds two last-mile lenses.
The design document is in [docs/DESIGN.md](docs/DESIGN.md); the evaluation report is
[data/eval/report.md](data/eval/report.md).

## Run it in three commands (no API keys needed)

The repo ships a snapshot of the real database (`data/snapshot/fli_snapshot.sqlite.gz`), so the UI is
useful immediately:

```bash
git clone <this repo> && cd frontier-lab-intel
uv sync --python 3.12          # https://docs.astral.sh/uv/ ; installs everything incl. Python 3.12
uv run fli db init && uv run fli snapshot import && uv run fli serve
```

Open http://127.0.0.1:8000 . `make demo` does the same.

Production-like stack (Postgres + web + scheduler worker), seeded from the same snapshot:

```bash
docker compose up -d --build   # web on http://localhost:8000
```

## What you can look at

| Page | What it answers |
|---|---|
| `/` | This week's top developments per audience, open alerts, latest reports, system health |
| `/feed/investment`, `/feed/ai-team` | Ranked developments with the audience lens: so-what, what to do, tickers / action |
| `/developments/{id}` | Every insight with its verbatim evidence (verified or not), sources, score and *why flagged*, exposure links, both lenses, label buttons |
| `/register/labs`, `/register/people` | Who is tracked and why, identities with how each link was established, affiliation history, contributor score |
| `/alerts`, `/reports` | Material alerts and weekly digests per audience |
| `/label` | Labelling queue (human ground truth for the scorer) |
| `/status` | Source freshness vs SLO, failures, job runs, ops events, LLM and API spend vs budget |
| `/api/developments`, `/api/alerts`, `/api/health` | JSON for integration with BIT's existing Signals & Alerts layer |

## Running the pipeline yourself

Copy `.env.example` to `.env` and add keys (only `ANTHROPIC_API_KEY` is needed for the LLM stages;
every other key is optional and its sources are skipped cleanly when missing).

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

## Layout

```
config/            what is tracked (files, not a UI): labs, people, sources_*, exposure map, materiality rules
src/fli/
  db/              SQLAlchemy schema (portable SQLite/Postgres) - DDL in docs/schema.sql
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
data/eval/         ground truth (known events, ER pairs), labels, metrics, the evaluation report
data/snapshot/     the committed real-data snapshot
deploy/aws/        one-command EC2 deployment (Docker Compose + Caddy TLS) and teardown
tests/             479 tests, no network and no LLM calls
```

## Access control

- Writes (labels, alert acks) need `FLI_BASIC_AUTH=user:password` in `.env` unless the request comes
  from a loopback client; labels are then recorded as `human:<user>`. The local `docker compose` stack
  counts as non-local (Docker NAT), so set `FLI_BASIC_AUTH` there if you want to label.
- `FLI_AUTH_ALL=1` puts every page and API read behind the same login (`/healthz` stays public).

## Deployment

`deploy/aws/deploy.sh` provisions one EC2 instance (eu-central-1, t4g.small, encrypted disk, IMDSv2,
SSH only from your current IP), ships the code and `.env`, and starts Postgres + web behind Caddy with
an automatic TLS certificate on `<ip>.sslip.io`. It refuses to run unless `.env` has `FLI_BASIC_AUTH`
with a password of at least 12 characters (`FLI_DEPLOY_CHECK_ONLY=1` checks that), and the AWS overlay
sets `FLI_AUTH_ALL=1`. The scheduler worker (continuous ingestion) starts only with `FLI_WORKER=1`.
`deploy/aws/teardown.sh` removes everything it created.

## Results at a glance (full detail in docs/DESIGN.md and data/eval/report.md)

| | |
|---|---|
| Corpus | 1,008 documents (incl. 6 curated talent moves), Aug 1 - Nov 4 2026; 689 insights; 477 developments |
| Citations | 99.8% of 1,255 evidence quotes verified verbatim in the stored source |
| Scoring (time-split hold-out) | AUC 0.87 investment / 0.82 AI team vs 0.80 / 0.75 for the policy prior |
| Alerts | 29 investment pushes (2.1/week), 56 AI-team pushes (4.1/week); P1 reserved for material items |
| Known events | 12/18 recalled (incl. the Discovery Loop founding and the DeepMind reshuffle via the curated register); 11 alerted their primary audience |
| Cost | $14.40 of LLM spend for the whole window incl. all re-runs and debugging; ~$0.60/week steady state |

## Tests

```bash
uv run pytest -q        # unit + integration tests; no network, no LLM
uv run ruff check src tests
```
