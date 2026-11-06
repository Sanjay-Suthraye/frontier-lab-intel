# Frontier Lab Intelligence - common commands. Everything runs through `uv run fli ...`.
.PHONY: setup demo serve test lint schema register backfill pipeline digests labels train eval health costs worker up down deploy-aws teardown-aws snapshot

setup:            ## install dependencies (Python 3.12 via uv)
	uv sync --python 3.12

demo: setup       ## load the committed snapshot (real data) into SQLite and open the UI - no API keys needed
	uv run fli db init
	uv run fli snapshot import
	uv run fli serve

serve:
	uv run fli serve

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

schema:           ## regenerate docs/schema.sql from the models
	uv run fli db schema

register:         ## load config/*.yaml into the database
	uv run fli db init && uv run fli register

backfill: register ## ingest + process the 3-month window from scratch (needs API keys; ~$10-15 of LLM spend)
	uv run fli ingest --since 2026-08-01 --force
	uv run fli resolve-entities
	uv run fli pipeline
	uv run fli contributors

pipeline:
	uv run fli pipeline

digests:
	uv run fli digest --backfill

labels:           ## LLM-judge silver labels, then export a CSV for human review
	uv run fli labels silver --n 300
	uv run fli labels export

train:
	uv run fli train

eval:
	uv run fli eval

health:
	uv run fli health

costs:
	uv run fli costs

worker:
	uv run fli worker

snapshot:         ## refresh the committed demo snapshot from the working DB
	uv run fli snapshot export

up:               ## production-like stack: Postgres + web + worker (seeded from the snapshot)
	docker compose up -d --build

down:
	docker compose down

deploy-aws:       ## provision/update the EC2 deployment (AWS CLI credentials required)
	deploy/aws/deploy.sh

teardown-aws:
	deploy/aws/teardown.sh
