# Job-board fixtures

- `greenhouse_departments_xai.json`: real `boards-api.greenhouse.io/v1/boards/xai/departments`, trimmed.
- `greenhouse_jobs_anthropic.json`: real `.../boards/anthropic/jobs?content=true`, 3 jobs, content shortened.
- `ashby_openai.json`: real `api.ashbyhq.com/posting-api/job-board/openai`, 7 postings, descriptions dropped.
- `lever_mistral_empty.json`: real response of `api.lever.co/v0/postings/mistral?mode=json` (empty board).
- `lever_postings.json`: **constructed** from the documented Lever postings schema (no tracked lab has a
  non-empty Lever board).
All recorded 2026-11-04.
