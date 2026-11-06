-- 2026-11-04: mark lab-level exposure links as method = 'lab'.
--
-- Since this change, fli.analyze.exposure stores a link that only the lab's curated relations
-- (config/labs.yaml public_exposure) support as method = 'lab': it describes the lab, not the
-- development, and score features count only event-specific method = 'rule' links. This one-off
-- update relabels the rows written before the change (the stage only ever upgrades or replaces its
-- own 'lab' rows, so it cannot fix older 'rule' rows itself).
--
-- A row is lab-level when its rationale (the "; "-joined evidence of every rule that fired) holds a
-- "<Lab> public exposure: ..." part and no part from the other two rules, "named in development
-- text: ..." (mention) or "theme '...': ..." (keywords). Matching the other rules' text, rather than
-- the absence of ';', keeps a row lab-level when a labs.yaml note itself contains a ';'.
-- On data/fli.sqlite as of 2026-11-04 it relabels 1,797 of 2,400 rows (1,689 at confidence 0.6,
-- 108 at 0.4 'verify'); all 2,400 rows are method 'rule' before it runs.
--
-- Portable (SQLite and Postgres), idempotent (a re-run matches no rows), no schema change.
-- Apply, then re-fit and re-score (features changed for every development with lab links):
--   sqlite3 data/fli.sqlite < scripts/migrations/20261104_exposure_method.sql    (or: psql "$DATABASE_URL" -f ...)
--   uv run fli train
-- Check: SELECT method, COUNT(*) FROM exposure_link GROUP BY method;  -- lab 1797, rule 603 on that data

UPDATE exposure_link
SET method = 'lab'
WHERE method = 'rule'
  AND rationale LIKE '%public exposure:%'
  AND rationale NOT LIKE '%named in development text:%'
  AND rationale NOT LIKE '%theme ''%';
