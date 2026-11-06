# Market, press and filings fixtures

- `openrouter_models.json`: real `openrouter.ai/api/v1/models` (2026-11-04), trimmed to eleven models
  (tracked labs, a `:batch` variant, a `~` alias, untracked providers, one model older than the window,
  and two models with time-of-day price overrides).
- `gdelt_anthropic.json`: real GDELT DOC 2.0 `artlist` response for an Anthropic event query (2026-10-27..11-03).
- `gdelt_throttled.txt`: real HTTP 429 plain-text body GDELT returns when requests come faster than 1 per 5 s.
- `gdelt_query_error.txt`: real plain-text rejection (HTTP 200) of a query with an unquoted dash (`S-1`).
- `edgar_fts_anthropic.json`: **constructed** from the documented EFTS search response (no SEC user agent
  is configured, so no live call was made). Issuers and CIKs are fictitious.
