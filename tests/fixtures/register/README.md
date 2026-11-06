Fixtures for tests/test_register.py (no network, no LLM).

- `config/labs.yaml`, `config/people.yaml`: a 9-person register in the real schema (loader round trip,
  attribution and contributor tests).
- `github_user_karpathy.json`: GitHub REST `GET /users/karpathy`, recorded 2026-11-04, trimmed to the
  fields entity resolution reads. Real quirk it captures: `twitter_username` is null and the X link sits
  in `blog`.
- `openalex_authors_jeff_dean.json`: OpenAlex `GET /authors?search=Jeff Dean`, recorded 2026-11-04,
  first 6 results, trimmed. Captures split author profiles and namesakes ("Jeffry L. Dean").
- `s2_rate_limited.json`: Semantic Scholar `GET /graph/v1/author/search` body as returned to an
  unauthenticated client on 2026-11-04 (HTTP 429 on 4 of 4 attempts) - the reason OpenAlex is the fallback.
