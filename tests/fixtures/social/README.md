# Social fixtures

- `bsky_feed_page1.json`, `bsky_feed_page2.json`: real `app.bsky.feed.getAuthorFeed` response
  (public AppView, `filter=posts_no_replies`, recorded 2026-11-04), trimmed to five items covering
  link facets, an external embed, images, a quote (`app.bsky.embed.record`) and a repost. Split into
  two pages; the page-1 `cursor` value is synthetic.
- `x_*.json`: **constructed** from the X API v2 response schema (users/by/username, users/:id/tweets).
  The live check on 2026-11-04 got `403 client-not-enrolled` on its first request (the configured app
  is not attached to a Project), so no timeline could be recorded. Post texts are placeholders.
  `x_error_403.json` keeps the live `detail` message verbatim with `client_id` redacted.
