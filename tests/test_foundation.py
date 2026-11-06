from fli.ingest.records import COLLECTORS, IngestRecord, collector
from fli.llm.client import cost_usd
from fli.textutil import canonical_url, content_hash


def test_canonical_url_collapses_platform_variants():
    assert canonical_url("http://www.arxiv.org/pdf/2509.01234v3") == "https://arxiv.org/abs/2509.01234"
    assert canonical_url("https://twitter.com/OpenAI/status/123?s=20") == "https://x.com/i/status/123"
    assert canonical_url("https://openai.com/news/x/?utm_source=a&id=1#top") == "https://openai.com/news/x?id=1"


def test_content_hash_ignores_case_and_whitespace():
    assert content_hash("Hello   World") == content_hash("hello world")


def test_haiku_long_context_rate_card():
    assert cost_usd("claude-haiku-5-5", 50_000, 0) == 50_000 * 0.10 / 1_000_000  # <=100K prompt: short rate card
    assert cost_usd("claude-haiku-5-5", 200_000, 0) == 200_000 * 0.50 / 1_000_000  # >100K prompt: long rate card
    assert cost_usd("claude-opus-5-5", 1000, 1000, batch=True) == (1000 * 4 + 1000 * 20) / 1_000_000 / 2


def test_collector_registry_and_persist(db):
    from fli.db.models import Document, Source
    from fli.ingest.runner import persist_records

    @collector("test_echo")
    def _echo(source, window, settings):
        return []

    assert "test_echo" in COLLECTORS
    with db() as s:
        src = Source(id="t:1", adapter="test_echo", kind="official_blog", source_class="official")
        s.add(src)
        s.flush()
        recs = [IngestRecord(natural_key="a", url="https://x.com/i/status/9", title="T", content_text="body " * 80)]
        assert persist_records(s, src, recs) == (1, 0)
        assert persist_records(s, src, recs) == (0, 0)  # idempotent
        s.commit()
        assert s.query(Document).count() == 1
