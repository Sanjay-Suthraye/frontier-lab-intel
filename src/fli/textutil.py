"""Text and URL normalisation shared by ingestion, dedup and quote verification."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|s$|si$|igshid$|cmpid$|source$)", re.I)
_WS = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")


def canonical_url(url: str) -> str:
    """Stable identity for a URL: https, no www, no tracking params, no fragment, and
    platform-specific collapsing (arXiv versions, X status ids)."""
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = parts.path or "/"
    if host.endswith("arxiv.org"):
        m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", path)
        if m:
            return f"https://arxiv.org/abs/{m.group(1)}"
    if host in {"twitter.com", "x.com", "mobile.twitter.com", "mobile.x.com"}:
        m = re.search(r"/status(?:es)?/(\d+)", path)
        if m:
            return f"https://x.com/i/status/{m.group(1)}"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)])
    scheme = "https" if parts.scheme in {"http", "https", ""} else parts.scheme
    return urlunsplit((scheme, host, path, query, ""))


def clean_text(text: str | None) -> str:
    """NFKC, strip zero-width/control chars, collapse spaces, keep paragraph breaks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NL.sub("\n\n", text).strip()


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation used to compare a quote with its source (quote verification)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_match(text).encode("utf-8")).hexdigest()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_break = max(cut.rfind("\n\n"), cut.rfind(". "))
    return (cut[: last_break + 1] if last_break > max_chars * 0.8 else cut).rstrip() + "\n[truncated]"


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "x"
