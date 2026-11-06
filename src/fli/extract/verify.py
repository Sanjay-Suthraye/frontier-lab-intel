"""Evidence-quote verification: the main hallucination control on extracted insights.

Each quote the extractor returns must occur in the stored document text (title + content). Both
sides go through `normalize_for_match` (NFKC, case, typographic quotes/dashes, whitespace) plus
removal of markdown emphasis characters, then:

  1. exact substring                                  -> verified, method 'exact'
  2. rapidfuzz partial_ratio on the best-matching window >= 92, AND every salient token of the
     quote (each number, compared without separators, and each word of 4+ letters) occurs in that
     window                                           -> verified, method 'fuzzy'
     (absorbs punctuation, markup and dropped short words, but not meaning-changing edits:
     measured on a recorded Gemini 4 quote, "Carbon" -> "Cobalt" still scores ~97 and
     "88.2%" -> "86.2%" ~99, so the score alone would verify a falsified quote)
  3. otherwise                                        -> not verified ('none')

A quote with an ellipsis ("A ... B") is checked part by part and verified only if every part is.
Very short quotes are not evidence ('too_short'), and a quote longer than the document cannot be
fuzzy-matched against it (partial_ratio would otherwise match the document inside the quote).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import asdict, dataclass

from rapidfuzz import fuzz

from fli.textutil import normalize_for_match

FUZZY_THRESHOLD = 92.0
MIN_QUOTE_CHARS = 8  # shorter normalised quotes ("GPT-6") prove nothing
MIN_FUZZY_CHARS = 20  # fuzzy scores on short strings are unreliable
_ELLIPSIS = re.compile(r"\s*\[?\.\.\.+\]?\s*")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")
WINDOW_PAD = 24  # chars added around the aligned window so boundary words are not cut
_MARKUP = re.compile(r"[*_`#>|]+")
_SPACES = re.compile(r"\s+")


def squash(text: str) -> str:
    text = _MARKUP.sub(" ", normalize_for_match(text))
    return _SPACES.sub(" ", text).strip()


@dataclass
class QuoteCheck:
    quote: str
    verified: bool
    method: str  # exact | fuzzy | parts | none | too_short | empty
    score: float
    matched: str | None = None  # best-matching window (normalised) when fuzzy
    missing: list[str] | None = None  # salient quote tokens absent from that window


def salient_tokens(text: str) -> set[str]:
    """Numbers (separators removed: 100,000 == 100.000) and words of 4+ letters."""
    out = set()
    for tok in _TOKEN.findall(text):
        if tok[0].isdigit():
            out.add(tok.replace(",", "").replace(".", ""))
        elif len(tok) >= 4:
            out.add(tok)
    return out


def _check_part(part: str, source: str) -> QuoteCheck:
    if len(part) < MIN_QUOTE_CHARS:
        return QuoteCheck(part, False, "too_short", 0.0)
    if part in source:
        return QuoteCheck(part, True, "exact", 100.0)
    if len(part) < MIN_FUZZY_CHARS or len(part) > len(source):
        return QuoteCheck(part, False, "none", 0.0)
    al = fuzz.partial_ratio_alignment(part, source)
    if al is None:
        return QuoteCheck(part, False, "none", 0.0)
    window = source[max(0, al.dest_start - WINDOW_PAD):al.dest_end + WINDOW_PAD]
    missing = sorted(salient_tokens(part) - salient_tokens(window))
    ok = al.score >= FUZZY_THRESHOLD and not missing
    return QuoteCheck(part, ok, "fuzzy" if ok else "none", round(al.score, 1),
                      source[al.dest_start:al.dest_end][:300], missing or None)


def check_quote(quote: str, source_squashed: str) -> QuoteCheck:
    """Verify one quote against an already squashed source text."""
    q = squash(quote).strip(" \"'")
    if not q:
        return QuoteCheck(quote, False, "empty", 0.0)
    parts = [p.strip(" \"'") for p in _ELLIPSIS.split(q) if p.strip(" \"'")]
    if len(parts) > 1:
        checks = [_check_part(p, source_squashed) for p in parts]
        ok = all(c.verified for c in checks)
        missing = sorted({t for c in checks for t in (c.missing or [])})
        return QuoteCheck(quote, ok, "parts" if ok else "none", min(c.score for c in checks), None, missing or None)
    return dataclasses.replace(_check_part(q, source_squashed), quote=quote)


def verify_quotes(quotes: list[str], source_text: str) -> dict:
    """Per-quote details plus a summary.

    status: 'verified' (every quote verified), 'partial' (some), 'unverified' (none, or no quotes).
    `Insight.quotes_verified` is True only for 'verified'; 'unverified' insights are kept but
    flagged, and alerting ignores them.
    """
    source = squash(source_text)
    checks = [check_quote(q, source) for q in quotes]
    n_ok = sum(c.verified for c in checks)
    status = "verified" if checks and n_ok == len(checks) else ("partial" if n_ok else "unverified")
    return {
        "status": status,
        "all_verified": status == "verified",
        "n_quotes": len(checks),
        "n_verified": n_ok,
        "threshold": FUZZY_THRESHOLD,
        "quotes": [{k: v for k, v in asdict(c).items() if v is not None} for c in checks],
    }


def verified_quotes(verification: dict | None) -> list[str]:
    """Quotes of an insight that passed verification (what lenses and digests may cite)."""
    return [q["quote"] for q in (verification or {}).get("quotes", []) if q.get("verified")]
