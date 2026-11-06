"""Filings collector: SEC EDGAR full-text search (efts.sec.gov).

Public companies disclose lab relationships (compute contracts, investments, supply agreements) in
8-K / 10-Q / 10-K / S-1 filings, often before or in more detail than the press. A filing is a
primary source with a legal-liability standard, so these records are strong evidence downstream.

SEC fair-access policy: a descriptive User-Agent with a contact address is mandatory (hence
requires=("sec_user_agent",)) and traffic stays far below 10 requests/second.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fli.config import Settings
from fli.ingest.records import IngestRecord, SourceInfo, Window, collector
from fli.timeutil import to_naive_utc

log = logging.getLogger("fli.ingest.filings")

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
DEFAULT_FORMS = "8-K,S-1,10-Q,10-K"
PAGE_SIZE = 100  # EFTS returns up to 100 hits per page; paginate with `from`
REQUEST_GAP_S = 0.25
ITEM_NAMES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "3.02": "Unregistered Sales of Equity Securities",
    "5.02": "Departure or Appointment of Directors or Officers",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}
_TAGS = re.compile(r"</?[^>]+>")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


@retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _search_page(client: httpx.Client, params: dict) -> dict:
    resp = client.get(EFTS_URL, params=params)
    resp.raise_for_status()
    return resp.json()


def filer_name(display_names: list[str] | None) -> str:
    """'AMAZON COM INC  (AMZN)  (CIK 0001018724)' -> 'AMAZON COM INC (AMZN)'."""
    name = (display_names or ["Unknown filer"])[0]
    return " ".join(re.sub(r"\(CIK \d+\)", "", name).split())


def filing_url(cik: str, adsh: str, filename: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{filename}"


def _snippet(hit: dict) -> str | None:
    """EFTS may return highlight fragments; use them when present (tags stripped)."""
    frags = [f for v in (hit.get("highlight") or {}).values() for f in (v or [])]
    text = " … ".join(" ".join(_TAGS.sub("", f).split()) for f in frags[:3])
    return text or None


def hits_to_records(hits: list[dict], *, query: dict) -> list[IngestRecord]:
    """One record per filing (accession number); exhibits that also matched are listed in meta."""
    by_adsh: dict[str, list[dict]] = {}
    for hit in hits:
        src = hit.get("_source") or {}
        adsh = src.get("adsh") or str(hit.get("_id", "")).split(":")[0]
        if adsh:
            by_adsh.setdefault(adsh, []).append(hit)
    records = []
    for adsh, group in by_adsh.items():
        # prefer the main document over exhibits, then relevance (EFTS order)
        best = next((h for h in group if (h.get("_source") or {}).get("file_type") == (h.get("_source") or {}).get("form")),
                    group[0])
        src = best.get("_source") or {}
        filename = str(best.get("_id", "")).split(":", 1)[-1]
        ciks = src.get("ciks") or []
        if not ciks or not filename:
            continue
        filer = filer_name(src.get("display_names"))
        form = src.get("form") or (src.get("root_forms") or ["?"])[0]
        filed = src.get("file_date")
        items = [f"{i} {ITEM_NAMES.get(i, '')}".strip() for i in src.get("items") or []]
        desc = (src.get("file_description") or "").strip()
        doc_label = src.get("file_type") or form
        lines = [(f"{filer} filed a {form} with the SEC on {filed}. The matching document "
                  f"({doc_label}{': ' + desc if desc and desc != doc_label else ''}) mentions {query['q']}.")]
        if items:
            lines.append("8-K items: " + "; ".join(items) + ".")
        if src.get("period_ending"):
            lines.append(f"Period ending {src['period_ending']}.")
        snippet = _snippet(best)
        if snippet:
            lines.append(f"Excerpt: {snippet}")
        others = [str(h.get("_id", "")).split(":", 1)[-1] for h in group if h is not best]
        if others:
            lines.append(f"Other matching documents in this filing: {', '.join(others[:10])}.")
        records.append(IngestRecord(
            natural_key=f"sec:{adsh}",
            url=filing_url(ciks[0], adsh, filename),
            title=f"{filer} {form} ({filed})" + (f": {desc}" if desc and desc not in (form, doc_label) else ""),
            content_text="\n".join(lines),
            published_at=to_naive_utc(filed),
            author=filer,
            lab_id=query.get("lab"),
            source_class="filing",
            meta={"adsh": adsh, "cik": ciks[0], "form": form, "file_type": src.get("file_type"),
                  "file_date": filed, "items": src.get("items") or [], "period_ending": src.get("period_ending"),
                  "query": query["q"], "matched_files": [filename, *others][:20], "sics": src.get("sics") or [],
                  "snippet": snippet},
        ))
    return records


def search_filings(client: httpx.Client, query: dict, *, forms: str, start: str, end: str, cap: int) -> list[dict]:
    hits: list[dict] = []
    offset = 0
    while len(hits) < cap:
        params: dict[str, Any] = {"q": query["q"], "forms": forms, "dateRange": "custom", "startdt": start,
                                  "enddt": end}
        if offset:
            params["from"] = offset
        data = _search_page(client, params)
        page = (data.get("hits") or {}).get("hits") or []
        hits += page
        total = ((data.get("hits") or {}).get("total") or {}).get("value", 0)
        offset += len(page)
        if not page or offset >= total:
            break
        time.sleep(REQUEST_GAP_S)
    return hits[:cap]


def _queries(source: SourceInfo) -> list[dict]:
    raw = source.config.get("queries") or ([source.config["query"]] if source.config.get("query") else [])
    out = [{"q": q, "lab": source.lab_id} if isinstance(q, str) else {"lab": source.lab_id, **q} for q in raw]
    if not out:
        raise ValueError(f"{source.id}: config.queries is required")
    return out


@collector("edgar_fts", requires=("sec_user_agent",), description="SEC EDGAR full-text search for lab mentions")
def edgar_fts(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    forms = source.config.get("forms") or DEFAULT_FORMS
    if isinstance(forms, list):
        forms = ",".join(forms)
    cap = int(source.config.get("max_per_query") or 200)
    if window.max_items:
        cap = min(cap, window.max_items)
    start, end = f"{window.since:%Y-%m-%d}", f"{window.until:%Y-%m-%d}"
    headers = {"User-Agent": settings.sec_user_agent or "", "Accept": "application/json"}
    records: dict[str, IngestRecord] = {}
    with httpx.Client(timeout=settings.http_timeout_s, headers=headers) as client:
        for query in _queries(source):
            hits = search_filings(client, query, forms=forms, start=start, end=end, cap=cap)
            for rec in hits_to_records(hits, query=query):
                records.setdefault(rec.natural_key, rec)  # first query wins attribution
            log.info("edgar %s: %d hits for %s", source.id, len(hits), query["q"])
            time.sleep(REQUEST_GAP_S)
    return list(records.values())
