"""Entity resolution: who wrote a document, and which accounts belong to the same person.

Three passes, cheapest first. Every link records how it was made (Identity.method and
Document.meta['people_matched'][*].method), so attribution stays auditable:

1. Document attribution. Author names (paper author lists, blog bylines) and author handles (posts,
   release authors) are matched to tracked people: exact normalised name/handle first, otherwise
   rapidfuzz token_sort_ratio >= 93 *and* lab agreement (the document's lab, or the lab of a
   co-author matched exactly, is one of the person's labs). People flagged meta.common_name - and
   every name on an off-domain (non cs/stat/eess) paper - need lab agreement even for an exact
   match, because namesakes are the dominant error.
2. Cross-platform links. A GitHub profile is self-asserted evidence: its twitter_username confirms
   (or proposes, as a candidate) an X identity; blog and company are stored as evidence.
3. Gray zone (85 <= score < 93, with lab agreement). Haiku answers same/different/unsure and must
   quote overlapping facts; 'same' is accepted only if a quoted fact appears verbatim in both records.

Idempotent and incremental: resolved documents carry meta['er'] = {v, reg, gray}, where `reg`
fingerprints the register, so unmatched documents are re-scanned only after the register changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml
from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from fli.config import CONFIG_DIR, Settings, get_settings
from fli.db.models import Affiliation, Document, Identity, Person, Source
from fli.db.session import session_scope
from fli.textutil import normalize_for_match
from fli.timeutil import to_naive_utc, utcnow

log = logging.getLogger("fli.register.er")

ER_VERSION = 1
FUZZY_MIN, GRAY_MIN = 93.0, 85.0
INITIAL_VARIANT_SCORE = 95.0  # "Quoc V. Le" vs "Quoc Le": identical once middle initials are dropped
FIRST_INITIAL_SCORE = 88.0  # "J. Dean" vs "Jeff Dean": plausible, so it goes to adjudication
TIER_RANK = {"A": 0, "B": 1, "C": 2}
METHOD_RANK = {"handle": 0, "exact": 0, "fuzzy": 1, "llm_adjudicated": 2}
HANDLE_KINDS = {"x_account": "x", "bluesky_account": "bluesky", "github_org": "github", "hf_org": "hf"}
SOCIAL = {"x", "github", "hf", "bluesky"}
ML_CATEGORIES = ("cs.", "stat.", "eess.")
LLM_CAP = 50  # adjudications per run (Haiku, ~$0.0002 each)
GITHUB_CAP = 100  # GitHub user lookups per run
GITHUB_REFRESH = timedelta(days=30)
_BOT = re.compile(r"\[bot\]$|-bot$|^github-actions", re.IGNORECASE)
_SPLIT = re.compile(r"\s*(?:;|,|&|\band\b)\s*")
_ET_AL = re.compile(r"\s*\bet al\.?\s*$", re.IGNORECASE)
_X_URL = re.compile(r"^(?:https?://)?(?:www\.)?(?:twitter|x)\.com/@?(\w{1,15})/?$", re.IGNORECASE)
_FOLD = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss",
                       "đ": "d", "ı": "i"})
ER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["same", "different", "unsure"]},
        "overlapping_facts": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "overlapping_facts", "reason"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------------------ name matching
def norm_name(name: str) -> str:
    """ASCII-folded lowercase words: 'Łukasz  Kaiser' -> 'lukasz kaiser', 'Quoc V. Le' -> 'quoc v le'."""
    s = unicodedata.normalize("NFKD", (name or "").translate(_FOLD)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z]+", " ", re.sub(r"['`]", "", s.lower())).split())


def name_score(a: str, b: str) -> float:
    """Similarity of two normalised names on rapidfuzz's 0-100 scale, plus two name-aware rules:
    a middle-initial variant scores 95 and a first-initial form ('j dean') scores at least 88."""
    if a == b:
        return 100.0
    ta, tb = a.split(), b.split()
    if len(ta) < 2 or len(tb) < 2:
        return 0.0
    score = float(fuzz.token_sort_ratio(a, b))
    if ta[-1] == tb[-1]:
        mid_a, mid_b = set(ta[1:-1]), set(tb[1:-1])
        if ta[0] == tb[0] and not (mid_a and mid_b and mid_a != mid_b):
            score = max(score, INITIAL_VARIANT_SCORE)
        elif (len(ta[0]) == 1 and tb[0].startswith(ta[0])) or (len(tb[0]) == 1 and ta[0].startswith(tb[0])):
            score = max(score, FIRST_INITIAL_SCORE)
    return score


@dataclass(frozen=True)
class PersonRef:
    id: str
    name: str
    tier: str
    labs: frozenset[str]
    common_name: bool
    names: tuple[str, ...]  # normalised: the person's name plus confirmed arxiv_name identities


@dataclass
class Match:
    person_id: str
    tier: str
    method: str  # exact | fuzzy | handle | llm_adjudicated
    score: float
    mention: str
    position: int = 0

    def rank(self) -> tuple:
        return TIER_RANK.get(self.tier, 3), METHOD_RANK.get(self.method, 3), self.position, self.person_id


@dataclass
class Gray:
    person_id: str
    mention: str
    position: int
    score: float


def dedupe(matches: Iterable[Match]) -> list[Match]:
    """One match per person (the strongest), most senior first: tier A > B > C, then method."""
    best: dict[str, Match] = {}
    for m in sorted(matches, key=Match.rank):
        best.setdefault(m.person_id, m)
    return sorted(best.values(), key=Match.rank)


class PeopleIndex:
    """Tracked people keyed by normalised name, name token and (platform, handle)."""

    def __init__(self, people: Iterable[PersonRef], handles: dict[tuple[str, str], str] | None = None):
        self.people = {p.id: p for p in people}
        self.handles = dict(handles or {})
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.by_token: dict[str, set[str]] = defaultdict(set)
        for p in self.people.values():
            for variant in p.names:
                self.by_name[variant].add(p.id)
                for tok in variant.split():
                    if len(tok) > 1:
                        self.by_token[tok].add(p.id)

    @classmethod
    def from_db(cls, s: Session) -> PeopleIndex:
        labs: dict[str, set[str]] = defaultdict(set)
        for pid, lab in s.execute(select(Affiliation.person_id, Affiliation.lab_id)):
            labs[pid].add(lab)
        names: dict[str, set[str]] = defaultdict(set)
        handles: dict[tuple[str, str], str] = {}
        idents = select(Identity).where(Identity.person_id.is_not(None), Identity.status == "confirmed")
        for i in s.scalars(idents):
            if i.platform == "arxiv_name":
                names[i.person_id].add(norm_name(i.handle))
            elif i.platform in SOCIAL:
                handles[(i.platform, i.handle.lower())] = i.person_id
        return cls([_ref(p.id, p.name, p.tier, {p.lab_id} | labs[p.id], (p.meta or {}).get("common_name"),
                         names[p.id]) for p in s.scalars(select(Person))], handles)

    @classmethod
    def from_config(cls, people_cfg: list[dict]) -> PeopleIndex:
        """The same index built straight from people.yaml (offline evaluation, no database)."""
        people, handles = [], {}
        for p in people_cfg:
            idents = p.get("identities") or []
            labs = {p.get("lab")} | {a.get("lab") for a in p.get("affiliations") or []}
            arxiv = {norm_name(i["handle"]) for i in idents if i["platform"] == "arxiv_name"}
            common = (p.get("meta") or {}).get("common_name")
            people.append(_ref(p["id"], p["name"], p.get("tier"), labs, common, arxiv))
            for i in idents:
                if i["platform"] in SOCIAL:
                    handles[(i["platform"], str(i["handle"]).lstrip("@").lower())] = p["id"]
        return cls(people, handles)

    def fingerprint(self) -> str:
        people = sorted((p.id, p.tier, sorted(p.labs), p.common_name, p.names) for p in self.people.values())
        payload = [ER_VERSION, people, sorted(f"{k[0]}:{k[1]}={v}" for k, v in self.handles.items())]
        return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:12]

    def resolve_handle(self, platform: str, handle: str) -> Match | None:
        pid = self.handles.get((platform, handle.strip().lstrip("@").lower()))
        return Match(pid, self.people[pid].tier, "handle", 100.0, handle) if pid in self.people else None

    def resolve_names(self, names: Sequence[str], doc_labs: Iterable[str] = (), strict: bool = False
                      ) -> tuple[list[Match], list[Gray]]:
        """Distinctive exact names first, so their labs can corroborate the rest of the author list.
        strict=True (off-domain paper) makes every name need corroboration."""
        exact: list[Match] = []
        pending: list[tuple[int, str, str]] = []
        for pos, raw in enumerate(names):
            n = norm_name(raw)
            if len(n.split()) < 2:
                continue
            sure = [] if strict else [self.people[i] for i in self.by_name.get(n, ())
                                      if not self.people[i].common_name]
            exact += [Match(p.id, p.tier, "exact", 100.0, raw, pos) for p in sure]
            if not sure:
                pending.append((pos, raw, n))
        context = set(doc_labs) | {lab for m in exact for lab in self.people[m.person_id].labs}
        matched, gray = list(exact), []
        for pos, raw, n in pending:
            for pid in sorted({i for tok in n.split() if len(tok) > 1 for i in self.by_token.get(tok, ())}):
                p = self.people[pid]
                if not p.labs & context:
                    continue  # uncorroborated: never auto-matched, never sent to (paid) adjudication
                score = max(name_score(n, v) for v in p.names)
                if score >= FUZZY_MIN:
                    matched.append(Match(pid, p.tier, "exact" if n in p.names else "fuzzy", score, raw, pos))
                elif score >= GRAY_MIN:
                    gray.append(Gray(pid, raw, pos, score))
        matched = dedupe(matched)
        return matched, [g for g in gray if g.person_id not in {m.person_id for m in matched}]


def _ref(pid: str, name: str, tier: str | None, labs: set, common: Any, arxiv: set[str]) -> PersonRef:
    names = tuple(sorted({norm_name(name)} | set(arxiv)))
    return PersonRef(pid, name, tier or "C", frozenset(labs - {None}), bool(common), names)


# --------------------------------------------------------------------------- document mentions
def document_mentions(doc: Document, source_kind: str | None) -> tuple[list[str], tuple[str, str] | None]:
    """(author names, (platform, handle)) of one document; bots and empty bylines yield nothing."""
    meta = doc.meta or {}
    if isinstance(meta.get("authors"), list) and meta["authors"]:
        return [str(a) for a in meta["authors"] if a], None
    author = _ET_AL.sub("", (doc.author or "").strip())
    if not author or _BOT.search(author):
        return [], None
    platform = meta.get("platform") if meta.get("platform") in {"x", "bluesky"} else None
    platform = platform or HANDLE_KINDS.get(source_kind or "")
    if platform or meta.get("kind") in {"release", "repo"}:
        return [], (platform or "github", author)
    return [n for n in _SPLIT.split(author) if n.strip()], None


def is_off_domain(categories: Sequence[str] | None) -> bool:
    return bool(categories) and not any(str(c).startswith(ML_CATEGORIES) for c in categories)


def corroborating_labs(doc: Document) -> set[str]:
    """The document's lab is evidence unless it was itself inferred from an author-name match (the
    arXiv author feed and HF 'tracked_author' attribution do that) - otherwise agreement is circular."""
    meta = doc.meta or {}
    via = (meta.get("lab_match") or {}).get("via")
    derived = via == "tracked_author" or (via is None and bool(meta.get("tracked_people")))
    return set() if not doc.lab_id or derived else {doc.lab_id}


def resolve_document(index: PeopleIndex, doc: Document, source_kind: str | None
                     ) -> tuple[list[Match], list[Gray]]:
    names, handle = document_mentions(doc, source_kind)
    if handle:
        m = index.resolve_handle(*handle)
        return ([m] if m else []), []
    strict = is_off_domain((doc.meta or {}).get("categories"))
    return index.resolve_names(names, corroborating_labs(doc), strict=strict)


def _apply(doc: Document, matches: Iterable[Match], reg: str, gray: Sequence[Gray] = ()) -> None:
    """The most senior match becomes the document's person; all matches stay in meta for features."""
    matches = dedupe(matches)
    meta = dict(doc.meta or {})
    meta["people_matched"] = [asdict(m) for m in matches]
    meta["er"] = {"v": ER_VERSION, "reg": reg, "gray": [asdict(g) for g in gray]}
    doc.meta = meta
    if matches:
        doc.person_id = matches[0].person_id


# ---------------------------------------------------------------------------- stage entry point
def run_entity_resolution(limit: int | None = None, *, use_llm: bool = True, use_github: bool = True,
                          gateway: Any = None, github_client: httpx.Client | None = None, **_: Any) -> dict:
    """Attribute unattributed documents, adjudicate gray-zone names, link GitHub profiles to X."""
    settings = get_settings()
    counts: Counter[str] = Counter(dict.fromkeys(("documents_scanned", "attributed", "skipped_resolved",
                                                  "gray_pairs"), 0))
    with session_scope() as s:
        index = PeopleIndex.from_db(s)
        reg = index.fingerprint()
        jobs = attribute_documents(s, index, reg, limit, counts)
    if jobs:
        adjudicate_jobs(jobs, reg, settings, counts, gateway=gateway, enabled=use_llm)
    if use_github:
        link_github(settings, counts, client=github_client)
    return {"register_fingerprint": reg, **dict(sorted(counts.items()))}


def attribute_documents(s: Session, index: PeopleIndex, reg: str, limit: int | None,
                        counts: Counter) -> list[dict]:
    """Pass 1 (deterministic). Returns the gray-zone pairs for the LLM pass."""
    rows = s.execute(select(Document, Source.kind).join(Source, Source.id == Document.source_id)
                     .where(Document.person_id.is_(None), Document.status != "duplicate")
                     .order_by(Document.id))
    jobs: list[dict] = []
    for doc, kind in rows:
        er = (doc.meta or {}).get("er") or {}
        if er.get("reg") == reg and not er.get("gray"):
            counts["skipped_resolved"] += 1
            continue
        if limit is not None and counts["documents_scanned"] >= limit:
            break
        counts["documents_scanned"] += 1
        matches, gray = resolve_document(index, doc, kind)
        _apply(doc, matches, reg, gray)
        counts.update(f"matches_{m.method}" for m in matches)
        counts["attributed"] += bool(matches)
        jobs += [{"document_id": doc.id, **asdict(g)} for g in gray]
    counts["gray_pairs"] += len(jobs)
    return jobs


# ------------------------------------------------------------------------ gray-zone adjudication
def accept_adjudication(data: dict | None, record_a: str, record_b: str, names: Iterable[str] = (),
                        trivial: Iterable[str] = ()) -> tuple[bool, list[str]]:
    """'same' counts only with >= 1 quoted fact found verbatim in BOTH records that is neither part of
    the names being compared nor a `trivial` string (the lab ids: lab agreement is the precondition
    for adjudication, so it cannot also be the evidence). The model judges; it may not invent proof."""
    if not data or data.get("verdict") != "same":
        return False, []
    a, b = normalize_for_match(record_a), normalize_for_match(record_b)
    excluded = [normalize_for_match(n) for n in (*names, *trivial) if n]
    kept = []
    for fact in data.get("overlapping_facts") or []:
        f = normalize_for_match(str(fact)).strip(" .,;:'\"")
        if len(f) >= 4 and f in a and f in b and not any(f in n for n in excluded):
            kept.append(str(fact))
    return bool(kept), kept


def person_record(s: Session, person_id: str) -> str:
    """Record A: the register's facts about the person, plus tracked colleagues who share a lab
    (a co-author found in both records is checkable evidence; world knowledge is not)."""
    p = s.get(Person, person_id)
    lines = [f"name: {p.name}", f"current lab: {p.lab_id or 'none'}", f"title: {p.title or 'unknown'}"]
    labs = {p.lab_id} - {None}
    for a in s.scalars(select(Affiliation).where(Affiliation.person_id == person_id)):
        labs |= {a.lab_id} - {None}
        lines.append(f"affiliation: {a.lab_id or a.org_name} | {a.role or 'role unknown'} | {_span(a)}")
    for handle in s.scalars(select(Identity.handle).where(  # confirmed only: an earlier LLM decision
            Identity.person_id == person_id, Identity.platform == "arxiv_name",  # must not become evidence
            Identity.status == "confirmed")):
        lines.append(f"publishes as: {handle}")
    colleagues = s.scalars(select(Person.name).join(Affiliation, Affiliation.person_id == Person.id)
                           .where(Affiliation.lab_id.in_(labs), Person.id != person_id).distinct()
                           .order_by(Person.name).limit(40))
    lines.append(f"tracked colleagues (shared lab): {', '.join(colleagues) or 'none'}")
    return "\n".join(lines)


def _span(a: Affiliation) -> str:
    start, end = (f"{d:%Y-%m}" if d else None for d in (a.start_date, a.end_date))
    if start and end:
        return f"{start} to {end}"
    return f"since {start}" if start else f"until {end}" if end else "dates unknown"


def mention_record(s: Session, doc: Document, mention: str) -> str:
    """Record B: the mention in its document context, including tracked co-authors matched in pass 1
    (with their labs), because a shared co-author or lab is the evidence the acceptance rule checks."""
    names, _ = document_mentions(doc, None)
    others = ", ".join(n for n in names if n != mention)[:1500] or "none"
    published = f"{doc.published_at:%Y-%m-%d}" if doc.published_at else "unknown"
    lines = [f"author as printed: {mention}", f"document title: {doc.title or 'untitled'}",
             f"document lab: {doc.lab_id or 'unknown'}", f"source: {doc.source_id}",
             f"published: {published}", f"other authors: {others}"]
    for m in (doc.meta or {}).get("people_matched") or []:
        if p := s.get(Person, m["person_id"]):
            labs = ", ".join(sorted({a.lab_id or a.org_name for a in p.affiliations} - {None})) or "unknown"
            lines.append(f"tracked co-author: {m['mention']} (register name: {p.name}; labs: {labs})")
    return "\n".join(lines)


def adjudication_input(record_a: str, record_b: str) -> str:
    return f"RECORD A (tracked person)\n{record_a}\n\nRECORD B (author mention)\n{record_b}"


def adjudicate_jobs(jobs: list[dict], reg: str, settings: Settings, counts: Counter, *, gateway: Any = None,
                    enabled: bool = True) -> None:
    """Pass 3: Haiku adjudicates gray-zone pairs (<= LLM_CAP per run). Pairs not decided (no key,
    budget, API error) stay pending in meta['er']['gray'] and are retried on the next run."""
    if not enabled or (gateway is None and not settings.anthropic_api_key):
        counts["gray_deferred"] += len(jobs)
        return
    from fli.llm.client import BudgetExceeded, LLMGateway, run_async
    from fli.llm.prompts import load_prompt

    batch, counts["gray_deferred"] = jobs[:LLM_CAP], counts["gray_deferred"] + max(0, len(jobs) - LLM_CAP)
    with session_scope() as s:
        for j in batch:
            j["record_a"] = person_record(s, j["person_id"])
            j["record_b"] = mention_record(s, s.get(Document, j["document_id"]), j["mention"])
            j["person_name"] = s.get(Person, j["person_id"]).name
    prompt = load_prompt("er_adjudicate")
    calls = [{"purpose": "er", "prompt": prompt, "schema": ER_SCHEMA, "model": settings.model_er,
              "user": adjudication_input(j["record_a"], j["record_b"]), "max_tokens": 1024, "effort": "low",
              "ref": {"document_id": j["document_id"], "person_id": j["person_id"]}} for j in batch]
    try:
        results = run_async((gateway or LLMGateway(settings)).structured_many(calls))
    except BudgetExceeded as exc:
        log.warning("gray-zone adjudication stopped: %s", exc)
        counts["gray_deferred"] += len(batch)
        return
    with session_scope() as s:
        for j, res in zip(batch, results, strict=True):
            _record_decision(s, j, res, reg, counts)


def _record_decision(s: Session, job: dict, res: Any, reg: str, counts: Counter) -> None:
    if res.data is None:  # refusal / error: leave the pair pending
        counts["llm_errors"] += 1
        return
    doc = s.get(Document, job["document_id"])
    person = s.get(Person, job["person_id"])
    labs = {doc.lab_id, person.lab_id} | {a.lab_id or a.org_name for a in person.affiliations}
    ok, facts = accept_adjudication(res.data, job["record_a"], job["record_b"],
                                    (job["mention"], job["person_name"]), labs - {None})
    counts["llm_accepted" if ok else "llm_rejected"] += 1
    meta = doc.meta or {}
    matches = [Match(**m) for m in meta.get("people_matched") or []]
    left = [Gray(**g) for g in (meta.get("er") or {}).get("gray") or []
            if (g["person_id"], g["mention"]) != (job["person_id"], job["mention"])]
    if ok:
        matches.append(Match(person.id, person.tier, "llm_adjudicated", job["score"], job["mention"],
                             job["position"]))
        evidence = f"LLM-adjudicated on document {doc.id}; facts in both records: {'; '.join(facts)}"[:1000]
        _link(s, "arxiv_name", job["mention"].strip().lower(), person.id, None, evidence, counts,
              method="llm_adjudicated", confidence=0.8)
    _apply(doc, matches, reg, left)


# ------------------------------------------------------------------------ GitHub self-assertion
def link_github(settings: Settings, counts: Counter, client: httpx.Client | None = None) -> None:
    """Read public GitHub profiles of tracked people (<= GITHUB_CAP calls, refreshed monthly):
    twitter_username is a self-asserted link to X; blog and company are kept as evidence."""
    if not settings.github_token and client is None:
        counts["github_skipped_no_token"] += 1
        return
    now = utcnow()
    with session_scope() as s:
        query = select(Identity).where(Identity.platform == "github", Identity.person_id.is_not(None),
                                       Identity.status == "confirmed").order_by(Identity.id)
        due = [(i.id, i.person_id, i.handle) for i in s.scalars(query) if _github_due(i, now)]
    own = client is None
    client = client or httpx.Client(timeout=settings.http_timeout_s, headers={
        "Authorization": f"Bearer {settings.github_token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": settings.user_agent})
    try:
        for ident_id, person_id, login in due[:GITHUB_CAP]:
            try:
                resp = client.get(f"https://api.github.com/users/{login}")
            except httpx.HTTPError as exc:
                counts["github_errors"] += 1
                log.warning("github lookup %s failed: %s", login, exc)
                continue
            counts["github_calls"] += 1
            if resp.status_code in (403, 429):
                counts["github_rate_limited"] += 1
                break
            if resp.status_code not in (200, 404):
                counts["github_errors"] += 1
                continue
            with session_scope() as s:
                apply_github_profile(s, ident_id, person_id, login,
                                     resp.json() if resp.status_code == 200 else None, counts)
    finally:
        if own:
            client.close()


def _github_due(ident: Identity, now: datetime) -> bool:
    fetched = to_naive_utc(((ident.meta or {}).get("github") or {}).get("fetched_at"))
    return fetched is None or now - fetched >= GITHUB_REFRESH


def apply_github_profile(s: Session, ident_id: int, person_id: str, login: str, profile: dict | None,
                         counts: Counter) -> None:
    gh = s.get(Identity, ident_id)
    snap: dict[str, Any] = {"fetched_at": utcnow().isoformat(timespec="seconds"),
                            "found": profile is not None}
    snap.update({k: (profile or {}).get(k) for k in ("name", "company", "blog", "twitter_username")})
    if profile is None:
        gh.meta = {**(gh.meta or {}), "github": snap}
        counts["github_not_found"] += 1
        log.warning("github identity %s of %s does not exist (404)", login, person_id)
        return
    gh.meta = {**(gh.meta or {}), "github": snap}  # `company` is kept as evidence only: it is often stale
    tw = str(profile.get("twitter_username") or "").lstrip("@").lower()
    links = [("x", tw, f"https://x.com/{tw}", "twitter_username")]
    if blog := str(profile.get("blog") or "").strip():
        links.append((*classify_url(blog), _http(blog), "blog"))
    seen: set[tuple[str, str]] = {("github", login.lower())}  # a profile linking itself proves nothing
    for platform, handle, url, field in links:
        if handle and (platform, handle) not in seen:
            seen.add((platform, handle))
            _link(s, platform, handle, person_id, url, f"github.com/{login} lists {field}={handle}", counts)


def classify_url(url: str) -> tuple[str, str]:
    """(platform, normalised handle) of a profile URL: X, GitHub, Google Scholar, else a homepage."""
    if m := _X_URL.match(url):
        return "x", m.group(1).lower()
    bare = re.sub(r"^https?://(www\.)?", "", url.strip(), flags=re.IGNORECASE).rstrip("/")
    if m := re.fullmatch(r"github\.com/([\w-]+)", bare, flags=re.IGNORECASE):
        return "github", m.group(1).lower()
    scholar = re.search(r"scholar\.google\.[a-z.]+/citations\?(?:.*&)?user=([\w-]+)", bare, re.IGNORECASE)
    if scholar:
        return "scholar", scholar.group(1).lower()  # lowercased like the loader; the URL keeps case
    return "homepage", bare.lower()


def _http(url: str) -> str:
    return url if url.lower().startswith("http") else f"https://{url}"


def _link(s: Session, platform: str, handle: str, person_id: str, url: str | None, evidence: str,
          counts: Counter, method: str = "self_asserted", confidence: float = 0.9) -> None:
    """An existing identity of the same person is upgraded to self_asserted/1.0; a new one is only a
    candidate (only confirmed identities attribute documents); a clash is left for review."""
    ident = s.scalar(select(Identity).where(Identity.platform == platform, Identity.handle == handle))
    if ident is None:
        s.add(Identity(platform=platform, handle=handle, person_id=person_id, url=url, method=method,
                       status="candidate", confidence=confidence, evidence=evidence))
        counts[f"{platform}_candidates_added"] += 1
    elif ident.person_id == person_id:
        if method == "self_asserted" and (ident.method != "self_asserted" or ident.confidence < 1.0):
            ident.method, ident.confidence, ident.evidence = "self_asserted", 1.0, evidence
            counts[f"{platform}_self_asserted"] += 1
    else:
        counts[f"{platform}_conflicts"] += 1
        log.warning("%s:%s asserted for %s belongs to %s", platform, handle, person_id,
                    ident.person_id or ident.lab_id)


# --------------------------------------------------------------------------------- evaluation
def evaluate_er(pairs_path: Path | str, people_path: Path | str | None = None) -> dict:
    """Precision/recall of the deterministic matcher on hand-labelled (person, mention) pairs.
    Gray-zone outcomes count as 'not matched' (the LLM is not called), so recall is a lower bound."""
    pairs = (yaml.safe_load(Path(pairs_path).read_text(encoding="utf-8")) or {}).get("pairs", [])
    people = yaml.safe_load(Path(people_path or CONFIG_DIR / "people.yaml").read_text(encoding="utf-8")) or {}
    index = PeopleIndex.from_config(people.get("people", []))
    rows = [(str(p.get("id")), str(p["label"]), predict_pair(index, p)) for p in pairs]
    tp = sum(label == "same" and pred == "same" for _, label, pred in rows)
    fp = sum(label == "different" and pred == "same" for _, label, pred in rows)
    fn = sum(label == "same" and pred != "same" for _, label, pred in rows)
    gray_same = sum(label == "same" and pred == "gray" for _, label, pred in rows)
    precision, recall = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": len(rows) - tp - fp - fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "gray_zone": sum(pred == "gray" for *_, pred in rows),
        "recall_if_gray_accepted": round((tp + gray_same) / (tp + fn), 3) if tp + fn else 0.0,
        "errors": [{"id": i, "label": label, "predicted": pred} for i, label, pred in rows
                   if (label == "same") != (pred == "same")],
    }


def predict_pair(index: PeopleIndex, pair: dict) -> str:
    """'same' | 'gray' | 'different' for one labelled pair (only the labelled mention counts)."""
    pid = pair["person"]
    if pair.get("platform"):
        m = index.resolve_handle(pair["platform"], pair["mention"])
        return "same" if m and m.person_id == pid else "different"
    names = [pair["mention"], *(pair.get("coauthors") or [])]
    strict = is_off_domain(pair.get("categories"))
    matches, gray = index.resolve_names(names, {pair.get("lab")} - {None}, strict=strict)
    if any(m.person_id == pid and m.position == 0 for m in matches):
        return "same"
    return "gray" if any(g.person_id == pid and g.position == 0 for g in gray) else "different"
