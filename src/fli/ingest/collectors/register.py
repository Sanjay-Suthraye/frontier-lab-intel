"""Register collector: curated, cited talent moves become documents of their own.

The people register (config/people.yaml -> person / affiliation tables) records dated moves of
tracked people, each with the URL it was curated from. Most of these moves never appear on an
official lab channel, and press reaches the corpus only as headlines, so without this collector a
founder's exit stays a register row that no development, alert or digest ever sees. It reads the
LOCAL database only (no network, no LLM) and builds every word of the text from register fields, so
citations stay code-owned.

An *event* is what one person did on one day:

  event        the person's changes that day                      title
  co_founding  starts a founder role at a new organisation        '<Name> leaves <Lab> to co-found <Org>' or '<Name> co-founds <Org>'
  move         ends a role at A and starts one at B               '<Name> leaves <A> to join <B> as <role>'
  join         starts a role (no same-day end)                    '<Name> joins <Org> as <role>'
  leave        ends a role (no same-day start)                    '<Name> leaves <Org> (<role>)'
  role_change  ends and starts roles at the same organisation     '<Name> becomes <role> at <Org> (was <role>)'

People with the same event on the same day are one record: a co-founding team (whatever each founder
left), a team leaving one lab for one destination, a same-day reorganisation. A real-world event is
then triaged, extracted, scored and alerted once, not once per person.

Record fields: natural_key 'aff:<id>:<start|end>' of the event's anchor, the lowest affiliation id
among the rows that define it, so the key survives members being added to the register later;
url = the anchor's source_url (every other source is listed in the text); published_at = the day of
the move; person_id = the most senior person; lab_id = the lab joined (start) or left (end), else
the other side's register lab; source_class 'curated'. Moves the register marks as reported by
secondary sources only (confidence < 1) say "reportedly" and "not confirmed", so extraction can flag
them as rumours.

Window: a move is collected when its date falls in [since, until] (whole days). It is also collected
when its affiliation row was added to the register after `since` and the move lies in the tracked
period (settings.backfill_start onwards): analysts often record a move weeks after it happened, and a
scheduled run only looks back two days. Re-emitting a move is free (natural-key upserts).
Affiliations without a source_url are skipped: an uncited fact is not a source.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from fli.config import Settings
from fli.db.models import Affiliation, Lab, Person
from fli.db.session import session_scope
from fli.ingest.records import IngestRecord, SourceInfo, Window, collector

log = logging.getLogger("fli.ingest.register")

SOURCE_CLASS = "curated"  # fli.taxonomy.SOURCE_CLASSES: analyst-maintained register facts, each with a source URL
TIER_RANK = {"A": 0, "B": 1, "C": 2}
TIER_GLOSS = {  # config/people.yaml, "WHO COUNTS AS KEY"
    "A": "founders, chief scientists and heads of research",
    "B": "named leads of flagship model work",
    "C": "high-signal contributors",
}
FOUNDER = re.compile(r"founder\b", re.IGNORECASE)  # Co-founder, Cofounder, Founder and CEO; not "Founding team"
CO_FOUNDER = re.compile(r"\bco-?founder\b", re.IGNORECASE)
BARE_FOUNDER = re.compile(r"^(co-?)?founder$", re.IGNORECASE)
_TRAILING_GLOSS = re.compile(r"\s*\([^()]*\)\s*$")
_REPORTED_SUFFIX = re.compile(r"\s*\(reported\)\s*$", re.IGNORECASE)


def short_name(name: str | None) -> str:
    """'Meta AI (Superintelligence Labs / FAIR)' -> 'Meta AI': headlines read better without the gloss."""
    name = (name or "").strip()
    return _TRAILING_GLOSS.sub("", name).strip() or name


@dataclass(frozen=True)
class Change:
    """One dated edge of one affiliation: the person started (or ended) a role at an organisation."""

    affiliation_id: int
    change: str  # "start" | "end"
    day: date
    person_id: str
    name: str
    tier: str
    lab_id: str | None  # set when the organisation is a register lab
    org: str  # display name
    role: str | None
    start: date | None
    end: date | None
    source_url: str
    evidence: str | None = None
    confidence: float = 1.0

    @property
    def org_key(self) -> str:
        return f"lab:{self.lab_id}" if self.lab_id else f"org:{self.org.casefold()}"

    @property
    def founds(self) -> bool:
        return self.change == "start" and bool(FOUNDER.search(self.role or ""))

    @property
    def reported(self) -> bool:
        return self.confidence < 1


@dataclass
class PersonDay:
    """Everything one person did on one day; classified into one event."""

    person_id: str
    name: str
    tier: str
    day: date
    ends: list[Change] = field(default_factory=list)
    starts: list[Change] = field(default_factory=list)
    event: str = ""
    primary: list[Change] = field(default_factory=list)  # the changes that define the event

    def classify(self) -> None:
        left = {c.org_key for c in self.ends}
        new = [c for c in self.starts if c.org_key not in left]
        founded = [c for c in new if c.founds]
        if founded:
            self.event, self.primary = "co_founding", founded
        elif new:
            self.event, self.primary = ("move" if self.ends else "join"), new
        elif self.starts:  # every new role is at an organisation the person also left a role at
            self.event, self.primary = "role_change", list(self.starts)
        else:
            self.event, self.primary = "leave", list(self.ends)

    def group_key(self) -> tuple:
        target = tuple(sorted({c.org_key for c in self.primary}))
        if self.event == "move":  # people leaving the same lab for the same organisation
            return (self.day, self.event, tuple(sorted({c.org_key for c in self.ends})), target)
        return (self.day, self.event, target)  # a co-founding team is one event, whatever each one left

    @property
    def changes(self) -> list[Change]:
        return sorted(self.ends, key=lambda c: c.affiliation_id) + sorted(self.starts, key=lambda c: c.affiliation_id)

    @property
    def reported(self) -> bool:
        return any(c.reported for c in self.changes)

    @property
    def rank(self) -> tuple[int, int]:
        return TIER_RANK.get(self.tier, len(TIER_RANK)), min(c.affiliation_id for c in self.primary)


# ------------------------------------------------------------------------------- loading
def _day(value: datetime | None) -> date | None:
    return value.date() if value is not None else None


def in_window(day: date, created_at: datetime | None, window: Window, floor: date) -> bool:
    """Whole-day window on the move date, plus moves recorded late (see the module docstring)."""
    if day > window.until.date():
        return False  # not happened yet: collected once its day arrives
    if day >= window.since.date():
        return True
    return created_at is not None and created_at >= window.since and day >= floor


def load_changes(session: Session, window: Window, floor: date) -> list[Change]:
    """Affiliation edges (start or end) inside the window, with person and lab, cited rows only."""
    earliest = datetime.combine(min(window.since.date(), floor), time.min)
    rows = session.execute(
        select(Affiliation, Person.name, Person.tier, Lab.name)
        .join(Person, Person.id == Affiliation.person_id)
        .outerjoin(Lab, Lab.id == Affiliation.lab_id)
        .where(Affiliation.source_url.is_not(None), Affiliation.source_url != "",
               or_(Affiliation.start_date >= earliest, Affiliation.end_date >= earliest))
        .order_by(Affiliation.id)
    ).all()
    out: list[Change] = []
    for aff, name, tier, lab_name in rows:
        url = aff.source_url.strip()
        org = short_name(lab_name or aff.lab_id) if aff.lab_id else short_name(aff.org_name)
        if not url:
            continue
        if not org:
            log.warning("affiliation %s of %s names neither a register lab nor an organisation", aff.id,
                        aff.person_id)
            continue
        start, end = _day(aff.start_date), _day(aff.end_date)
        for change, day in (("start", start), ("end", end)):
            if day is None or (change == "end" and day == start):  # zero-length row: keep the start only
                continue
            if not in_window(day, aff.created_at, window, floor):
                continue
            out.append(Change(
                affiliation_id=aff.id, change=change, day=day, person_id=aff.person_id, name=name,
                tier=tier or "C", lab_id=aff.lab_id, org=org, role=(aff.role or "").strip() or None,
                start=start, end=end, source_url=url,
                evidence=(aff.evidence or "").strip() or None,
                confidence=float(aff.confidence if aff.confidence is not None else 1.0),
            ))
    return out


# ------------------------------------------------------------------------------- wording
def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _orgs(changes: list[Change]) -> str:
    return _join(list(dict.fromkeys(c.org for c in changes)))


def _role(role: str | None, reported: bool = False) -> str | None:
    """The curated role; ' (reported)' is dropped where the sentence already says 'reportedly'."""
    if not role or not reported:
        return role
    return _REPORTED_SUFFIX.sub("", role).strip() or role


def _roles(changes: list[Change], reported: bool = False) -> str:
    roles = (_role(c.role, reported) for c in changes if c.role)
    return "; ".join(dict.fromkeys(roles)) or "an unrecorded role"


def _at(changes: list[Change]) -> str:
    """'Google DeepMind (Chief Scientist)', or 'A (r1; r2) and B (r3)' for several organisations."""
    roles: dict[str, list[str]] = {}
    for c in changes:
        org_roles = roles.setdefault(c.org, [])
        if c.role and c.role not in org_roles:
            org_roles.append(c.role)
    return _join([f"{org} ({'; '.join(r)})" if r else org for org, r in roles.items()])


def _as(changes: list[Change], reported: bool = False) -> str:
    return _join([f"{c.org} as {_role(c.role, reported)}" if c.role else c.org for c in changes])


def _founded(changes: list[Change]) -> str:
    return _join([c.org if BARE_FOUNDER.match(c.role or "") else f"{c.org} ({c.role})" for c in changes])


def _found_verb(members: list[PersonDay]) -> str:
    if len(members) > 1 or any(CO_FOUNDER.search(c.role or "") for c in members[0].primary):
        return "co-found"
    return "found"


def title_for(members: list[PersonDay]) -> str:
    lead, event, single = members[0], members[0].event, len(members) == 1
    names = _join([m.name for m in members])
    reported = all(m.reported for m in members)
    adv, s = ("reportedly " if reported else ""), ("s" if single else "")
    target = _orgs(lead.primary)  # the same organisations for every member (group key)
    role = _roles(lead.primary, reported) if single and any(c.role for c in lead.primary) else None
    if event == "co_founding":
        verb = _found_verb(members)
        if all(m.ends for m in members):
            return f"{names} {adv}leave{s} {_orgs([c for m in members for c in m.ends])} to {verb} {target}"
        return f"{names} {adv}{verb}{s} {target}"
    if event == "move":
        return f"{names} {adv}leave{s} {_orgs(lead.ends)} to join {target}" + (f" as {role}" if role else "")
    if event == "join":
        return f"{names} {adv}join{s} {target}" + (f" as {role}" if role else "")
    if event == "leave":
        return f"{names} {adv}leave{s} {target}" + (f" ({_roles(lead.ends)})" if single else "")
    if single:
        return f"{names} {adv}becomes {role or _roles(lead.primary)} at {target} (was {_roles(lead.ends)})"
    return f"{names} {adv}change roles at {target}"  # the roles are in the text


def _summary(members: list[PersonDay]) -> str:
    """One sentence for the whole group (only written when several people share the event)."""
    lead, event, day = members[0], members[0].event, members[0].day.isoformat()
    names = _join([m.name for m in members])
    adv = "reportedly " if all(m.reported for m in members) else ""
    target = _orgs(lead.primary)
    if event == "co_founding":
        if all(m.ends for m in members):
            origins = _orgs([c for m in members for c in m.ends])
            return f"{names} {adv}left {origins} to co-found {target} on {day}."
        return f"{names} {adv}co-founded {target} on {day}."
    if event == "move":
        return f"{names} {adv}left {_orgs(lead.ends)} to join {target} on {day}."
    if event == "join":
        return f"{names} {adv}joined {target} on {day}."
    if event == "leave":
        return f"{names} {adv}left {target} on {day}."
    return f"{names} {adv}changed roles at {target} on {day}."


def _sentence(m: PersonDay, verb: str) -> str:
    """One person's part of the event; `verb` is the group's founding verb (co-found / found)."""
    adv, day = ("reportedly " if m.reported else ""), m.day.isoformat()
    if m.event == "co_founding":
        if m.ends:
            return f"{m.name} {adv}left {_at(m.ends)} on {day} to {verb} {_founded(m.primary)}."
        return f"{m.name} {adv}{verb}ed {_founded(m.primary)} on {day}."
    if m.event == "move":
        return f"{m.name} {adv}left {_at(m.ends)} on {day} to join {_as(m.primary, m.reported)}."
    if m.event == "join":
        return f"{m.name} {adv}joined {_as(m.primary, m.reported)} on {day}."
    if m.event == "leave":
        return f"{m.name} {adv}left {_at(m.ends)} on {day}."
    return (f"{m.name} {adv}changed role at {_orgs(m.primary)} on {day}, "
            f"from {_roles(m.ends, m.reported)} to {_roles(m.primary, m.reported)}.")


def _status(members: list[PersonDay]) -> str:
    unconfirmed = [(m, c) for m in members for c in m.changes if c.reported]
    if not unconfirmed:
        return "Status: confirmed in the register (confidence 1)."
    items = "; ".join(f"{m.name}, {c.role or 'role'} at {c.org} (register confidence {c.confidence:g})"
                      for m, c in unconfirmed)
    return f"Status: reported by secondary sources only, not confirmed: {items}."


def _entry(m: PersonDay, c: Change) -> str:
    """The structured line: person, tier, change, organisation, role, dates, confidence, source."""
    org = f"{c.org} (register lab {c.lab_id})" if c.lab_id else f"{c.org} (not in the lab register)"
    if c.start and c.end:
        span = f"{c.start.isoformat()} to {c.end.isoformat()}"
    else:
        span = f"from {c.start.isoformat()}" if c.start else f"until {c.end.isoformat() if c.end else '?'}"
    what = "started" if c.change == "start" else "ended"
    return (f"- {m.name} | tier {m.tier} | {what} | {org} | role: {c.role or 'not recorded'} | {span} | "
            f"confidence {c.confidence:g} | source: {c.source_url}")


def text_for(members: list[PersonDay]) -> str:
    lines = [_summary(members)] if len(members) > 1 else []
    verb = _found_verb(members)
    lines += [_sentence(m, verb) for m in members]
    for m in members:
        for c in m.changes:
            if c.evidence:
                lines.append(f"Register note on {m.name}, {c.role or 'role'} at {c.org}: {c.evidence.rstrip('.')}.")
    lines.append(_status(members))
    tiers = sorted({m.tier for m in members if m.tier in TIER_GLOSS}, key=TIER_RANK.__getitem__)
    gloss = "; ".join(f"tier {t} = {TIER_GLOSS[t]}" for t in tiers)
    lines += ["", "Curated register entries (FLI people register, config/people.yaml"
              + (f"; {gloss}" if gloss else "") + "):"]
    lines += [_entry(m, c) for m in members for c in m.changes]
    return "\n".join(lines)


# ------------------------------------------------------------------------------- records
def record_for(members: list[PersonDay]) -> IngestRecord:
    """One record for a group of people who share an event (members sorted most senior first)."""
    lead = members[0]
    changes = [c for m in members for c in m.changes]
    anchor = min((c for m in members for c in m.primary), key=lambda c: c.affiliation_id)
    lab_id = anchor.lab_id or next((c.lab_id for c in changes if c.lab_id), None)  # prefer a register lab
    return IngestRecord(
        natural_key=f"aff:{anchor.affiliation_id}:{anchor.change}",
        url=anchor.source_url,
        title=title_for(members),
        content_text=text_for(members),
        published_at=datetime.combine(lead.day, time.min),
        person_id=lead.person_id,
        lab_id=lab_id,
        source_class=SOURCE_CLASS,
        meta={
            "affiliation_id": anchor.affiliation_id, "change": anchor.change, "role": anchor.role,
            "tier": lead.tier, "source_url": anchor.source_url, "event": lead.event,
            "affiliation_ids": sorted({c.affiliation_id for c in changes}),
            "people": [m.person_id for m in members],
            "person_tiers": sorted({m.tier for m in members}),
            "labs": sorted({c.lab_id for c in changes if c.lab_id}),
            "sources": list(dict.fromkeys(c.source_url for c in changes)),
            "confidence": min(c.confidence for c in changes),
        },
    )


def build_records(changes: list[Change]) -> list[IngestRecord]:
    """Changes -> person-days -> events -> one record per group of people sharing an event."""
    days: dict[tuple[str, date], PersonDay] = {}
    for c in sorted(changes, key=lambda c: (c.day, c.affiliation_id, c.change)):
        pd = days.get((c.person_id, c.day))
        if pd is None:
            pd = days[(c.person_id, c.day)] = PersonDay(c.person_id, c.name, c.tier, c.day)
        (pd.starts if c.change == "start" else pd.ends).append(c)
    groups: dict[tuple, list[PersonDay]] = defaultdict(list)
    for pd in days.values():
        pd.classify()
        groups[pd.group_key()].append(pd)
    records = [record_for(sorted(members, key=lambda m: m.rank)) for members in groups.values()]
    return sorted(records, key=lambda r: (r.published_at, r.natural_key))


@collector("register_affiliations",
           description="People register (local DB) -> one curated document per dated, cited talent move")
def register_affiliations(source: SourceInfo, window: Window, settings: Settings) -> list[IngestRecord]:
    with session_scope() as s:
        changes = load_changes(s, window, settings.backfill_start)
    records = build_records(changes)
    log.info("register %s: %d affiliation changes in window -> %d records", source.id, len(changes), len(records))
    return records[-window.max_items:] if window.max_items else records
