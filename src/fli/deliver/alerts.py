"""Alert stage: evaluate recently touched developments against the materiality policy and deliver
audience-specific alerts (always in-app; Slack when SLACK_WEBHOOK_URL is set).

Idempotent and incremental, so the scheduler can run it every 30 minutes:
  * one Alert row per (audience, development): dedup_key '<audience>:<development_id>'
  * a later run re-alerts only when the guarded severity increases (P3 -> P2 -> P1)
  * follow-ups: a new material sub-event that joins an already-alerted development (e.g. ChatGPT
    for Financial Services inside the GPT-6 Astra launch) gets its own alert while the development
    still qualifies for a push: dedup_key '<audience>:<development_id>:<insight_id>', title
    'Follow-up: ...', at most FOLLOWUP_MAX per (audience, development). What counts is in
    followup_reasons and followup_hits: a re-report of the alerted launch (another channel, a rollout
    post, a market listing, the same morning's burst of posts) names nothing new and does not
    re-alert. A development whose alert a reader suppressed gets none
  * every alert records the insights it was rendered from (delivered['covers']): "new since the
    alert" means not among them. Alerts from before that record cover the insights extracted before
    they were rendered (delivered['in_app'])
  * a daily push budget of `daily_cap` P1+P2 alerts per audience per UTC day, follow-ups included.
    P1 is never demoted (a landmark that arrives after the budget is spent still pushes) but uses up
    budget; a P2 beyond it is stored as P3 (digest only). Within a run, candidates are processed
    most severe first, new developments before follow-ups, then by lens verdict (update thesis /
    adopt before watch / assess before abstentions), then highest p
  * before anything is pushed the lens is generated if missing and regenerated if it is older than
    the development's evidence (lens.ensure_lens(refresh=True)); the re-plan then applies its verdict
  * the alert body says which facts made each rule fire (materiality.condition_reasons); tickers
    are the lens's own when a lens exists, exposure links only when it does not

Live vs historical: when a development's newest evidence is older than `live_horizon_hours` the
alert is a backfill of history. It is dated at the development's first sighting (so the per-day cap
and the in-app timeline mirror when it would have fired) and is never pushed to Slack. A follow-up
is historical when its own insight is that old, and is then dated when the insight was published.
A historical alert whose lens was regenerated after it was rendered is re-rendered from the new
lens (rerender_refreshed); a pushed alert stays a record of what was sent.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from fli.analyze.artifacts import UMBRELLA_KEYS, guess_kind, same_artifact, unique_normalized
from fli.config import get_settings
from fli.db.models import Alert, Development, Document, Insight, Lens, Score
from fli.db.session import session_scope
from fli.deliver import notify
from fli.deliver.materiality import (
    BREAKING_CHANGE,
    INACTIVE_STATUSES,
    SEVERITY_RANK,
    DevContext,
    Evaluation,
    Policy,
    evaluate,
    is_lab_level,
    load_contexts,
    load_policy,
    rule_reasons,
    verified_quotes,
)
from fli.taxonomy import AUDIENCES
from fli.timeutil import utcnow

log = logging.getLogger("fli.deliver.alerts")

PUSH = ("P1", "P2")
SLACK_MAX_ATTEMPTS = 3
SLACK_RETRY_WINDOW = timedelta(hours=24)
# Order within a severity when the daily push budget is contended: what the lens says to act on first.
# No lens yet ranks with watch / assess (the rules already passed it).
ACTION_RANK: dict[str | None, int] = {"update_thesis": 3, "review_position": 3, "ask_company": 2, "watch": 1,
                                      "no_action": 0, "adopt": 3, "trial": 2, "assess": 1, "hold": 0, None: 1}

# Follow-ups (sub-events of an already-alerted development).
FOLLOWUP_MAX = 2  # per (audience, development), capped ones included
FOLLOWUP_MIN_MAGNITUDE = 3  # extractor rubric: 3 significant, 4 major, 5 landmark
FOLLOWUP_P1_MAGNITUDE = 5  # a landmark sub-event of a P1 development is itself P1; every other follow-up is P2
FOLLOWUP_QUIET = timedelta(hours=12)  # insights published this soon after an alert belong to its news burst
FOLLOWUP_SKIP_CLASSES = frozenset({"market_data"})  # listings enrich a development (price, context); not news
FOLLOWUP_RULE = "followup.sub_event"
FOLLOWUP_DESCRIPTION = ("new material insight in a development already alerted to this audience: verified, "
                        "not a rumor, magnitude >= 3 (AI team: or an official deprecation / breaking change), "
                        "of an event type or naming an artifact the earlier alert did not cover, published "
                        "more than 12 hours after it and not a market listing")
WILDCARD_EVENT = "other"  # the extractor's catch-all type (cluster.WILDCARD_EVENT): never a new event type


def dedup_key(audience: str, development_id: int) -> str:
    return f"{audience}:{development_id}"


def followup_key(audience: str, development_id: int, insight_id: int) -> str:
    return f"{audience}:{development_id}:{insight_id}"


def is_followup_key(key: str | None) -> bool:
    return (key or "").count(":") == 2


@dataclass
class FollowUp:
    parent: Alert  # the development's alert for this audience
    hits: list[tuple[Insight, list[str]]]  # qualifying new insights with their reasons, the trigger first
    covers: list[int]  # insight ids the follow-up is rendered from (all current members)
    number: int  # 1..FOLLOWUP_MAX

    @property
    def insight(self) -> Insight:
        return self.hits[0][0]

    @property
    def reasons(self) -> list[str]:
        return self.hits[0][1]


@dataclass
class Candidate:
    ctx: DevContext
    ev: Evaluation  # for a follow-up: the development's evaluation with the follow-up's severity
    when: datetime  # what the alert is dated: now (live) or first sighting (historical)
    live: bool
    followup: FollowUp | None = None

    @property
    def key(self) -> str:
        if self.followup is not None:
            return followup_key(self.ev.audience, self.ctx.dev.id, self.followup.insight.id)
        return dedup_key(self.ev.audience, self.ctx.dev.id)


# --------------------------------------------------------------------------------- selection
def candidate_ids(session: Session, *, now: datetime, backfill: bool, lookback_days: int,
                  limit: int | None = None) -> list[int]:
    """Developments touched (updated, new evidence, or re-scored) within the lookback; all when backfilling."""
    q = select(Development.id).where(Development.status.not_in(INACTIVE_STATUSES))
    if not backfill:
        cutoff = now - timedelta(days=lookback_days)
        rescored = select(Score.development_id).where(Score.created_at >= cutoff)
        q = q.where(or_(Development.updated_at >= cutoff, Development.last_seen_at >= cutoff,
                        Development.id.in_(rescored)))
    q = q.order_by(Development.last_seen_at.desc(), Development.id.desc())
    if limit:
        q = q.limit(limit)
    return list(session.scalars(q))


def alert_time(dev: Development, now: datetime, policy: Policy) -> tuple[datetime, bool]:
    newest = dev.last_seen_at or dev.first_seen_at or now
    if newest >= now - timedelta(hours=policy.live_horizon_hours):
        return now, True
    return min(dev.first_seen_at or newest, now), False


def _prior_guarded(alert: Alert | None) -> str | None:
    if alert is None:
        return None
    return (alert.delivered or {}).get("guarded_severity") or alert.severity


def _parse_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def covered_ids(alert: Alert, insights: list[Insight]) -> set[int]:
    """Insights an alert was rendered from: its recorded `covers`, else (alerts from before that
    record) the insights extracted before it was rendered."""
    recorded = (alert.delivered or {}).get("covers")
    if isinstance(recorded, list):
        return {int(i) for i in recorded if isinstance(i, int) and not isinstance(i, bool)}
    at = _parse_time((alert.delivered or {}).get("in_app")) or alert.created_at
    return {i.id for i in insights if i.created_at is None or at is None or i.created_at <= at}


def _artifact_keys(names: Any) -> dict[str, str]:
    """Normalised artifact names (key -> display) that can mark a new subject: datasets and umbrella
    brands ('ChatGPT', 'Gemini') name nothing specific."""
    return {k: d for k, d in unique_normalized(names) if k not in UMBRELLA_KEYS and guess_kind(d) != "dataset"}


def _named(key: str, covered: set[str]) -> bool:
    """`key` is already named: the same artifact, or a family whose variant is ('Timestral' after
    'Timestral-5M'). A new variant of a named family ('GPT-6 Astra Pro', a '-1017' snapshot) is not."""
    return any(same_artifact(key, c) or c.startswith(key + " ") for c in covered)


def followup_reasons(ins: Insight, covered: list[Insight], audience: str) -> list[str]:
    """Why `ins` (not among the alert's `covered` insights) deserves a follow-up; [] when it does not.

    Eligible: at least one verified quote, not a rumor. Material: magnitude >= FOLLOWUP_MIN_MAGNITUDE
    or, for the AI team, an official deprecation / breaking change (the rubric rates those 2-3, and
    ai.p2.breaking_change exists for them). New: an event type no covered insight has, or an artifact
    named for the first time in the corpus (Insight.first_seen_artifacts) that no covered insight
    names. Magnitude alone is not news: a launch re-reported at magnitude 4-5 on another channel
    names nothing new."""
    if ins.is_rumor or not verified_quotes(ins):
        return []
    magnitude = ins.magnitude or 0
    breaking = (audience == "ai_team" and bool(ins.is_official)
                and BREAKING_CHANGE.search(f"{ins.title or ''} {ins.claim or ''}") is not None)
    if magnitude < FOLLOWUP_MIN_MAGNITUDE and not breaking:
        return []
    new_type = ins.event_type not in {c.event_type for c in covered} and ins.event_type != WILDCARD_EVENT
    named = {k for c in covered for k in _artifact_keys(c.artifacts)}
    first = [d for k, d in _artifact_keys(ins.first_seen_artifacts).items() if not _named(k, named)]
    if not (new_type or first):
        return []
    reasons = [f"new event type {ins.event_type}"] if new_type else []
    if first:
        reasons.append("first mention of " + ", ".join(first[:3]))
    reasons.append(f"magnitude {magnitude}" if magnitude else "magnitude unrated")
    if breaking:
        reasons.append("official deprecation / breaking change")
    return reasons


def _insight_time(ins: Insight) -> datetime | None:
    return ins.published_at or ins.created_at


def followup_hits(ctx: DevContext, covered: set[int], anchor: datetime | None,
                  audience: str) -> list[tuple[Insight, list[str]]]:
    """Insights outside `covered` that trigger a follow-up (followup_reasons), most material first.
    Insights published up to FOLLOWUP_QUIET after `anchor` (the latest alert) are that alert's news
    burst (the launch post, the API docs and the weights within hours): they never trigger, and
    count as covered when judging what is new. Market listings never trigger."""
    quiet_end = anchor + FOLLOWUP_QUIET if anchor is not None else None
    burst = {i.id for i in ctx.insights if i.id not in covered and quiet_end is not None
             and (_insight_time(i) or quiet_end) <= quiet_end}
    known = [i for i in ctx.insights if i.id in covered or i.id in burst]
    hits = []
    for ins in ctx.insights:
        doc = ctx.documents.get(ins.document_id)
        if ins.id in covered or ins.id in burst or (doc is not None and doc.source_class in FOLLOWUP_SKIP_CLASSES):
            continue
        if why := followup_reasons(ins, known, audience):
            hits.append((ins, why))
    hits.sort(key=lambda h: (-(h[0].magnitude or 0), not h[0].is_official, _insight_time(h[0]) or datetime.max,
                             h[0].id))
    return hits


def plan_followup(ctx: DevContext, ev: Evaluation, parent: Alert | None, prior: list[Alert], now: datetime,
                  policy: Policy) -> Candidate | None:
    """A follow-up candidate for a development whose alert stands (`parent`, with earlier follow-ups
    `prior`) and that still qualifies for a push, when insights that none of those alerts covered
    include a follow-up trigger (followup_hits). The trigger is the most material one; the follow-up
    covers every current insight, so one wave of coverage yields one follow-up."""
    if parent is None or parent.status == "suppressed" or ev.severity not in PUSH or ev.unconfirmed:
        return None
    if len(prior) >= FOLLOWUP_MAX:
        return None
    seen: set[int] = set()
    for alert in (parent, *prior):
        seen |= covered_ids(alert, ctx.insights)
    anchor = max((a.created_at for a in (parent, *prior) if a.created_at is not None), default=None)
    hits = followup_hits(ctx, seen, anchor, ev.audience)
    if not hits:
        return None
    trigger = hits[0][0]
    severity = "P1" if ev.severity == "P1" and (trigger.magnitude or 0) >= FOLLOWUP_P1_MAGNITUDE else "P2"
    published = _insight_time(trigger) or now
    live = published >= now - timedelta(hours=policy.live_horizon_hours)
    when = now if live else min(max(published, parent.created_at or published), now)
    follow = FollowUp(parent, hits, sorted(i.id for i in ctx.insights), len(prior) + 1)
    return Candidate(ctx, replace(ev, severity=severity), when, live, follow)


def _plan(session: Session, ids: list[int], policy: Policy, now: datetime) -> tuple[list[Candidate], int]:
    """Evaluate both audiences; keep decisions that create or escalate an alert, and follow-ups.
    Returns (candidates, number of alert-worthy decisions already alerted at that severity)."""
    ctxs = load_contexts(session, ids)
    existing: dict[str, Alert] = {}
    followups: dict[tuple[str, int], list[Alert]] = defaultdict(list)
    for i in range(0, len(ids), 500):
        for a in session.scalars(select(Alert).where(Alert.development_id.in_(ids[i : i + 500]))):
            if is_followup_key(a.dedup_key):
                followups[(a.audience, a.development_id)].append(a)
            else:
                existing[a.dedup_key] = a
    out, unchanged = [], 0
    for ctx in ctxs.values():
        for audience in AUDIENCES:
            ev = evaluate(ctx, audience, policy)
            if not ev.severity:
                continue
            parent = existing.get(dedup_key(audience, ctx.dev.id))
            if SEVERITY_RANK[ev.severity] <= SEVERITY_RANK[_prior_guarded(parent)]:
                unchanged += 1
                follow = plan_followup(ctx, ev, parent, followups.get((audience, ctx.dev.id), []), now, policy)
                if follow is not None:
                    out.append(follow)
                continue
            when, live = alert_time(ctx.dev, now, policy)
            out.append(Candidate(ctx, ev, when, live))
    out.sort(key=lambda c: (c.when.date(), -SEVERITY_RANK[c.ev.severity], c.followup is not None,
                            -ACTION_RANK.get(c.ev.facts.get("lens_action"), 1), -(c.ev.facts.get("p") or 0.0),
                            c.ctx.dev.id))
    return out, unchanged


def _ensure_lenses(cands: list[Candidate]) -> dict[str, Any]:
    """Make sure P1/P2 candidates have a lens (so_what / what_to_do) and that it is not older than
    the evidence being pushed (ensure_lens(refresh=True) regenerates only stale lenses). Failure is
    not fatal: an alert built from the development summary beats no alert."""
    missing: dict[str, set[int]] = {}
    present: dict[str, set[int]] = {}
    for c in cands:
        if c.ev.severity in PUSH:
            group = present if c.ev.audience in c.ctx.lenses else missing
            group.setdefault(c.ev.audience, set()).add(c.ctx.dev.id)
    if not missing and not present:
        return {}
    try:
        from fli.deliver.lens import ensure_lens
    except ImportError as exc:
        log.warning("lens module unavailable (%s); alerts fall back to development summaries", exc)
        return {"status": "unavailable"}
    out: dict[str, Any] = {}
    for audience, ids in missing.items():
        try:
            ensure_lens(sorted(ids), audience)
            out[audience] = len(ids)
        except Exception as exc:  # includes BudgetExceeded
            log.warning("ensure_lens(%s, %d developments) failed: %s", audience, len(ids), exc)
            out[audience] = f"error: {exc.__class__.__name__}"
    for audience, ids in present.items():
        try:
            if refreshed := ensure_lens(sorted(ids), audience, refresh=True):
                out[f"{audience}_refreshed"] = refreshed
        except Exception as exc:  # includes BudgetExceeded; the existing lens is used
            log.warning("lens refresh (%s, %d developments) failed: %s", audience, len(ids), exc)
            out[f"{audience}_refreshed"] = f"error: {exc.__class__.__name__}"
    return out


# --------------------------------------------------------------------------------- stage
def run_alerts(limit: int | None = None, backfill: bool = False, now: datetime | None = None,
               ensure_lenses: bool = True, **_kwargs: Any) -> dict:
    """Pipeline stage. Returns counts: created / escalated / followups / capped / unchanged / slack."""
    policy = load_policy()
    now = now or utcnow()
    with session_scope() as s:
        ids = candidate_ids(s, now=now, backfill=backfill, lookback_days=policy.lookback_days, limit=limit)
        planned, unchanged = _plan(s, ids, policy, now)
    lens_info = _ensure_lenses(planned) if ensure_lenses else {}
    slack_on = bool(get_settings().slack_webhook_url)
    stats: Counter[str] = Counter({"created": 0, "escalated": 0, "followups": 0, "capped": 0, "unchanged": unchanged})
    severities: Counter[str] = Counter()
    with session_scope() as s:
        # re-plan inside the write transaction: fresh lenses, and safe against a concurrent run
        fresh, _ = _plan(s, sorted({c.ctx.dev.id for c in planned}), policy, now)
        for cand in fresh:
            outcome, severity = _apply(s, cand, policy, now, slack_on)
            stats[outcome] += 1
            if outcome in ("created", "escalated", "followups"):
                severities[f"{cand.ev.audience}:{severity}"] += 1
        if rerendered := rerender_refreshed(s, policy):
            stats["rerendered"] = rerendered
    slack = deliver_pending(now) if slack_on else {"status": "not configured"}
    result = {"evaluated": len(ids), "candidates": len(planned), **dict(stats), "by_severity": dict(severities),
              "slack": slack}
    if lens_info:
        result["lens"] = lens_info
    log.info("alerts: %s", result)
    return result


def _pushed_on_day(s: Session, audience: str, when: datetime, exclude_key: str) -> int:
    """P1 and P2 alerts (follow-ups included) already dated on that UTC day: all use up the push budget."""
    day = datetime(when.year, when.month, when.day)
    return int(s.scalar(select(func.count(Alert.id)).where(
        Alert.audience == audience, Alert.severity.in_(PUSH), Alert.created_at >= day,
        Alert.created_at < day + timedelta(days=1), Alert.dedup_key != exclude_key)) or 0)


def _covers(ctx: DevContext) -> list[int]:
    return sorted(i.id for i in ctx.insights)


def _apply(s: Session, cand: Candidate, policy: Policy, now: datetime, slack_on: bool) -> tuple[str, str | None]:
    if cand.followup is not None:
        return _apply_followup(s, cand, policy, now, slack_on)
    ev = cand.ev
    alert = s.scalar(select(Alert).where(Alert.dedup_key == cand.key))
    if SEVERITY_RANK[ev.severity] <= SEVERITY_RANK[_prior_guarded(alert)]:
        return "unchanged", None
    final, rules = ev.severity, list(ev.rules)
    # P1 is never demoted: a landmark that arrives after the day's budget is spent would otherwise
    # become digest-only for good (the guarded severity is recorded and never re-pushed).
    if final == "P2" and _pushed_on_day(s, ev.audience, cand.when, cand.key) >= policy.daily_cap:
        final = "P3"
        rules.append("guard.daily_cap")
    old = dict(alert.delivered or {}) if alert is not None else {}
    if alert is not None and SEVERITY_RANK[final] <= SEVERITY_RANK[alert.severity]:
        # escalated but swallowed by the cap: remember it, so tomorrow does not re-push stale news
        alert.delivered = {**old, "guarded_severity": ev.severity}
        alert.rules = rules
        return "capped", None
    title, body = render_alert(cand.ctx, ev, final, rules, policy)
    mode = "live" if cand.live else "historical"
    entry = {"at": now.isoformat(timespec="seconds"), "severity": final, "guarded": ev.severity, "mode": mode}
    outcome = "escalated" if alert is not None else "created"
    if alert is None:
        alert = Alert(development_id=cand.ctx.dev.id, audience=ev.audience, dedup_key=cand.key)
        s.add(alert)
    alert.severity, alert.title, alert.body_md, alert.rules = final, title, body, rules
    alert.status, alert.created_at = "open", cand.when
    delivered = {**old, "in_app": entry["at"], "guarded_severity": ev.severity, "mode": mode,
                 "log": [*old.get("log", []), entry], "covers": _covers(cand.ctx)}
    if slack_on and cand.live and final in PUSH:
        delivered.update(slack_pending=True, slack_attempts=0)
    alert.delivered = delivered
    s.flush()
    return outcome, final


def _apply_followup(s: Session, cand: Candidate, policy: Policy, now: datetime,
                    slack_on: bool) -> tuple[str, str | None]:
    """Create a follow-up alert (never updated afterwards: the next sub-event gets its own row)."""
    ev, follow = cand.ev, cand.followup
    assert follow is not None
    siblings = [k for k in s.scalars(select(Alert.dedup_key).where(
        Alert.development_id == cand.ctx.dev.id, Alert.audience == ev.audience)) if is_followup_key(k)]
    if cand.key in siblings or len(siblings) >= FOLLOWUP_MAX:  # a concurrent run got there first
        return "unchanged", None
    final, rules = ev.severity, [FOLLOWUP_RULE, *ev.rules]
    if final == "P2" and _pushed_on_day(s, ev.audience, cand.when, cand.key) >= policy.daily_cap:
        final = "P3"
        rules.append("guard.daily_cap")
    title, body = render_followup(cand.ctx, ev, final, rules, policy, follow)
    mode = "live" if cand.live else "historical"
    entry = {"at": now.isoformat(timespec="seconds"), "severity": final, "guarded": ev.severity, "mode": mode}
    delivered: dict[str, Any] = {"in_app": entry["at"], "guarded_severity": ev.severity, "mode": mode,
                                 "log": [entry], "followup_of": follow.parent.id, "insight_id": follow.insight.id,
                                 "covers": follow.covers}
    if slack_on and cand.live and final in PUSH:
        delivered.update(slack_pending=True, slack_attempts=0)
    s.add(Alert(development_id=cand.ctx.dev.id, audience=ev.audience, dedup_key=cand.key, severity=final,
                title=title, body_md=body, rules=rules, status="open", delivered=delivered, created_at=cand.when))
    s.flush()
    return "followups", final


def rerender_refreshed(s: Session, policy: Policy) -> int:
    """Historical primary alerts (backfilled, never pushed) whose lens was regenerated after they
    were rendered get their title and body rendered again from the current lens and evidence: that
    text is what the in-app timeline shows, and it must not keep analysis the lens has replaced.
    Pushed (live) alerts stay a record of what was sent. Severity, rules and dates are unchanged.
    Returns how many were re-rendered; idempotent (delivered['lens_refresh_at'] records which lens
    refresh the text reflects)."""
    refreshed: dict[tuple[int, str], datetime] = {}
    for lens in s.scalars(select(Lens)):
        record = (lens.checks or {}).get("refresh")
        if isinstance(record, dict) and (at := _parse_time(record.get("at"))) is not None:
            refreshed[(lens.development_id, lens.audience)] = at
    if not refreshed:
        return 0
    stale: list[tuple[Alert, datetime]] = []
    dev_ids = sorted({d for d, _ in refreshed})
    for i in range(0, len(dev_ids), 500):
        for a in s.scalars(select(Alert).where(Alert.development_id.in_(dev_ids[i : i + 500]))):
            d = a.delivered or {}
            reflects = max((t for t in (_parse_time(d.get("lens_refresh_at")), _parse_time(d.get("in_app"))) if t),
                           default=None)  # the last render, or the lens refresh it was re-rendered from
            at = refreshed.get((a.development_id, a.audience))
            if (at is not None and d.get("mode") == "historical" and not is_followup_key(a.dedup_key)
                    and (reflects is None or reflects < at)):
                stale.append((a, at))
    if not stale:
        return 0
    ctxs = load_contexts(s, {a.development_id for a, _ in stale})
    for a, at in stale:
        if (ctx := ctxs.get(a.development_id)) is None:
            continue
        a.title, a.body_md = render_alert(ctx, evaluate(ctx, a.audience, policy), a.severity, list(a.rules or []),
                                          policy)
        a.delivered = {**(a.delivered or {}), "lens_refresh_at": at.isoformat(timespec="seconds"),
                       "covers": _covers(ctx)}
    log.info("re-rendered %d historical alerts from refreshed lenses", len(stale))
    return len(stale)


def deliver_pending(now: datetime | None = None) -> dict:
    """Push alerts marked slack_pending (new ones and retries of recent failures)."""
    now = now or utcnow()
    with session_scope() as s:
        rows = s.scalars(select(Alert).where(Alert.created_at >= now - SLACK_RETRY_WINDOW)).all()
        due = [(a.id, notify.alert_message(a)) for a in rows if (a.delivered or {}).get("slack_pending")]
    sent = failed = 0
    for alert_id, payload in due:
        res = notify.post_slack(payload["text"], payload["blocks"])
        with session_scope() as s:
            alert = s.get(Alert, alert_id)
            d = dict(alert.delivered or {})
            d["slack_attempts"] = int(d.get("slack_attempts", 0)) + 1
            if res.ok:
                sent += 1
                d.update(slack=now.isoformat(timespec="seconds"), slack_pending=False)
                d.pop("slack_error", None)
                if d.get("log"):
                    d["log"] = [*d["log"][:-1], {**d["log"][-1], "channels": ["in_app", "slack"]}]
            else:
                failed += 1
                d["slack_error"] = res.error
                d["slack_pending"] = d["slack_attempts"] < SLACK_MAX_ATTEMPTS
            alert.delivered = d
    return {"sent": sent, "failed": failed}


# --------------------------------------------------------------------------------- rendering
def md_label(text: str) -> str:
    return " ".join(text.replace("[", "(").replace("]", ")").split())


def _domain(url: str | None) -> str:
    host = urlsplit(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host or "source"


def source_ref(doc: Document | None) -> str:
    """'[title](url), official, 2026-10-14' - URL and date come from source metadata, never an LLM."""
    if doc is None:
        return "source unavailable"
    label = md_label(doc.title or _domain(doc.url))[:90]
    date = f", {doc.published_at:%Y-%m-%d}" if doc.published_at else ""
    return f"[{label}]({doc.url}), {doc.source_class}{date}"


def top_contributions(score: Score, k: int = 3) -> list[tuple[str, float]]:
    items = [(str(n), float(v)) for n, v in (score.contributions or {}).items()
             if isinstance(v, int | float) and not isinstance(v, bool)]
    return sorted(items, key=lambda kv: -abs(kv[1]))[:k]


def _default_action(ev: Evaluation) -> str:
    if ev.audience == "investment":
        held = ev.facts.get("held_tickers") or []
        if held:
            return f"Read the primary source and check the thesis for {', '.join(held)}."
        return "Read the primary source; no BIT holding is directly exposed."
    return "Read the primary source and decide whether it merits a time-boxed trial."


def _ticker_line(ctx: DevContext, lens: Lens | None) -> str:
    """The lens's own tickers when a lens exists (none when it abstained: no padding, so a dismissed
    item never shows a row of holdings); exposure links only when no lens exists, event-specific
    first and lab-level relations marked as such."""
    out, seen = [], set()

    def held(sym: str) -> str:
        return f", BIT {ctx.holdings[sym]:.1f}%" if sym in ctx.holdings else ""

    if lens is not None:
        for t in (lens.tickers or []) if lens.action != "no_action" else []:
            if isinstance(t, dict) and t.get("symbol") and str(t["symbol"]).upper() not in seen:
                sym = str(t["symbol"]).upper()
                seen.add(sym)
                why = f": {str(t['why']).rstrip('. ')}" if t.get("why") else ""
                out.append(f"{sym} ({t.get('direction') or 'unclear'}{held(sym)}){why}")
        return "; ".join(out)
    for x in sorted(ctx.exposures, key=lambda x: (is_lab_level(x), x.ticker not in ctx.holdings,
                                                  -(x.confidence or 0.0), x.ticker)):
        if x.ticker not in seen and len(out) < 6:
            seen.add(x.ticker)
            kind = "lab-level link" if is_lab_level(x) else (x.direction or "unclear")
            out.append(f"{x.ticker} ({kind}{held(x.ticker)})")
    return "; ".join(out)


def _lens_lines(ctx: DevContext, ev: Evaluation, lens: Lens | None) -> list[str]:
    """So what / what to do from the lens (the development summary and a default action without
    one), then the investment tickers."""
    lines = []
    if lens is not None:
        lines.append(f"- **So what:** {lens.so_what}")
        action = f" (action: {lens.action.replace('_', ' ')})" if lens.action else ""
        lines.append(f"- **What to do:** {lens.what_to_do}{action}")
    else:
        summary = ctx.dev.summary or next((i.claim for i in ctx.insights if i.claim), "")
        if summary:
            lines.append(f"- **What happened:** {summary}")
        lines.append(f"- **What to do:** {_default_action(ev)}")
    if ev.audience == "investment" and (tickers := _ticker_line(ctx, lens)):
        lines.append(f"- **Tickers:** {tickers}")
    return lines


def _quote_lines(pairs: list[tuple[str, Document | None]]) -> list[str]:
    out = []
    for n, (quote, doc) in enumerate(pairs, 1):
        q = " ".join(quote.split())
        out.append(f'{n}. "{q if len(q) <= 280 else q[:277].rstrip() + "..."}" ({source_ref(doc)})')
    return out


def _rule_lines(ctx: DevContext, ev: Evaluation, rules: list[str], policy: Policy) -> list[str]:
    """Each rule with its description and matched facts, then the score and the coverage."""
    facts, lines = ev.facts, []
    for r in rules:
        matched = rule_reasons(r, facts, policy) if r in ev.fired else []
        lines.append(f"- `{r}`: {policy.describe(r)}" + (f". Matched: {'; '.join(matched)}" if matched else ""))
    if (score := ctx.scores.get(ev.audience)) is not None:
        contrib = ", ".join(f"{n} ({v:+.2f})" for n, v in top_contributions(score))
        lines.append(f"- Score: p_material {score.p_material:.2f} ({score.model_version})"
                     + (f"; top contributions: {contrib}" if contrib else ""))
    classes = ", ".join(facts.get("source_classes") or ["unknown"])
    lines.append(f"- Coverage: {facts.get('source_count') or len(ctx.documents)} source(s) across {classes}; "
                 f"first seen {ctx.dev.first_seen_at:%Y-%m-%d %H:%M} UTC")
    return lines


def render_alert(ctx: DevContext, ev: Evaluation, severity: str, rules: list[str],
                 policy: Policy) -> tuple[str, str]:
    """Title and Markdown body: lens, tickers, top verified quotes with links, and why it was flagged."""
    lens, facts = ctx.lenses.get(ev.audience), ev.facts
    headline = (lens.headline if lens is not None else None) or ctx.dev.title
    title = (("Unconfirmed: " if ev.unconfirmed else "") + headline)[:400]
    parts = [f"**{headline}**"]
    if ev.unconfirmed:
        why = ("only personal posts so far; no official, press or filing source yet" if facts.get("personal_only")
               else "reported, but not yet corroborated by independent source classes")
        parts.append(f"> Unconfirmed: {why}.")
    parts.append("\n".join(_lens_lines(ctx, ev, lens)))
    if quotes := _quote_lines(ctx.evidence(3)):
        parts.append("**Evidence**\n\n" + "\n".join(quotes))
    why_lines = [f"- Severity {severity}" + (f" (rules: {ev.rule_severity})" if ev.rule_severity != severity else "")]
    why_lines += _rule_lines(ctx, ev, rules, policy)
    parts.append("**Why flagged**\n\n" + "\n".join(why_lines))
    return title, "\n\n".join(p for p in parts if p)


def _lens_predates(lens: Lens, ins: Insight) -> bool:
    basis = (lens.checks or {}).get("basis")
    if isinstance(basis, dict) and isinstance(basis.get("newest_insight_id"), int):
        return ins.id > basis["newest_insight_id"]
    return bool(lens.created_at and ins.created_at and ins.created_at > lens.created_at)


def render_followup(ctx: DevContext, ev: Evaluation, severity: str, rules: list[str], policy: Policy,
                    follow: FollowUp) -> tuple[str, str]:
    """Title and Markdown body of a follow-up: what is new (claims of the new insights with their
    sources), the alert it follows, the development lens, new evidence first, and why it fired."""
    lens, trigger, parent = ctx.lenses.get(ev.audience), follow.insight, follow.parent
    subject = " ".join((trigger.title or trigger.claim or ctx.dev.title or "").split())
    title = f"Follow-up: {subject}"[:400]
    parts = [f"**{title}**",
             f"> New since {parent.severity} alert #{parent.id} of {parent.created_at:%Y-%m-%d}: "
             f"{md_label(parent.title or '')[:160]}"]
    lines = []
    for n, (ins, _) in enumerate(follow.hits[:3]):
        claim = " ".join((ins.claim or ins.title or "").split())
        lines.append(f"- **{'What is new' if n == 0 else 'Also new'}:** {claim} "
                     f"({source_ref(ctx.documents.get(ins.document_id))})")
    if lens is not None and _lens_predates(lens, trigger):
        lines.append("- _The lens below was written before this update._")
    parts.append("\n".join(lines + _lens_lines(ctx, ev, lens)))
    pairs, seen = [], set()
    news = [(q, ctx.documents.get(i.document_id)) for i, _ in follow.hits for q in verified_quotes(i)]
    for quote, doc in news + ctx.evidence(6):
        norm = " ".join(quote.lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            pairs.append((quote, doc))
    if quotes := _quote_lines(pairs[:3]):
        parts.append("**Evidence**\n\n" + "\n".join(quotes))
    why_lines = [f"- Severity {severity} (follow-up {follow.number} of at most {FOLLOWUP_MAX}; "
                 f"development rules: {ev.rule_severity})",
                 f"- `{FOLLOWUP_RULE}`: {FOLLOWUP_DESCRIPTION}. Matched: {'; '.join(follow.reasons)}"]
    why_lines += _rule_lines(ctx, ev, [r for r in rules if r != FOLLOWUP_RULE], policy)
    parts.append("**Why flagged**\n\n" + "\n".join(why_lines))
    return title, "\n\n".join(p for p in parts if p)
