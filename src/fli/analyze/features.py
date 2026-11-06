"""Interpretable development features, shared by both audiences' materiality models.

Every feature is a plain number with an obvious meaning, so a fitted coefficient reads as "how much
this property moves the odds that the audience wants to see the item". The same vector serves both
audiences; they differ only in their separately trained coefficients.

Only facts recorded by earlier stages are used (source metadata, extractor fields, novelty, rule
exposure links). Two kinds of exposure link are deliberately ignored:
  - lab-level links (method 'lab', see fli.analyze.exposure): they come only from the lab's curated
    relations, so every development of a lab carries the same ones (every OpenAI item had 8 tickers).
    Counting them made n_tickers a lab identifier, and the investment model learnt a spurious negative
    weight for it while the event-specific signal was diluted.
  - LLM-made links (lens stage): the lens runs on high-scoring developments, so counting its links
    would feed the score back into itself.
The remaining event-specific links are weighted by their confidence, so a company the development
names (0.65+) counts more than a theme-keyword guess (0.35-0.55).

Attention uses only counters that describe the item itself: a paper's upvotes, a model's likes and
downloads, a post's likes or reactions. GitHub `stars` are not used: on a release they are the
repository's lifetime stars, so every point release of a popular SDK inherited them (openai-python:
32k) and scored as if that tag had drawn attention, which lifted SDK point releases over the AI
team's P1 bar; on a new-repo document they are the count at the latest crawl (the collector keeps
refreshing it), so a backfill sees months of later popularity a live run never has when it first
sees the repo. Upvotes, likes and downloads are read at crawl time too, but they measure the item.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from fli.db.models import Development, Document, ExposureLink, Insight, Lab, Person, Ticker

log = logging.getLogger("fli.features")

EVENT_GROUPS: dict[str, tuple[str, ...]] = {
    "release": ("model_release", "open_weight_release", "capability_claim"),
    "research": ("research_paper", "technique"),
    "capital": ("funding_ipo", "new_venture"),
    "infra": ("compute_deal", "energy_deal", "custom_silicon"),
    "people": ("talent_move",),
    "strategy": ("strategy_shift", "partnership", "product_launch", "pricing_change"),
    "policy": ("safety_policy",),
    "hiring": ("hiring_signal",),
}
_GROUP_OF = {et: group for group, types in EVENT_GROUPS.items() for et in types}
PERSON_TIER_VALUE = {"A": 1.0, "B": 0.6, "C": 0.3}
PRESS_LIKE = {"press", "aggregator"}
# per-item engagement counters collectors put in document.meta (top level or one dict deep, e.g. X
# public_metrics, Bluesky metrics); repository-level counters (stars, forks) are excluded, see above
ATTENTION_KEYS = ("upvotes", "likes", "like_count", "likeCount", "favorite_count", "reactions", "downloads")
LINK_METHOD = "rule"  # event-specific rule links only: not 'lab' (lab-level) or 'llm' (lens) links
DEFAULT_LINK_CONFIDENCE = 0.5  # the column default, for rows written without one
# imputation for values earlier stages did not produce: "low-moderate" magnitude, neutral novelty/confidence
DEFAULT_MAGNITUDE, DEFAULT_NOVELTY, DEFAULT_CONFIDENCE = 2.0, 0.5, 0.5

FEATURE_NAMES: list[str] = [
    "has_official", "log_sources", "n_source_classes", "has_verified",
    "max_magnitude", "mean_confidence", "rumor_share", "max_novelty", "first_seen_artifact",
    "lab_tier_1", "lab_tier_2", "lab_tier_3", "person_tier",
    *[f"grp_{g}" for g in EVENT_GROUPS],
    "exposure_bit", "n_tickers", "attention", "press_only",
]
# judgements made by the extraction LLM (ablated in training to show what the model gains from them)
LLM_DERIVED: tuple[str, ...] = ("max_magnitude", "mean_confidence")

FEATURE_LABELS: dict[str, str] = {
    "has_official": "Published on an official lab channel",
    "log_sources": "Independent sources (log)",
    "n_source_classes": "Kinds of source (official, personal, press ...)",
    "has_verified": "Evidence quotes verified against the source",
    "max_magnitude": "Extractor magnitude judgement (1-5)",
    "mean_confidence": "Extractor confidence",
    "rumor_share": "Share of rumoured / unconfirmed claims",
    "max_novelty": "New relative to the lab's recent history",
    "first_seen_artifact": "Names a model/product never seen before",
    "lab_tier_1": "Tier-1 frontier lab",
    "lab_tier_2": "Tier-2 lab (challenger / open-weight)",
    "lab_tier_3": "Tier-3 alumni startup",
    "person_tier": "Involves a key person (A=1, B=0.6, C=0.3)",
    "grp_release": "Model release / capability claim",
    "grp_research": "Research paper / technique",
    "grp_capital": "Funding, IPO or new venture",
    "grp_infra": "Compute, energy or custom silicon",
    "grp_people": "Talent move",
    "grp_strategy": "Strategy, partnership, product or pricing",
    "grp_policy": "Safety / policy",
    "grp_hiring": "Hiring signal",
    "exposure_bit": "BIT holdings the event itself links to (fund weight % x link confidence, / 10)",
    "n_tickers": "Public companies the event itself links to (confidence-weighted)",
    "attention": "Attention to the item itself (upvotes / likes / downloads, log)",
    "press_only": "Reported only by press / aggregators",
}


def event_group(event_type: str | None) -> str | None:
    return _GROUP_OF.get(event_type or "")


@dataclass
class FeatureContext:
    """Small register lookups loaded once per run instead of once per development."""

    lab_tiers: dict[str, int] = field(default_factory=dict)
    person_tiers: dict[str, str] = field(default_factory=dict)
    bit_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, session: Session) -> FeatureContext:
        return cls(
            lab_tiers={lab_id: int(tier or 0) for lab_id, tier in session.execute(select(Lab.id, Lab.tier))},
            person_tiers={pid: tier for pid, tier in session.execute(select(Person.id, Person.tier))},
            bit_weights={sym: float(w) for sym, w in session.execute(select(Ticker.symbol, Ticker.bit_weight))
                         if w},
        )


def engagement(meta: Mapping | None) -> float:
    """Largest per-item engagement counter in a document's meta (0 when none is present)."""
    if not isinstance(meta, Mapping):
        return 0.0
    best = 0.0
    scopes = [meta, *(v for v in meta.values() if isinstance(v, Mapping))]
    for scope in scopes:
        for key in ATTENTION_KEYS:
            try:
                best = max(best, float(scope.get(key) or 0))
            except (TypeError, ValueError):
                continue
    return best


def _mean(values: Iterable[float], default: float) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else default


def link_weight(link: ExposureLink) -> float:
    """A link's confidence, clipped to [0, 1]: the weight it gets in the exposure features."""
    conf = DEFAULT_LINK_CONFIDENCE if link.confidence is None else float(link.confidence)
    return min(1.0, max(0.0, conf))


def development_features(session: Session, dev: Development, ctx: FeatureContext | None = None) -> dict[str, float]:
    """Feature vector for one development (see FEATURE_LABELS for meanings)."""
    ctx = ctx or FeatureContext.load(session)
    insights = list(session.scalars(select(Insight).where(Insight.development_id == dev.id)))
    doc_ids = {i.document_id for i in insights}
    docs = list(session.scalars(select(Document).where(Document.id.in_(doc_ids)))) if doc_ids else []
    links = list(session.scalars(select(ExposureLink).where(
        ExposureLink.development_id == dev.id, ExposureLink.method == LINK_METHOD)))

    # source facts: insights' documents are the ground truth, development columns the fallback
    classes = set(dev.source_classes or []) | {d.source_class for d in docs if d.source_class}
    n_sources = max(int(dev.source_count or 0), len({d.source_id for d in docs}))
    has_official = bool(dev.has_official) or any(i.is_official for i in insights) or "official" in classes
    has_verified = bool(dev.has_verified) or any(i.quotes_verified for i in insights)

    magnitudes = [float(i.magnitude) for i in insights if i.magnitude is not None]
    novelties = [float(i.novelty_entity) for i in insights if i.novelty_entity is not None]
    tier = ctx.lab_tiers.get(dev.lab_id or "", 0)
    person_ids = set(dev.person_ids or []) | {p for i in insights for p in (i.person_ids or [])}
    group = event_group(dev.event_type)
    feats: dict[str, float] = {
        "has_official": float(has_official),
        "log_sources": math.log1p(n_sources),
        "n_source_classes": float(len(classes)),
        "has_verified": float(has_verified),
        "max_magnitude": max(magnitudes) if magnitudes else DEFAULT_MAGNITUDE,
        "mean_confidence": _mean((float(i.confidence) for i in insights if i.confidence is not None),
                                 DEFAULT_CONFIDENCE),
        "rumor_share": _mean((float(bool(i.is_rumor)) for i in insights), 0.0),
        "max_novelty": max(novelties) if novelties else DEFAULT_NOVELTY,
        "first_seen_artifact": float(any(i.first_seen_artifacts for i in insights)),
        "lab_tier_1": float(tier == 1),
        "lab_tier_2": float(tier == 2),
        "lab_tier_3": float(tier == 3),
        "person_tier": max((PERSON_TIER_VALUE.get(ctx.person_tiers.get(p, ""), 0.0) for p in person_ids),
                           default=0.0),
        **{f"grp_{g}": float(g == group) for g in EVENT_GROUPS},
        # one row per (development, ticker), so the sums count each company once
        "exposure_bit": sum(ctx.bit_weights.get(link.ticker, 0.0) * link_weight(link) for link in links) / 10.0,
        "n_tickers": sum(link_weight(link) for link in links),
        "attention": math.log1p(max((engagement(d.meta) for d in docs), default=0.0)),
        "press_only": float(bool(classes) and classes <= PRESS_LIKE),
    }
    return {name: round(float(feats[name]), 6) for name in FEATURE_NAMES}


def describe_contributions(contributions: Mapping[str, float]) -> list[dict]:
    """Human-readable 'why flagged' rows, largest effect first (for the web UI and alerts)."""
    rows = [{"feature": k, "label": FEATURE_LABELS.get(k, k), "contribution": round(float(v), 3)}
            for k, v in contributions.items()]
    return sorted(rows, key=lambda r: -abs(r["contribution"]))
