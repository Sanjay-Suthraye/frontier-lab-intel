"""Shared vocabulary. Every stage (LLM schemas, scoring features, materiality rules, UI) uses these
exact strings, so changing a value here is a schema change."""

from __future__ import annotations

AUDIENCES: tuple[str, ...] = ("investment", "ai_team")

EVENT_TYPES: dict[str, str] = {
    "model_release": "A new model or major model version is released or announced (API or product).",
    "open_weight_release": "Open-weight model, dataset or major code release that others can run.",
    "capability_claim": "Benchmark result, capability jump or notable demonstrated ability (incl. self-reported).",
    "research_paper": "Notable paper, technical report or research result.",
    "technique": "Engineering technique or tooling for building AI systems (agents, evals, orchestration, inference, training).",
    "product_launch": "New product, feature, API capability or distribution channel.",
    "pricing_change": "API/product price change or new pricing tier.",
    "talent_move": "A key person joins, leaves or changes role.",
    "new_venture": "A spinout / new startup by lab alumni, or a stealth company surfaces.",
    "funding_ipo": "Funding round, valuation, IPO filing, M&A or secondary sale.",
    "compute_deal": "Compute / cloud / datacenter / chip-supply agreement, or a lab's own capacity build-out milestone (MW/GW, GPU count, campus).",
    "energy_deal": "Power or energy agreement for AI infrastructure.",
    "custom_silicon": "A lab's own chip / hardware programme news.",
    "strategy_shift": "New strategic direction, reorganisation, new business line (devices, ads, robotics ...).",
    "partnership": "Commercial partnership or distribution deal (non-compute).",
    "safety_policy": "Safety framework, policy/regulatory engagement, government contract, export controls.",
    "hiring_signal": "Hiring trend derived from public job boards.",
    "other": "Relevant but none of the above.",
}

# Triage noise categories: items flagged with one of these (and not relevant) are suppressed.
NOISE_REASONS: tuple[str, ...] = (
    "none",
    "marketing",
    "customer_story",
    "event_promo",
    "hiring_post",
    "minor_update",
    "off_topic",
    "opinion_chatter",
    "repost",
)

# How much a source class is trusted as a *primary* source for attribution.
SOURCE_CLASSES: tuple[str, ...] = ("official", "research", "personal", "filing", "press", "aggregator", "market_data",
                                  "curated")  # curated = analyst-maintained register facts, each with a source URL

INVESTMENT_ACTIONS: tuple[str, ...] = ("watch", "review_position", "update_thesis", "ask_company", "no_action")
AI_TEAM_ACTIONS: tuple[str, ...] = ("adopt", "trial", "assess", "hold")  # tech-radar rings
DIRECTIONS: tuple[str, ...] = ("positive", "negative", "mixed", "unclear")
