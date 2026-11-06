"""Load the register (config/*.yaml) into the database. Idempotent: safe to run on every deploy.

Config files (tracked configuration lives in files, not a UI - per the brief):
  config/labs.yaml      labs, their public exposure, identities and official sources
  config/people.yaml    tracked people: tier, why they are key, affiliations, identities
  config/sources.yaml   global sources not owned by one lab (papers, press, filings, model index)
  config/exposure.yaml  tickers, themes and BIT Capital holdings (closed list for ticker mapping)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from fli.config import CONFIG_DIR
from fli.db.models import Affiliation, Identity, Lab, Person, Source, Ticker
from fli.db.session import session_scope
from fli.timeutil import to_naive_utc

log = logging.getLogger("fli.register")

OFFICIAL_KINDS = {
    "official_blog", "research_blog", "engineering_blog", "release_notes", "model_cards",
    "github_org", "hf_org", "job_board", "docs_changelog",
}
CLASS_BY_KIND = {"arxiv": "research", "hf_papers": "research", "press": "press", "sec": "filing",
                 "model_index": "market_data"}
# identity platforms that become pollable person sources when `track: true`
TRACKABLE = {"x": ("x_timeline", "x_account"), "bluesky": ("bluesky_feed", "bluesky_account")}


def _read(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def norm_handle(platform: str, handle: str) -> str:
    handle = handle.strip()
    if platform in {"x", "github", "hf", "bluesky"}:
        return handle.lstrip("@").lower()
    return handle.lower()


def _upsert(session: Session, model, pk: Any, **fields):
    obj = session.get(model, pk)
    if obj is None:
        obj = model(**fields)
        session.add(obj)
    else:
        for k, v in fields.items():
            setattr(obj, k, v)
    return obj


def _upsert_identity(session: Session, *, platform: str, handle: str, person_id: str | None = None,
                     lab_id: str | None = None, url: str | None = None, method: str = "seed",
                     evidence: str | None = None, confidence: float = 1.0, meta: dict | None = None) -> Identity:
    h = norm_handle(platform, handle)
    ident = session.scalar(select(Identity).where(Identity.platform == platform, Identity.handle == h))
    if ident is None:
        ident = Identity(platform=platform, handle=h)
        session.add(ident)
    ident.person_id, ident.lab_id = person_id, lab_id
    ident.url = url or ident.url
    if ident.method in (None, "seed") or method != "seed":  # never downgrade a manual/verified link
        ident.method, ident.status, ident.confidence = method, "confirmed", confidence
        ident.evidence = evidence or ident.evidence
    elif ident.status == "candidate":  # the curated register confirms a machine-proposed candidate
        ident.status = "confirmed"
    if meta:
        ident.meta = {**(ident.meta or {}), **meta}
    return ident


def _source_fields(spec: dict, *, lab_id: str | None = None, person_id: str | None = None,
                   default_class: str | None = None) -> dict:
    kind = spec["kind"]
    source_class = spec.get("source_class") or default_class or (
        "official" if kind in OFFICIAL_KINDS else CLASS_BY_KIND.get(kind, "official" if lab_id else "personal")
    )
    config = dict(spec.get("config") or {})
    for k in ("handle", "org", "board", "query"):
        if k in spec and k not in config:
            config[k] = spec[k]
    return dict(
        lab_id=lab_id, person_id=person_id, adapter=spec["adapter"], kind=kind, source_class=source_class,
        url=spec.get("url"), config=config, cadence_minutes=int(spec.get("cadence_minutes", 360)),
        freshness_slo_hours=int(spec.get("freshness_slo_hours", 24 * 14)), enabled=bool(spec.get("enabled", True)),
    )


def load_register(config_dir: Path | None = None) -> dict:
    global CONFIG_DIR
    if config_dir is not None:
        CONFIG_DIR = Path(config_dir)
    labs_cfg, people_cfg, exposure_cfg = _read("labs.yaml"), _read("people.yaml"), _read("exposure.yaml")
    # sources are split by owner/domain: sources_feeds.yaml, sources_research.yaml, sources_social.yaml ...
    sources_cfg = {"sources": [spec for f in sorted(CONFIG_DIR.glob("sources*.yaml"))
                               for spec in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("sources", [])]}
    counts = {"labs": 0, "people": 0, "identities": 0, "sources": 0, "tickers": 0}
    with session_scope() as s:
        # --- tickers (needed before labs reference them) ---------------------------------------
        weights = {h["symbol"]: h.get("weight") for h in exposure_cfg.get("bit_holdings", [])}
        funds = {h["symbol"]: h.get("funds", []) for h in exposure_cfg.get("bit_holdings", [])}
        theme_members: dict[str, list[str]] = {}
        for theme in exposure_cfg.get("themes", []):
            for sym in theme.get("tickers", []):
                theme_members.setdefault(sym, []).append(theme["id"])
        for t in exposure_cfg.get("tickers", []):
            sym = t["symbol"]
            _upsert(s, Ticker, sym, symbol=sym, company=t["company"], sector=t.get("sector"),
                    themes=sorted(set(theme_members.get(sym, []) + t.get("themes", []))),
                    bit_weight=weights.get(sym), bit_funds=funds.get(sym, []), notes=t.get("notes"))
            counts["tickers"] += 1

        # --- labs ---------------------------------------------------------------------------
        for lab in labs_cfg.get("labs", []):
            _upsert(s, Lab, lab["id"], id=lab["id"], name=lab["name"], kind=lab.get("kind", "frontier_lab"),
                    tier=int(lab.get("tier", 2)), parent=lab.get("parent"), homepage=lab.get("homepage"),
                    description=lab.get("description"), public_exposure=lab.get("public_exposure", []),
                    meta=lab.get("meta", {}))
            counts["labs"] += 1
        s.flush()
        for lab in labs_cfg.get("labs", []):
            for ident in lab.get("identities", []):
                _upsert_identity(s, platform=ident["platform"], handle=ident["handle"], lab_id=lab["id"],
                                 url=ident.get("url"), evidence="official lab account (register)")
                counts["identities"] += 1
            for spec in lab.get("sources", []):
                sid = f"{lab['id']}:{spec['key']}"
                _upsert(s, Source, sid, id=sid, **_source_fields(spec, lab_id=lab["id"]))
                counts["sources"] += 1

        # --- people ---------------------------------------------------------------------------
        for p in people_cfg.get("people", []):
            _upsert(s, Person, p["id"], id=p["id"], name=p["name"], lab_id=p.get("lab"), title=p.get("title"),
                    tier=p.get("tier", "C"), why_key=p.get("why_key"), inclusion_rule=p.get("inclusion_rule"),
                    status=p.get("status", "active"), meta=p.get("meta", {}))
            counts["people"] += 1
        s.flush()
        for p in people_cfg.get("people", []):
            existing = {(a.lab_id, a.org_name, a.role) for a in s.scalars(
                select(Affiliation).where(Affiliation.person_id == p["id"]))}
            for a in p.get("affiliations", []):
                key = (a.get("lab"), a.get("org"), a.get("role"))
                if key in existing:
                    continue
                s.add(Affiliation(person_id=p["id"], lab_id=a.get("lab"), org_name=a.get("org"), role=a.get("role"),
                                  start_date=to_naive_utc(a.get("start")), end_date=to_naive_utc(a.get("end")),
                                  source_url=a.get("source_url"), evidence=a.get("evidence"),
                                  confidence=float(a.get("confidence", 1.0))))
            for ident in p.get("identities", []):
                _upsert_identity(s, platform=ident["platform"], handle=ident["handle"], person_id=p["id"],
                                 url=ident.get("url"), evidence=ident.get("evidence", "curated seed (register)"),
                                 meta=ident.get("meta"))
                counts["identities"] += 1
                if ident.get("track") and ident["platform"] in TRACKABLE:
                    adapter, kind = TRACKABLE[ident["platform"]]
                    sid = f"{ident['platform']}:{norm_handle(ident['platform'], ident['handle'])}"
                    spec = {"adapter": adapter, "kind": kind, "handle": ident["handle"],
                            "cadence_minutes": ident.get("cadence_minutes", 720),
                            "freshness_slo_hours": ident.get("freshness_slo_hours", 24 * 30)}
                    cfg = {k: ident[k] for k in ("max_posts", "page_size") if k in ident}
                    if cfg:  # per-account caps (X reads are metered)
                        spec["config"] = cfg
                    _upsert(s, Source, sid, id=sid, **_source_fields(spec, person_id=p["id"],
                                                                     default_class="personal"))
                    counts["sources"] += 1

        # --- seed identities removed from people.yaml are rejected (the loader never deletes) ---
        for p in people_cfg.get("people", []):
            listed = {(i["platform"], norm_handle(i["platform"], i["handle"])) for i in p.get("identities", [])}
            for ident in s.scalars(select(Identity).where(Identity.person_id == p["id"], Identity.method == "seed",
                                                          Identity.status != "rejected")):
                if (ident.platform, ident.handle) not in listed:
                    ident.status = "rejected"

        # --- global sources -------------------------------------------------------------------
        for spec in sources_cfg.get("sources", []):
            _upsert(s, Source, spec["id"], id=spec["id"],
                    **_source_fields(spec, lab_id=spec.get("lab"), person_id=spec.get("person")))
            counts["sources"] += 1
    log.info("register loaded: %s", counts)
    return counts
