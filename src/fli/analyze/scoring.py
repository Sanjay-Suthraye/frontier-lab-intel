"""Materiality scoring: one small, interpretable logistic model per audience.

p_material = P(the audience labels the development >= 1, "worth knowing" | features). Model:
StandardScaler + L2 logistic regression (class_weight='balanced', C by stratified CV), trained on
labels with human labels preferred over LLM-judge labels (the source is recorded, never mixed
silently) and validated on a time-ordered hold-out (first 70% by first_seen_at train, last 30% test)
against one-feature baselines, the hand-set prior and an ablation without the extractor's LLM
judgements. Balanced weights distort probabilities, so the intercept gets the King-Zeng prior
correction (+ln(n1/n0)); ranking metrics are unaffected, Brier/calibration become meaningful.

Day one (no labels): 'prior-v0', hand-set coefficients encoding the materiality policy and marked
validated=False wherever shown. It keeps the system usable and gives the label sampler a score to
stratify on; once labels exist, `fli train` measures how good it actually was.

"Why flagged": contributions are coef_i x standardised x_i. With the intercept they sum exactly to
the logit, so the panel explains the score instead of decorating it.

Deployed models are data/models/scoring_<audience>.json, written atomically (temp file + os.replace),
so a reader never sees a half-written model. A deployed file that cannot be read halts scoring for
that audience (ModelFileError, plus a crit 'model_unreadable' ops event) instead of silently scoring
with prior-v0: that fallback would re-stamp the unvalidated prior's rows as newest, and alerts and
digests (which read the newest row) would switch models while nobody noticed. Consumers that filter
Score rows by version (the web) ask current_versions(), which never raises.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fli.analyze.exposure import SKIP_STATUSES
from fli.analyze.features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    LLM_DERIVED,
    FeatureContext,
    development_features,
)
from fli.config import DATA_DIR
from fli.db.models import Development, OpsEvent, Score
from fli.db.session import session_scope
from fli.taxonomy import AUDIENCES
from fli.timeutil import utcnow

log = logging.getLogger("fli.scoring")

MODELS_DIR = DATA_DIR / "models"
EVAL_DIR = DATA_DIR / "eval"
PRIOR_VERSION = "prior-v0"
TOP_CONTRIBUTIONS = 6
RECHECK_DAYS = 7  # recent developments are re-featurised each run: novelty/links may land later
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CV_AUC_TOLERANCE = 0.01  # CV AUC differences smaller than this are fold noise (see choose_c)
TRAIN_FRACTION = 0.7
MIN_LABELS, MIN_PER_CLASS = 40, 5
MAX_AUC_REGRESSION = 0.02  # deploy a trained model only if it is not worse than the prior on the hold-out
OPS_KIND = "model_unreadable"  # ops_event kind for a deployed model file that cannot be read


class ModelFileError(RuntimeError):
    """A deployed model file exists but cannot be used. Scoring for that audience stops (the last good
    scores stay the newest rows) rather than falling back to the unvalidated prior."""

    def __init__(self, audience: str, path: Path, reason: str):
        super().__init__(f"scoring model for {audience} at {path} is unreadable ({reason}); not scoring {audience}"
                         f" until it is fixed (re-run `fli train`, or delete the file to use {PRIOR_VERSION})")
        self.audience, self.path, self.reason = audience, path, reason


# --------------------------------------------------------------------------------------- model


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


@dataclass
class ScoringModel:
    audience: str
    version: str
    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    validated: bool = False
    meta: dict = field(default_factory=dict)

    def contributions(self, feats: Mapping[str, float]) -> dict[str, float]:
        """coef_i x standardised x_i for every feature; intercept + sum == logit."""
        return {n: c * (float(feats.get(n, 0.0)) - m) / (s or 1.0)
                for n, m, s, c in zip(self.feature_names, self.mean, self.scale, self.coef, strict=True)}

    def logit(self, feats: Mapping[str, float]) -> float:
        return self.intercept + sum(self.contributions(feats).values())

    def predict(self, feats: Mapping[str, float]) -> float:
        return sigmoid(self.logit(feats))

    def explain(self, feats: Mapping[str, float], top: int = TOP_CONTRIBUTIONS) -> dict[str, float]:
        ranked = sorted(self.contributions(feats).items(), key=lambda kv: -abs(kv[1]))
        return {k: round(v, 4) for k, v in ranked[:top] if abs(v) >= 1e-4}

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """Vectorised predict; X columns follow self.feature_names."""
        scale = np.where(np.asarray(self.scale) == 0, 1.0, self.scale)
        z = self.intercept + ((np.asarray(X, float) - np.asarray(self.mean)) / scale) @ np.asarray(self.coef)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def to_dict(self) -> dict:
        core = {"audience": self.audience, "version": self.version, "validated": self.validated,
                "feature_names": self.feature_names, "scaler_mean": self.mean, "scaler_scale": self.scale,
                "coefficients": self.coef, "intercept": self.intercept}
        return {**core, **self.meta}

    @classmethod
    def from_dict(cls, d: Mapping) -> ScoringModel:
        """Parse and check a saved model; ValueError/KeyError/TypeError when it is unusable."""
        core = {"audience", "version", "validated", "feature_names", "scaler_mean", "scaler_scale",
                "coefficients", "intercept"}
        model = cls(audience=str(d["audience"]), version=str(d["version"]), feature_names=list(d["feature_names"]),
                    mean=[float(x) for x in d["scaler_mean"]], scale=[float(x) for x in d["scaler_scale"]],
                    coef=[float(x) for x in d["coefficients"]], intercept=float(d["intercept"]),
                    validated=bool(d.get("validated", False)), meta={k: v for k, v in d.items() if k not in core})
        n = len(model.feature_names)
        if not model.version or not n or {len(model.mean), len(model.scale), len(model.coef)} != {n}:
            raise ValueError(f"inconsistent model: {n} features, {len(model.mean)} means, {len(model.scale)} scales, "
                             f"{len(model.coef)} coefficients, version {model.version!r}")
        if not all(math.isfinite(x) for x in (*model.mean, *model.scale, *model.coef, model.intercept)):
            raise ValueError("non-finite model parameters")
        return model


# --------------------------------------------------------------------------------------- prior
# Contributions are measured from a "typical development" so the panel reads "why above/below
# typical" and the intercept is the logit of a typical item (about 0.17). exposure_bit, n_tickers and
# attention are the corpus means since they count only event-specific links and per-item attention
# (2026-11-04: 0.17, 1.0, 1.04; they were 1.0, 4.0 and 2.0 while lab-level links and repo stars counted).
PRIOR_REFERENCE: dict[str, float] = {
    "has_official": 0.4, "log_sources": 0.8, "n_source_classes": 1.2, "has_verified": 0.6,
    "max_magnitude": 2.5, "mean_confidence": 0.7, "rumor_share": 0.1, "max_novelty": 0.5,
    "first_seen_artifact": 0.15, "lab_tier_1": 0.55, "lab_tier_2": 0.3, "lab_tier_3": 0.1, "person_tier": 0.3,
    "grp_release": 0.2, "grp_research": 0.3, "grp_capital": 0.05, "grp_infra": 0.05, "grp_people": 0.05,
    "grp_strategy": 0.25, "grp_policy": 0.05, "grp_hiring": 0.03,
    "exposure_bit": 0.2, "n_tickers": 1.0, "attention": 1.0, "press_only": 0.15,
}
# Per raw unit. Policy: confirmation (official, verified, several sources), magnitude and novelty
# raise both audiences; capital/infra/people events and BIT-weighted exposure matter to the
# investment team, releases/research and community attention to the AI team; rumours and
# press-only items are discounted.
PRIOR_WEIGHTS: dict[str, dict[str, float]] = {
    "investment": {
        "has_official": 0.8, "log_sources": 0.7, "n_source_classes": 0.2, "has_verified": 0.3,
        "max_magnitude": 0.8, "mean_confidence": 0.5, "rumor_share": -0.8, "max_novelty": 0.8,
        "first_seen_artifact": 0.4, "lab_tier_1": 0.6, "lab_tier_2": -0.2, "lab_tier_3": 0.3, "person_tier": 0.8,
        "grp_release": 0.3, "grp_research": -1.2, "grp_capital": 1.0, "grp_infra": 1.2, "grp_people": 0.6,
        "grp_strategy": 0.2, "grp_policy": 0.1, "grp_hiring": -0.4,
        "exposure_bit": 0.3, "n_tickers": 0.05, "attention": 0.05, "press_only": -0.3,
    },
    "ai_team": {
        "has_official": 0.6, "log_sources": 0.5, "n_source_classes": 0.1, "has_verified": 0.3,
        "max_magnitude": 0.7, "mean_confidence": 0.5, "rumor_share": -1.0, "max_novelty": 0.8,
        "first_seen_artifact": 0.6, "lab_tier_1": 0.4, "lab_tier_2": 0.3, "lab_tier_3": 0.0, "person_tier": 0.3,
        "grp_release": 1.0, "grp_research": 0.6, "grp_capital": -2.0, "grp_infra": -2.0, "grp_people": -0.6,
        "grp_strategy": 0.0, "grp_policy": -0.2, "grp_hiring": -1.2,
        "exposure_bit": 0.0, "n_tickers": 0.0, "attention": 0.25, "press_only": -0.6,
    },
}
PRIOR_INTERCEPT: dict[str, float] = {"investment": -1.6, "ai_team": -1.6}


def prior_model(audience: str) -> ScoringModel:
    return ScoringModel(
        audience=audience, version=PRIOR_VERSION, feature_names=list(FEATURE_NAMES),
        mean=[PRIOR_REFERENCE[n] for n in FEATURE_NAMES], scale=[1.0] * len(FEATURE_NAMES),
        coef=[PRIOR_WEIGHTS[audience].get(n, 0.0) for n in FEATURE_NAMES], intercept=PRIOR_INTERCEPT[audience],
        validated=False, meta={"kind": "hand-set policy prior",
                               "note": "UNVALIDATED: coefficients encode the materiality policy, not fitted to labels"},
    )


# --------------------------------------------------------------------------------------- loading


def model_path(audience: str, models_dir: Path | None = None) -> Path:
    return Path(models_dir or MODELS_DIR) / f"scoring_{audience}.json"


@lru_cache(maxsize=8)
def _read_model(path: str, _mtime_ns: int) -> ScoringModel:
    model = ScoringModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    if missing := set(FEATURE_NAMES) - set(model.feature_names):
        log.warning("%s predates features %s (scored as 0 effect); retrain with `fli train`", path, sorted(missing))
    return model


def ops_subject(audience: str) -> str:
    """OpsEvent subject for an audience's model file (kind OPS_KIND), shared with fli.ops.health."""
    return f"scoring_model:{audience}"


_recorded: set[tuple[str, str, int]] = set()  # (dedup key, path, mtime) already recorded by this process


def _record_unreadable(err: ModelFileError, mtime_ns: int) -> None:
    """Open a crit ops_event for an unreadable model file, at most once per day and file version.
    Best effort: raising ModelFileError is what protects the scores, so a DB problem here only logs."""
    key = f"{OPS_KIND}:{ops_subject(err.audience)}:{utcnow():%Y-%m-%d}"[:240]
    if (key, str(err.path), mtime_ns) in _recorded:
        return
    try:
        with session_scope() as s:
            if s.scalar(select(OpsEvent.id).where(OpsEvent.dedup_key == key)) is None:
                s.add(OpsEvent(kind=OPS_KIND, severity="crit", subject=ops_subject(err.audience),
                               message=str(err)[:2000], dedup_key=key, delivered={}))
        _recorded.add((key, str(err.path), mtime_ns))
    except Exception:  # noqa: BLE001 - never let a ledger problem mask the ModelFileError
        log.exception("could not record the %s ops event for %s", OPS_KIND, err.path)


def load_model(audience: str, models_dir: Path | None = None) -> ScoringModel:
    """The deployed model for an audience; the unvalidated prior when none is deployed (no file).

    Raises ModelFileError, after logging and recording a crit ops event, when a deployed file exists but
    cannot be used (truncated or invalid JSON, inconsistent or non-finite parameters, another audience's
    model): silently scoring with prior-v0 instead would switch alerts to the unvalidated prior."""
    path = model_path(audience, models_dir)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except FileNotFoundError:
        return prior_model(audience)
    except OSError as exc:  # exists but cannot be inspected (permissions, I/O)
        mtime_ns, problem = 0, exc
    else:
        try:
            model = _read_model(str(path), mtime_ns)
            if model.audience == audience:
                return model
            problem = ValueError(f"the file holds the {model.audience!r} model")
        except FileNotFoundError:  # removed since the stat: nothing deployed any more
            return prior_model(audience)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problem = exc
    err = ModelFileError(audience, path, f"{problem.__class__.__name__}: {problem}")
    log.error("%s", err)
    _record_unreadable(err, mtime_ns)
    raise err from problem


def model_file_problems(models_dir: Path | None = None) -> list[ModelFileError]:
    """Deployed model files that cannot be used: for a health check (OpsEvent kind OPS_KIND, subject
    ops_subject(audience)) so the event stays open, and is announced, until the file is fixed."""
    problems = []
    for aud in AUDIENCES:
        try:
            load_model(aud, models_dir)
        except ModelFileError as exc:
            problems.append(exc)
    return problems


def current_versions(session: Session | None = None, *, models_dir: Path | None = None) -> dict[str, str]:
    """Live model version per audience: the version whose Score rows a page should show. Never raises.

    Normally the deployed model's version (prior-v0 when none is deployed); the scoring stage keeps that
    version's rows the newest per development, which is what alerts, digests and the lens read. Two
    transitional cases fall back to the version of the audience's newest Score row (what the alert stage
    reads right now): the deployed file is unreadable (scoring is halted, see load_model), or its version
    has no rows yet (deployed moments ago, before `fli train` finished re-scoring)."""
    if session is None:
        with session_scope() as s:
            return current_versions(s, models_dir=models_dir)
    out: dict[str, str] = {}
    for aud in AUDIENCES:
        try:
            deployed: str | None = load_model(aud, models_dir).version
        except ModelFileError:
            deployed = None
        if deployed is not None and session.scalar(select(Score.id).where(
                Score.audience == aud, Score.model_version == deployed).limit(1)) is not None:
            out[aud] = deployed
            continue
        newest = session.scalar(select(Score.model_version).where(Score.audience == aud)
                                .order_by(Score.created_at.desc(), Score.id.desc()).limit(1))
        out[aud] = newest or deployed or PRIOR_VERSION
    return out


def current_scores(session: Session, audience: str, models_dir: Path | None = None) -> dict[int, Score]:
    """Scores of the live model version (current_versions), keyed by development id."""
    version = current_versions(session, models_dir=models_dir)[audience]
    q = select(Score).where(Score.audience == audience, Score.model_version == version)
    return {sc.development_id: sc for sc in session.scalars(q)}


# --------------------------------------------------------------------------------------- stage


def _stale(dev: Development, sc: Score | None, newest: datetime | None, recheck_after: datetime) -> bool:
    """Unscored; changed since scored; still recent (novelty/links may land later); or a row of another
    model version is newer (UI, lens and alerts read the newest row, so it must be the live model's)."""
    if sc is None:
        return True
    return bool((dev.updated_at and dev.updated_at > sc.created_at) or (newest and newest > sc.created_at)
                or (dev.last_seen_at and dev.last_seen_at >= recheck_after))


def _stamp(newest: datetime | None) -> datetime:
    """A created_at strictly after every other row for this key, even under clock skew."""
    now = utcnow()
    return max(now, newest + timedelta(microseconds=1)) if newest else now


def _upsert_score(s: Session, dev: Development, model: ScoringModel, feats: dict[str, float],
                  existing: Score | None, newest: datetime | None, force: bool) -> str:
    p = round(model.predict(feats), 6)
    if existing is None:
        s.add(Score(development_id=dev.id, audience=model.audience, p_material=p, features=feats,
                    contributions=model.explain(feats), model_version=model.version, created_at=_stamp(newest)))
        return "new"
    if force or existing.features != feats or existing.p_material != p:
        existing.p_material, existing.features = p, feats
        existing.contributions, existing.created_at = model.explain(feats), _stamp(newest)
        return "updated"
    if (dev.updated_at and dev.updated_at > existing.created_at) or (newest and newest > existing.created_at):
        existing.created_at = _stamp(newest)  # unchanged, but keep it the newest row and stop re-checking it
    return "unchanged"


def run_scoring(limit: int | None = None, rescore: bool = False, models_dir: Path | None = None,
                **_: object) -> dict:
    """Pipeline stage: score developments lacking a score for each audience's live model version and
    refresh stale ones. Idempotent: one row per (development, audience, version), no writes if unchanged.

    An audience whose deployed model file is unreadable is not scored at all (its newest rows stay the
    last good model's); the other audience is scored and committed, then the ModelFileError is raised
    so the stage is recorded as failed (job_run, and the health check's job_failed event)."""
    models: dict[str, ScoringModel] = {}
    halted: list[ModelFileError] = []
    for aud in AUDIENCES:
        try:
            models[aud] = load_model(aud, models_dir)
        except ModelFileError as exc:
            halted.append(exc)
    stats = {aud: {"version": m.version, "validated": m.validated, "new": 0, "updated": 0, "unchanged": 0}
             for aud, m in models.items()}
    checked = _score_developments(models, stats, limit, rescore) if models else 0
    log.info("scoring: %d developments checked (%s)", checked,
             ", ".join(f"{a} {v['version']}" for a, v in stats.items()))
    if halted:
        for exc in halted[1:]:
            log.error("also halted: %s", exc)
        raise halted[0]
    return {"developments": checked, **stats}


def _score_developments(models: Mapping[str, ScoringModel], stats: dict, limit: int | None, rescore: bool) -> int:
    recheck_after = utcnow() - timedelta(days=RECHECK_DAYS)
    with session_scope() as s:
        current: dict[tuple[int, str], Score] = {}
        for aud, m in models.items():
            for sc in s.scalars(select(Score).where(Score.audience == aud, Score.model_version == m.version)):
                current[(sc.development_id, aud)] = sc
        newest = {(d, a): t for d, a, t in s.execute(
            select(Score.development_id, Score.audience, func.max(Score.created_at))
            .group_by(Score.development_id, Score.audience))}
        devs = s.scalars(select(Development).where(Development.status.not_in(SKIP_STATUSES))
                         .order_by(Development.first_seen_at.desc(), Development.id.desc())).all()
        todo = [d for d in devs if rescore or any(
            _stale(d, current.get((d.id, a)), newest.get((d.id, a)), recheck_after) for a in models)]
        todo = todo[:limit] if limit else todo
        ctx = FeatureContext.load(s)
        for dev in todo:
            feats = development_features(s, dev, ctx)
            for aud, model in models.items():
                key = (dev.id, aud)
                stats[aud][_upsert_score(s, dev, model, feats, current.get(key), newest.get(key), rescore)] += 1
    return len(todo)


# --------------------------------------------------------------------------------------- metrics


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or math.isnan(float(x)) else round(float(x), nd)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Mann-Whitney ROC-AUC (ties count half); None when only one class is present."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci95(y: np.ndarray, s: np.ndarray, n_boot: int = 500, seed: int = 0) -> list[float] | None:
    """Bootstrap interval: with ~100 test items an AUC is only good to about +-0.1, so say so."""
    rng = np.random.default_rng(seed)
    vals = [a for idx in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
            if (a := roc_auc(y[idx], s[idx])) is not None]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float | None:
    """Expected precision of the top k under random tie-breaking (rule baselines are full of ties)."""
    k = min(k, len(y))
    if k == 0:
        return None
    threshold = np.sort(s)[::-1][k - 1]
    above, tied = s > threshold, s == threshold
    return float((y[above].sum() + (k - above.sum()) * y[tied].mean()) / k)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return [{"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int((idx == b).sum()),
             "mean_predicted": _r(p[idx == b].mean()) if (idx == b).any() else None,
             "observed_rate": _r(y[idx == b].mean()) if (idx == b).any() else None} for b in range(bins)]


def evaluate(y, s, y2=None, probabilistic: bool = True) -> dict:
    """Hold-out metrics for one scorer. y: label >= 1; y2: label == 2 (does it also rank alert-worthy items?)."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    both = 0 < y.sum() < len(y)
    out: dict = {"n": len(y), "positives": int(y.sum()), "roc_auc": _r(roc_auc(y, s)),
                 "auc_ci95": auc_ci95(y, s) if both else None,
                 "average_precision": _r(average_precision_score(y, s)) if both else None,
                 "precision_at_10": _r(precision_at_k(y, s, 10)), "precision_at_20": _r(precision_at_k(y, s, 20))}
    if probabilistic:
        out["brier"] = _r(brier_score_loss(y, s, labels=[0, 1]))
        out["calibration"] = calibration_table(y, s)
    if y2 is not None:
        y2 = np.asarray(y2, int)
        out["material_n"], out["roc_auc_material"] = int(y2.sum()), _r(roc_auc(y2, s))
    return out


# --------------------------------------------------------------------------------------- training


def _pipeline(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def choose_c(X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, dict]]:
    """C by stratified k-fold CV inside the training window: every C within CV_AUC_TOLERANCE of the
    best ROC-AUC counts as equally good (finer differences are fold noise), and the best log-loss among
    those wins, so calibration decides between equally good rankers (ties: the stronger penalty).
    A band rather than rounding to 0.01: rounding called 0.9144 and 0.9194 different but 0.9099 and
    0.9144 equal, so C (and how extreme the probabilities are) flipped with tiny feature changes."""
    folds = int(min(5, np.bincount(y, minlength=2).min()))
    if folds < 2:
        return 1.0, {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    res = {c: cross_validate(_pipeline(c), X, y, cv=cv, scoring={"auc": "roc_auc", "ll": "neg_log_loss"})
           for c in C_GRID}
    table = {c: {"roc_auc": round(float(r["test_auc"].mean()), 4), "log_loss": round(-float(r["test_ll"].mean()), 4)}
             for c, r in res.items()}
    top = max(v["roc_auc"] for v in table.values())
    near = [c for c in C_GRID if table[c]["roc_auc"] >= top - CV_AUC_TOLERANCE - 1e-9]
    best = min(near, key=lambda c: (table[c]["log_loss"], c))
    return best, {str(c): v for c, v in table.items()}


def fit_model(audience: str, X: np.ndarray, y: np.ndarray, C: float, names: list[str] | None = None,
              version: str = "holdout") -> ScoringModel:
    pipe = _pipeline(C).fit(X, y)
    scaler, lr = pipe[0], pipe[-1]
    correction = math.log(int(y.sum()) / (len(y) - int(y.sum())))  # undo the balanced-weight prior shift
    return ScoringModel(audience=audience, version=version, feature_names=list(names or FEATURE_NAMES),
                        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=lr.coef_[0].tolist(),
                        intercept=float(lr.intercept_[0]) + correction,
                        meta={"C": C, "intercept_correction": round(correction, 6)})


@dataclass
class LabeledSet:
    X: np.ndarray
    y: np.ndarray
    y2: np.ndarray
    sources: list[str]
    first_seen: list


def labeled_set(session: Session, audience: str, ctx: FeatureContext, cache: dict[int, dict]) -> LabeledSet:
    """Labelled developments in first_seen order (time split), features computed once per development."""
    from fli.analyze.labeling import training_labels  # labeling imports this module

    rows = sorted(training_labels(session, audience), key=lambda r: (r.dev.first_seen_at, r.dev.id))
    for r in rows:
        if r.dev.id not in cache:
            cache[r.dev.id] = development_features(session, r.dev, ctx)
    X = np.array([[cache[r.dev.id][n] for n in FEATURE_NAMES] for r in rows], float).reshape(len(rows), -1)
    return LabeledSet(X, np.array([r.y for r in rows], int), np.array([r.y2 for r in rows], int),
                      [r.source for r in rows], [r.dev.first_seen_at for r in rows])


def _problem(d: LabeledSet, n_train: int, min_labels: int) -> str | None:
    if len(d.y) < min_labels:
        return f"{len(d.y)} labels < {min_labels} required"
    counts = np.bincount(d.y[:n_train], minlength=2)
    if counts.min() < MIN_PER_CLASS:
        return f"training window has {counts[0]} negatives / {counts[1]} positives (< {MIN_PER_CLASS})"
    if not 0 < d.y[n_train:].sum() < len(d.y) - n_train:
        return "test window has a single class; cannot validate"
    return None


def train_audience(audience: str, d: LabeledSet, today, min_labels: int = MIN_LABELS,
                   force: bool = False) -> tuple[dict, ScoringModel | None]:
    """Hold-out evaluation against baselines, prior and ablation; returns (report, model to deploy or None)."""
    n = len(d.y)
    n_train = round(TRAIN_FRACTION * n)
    report: dict = {"n_labels": n, "label_sources": dict(Counter(d.sources)), "positives": int(d.y.sum()),
                    "material": int(d.y2.sum())}
    if problem := _problem(d, n_train, min_labels):
        return {**report, "status": "skipped", "reason": problem, "deployed": False}, None
    Xtr, ytr, Xte, yte, y2te = d.X[:n_train], d.y[:n_train], d.X[n_train:], d.y[n_train:], d.y2[n_train:]
    C, cv = choose_c(Xtr, ytr)
    holdout = fit_model(audience, Xtr, ytr, C)
    p = holdout.predict_matrix(Xte)
    metrics = evaluate(yte, p, y2te)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    z = np.log(pc / (1 - pc)) - holdout.meta["intercept_correction"]
    metrics["brier_without_intercept_correction"] = _r(brier_score_loss(yte, 1 / (1 + np.exp(-z)), labels=[0, 1]))
    col = FEATURE_NAMES.index
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in LLM_DERIVED]
    c_abl, _ = choose_c(Xtr[:, keep], ytr)
    ablation = fit_model(audience, Xtr[:, keep], ytr, c_abl, [FEATURE_NAMES[i] for i in keep])
    baselines = {
        "official_only": evaluate(yte, Xte[:, col("has_official")], y2te, probabilistic=False),
        "magnitude_only": evaluate(yte, Xte[:, col("max_magnitude")], y2te, probabilistic=False),
        "n_sources_only": evaluate(yte, Xte[:, col("log_sources")], y2te, probabilistic=False),
        PRIOR_VERSION: evaluate(yte, prior_model(audience).predict_matrix(Xte), y2te),
    }
    source = "human" if set(d.sources) == {"human"} else "llm" if set(d.sources) == {"llm"} else "mixed"
    final = fit_model(audience, d.X, d.y, C, version=f"lr-{audience}-{today:%Y%m%d}-{source}")
    auc_m, auc_p = metrics["roc_auc"], baselines[PRIOR_VERSION]["roc_auc"]
    deployed = force or auc_p is None or auc_m >= auc_p - MAX_AUC_REGRESSION
    split = {"train": n_train, "test": n - n_train, "test_from": str(d.first_seen[n_train]),
             "train_positives": int(ytr.sum()), "test_positives": int(yte.sum())}
    final.validated = True
    final.meta.update({"kind": "logistic_regression", "trained_at": today.isoformat(timespec="seconds"),
                       "label_source": source, "label_counts": report["label_sources"], "split": split,
                       "metrics": {k: v for k, v in metrics.items() if k != "calibration"},
                       "note": "coefficients refit on all labels after the time-split hold-out evaluation"})
    report.update({
        "status": "trained", "version": final.version, "label_source": source, "split": split, "C": C,
        "cv_by_C": cv, "model": metrics, "baselines": baselines,
        "ablation_without_llm_features": {"dropped": list(LLM_DERIVED), "C": c_abl,
                                          **evaluate(yte, ablation.predict_matrix(Xte[:, keep]), y2te)},
        "coefficients_per_sd": sorted(({"feature": nm, "label": FEATURE_LABELS.get(nm, nm), "coef": round(c, 4)}
                                       for nm, c in zip(final.feature_names, final.coef, strict=True)),
                                      key=lambda r: -abs(r["coef"])),
        "deployed": deployed,
        "deploy_reason": (f"hold-out AUC {auc_m} vs {PRIOR_VERSION} {auc_p}"
                          + ("" if deployed else f": worse by > {MAX_AUC_REGRESSION}, keeping the current model")),
    })
    return report, final if deployed else None


def write_atomic(path: Path, text: str) -> Path:
    """Write a file so readers see the old content or the new, never a partial one: a temp file in the
    same directory, flushed to disk, then os.replace (atomic on POSIX and Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:  # mkstemp creates 0600: keep the old file's mode (else 0644), so e.g. a web worker can still read it
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o644
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def save_model(model: ScoringModel, models_dir: Path | None = None) -> Path:
    """Deploy a model file atomically (see write_atomic): a crash or a concurrent reader never sees a
    half-written model, which load_model would refuse."""
    return write_atomic(model_path(model.audience, models_dir),
                        json.dumps(model.to_dict(), indent=2, default=str) + "\n")


def train_models(models_dir: Path | None = None, eval_dir: Path | None = None, min_labels: int = MIN_LABELS,
                 force: bool = False, **_: object) -> dict:
    """`fli train`: per audience fit, validate and deploy (unless it regresses vs the prior), write
    data/eval/scoring_metrics.json, then re-score every development with the live models."""
    today = utcnow()
    report: dict = {"generated_at": today.isoformat(timespec="seconds"),
                    "target": "label >= 1 (worth knowing or material); label == 2 reported as roc_auc_material",
                    "label_policy": "per development: human labels (majority) preferred over the LLM judge",
                    "split": f"time-ordered: first {TRAIN_FRACTION:.0%} by first_seen_at train, rest test",
                    "audiences": {}}
    with session_scope() as s:
        ctx, cache = FeatureContext.load(s), {}
        for aud in AUDIENCES:
            aud_report, model = train_audience(aud, labeled_set(s, aud, ctx, cache), today, min_labels, force)
            if model is not None:
                aud_report["model_file"] = str(save_model(model, models_dir))
            report["audiences"][aud] = aud_report
    out = write_atomic(Path(eval_dir or EVAL_DIR) / "scoring_metrics.json", json.dumps(report, indent=2, default=str) + "\n")
    summary = {aud: {"status": r["status"], "n_labels": r["n_labels"], "deployed": r["deployed"],
                     "version": r.get("version"), "reason": r.get("reason") or r.get("deploy_reason"),
                     "test_roc_auc": r.get("model", {}).get("roc_auc"),
                     "prior_roc_auc": r.get("baselines", {}).get(PRIOR_VERSION, {}).get("roc_auc")}
               for aud, r in report["audiences"].items()}
    return {"metrics_file": str(out), **summary, "rescored": run_scoring(rescore=True, models_dir=models_dir)}
