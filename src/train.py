"""
train.py — Industry-grade model training for drug toxicity prediction.

Key design decisions for class imbalance (Tox21 has 4-16% positive rates):
  - scale_pos_weight boosted by 2x to aggressively recover sensitivity
  - Platt (sigmoid) calibration instead of isotonic: isotonic overfits on
    small val sets and collapses probabilities toward the prior, making the
    model "too scared" to predict toxic. Platt fits only 2 params, stable.
  - Per-assay LightGBM models with assay-specific class weights
  - Ensemble weights optimized on AUPR (not AUC-ROC) — AUPR penalises
    models that never predict toxic
"""

import numpy as np
import logging
import joblib
import json
import os
from datetime import datetime

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y: np.ndarray, sensitivity_boost: float = 2.0) -> float:
    """
    neg/pos ratio x sensitivity_boost.
    Default 2.0 doubles toxic-example weight so model actually predicts toxic.
    Without this boost, Tox21's severe imbalance causes the model to default
    to predicting everything safe.
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    ratio = (n_neg / max(n_pos, 1)) * sensitivity_boost
    logger.info(
        f"Class balance: neg/pos={n_neg/max(n_pos,1):.2f} "
        f"x boost={sensitivity_boost} = scale_pos_weight={ratio:.2f}"
    )
    return ratio


def cross_validate_model(model, X, y, cv=5, scoring="roc_auc"):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    logger.info(f"CV {scoring}: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_lgbm(scale_pos_weight: float = 1.0, params: dict = None) -> lgb.LGBMClassifier:
    default_params = dict(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=10,   # lowered from 20 so minority-class leaves form
        min_child_weight=1e-3,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.05,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
        class_weight=None,
    )
    if params:
        default_params.update(params)
    return lgb.LGBMClassifier(**default_params)


def build_xgb(scale_pos_weight: float = 1.0, params: dict = None) -> XGBClassifier:
    default_params = dict(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=50,
        missing=np.nan,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if params:
        default_params.update(params)
    return XGBClassifier(**default_params)


def build_rf(scale_pos_weight: float = 1.0) -> RandomForestClassifier:
    class_weight = {0: 1.0, 1: float(scale_pos_weight)}
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(model, X_val, y_val, method="sigmoid"):
    """
    Platt (sigmoid) calibration on pre-fitted model.

    WHY NOT ISOTONIC:
    Isotonic regression is non-parametric and can perfectly memorise the val
    set. On small val sets (<5000 samples) this collapses ALL predicted
    probabilities toward the empirical class prior -- the model becomes
    "too scared" and outputs tiny scores like 0.03 for toxic compounds.
    Sigmoid fits only 2 parameters (a*score + b inside logistic) and is
    robust on small sets. Use isotonic only when val set > 10k samples.

    sklearn >= 1.2: cv=None means estimator is already fitted (was cv='prefit').
    """
    import warnings
    cal = CalibratedClassifierCV(model, cv=None, method=method)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        cal.fit(X_val, y_val)
    return cal


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_lgbm(
    X_train, y_train, X_val, y_val,
    scale_pos_weight: float = None,
    calibrate: bool = True,
    params: dict = None,
    sensitivity_boost: float = 2.0,
):
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train, sensitivity_boost)

    model = build_lgbm(scale_pos_weight=scale_pos_weight, params=params)

    logger.info("Training LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    best_iter = model.best_iteration_
    logger.info(f"LightGBM best iteration: {best_iter}")

    val_prob = model.predict_proba(X_val)[:, 1]
    val_auc  = roc_auc_score(y_val, val_prob)
    val_ap   = average_precision_score(y_val, val_prob)
    logger.info(f"LightGBM (raw) val AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

    if calibrate:
        model = calibrate_model(model, X_val, y_val, method="sigmoid")
        cp = model.predict_proba(X_val)[:, 1]
        logger.info(
            f"LightGBM (calibrated) val AUC: {roc_auc_score(y_val, cp):.4f}, "
            f"AP: {average_precision_score(y_val, cp):.4f}"
        )

    return model, {"val_auc": val_auc, "val_ap": val_ap, "best_iter": best_iter}


def train_xgb(
    X_train, y_train, X_val, y_val,
    scale_pos_weight: float = None,
    params: dict = None,
    sensitivity_boost: float = 2.0,
):
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train, sensitivity_boost)

    model = build_xgb(scale_pos_weight=scale_pos_weight, params=params)

    logger.info("Training XGBoost...")
    # XGBoost QuantileDMatrix casts to float32 internally; huge float64 values
    # (e.g. RDKit Ipc ~1e300) overflow to inf after that cast. Fix: explicit cast.
    _F32_MAX = np.finfo(np.float32).max
    def _safe_f32(arr):
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, -_F32_MAX, _F32_MAX)
        return arr.astype(np.float32)
    X_tr32 = _safe_f32(X_train)
    X_va32 = _safe_f32(X_val)

    model.fit(X_tr32, y_train, eval_set=[(X_va32, y_val)], verbose=False)

    val_prob = model.predict_proba(X_va32)[:, 1]
    val_auc  = roc_auc_score(y_val, val_prob)
    val_ap   = average_precision_score(y_val, val_prob)
    logger.info(f"XGBoost val AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
    return model, {"val_auc": val_auc, "val_ap": val_ap}


def train_rf(X_train, y_train, X_val, y_val, scale_pos_weight: float = None,
             sensitivity_boost: float = 2.0):
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train, sensitivity_boost)
    model = build_rf(scale_pos_weight=scale_pos_weight)
    logger.info("Training Random Forest...")
    model.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    val_ap  = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    logger.info(f"Random Forest val AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
    return model, {"val_auc": val_auc, "val_ap": val_ap}


# ---------------------------------------------------------------------------
# Per-assay training
# ---------------------------------------------------------------------------

def train_per_assay_models(df_train, df_val, imputer, nonzero_mask, targets,
                            sensitivity_boost: float = 2.5):
    """
    Train one calibrated LightGBM per Tox21 assay.

    Each assay has a different positive rate (4-16%), so a single shared
    scale_pos_weight is always wrong for at least some assays.
    Per-assay models with per-assay class weights give dramatically better
    per-assay sensitivity (the key metric in drug safety).

    sensitivity_boost=2.5 (slightly higher than aggregate) because per-assay
    rates are lower and we need more aggressive upweighting.

    Returns: dict {assay_name: (calibrated_model, threshold)}
    """
    from src.features import transform_features
    from src.evaluate import find_optimal_threshold

    per_assay_models = {}
    X_train_all = transform_features(df_train, imputer, nonzero_mask)
    X_val_all   = transform_features(df_val,   imputer, nonzero_mask)

    logger.info(f"Training per-assay models for {len(targets)} targets...")
    for target in targets:
        train_mask = df_train[target].notna()
        val_mask   = df_val[target].notna()

        if train_mask.sum() < 50 or val_mask.sum() < 20:
            logger.warning(f"  {target}: skipped (insufficient data)")
            continue

        X_tr = X_train_all[train_mask.values]
        y_tr = df_train.loc[train_mask, target].values.astype(int)
        X_va = X_val_all[val_mask.values]
        y_va = df_val.loc[val_mask, target].values.astype(int)

        if y_tr.sum() < 5:
            logger.warning(f"  {target}: skipped (< 5 positives in train)")
            continue

        spw = compute_scale_pos_weight(y_tr, sensitivity_boost)
        model = build_lgbm(scale_pos_weight=spw)

        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=80, verbose=False),
                    lgb.log_evaluation(period=9999),
                ],
            )
            model = calibrate_model(model, X_va, y_va, method="sigmoid")

            val_prob  = model.predict_proba(X_va)[:, 1]
            auc       = roc_auc_score(y_va, val_prob)
            ap        = average_precision_score(y_va, val_prob)
            threshold, _ = find_optimal_threshold(y_va, val_prob, metric="mcc")

            logger.info(
                f"  {target:20s}: AUC={auc:.4f}, AP={ap:.4f}, "
                f"threshold={threshold:.3f}, pos_rate={y_tr.mean():.3f}"
            )
            per_assay_models[target] = (model, float(threshold))

        except Exception as e:
            logger.warning(f"  {target}: failed — {e}")

    return per_assay_models


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def weighted_ensemble(models: list, weights: list, X: np.ndarray) -> np.ndarray:
    assert len(models) == len(weights)
    weights = np.array(weights) / sum(weights)
    preds = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return (preds * weights).sum(axis=1)


def optimize_ensemble_weights(models: list, X_val: np.ndarray, y_val: np.ndarray) -> list:
    """
    Grid search ensemble weights maximising AUPR (not AUC-ROC).
    AUPR penalises models that never predict toxic — exactly what we want.
    """
    best_score  = 0
    best_weights = [1.0] * len(models)
    preds = [m.predict_proba(X_val)[:, 1] for m in models]

    step = 0.1
    grid = np.arange(0, 1 + step, step)

    for w1 in grid:
        for w2 in grid:
            if len(models) == 2:
                w = [w1, 1 - w1]
                combined = w[0] * preds[0] + w[1] * preds[1]
            elif len(models) == 3:
                w3 = max(0, 1 - w1 - w2)
                if abs(w1 + w2 + w3 - 1.0) > 1e-6:
                    continue
                w = [w1, w2, w3]
                combined = sum(wi * p for wi, p in zip(w, preds))
            else:
                break

            score = average_precision_score(y_val, combined)
            if score > best_score:
                best_score = score
                best_weights = w

    logger.info(
        f"Optimal ensemble weights: {[f'{w:.2f}' for w in best_weights]} "
        f"-> AUPR: {best_score:.4f}"
    )
    return best_weights


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def optuna_lgbm_search(X_train, y_train, X_val, y_val, n_trials=50):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed. pip install optuna")
        return {}

    spw = compute_scale_pos_weight(y_train, sensitivity_boost=2.0)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        m = build_lgbm(scale_pos_weight=spw, params=params)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best = study.best_params
    logger.info(f"Best Optuna params (AUPR={study.best_value:.4f}): {best}")
    return best


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str, metadata: dict = None):
    joblib.dump(model, path)
    if metadata:
        meta_path = path.replace(".pkl", "_metadata.json")
        metadata["saved_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)