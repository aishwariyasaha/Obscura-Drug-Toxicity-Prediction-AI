"""
run.py — Main training pipeline for Drug Toxicity Prediction.

Pipeline:
  1. Load & standardize Tox21 data
  2. Build comprehensive feature matrix
  3. Create aggregate binary labels
  4. Stratified train/val/test split
  5. (Optional) Optuna HPO
  6. Train LightGBM, XGBoost, Random Forest (with 2x sensitivity boost)
  7. Optimize ensemble weights on validation AUPR
  8. Train per-assay models (one per Tox21 target, own class weight)
  9. Evaluate on held-out test set
  10. SHAP explainability
  11. Save all artifacts

Usage:
    python run.py
    python run.py --optuna
    python run.py --label majority
    python run.py --no-rf
"""

import argparse
import logging
import numpy as np
import joblib
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

from src.preprocessing import (
    load_tox21, make_aggregate_label, stratified_split, TOX21_TARGETS
)
from src.features import build_feature_matrix
from src.train import (
    train_lgbm, train_xgb, train_rf,
    optimize_ensemble_weights, weighted_ensemble,
    compute_scale_pos_weight, save_model, optuna_lgbm_search,
    train_per_assay_models,
)
from sklearn.model_selection import train_test_split
from src.evaluate import (
    compute_all_metrics, plot_roc_pr_curves, plot_confusion_matrix,
    plot_probability_distribution, plot_per_assay_performance,
    save_metrics_report, find_optimal_threshold,
)
from src.explain import generate_all_shap_plots

# DELETE the EnsembleModel class from run.py and replace with:
from src.ensemble import EnsembleModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/raw/tox21.csv")
    parser.add_argument("--label",      default="any",
                        choices=["any", "majority", "sr_only"])
    parser.add_argument("--optuna",     action="store_true")
    parser.add_argument("--n-trials",   type=int, default=50)
    parser.add_argument("--no-rf",      action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir",  default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Step 1: Loading and standardizing Tox21 dataset")
    df = load_tox21(args.data)

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    logger.info("Step 2: Building comprehensive feature matrix")
    X, feature_names, imputer, nonzero_mask = build_feature_matrix(df)
    logger.info(f"Feature matrix shape: {X.shape}")

    # ── 3. Create labels ──────────────────────────────────────────────────────
    logger.info(f"Step 3: Creating binary labels (strategy='{args.label}')")
    y = make_aggregate_label(df, strategy=args.label)

    # ── 4. Three-way split ────────────────────────────────────────────────────
    logger.info("Step 4: Stratified train/val/test split (70/15/15)")
    idx = np.arange(len(df))
    idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
    val_frac = 0.15 / (1 - 0.15)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_frac, random_state=42,
                                           stratify=y[idx_temp])

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    df_train_sub = df.iloc[idx_train].reset_index(drop=True)
    df_val_sub   = df.iloc[idx_val].reset_index(drop=True)
    logger.info(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    spw = compute_scale_pos_weight(y_train, sensitivity_boost=3.5)

    # ── 5. Optional Optuna HPO ────────────────────────────────────────────────
    lgbm_params = None
    if args.optuna:
        logger.info(f"Step 5: Running Optuna HPO ({args.n_trials} trials)...")
        lgbm_params = optuna_lgbm_search(X_train, y_train, X_val, y_val,
                                          n_trials=args.n_trials)

    # ── 6. Train models ───────────────────────────────────────────────────────
    logger.info("Step 6: Training models (sensitivity_boost=2.0)")

    model_lgb, metrics_lgb = train_lgbm(
        X_train, y_train, X_val, y_val,
        calibrate=True, params=lgbm_params, sensitivity_boost=3.5
    )
    save_model(model_lgb, f"{args.model_dir}/lgbm_model.pkl",
               metadata={**metrics_lgb, "type": "LightGBM (Platt-calibrated)",
                         "label_strategy": args.label})

    model_xgb, metrics_xgb = train_xgb(
        X_train, y_train, X_val, y_val, sensitivity_boost=3.5
    )
    save_model(model_xgb, f"{args.model_dir}/xgb_model.pkl",
               metadata={**metrics_xgb, "type": "XGBoost"})

    models      = [model_lgb, model_xgb]
    model_names = ["LightGBM", "XGBoost"]

    if not args.no_rf:
        model_rf, metrics_rf = train_rf(
            X_train, y_train, X_val, y_val, sensitivity_boost=3.5
        )
        save_model(model_rf, f"{args.model_dir}/rf_model.pkl",
                   metadata={**metrics_rf, "type": "RandomForest"})
        models.append(model_rf)
        model_names.append("RandomForest")

    # ── 7. Ensemble weights (AUPR objective) ──────────────────────────────────
    logger.info("Step 7: Optimizing ensemble weights on validation AUPR")
    best_weights        = optimize_ensemble_weights(models, X_val, y_val)
    ensemble_pred_val   = weighted_ensemble(models, best_weights, X_val)
    ensemble_pred_test  = weighted_ensemble(models, best_weights, X_test)

    # Threshold: f1 on val (best for imbalanced)
    opt_threshold, _ = find_optimal_threshold(y_val, ensemble_pred_val, metric="f1")
    logger.info(f"Optimal classification threshold: {opt_threshold:.3f}")

    # ── 8. Per-assay models ───────────────────────────────────────────────────
    logger.info("Step 8: Training per-assay models...")
    per_assay_models = train_per_assay_models(
        df_train_sub, df_val_sub, imputer, nonzero_mask,
        TOX21_TARGETS, sensitivity_boost=2.5
    )
    logger.info(f"Trained {len(per_assay_models)} per-assay models")
    joblib.dump(per_assay_models, f"{args.model_dir}/per_assay_models.pkl")

    # ── Save ensemble ─────────────────────────────────────────────────────────
    # EnsembleModel is defined at module level (required for pickle)

    ensemble_model = EnsembleModel(models, best_weights, opt_threshold)
    save_model(ensemble_model, f"{args.model_dir}/ensemble_model.pkl",
               metadata={"weights": best_weights, "threshold": opt_threshold,
                         "label_strategy": args.label})

    # Save preprocessing artifacts
    joblib.dump(imputer,       f"{args.model_dir}/imputer.pkl")
    joblib.dump(nonzero_mask,  f"{args.model_dir}/nonzero_mask.pkl")
    joblib.dump(feature_names, f"{args.model_dir}/feature_names.pkl")
    with open(f"{args.model_dir}/threshold.json", "w") as f:
        json.dump({"optimal_threshold": opt_threshold}, f)
    logger.info("All preprocessing artifacts saved")

    # ── 9. Evaluation ─────────────────────────────────────────────────────────
    logger.info("Step 9: Evaluating on held-out test set")
    metrics = compute_all_metrics(y_test, ensemble_pred_test,
                                   threshold=opt_threshold, label="Ensemble Test")

    all_metrics = {"Ensemble": metrics}
    model_dict  = {"Ensemble": ensemble_model}
    for name, m in zip(model_names, models):
        prob = m.predict_proba(X_test)[:, 1]
        all_metrics[name] = compute_all_metrics(y_test, prob, label=name)
        model_dict[name]  = m

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    logger.info("Step 10: Generating evaluation plots")
    plot_roc_pr_curves(model_dict, X_test, y_test,
                        output_path=f"{args.output_dir}/roc_pr_curves.png")
    plot_confusion_matrix(y_test, ensemble_pred_test, threshold=opt_threshold,
                           model_name="Ensemble",
                           output_path=f"{args.output_dir}/confusion_matrix.png")
    plot_probability_distribution(y_test, ensemble_pred_test,
                                   model_name="Ensemble",
                                   output_path=f"{args.output_dir}/score_distribution.png")

    logger.info("Per-assay performance analysis...")
    per_assay_results = plot_per_assay_performance(
        df, model_lgb, imputer, nonzero_mask, feature_names,
        output_path=f"{args.output_dir}/per_assay_auc.png"
    )

    # ── 11. SHAP ──────────────────────────────────────────────────────────────
    logger.info("Step 11: SHAP explainability analysis")
    def _unwrap_calibrated(m):
        if hasattr(m, "calibrated_classifiers_"):
            return m.calibrated_classifiers_[0].estimator
        return m
    raw_lgb = _unwrap_calibrated(model_lgb)
    try:
        shap_vals, explainer, group_importance, top_desc = generate_all_shap_plots(
            raw_lgb, X_test, feature_names, output_dir=args.output_dir
        )
    except Exception as e:
        logger.warning(f"SHAP analysis failed (non-critical): {e}")
        group_importance, top_desc = {}, []

    # ── 12. Save metrics report ───────────────────────────────────────────────
    logger.info("Step 12: Saving metrics report")
    save_metrics_report(metrics, per_assay=per_assay_results,
                         output_path=f"{args.output_dir}/metrics.txt")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test ROC-AUC:      {metrics['roc_auc']:.4f}")
    logger.info(f"Test AUPR:         {metrics['avg_precision']:.4f}")
    logger.info(f"Test MCC:          {metrics['mcc']:.4f}")
    logger.info(f"Test Sensitivity:  {metrics['sensitivity']:.4f}")
    logger.info(f"Test Specificity:  {metrics['specificity']:.4f}")
    logger.info(f"Threshold:         {opt_threshold:.3f}")
    if per_assay_results:
        aucs = [v["auc"] for v in per_assay_results.values()]
        logger.info(f"Mean Per-Assay AUC: {np.mean(aucs):.4f}")
    logger.info("=" * 60)
    logger.info(f"Outputs: {args.output_dir}/  |  Models: {args.model_dir}/")


if __name__ == "__main__":
    main()