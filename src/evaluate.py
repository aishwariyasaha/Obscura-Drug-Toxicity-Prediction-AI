"""
evaluate.py — Comprehensive model evaluation for drug toxicity prediction.

Industry-standard metrics beyond simple ROC-AUC:
  - ROC-AUC (discrimination)
  - Average Precision / AUPR (precision-recall, better for imbalanced)
  - MCC (Matthews Correlation Coefficient — best single metric for imbalance)
  - F1, Precision, Recall at optimal threshold
  - Balanced Accuracy
  - Specificity (critical in drug safety — false negatives are dangerous)
  - Calibration metrics (Brier score, ECE)
  - Bootstrap confidence intervals
  - Per-assay multi-label metrics
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_curve, precision_recall_curve,
    brier_score_loss, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

TOX21_TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimization
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """Find the probability threshold that maximizes the chosen metric."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score, best_thresh = 0, 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Full metric suite
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_prob, threshold=None, label="") -> dict:
    """
    Compute the full industry-standard metric suite.
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)

    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_prob, metric="mcc")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / max(tp + fn, 1)   # recall / true positive rate
    specificity = tn / max(tn + fp, 1)   # true negative rate

    metrics = {
        "roc_auc":          roc_auc_score(y_true, y_prob),
        "avg_precision":    average_precision_score(y_true, y_prob),   # AUPR
        "mcc":              matthews_corrcoef(y_true, y_pred),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "precision":        precision_score(y_true, y_pred, zero_division=0),
        "recall":           recall_score(y_true, y_pred, zero_division=0),
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "balanced_acc":     balanced_accuracy_score(y_true, y_pred),
        "brier_score":      brier_score_loss(y_true, y_prob),
        "threshold":        threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_pos": int(y_true.sum()), "n_neg": int((1 - y_true).sum()),
    }

    if label:
        logger.info(f"\n{'='*50}\nMetrics [{label}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k:20s}: {v:.4f}")

    return metrics


def bootstrap_confidence_intervals(y_true, y_prob, metric_fn, n_bootstrap=1000, ci=0.95):
    """Bootstrap 95% CI for any metric function."""
    n = len(y_true)
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except Exception:
            pass
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return np.percentile(scores, [alpha * 100, (1 - alpha) * 100])


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(models: dict, X_test, y_test, output_path: str = None):
    """
    Plot ROC and PR curves for multiple models on the same axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Comparison: ROC and Precision-Recall Curves", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=2)

    # ROC panel
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR panel
    baseline = y_test.mean()
    axes[1].axhline(baseline, color="k", ls="--", lw=1, label=f"Random (AP={baseline:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC/PR plot saved to {output_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_prob, threshold=None, model_name="", output_path=None):
    """Confusion matrix with normalized display."""
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_prob, metric="mcc")
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Normalized"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=["Non-Toxic", "Toxic"],
                    yticklabels=["Non-Toxic", "Toxic"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{model_name} — Confusion Matrix ({title})")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_probability_distribution(y_true, y_prob, model_name="", output_path=None):
    """Probability distribution by true class — good for calibration inspection."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, label="Non-toxic", color="steelblue", density=True)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, label="Toxic", color="crimson", density=True)
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold=0.5")
    ax.set_xlabel("Predicted Toxicity Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Score Distribution by Class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_assay_performance(df_labels, model, imputer, nonzero_mask, feature_names,
                                output_path=None):
    """
    Per-assay AUC bar chart across all 12 Tox21 targets.
    Industry models report per-assay performance — single metric is misleading.
    """
    from src.features import transform_features
    import warnings

    results = {}
    for target in TOX21_TARGETS:
        mask = df_labels[target].notna()
        if mask.sum() < 50:
            continue
        sub_df = df_labels[mask].copy()
        X_sub = transform_features(sub_df, imputer, nonzero_mask)
        y_sub = sub_df[target].values.astype(int)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob = model.predict_proba(X_sub)[:, 1]
                auc = roc_auc_score(y_sub, prob)
                ap = average_precision_score(y_sub, prob)
                results[target] = {"auc": auc, "ap": ap, "n": int(mask.sum()),
                                   "pos_rate": float(y_sub.mean())}
        except Exception as e:
            logger.warning(f"Skipping {target}: {e}")

    if not results:
        return results

    df_res = pd.DataFrame(results).T.sort_values("auc", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_res.index, df_res["auc"], color=plt.cm.RdYlGn(df_res["auc"].values))
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Random")
    ax.axvline(0.8, color="green", ls="--", lw=1.5, alpha=0.6, label="Good (0.80)")

    for bar, (_, row) in zip(bars, df_res.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['auc']:.3f} (n={row['n']})", va="center", fontsize=9)

    ax.set_xlabel("ROC-AUC")
    ax.set_title("Per-Assay Toxicity Prediction Performance (Tox21)")
    ax.set_xlim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Per-assay plot saved to {output_path}")
    plt.close()
    return results


def save_metrics_report(metrics: dict, per_assay: dict = None, output_path: str = "outputs/metrics.txt"):
    """Save a comprehensive metrics report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DRUG TOXICITY PREDICTION — MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("[ AGGREGATE BINARY CLASSIFIER ]\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"  {k:25s}: {v:.4f}\n")
            else:
                f.write(f"  {k:25s}: {v}\n")

        if per_assay:
            f.write("\n[ PER-ASSAY ROC-AUC ]\n")
            for assay, vals in sorted(per_assay.items()):
                f.write(f"  {assay:20s}: AUC={vals['auc']:.4f}  AP={vals['ap']:.4f}  n={vals['n']}\n")

            aucs = [v["auc"] for v in per_assay.values()]
            f.write(f"\n  Mean AUC across assays: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")

    logger.info(f"Full metrics report saved to {output_path}")
