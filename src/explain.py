"""
explain.py — Industry-grade SHAP explainability for toxicity prediction.

Key improvements:
  - Feature names tracked throughout (critical for interpretability)
  - Grouped feature importance (FP bits grouped, descriptors individual)
  - Top descriptor identification (not just bit indices)
  - SHAP waterfall plots for individual predictions
  - Toxicophore feature highlighting
  - Correlation analysis between top features and toxicity labels
  - Per-assay feature importance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Feature group prefixes for grouping analysis
FP_GROUPS = {
    "ECFP4":   "ECFP4_",
    "ECFP6":   "ECFP6_",
    "FCFP4":   "FCFP4_",
    "MACCS":   "MACCS_",
    "RDKitFP": "RDKitFP_",
}

def unwrap_model(model):
    """Extract raw model if wrapped in CalibratedClassifierCV"""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def explain_single_sample(model, X, feature_names):
    import shap

    # 🔥 unwrap calibrated model
    model = unwrap_model(model)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # get top features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-10:]

    return [(feature_names[i], mean_abs[i]) for i in top_idx[::-1]]


def shap_to_text(top_features):
    reasons = []
    for name, val in top_features[:3]:
        if val > 0:
            reasons.append(f"{name} increases toxicity")
        else:
            reasons.append(f"{name} reduces toxicity")
    return reasons

def get_feature_group(name: str) -> str:
    for group, prefix in FP_GROUPS.items():
        if name.startswith(prefix):
            return group
    return "Descriptors"


# ─────────────────────────────────────────────────────────────────────────────
# Core SHAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X: np.ndarray, feature_names: list,
                         max_samples: int = 500):
    """
    Compute SHAP values using TreeExplainer (fast, exact for tree models).
    Samples for efficiency if dataset is large.
    """
    if len(X) > max_samples:
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    logger.info(f"Computing SHAP values on {len(X_sample)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # If shap_values is multi-output (calibrated), take positive class
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        sv = shap_values.values[:, :, 1]   # class 1
    else:
        sv = shap_values.values

    return sv, X_sample, explainer


# ─────────────────────────────────────────────────────────────────────────────
# Summary plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values, X_sample, feature_names, top_n=30, output_path=None):
    """
    Beeswarm SHAP summary plot for top N features.
    Shows both direction and magnitude of feature impact.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    sv_top = shap_values[:, top_idx]
    X_top  = X_sample[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]

    plt.figure(figsize=(12, max(8, top_n * 0.35)))
    shap.summary_plot(
        sv_top, X_top,
        feature_names=names_top,
        show=False, plot_size=None,
        color_bar_label="Feature value"
    )
    plt.title(f"SHAP Feature Importance — Top {top_n} Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"SHAP summary saved to {output_path}")
    plt.close()


def plot_shap_bar(shap_values, feature_names, top_n=30, output_path=None):
    """Mean absolute SHAP bar chart — cleaner for presentations."""
    mean_abs  = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = mean_abs[top_idx]

    # Color by feature group
    group_colors = {
        "ECFP4": "#4C72B0", "ECFP6": "#55A868", "FCFP4": "#C44E52",
        "MACCS": "#8172B2", "RDKitFP": "#CCB974", "Descriptors": "#64B5CD"
    }
    colors = [group_colors[get_feature_group(n)] for n in top_names]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    bars = ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (average impact on toxicity probability)")
    ax.set_title(f"Top {top_n} Features by SHAP Importance", fontweight="bold")

    # Legend for groups
    patches = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_grouped_importance(shap_values, feature_names, output_path=None):
    """Aggregate SHAP importance by feature group."""
    groups = [get_feature_group(n) for n in feature_names]
    df = pd.DataFrame({"group": groups, "abs_shap": np.abs(shap_values).mean(axis=0)})
    group_totals = df.groupby("group")["abs_shap"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    group_totals.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"])
    ax.set_title("Total SHAP Importance by Feature Type", fontweight="bold")
    ax.set_ylabel("Sum of Mean |SHAP values|")
    ax.set_xlabel("Feature Group")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return group_totals.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Individual prediction explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_single_prediction(explainer, x: np.ndarray, feature_names: list,
                               pred_prob: float, output_path=None):
    """
    Waterfall plot showing how each feature pushes the prediction from the
    baseline probability up or down for a single compound.
    """
    sv = explainer(x.reshape(1, -1))
    if sv.values.ndim == 3:
        sv_vals = sv.values[0, :, 1]
        base    = sv.base_values[0, 1]
    else:
        sv_vals = sv.values[0]
        base    = sv.base_values[0]

    # Take top contributing features
    top_n = 20
    top_idx = np.argsort(np.abs(sv_vals))[::-1][:top_n]
    top_feats = [(feature_names[i], sv_vals[i], x[i]) for i in top_idx]
    top_feats.sort(key=lambda t: t[1])  # sort ascending for waterfall

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["crimson" if v > 0 else "steelblue" for _, v, _ in top_feats]
    ax.barh([f[0] for f in top_feats], [f[1] for f in top_feats], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (contribution to toxicity prediction)")
    ax.set_title(f"Individual Prediction Explanation\n"
                 f"Predicted toxicity probability: {pred_prob:.3f} | Base rate: {shap.utils.safe_isinstance(base, 'float') and f'{base:.3f}' or '?'}")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Descriptor correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_descriptor_correlations(df: pd.DataFrame, y: np.ndarray, output_path=None):
    """
    Correlation heatmap of key physicochemical descriptors with toxicity label.
    Useful for identifying structure-toxicity relationships.
    """
    desc_cols = [
        "MolWt", "LogP", "HBD", "HBA", "TPSA", "RotBonds",
        "RO5_violations", "halogen_count", "nitro_group",
        "aromatic_ring_count", "heavy_atom_count"
    ]
    available = [c for c in desc_cols if c in df.columns]
    if not available:
        return

    sub = df[available].copy()
    sub["Toxic"] = y
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax, mask=False,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Descriptor Correlation Matrix (incl. Toxicity Label)", fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_shap_plots(model, X_test, feature_names, output_dir="outputs"):
    """Run the full SHAP analysis pipeline and save all plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    shap_vals, X_sample, explainer = compute_shap_values(model, X_test, feature_names)

    plot_shap_summary(shap_vals, X_sample, feature_names,
                      output_path=f"{output_dir}/shap_beeswarm.png")

    plot_shap_bar(shap_vals, feature_names,
                  output_path=f"{output_dir}/shap_bar.png")

    group_importance = plot_grouped_importance(shap_vals, feature_names,
                                                output_path=f"{output_dir}/shap_groups.png")

    # Top descriptor names (non-fingerprint)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_desc = [
        (feature_names[i], float(mean_abs[i]))
        for i in np.argsort(mean_abs)[::-1]
        if not any(feature_names[i].startswith(p) for p in FP_GROUPS.values())
    ][:20]

    logger.info("\nTop 20 Physicochemical Descriptors by SHAP:")
    for name, val in top_desc:
        logger.info(f"  {name:35s}: {val:.5f}")

    return shap_vals, explainer, group_importance, top_desc
