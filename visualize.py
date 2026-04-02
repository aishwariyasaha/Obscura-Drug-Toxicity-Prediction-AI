"""
visualize.py — Molecular property visualizations for drug toxicity prediction.

Generates:
  1. LogP vs Toxicity distribution
  2. Molecular Weight vs Toxicity distribution
  3. TPSA vs Toxicity scatter
  4. HBD/HBA vs Toxicity
  5. Property correlation heatmap with toxicity
  6. Toxicity probability distribution
  7. Structural alerts frequency chart

Usage:
    python visualize.py
    python visualize.py --data data/raw/tox21.csv --output outputs/
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Color palette ──────────────────────────────────────────────────────────────
TOXIC_COLOR    = "#E63946"   # red
NONTOXIC_COLOR = "#2A9D8F"   # teal
PALETTE        = [NONTOXIC_COLOR, TOXIC_COLOR]
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#2E3250",
    "axes.labelcolor":  "#C8D0E0",
    "xtick.color":      "#8890A8",
    "ytick.color":      "#8890A8",
    "text.color":       "#C8D0E0",
    "grid.color":       "#2E3250",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "figure.dpi":       130,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data_with_properties(tox21_path: str) -> pd.DataFrame:
    """Load Tox21 and compute RDKit properties for all molecules."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    logger.info(f"Loading {tox21_path}...")
    df = pd.read_csv(tox21_path)

    TOX21_TARGETS = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    # Parse molecules
    df["mol"] = df["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)))
    df = df[df["mol"].notna()].reset_index(drop=True)

    logger.info(f"Computing molecular properties for {len(df)} compounds...")

    def safe(fn, mol, default=np.nan):
        try:
            v = fn(mol)
            return float(v) if np.isfinite(float(v)) else default
        except Exception:
            return default

    df["MolWt"]       = df["mol"].apply(lambda m: safe(Descriptors.MolWt, m))
    df["LogP"]        = df["mol"].apply(lambda m: safe(Descriptors.MolLogP, m))
    df["TPSA"]        = df["mol"].apply(lambda m: safe(Descriptors.TPSA, m))
    df["HBD"]         = df["mol"].apply(lambda m: safe(Lipinski.NumHDonors, m))
    df["HBA"]         = df["mol"].apply(lambda m: safe(Lipinski.NumHAcceptors, m))
    df["RotBonds"]    = df["mol"].apply(lambda m: safe(Lipinski.NumRotatableBonds, m))
    df["AromaticRings"] = df["mol"].apply(lambda m: safe(rdMolDescriptors.CalcNumAromaticRings, m))
    df["HeavyAtoms"]  = df["mol"].apply(lambda m: float(m.GetNumHeavyAtoms()))
    df["Halogens"]    = df["mol"].apply(
        lambda m: float(sum(a.GetAtomicNum() in (9,17,35,53) for a in m.GetAtoms()))
    )
    df["HasNitro"]    = df["mol"].apply(
        lambda m: float(any(
            a.GetAtomicNum() == 7 and
            sum(n.GetAtomicNum() == 8 for n in a.GetNeighbors()) >= 2
            for a in m.GetAtoms()
        ))
    )

    # Aggregate toxicity label
    label_matrix = df[TOX21_TARGETS].apply(pd.to_numeric, errors="coerce").values
    y = np.nanmax(label_matrix, axis=1)
    df["Toxic"] = np.where(np.isnan(y), 0, y).astype(int)
    df["ToxicLabel"] = df["Toxic"].map({0: "Non-Toxic", 1: "Toxic"})

    logger.info(f"Toxic: {df['Toxic'].sum()} / {len(df)} ({df['Toxic'].mean():.1%})")
    return df


# ── Individual plots ───────────────────────────────────────────────────────────

def plot_logp_distribution(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LogP (Lipophilicity) vs Toxicity", fontsize=14, fontweight="bold", color="white", y=1.01)

    # KDE plot
    for label, color in [("Non-Toxic", NONTOXIC_COLOR), ("Toxic", TOXIC_COLOR)]:
        sub = df[df["ToxicLabel"] == label]["LogP"].dropna()
        sub = sub[(sub > -5) & (sub < 12)]
        axes[0].hist(sub, bins=50, alpha=0.55, color=color, label=label, density=True)
        sub.plot.kde(ax=axes[0], color=color, lw=2.5)

    axes[0].set_xlabel("LogP"); axes[0].set_ylabel("Density")
    axes[0].set_title("LogP Distribution by Toxicity Class")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].axvline(5, color="orange", ls="--", lw=1.5, label="Lipinski limit (5)")

    # Box plot
    plot_df = df[["LogP", "ToxicLabel"]].dropna()
    plot_df = plot_df[(plot_df["LogP"] > -5) & (plot_df["LogP"] < 12)]
    sns.boxplot(data=plot_df, x="ToxicLabel", y="LogP",
                palette={"Non-Toxic": NONTOXIC_COLOR, "Toxic": TOXIC_COLOR},
                ax=axes[1], width=0.5, linewidth=1.5)
    axes[1].set_xlabel(""); axes[1].set_ylabel("LogP")
    axes[1].set_title("LogP Spread by Class")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_mw_distribution(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Molecular Weight vs Toxicity", fontsize=14, fontweight="bold", color="white", y=1.01)

    for label, color in [("Non-Toxic", NONTOXIC_COLOR), ("Toxic", TOXIC_COLOR)]:
        sub = df[df["ToxicLabel"] == label]["MolWt"].dropna()
        sub = sub[(sub > 0) & (sub < 900)]
        axes[0].hist(sub, bins=50, alpha=0.55, color=color, label=label, density=True)

    axes[0].axvline(500, color="orange", ls="--", lw=1.5, label="Lipinski limit (500 Da)")
    axes[0].set_xlabel("Molecular Weight (Da)"); axes[0].set_ylabel("Density")
    axes[0].set_title("MW Distribution by Toxicity Class")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    plot_df = df[["MolWt", "ToxicLabel"]].dropna()
    plot_df = plot_df[(plot_df["MolWt"] > 0) & (plot_df["MolWt"] < 900)]
    sns.violinplot(data=plot_df, x="ToxicLabel", y="MolWt",
                   palette={"Non-Toxic": NONTOXIC_COLOR, "Toxic": TOXIC_COLOR},
                   ax=axes[1], inner="quartile", linewidth=1.2)
    axes[1].set_xlabel(""); axes[1].set_ylabel("Molecular Weight (Da)")
    axes[1].set_title("MW Distribution (Violin)")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_tpsa_scatter(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TPSA & Rotatable Bonds vs Toxicity", fontsize=14, fontweight="bold", color="white", y=1.01)

    # TPSA scatter vs LogP
    sample = df.dropna(subset=["TPSA", "LogP"]).sample(min(2000, len(df)), random_state=42)
    for label, color in [("Non-Toxic", NONTOXIC_COLOR), ("Toxic", TOXIC_COLOR)]:
        sub = sample[sample["ToxicLabel"] == label]
        axes[0].scatter(sub["LogP"], sub["TPSA"], alpha=0.35, s=12,
                        color=color, label=label)
    axes[0].axhline(140, color="orange", ls="--", lw=1.5, label="Veber limit (140 Å²)")
    axes[0].axvline(5,   color="yellow",  ls="--", lw=1.5, label="Lipinski LogP=5")
    axes[0].set_xlabel("LogP"); axes[0].set_ylabel("TPSA (Å²)")
    axes[0].set_title("Chemical Space: LogP vs TPSA")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    # TPSA distribution
    for label, color in [("Non-Toxic", NONTOXIC_COLOR), ("Toxic", TOXIC_COLOR)]:
        sub = df[df["ToxicLabel"] == label]["TPSA"].dropna()
        sub = sub[(sub >= 0) & (sub < 300)]
        axes[1].hist(sub, bins=40, alpha=0.55, color=color, label=label, density=True)
    axes[1].set_xlabel("TPSA (Å²)"); axes[1].set_ylabel("Density")
    axes[1].set_title("TPSA Distribution by Toxicity")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_hbond_analysis(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("H-Bond Donors & Acceptors vs Toxicity", fontsize=14, fontweight="bold", color="white", y=1.01)

    for ax, prop, limit, label_str in [
        (axes[0], "HBD", 5,  "H-Bond Donors (Lipinski ≤5)"),
        (axes[1], "HBA", 10, "H-Bond Acceptors (Lipinski ≤10)"),
    ]:
        plot_df = df[[prop, "ToxicLabel"]].dropna()
        plot_df = plot_df[plot_df[prop] <= 20]
        counts = plot_df.groupby([prop, "ToxicLabel"]).size().unstack(fill_value=0)
        counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100
        if "Non-Toxic" in counts_pct.columns and "Toxic" in counts_pct.columns:
            counts_pct[["Non-Toxic", "Toxic"]].plot(
                kind="bar", ax=ax,
                color=[NONTOXIC_COLOR, TOXIC_COLOR],
                alpha=0.85, width=0.7
            )
        ax.axvline(limit - 0.5, color="orange", ls="--", lw=1.5,
                   label=f"Lipinski limit ({limit})")
        ax.set_xlabel(label_str); ax.set_ylabel("% of compounds")
        ax.set_title(f"{prop} Distribution"); ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_property_correlation(df: pd.DataFrame, output_path: str):
    props = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
             "AromaticRings", "HeavyAtoms", "Halogens", "Toxic"]
    sub = df[props].dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.zeros_like(corr, dtype=bool)
    # Show full matrix but highlight Toxic column
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Molecular Property Correlation Matrix\n(incl. Toxicity Label)",
                 fontsize=13, fontweight="bold", pad=15)

    # Highlight Toxic row/col
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_structural_alerts(df: pd.DataFrame, output_path: str):
    """Bar chart of structural alert frequency in toxic vs non-toxic compounds."""
    alerts = {
        "Nitro Group":       "HasNitro",
        "Halogens ≥ 3":      None,
        "LogP > 5":          None,
        "MW > 500":          None,
        "HBD > 5":           None,
        "Aromatic Rings ≥ 4": None,
        "TPSA < 20":         None,
    }

    df2 = df.copy()
    df2["Halogens ≥ 3"]       = (df2["Halogens"] >= 3).astype(float)
    df2["LogP > 5"]            = (df2["LogP"] > 5).astype(float)
    df2["MW > 500"]            = (df2["MolWt"] > 500).astype(float)
    df2["HBD > 5"]             = (df2["HBD"] > 5).astype(float)
    df2["Aromatic Rings ≥ 4"]  = (df2["AromaticRings"] >= 4).astype(float)
    df2["TPSA < 20"]           = (df2["TPSA"] < 20).astype(float)
    df2["Nitro Group"]         = df2["HasNitro"]

    alert_cols = list(alerts.keys())
    results = {}
    for alert in alert_cols:
        toxic_rate    = df2[df2["Toxic"] == 1][alert].mean() * 100
        nontoxic_rate = df2[df2["Toxic"] == 0][alert].mean() * 100
        results[alert] = {"Toxic": toxic_rate, "Non-Toxic": nontoxic_rate}

    res_df = pd.DataFrame(results).T

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(res_df))
    width = 0.38
    ax.bar(x - width/2, res_df["Non-Toxic"], width, label="Non-Toxic",
           color=NONTOXIC_COLOR, alpha=0.85)
    ax.bar(x + width/2, res_df["Toxic"],     width, label="Toxic",
           color=TOXIC_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(res_df.index, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("% of compounds with alert")
    ax.set_title("Structural Alert Frequency: Toxic vs Non-Toxic Compounds",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate enrichment ratio
    for i, alert in enumerate(res_df.index):
        t = res_df.loc[alert, "Toxic"]
        nt = res_df.loc[alert, "Non-Toxic"]
        if nt > 0:
            ratio = t / nt
            ax.text(i, max(t, nt) + 1, f"{ratio:.1f}x", ha="center",
                    fontsize=8, color="white", alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_chemical_space_overview(df: pd.DataFrame, output_path: str):
    """MW vs LogP chemical space colored by toxicity — classic drug discovery plot."""
    fig, ax = plt.subplots(figsize=(11, 8))

    sample = df.dropna(subset=["MolWt", "LogP"])
    sample = sample[(sample["MolWt"] < 900) & (sample["LogP"].between(-5, 12))]
    sample = sample.sample(min(3000, len(sample)), random_state=42)

    for label, color, alpha, size in [
        ("Non-Toxic", NONTOXIC_COLOR, 0.3, 10),
        ("Toxic",     TOXIC_COLOR,    0.5, 14),
    ]:
        sub = sample[sample["ToxicLabel"] == label]
        ax.scatter(sub["MolWt"], sub["LogP"], c=color, alpha=alpha,
                   s=size, label=f"{label} (n={len(sub)})")

    # Lipinski box
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, -5), 500, 10, linewidth=1.5,
                     edgecolor="orange", facecolor="none",
                     linestyle="--", label="Lipinski space")
    ax.add_patch(rect)

    ax.set_xlabel("Molecular Weight (Da)", fontsize=11)
    ax.set_ylabel("LogP (Lipophilicity)", fontsize=11)
    ax.set_title("Chemical Space Overview: MW vs LogP\n(colored by toxicity)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved: {output_path}")


# ── Master runner ──────────────────────────────────────────────────────────────

def generate_all_visualizations(data_path: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = load_data_with_properties(data_path)

    plot_logp_distribution(df,       f"{output_dir}/viz_logp_toxicity.png")
    plot_mw_distribution(df,         f"{output_dir}/viz_mw_toxicity.png")
    plot_tpsa_scatter(df,            f"{output_dir}/viz_tpsa_scatter.png")
    plot_hbond_analysis(df,          f"{output_dir}/viz_hbond_toxicity.png")
    plot_property_correlation(df,    f"{output_dir}/viz_property_correlation.png")
    plot_structural_alerts(df,       f"{output_dir}/viz_structural_alerts.png")
    plot_chemical_space_overview(df, f"{output_dir}/viz_chemical_space.png")

    logger.info(f"\nAll visualizations saved to {output_dir}/")
    logger.info("Files generated:")
    for f in sorted(Path(output_dir).glob("viz_*.png")):
        logger.info(f"  {f.name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/raw/tox21.csv")
    parser.add_argument("--output", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_visualizations(args.data, args.output)