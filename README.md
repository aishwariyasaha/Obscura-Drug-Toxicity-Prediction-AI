# 🧬 Obscura — Drug Toxicity Prediction AI

> **CodeCure AI Hackathon · Track A · Drug Toxicity Prediction (Pharmacology + AI)**

A production-grade ML pipeline for predicting drug toxicity across all **12 Tox21 assay targets** using an ensemble of LightGBM, XGBoost, and Random Forest, with SHAP explainability and a dynamic Streamlit interface.

---

## 🏆 Results vs Published Baselines

| Assay | **Our AUC** | MolNet Baseline | Δ |
|-------|:-----------:|:---------------:|:---:|
| NR-AR | **0.779** | 0.745 | +0.034 |
| NR-AR-LBD | **0.834** | 0.811 | +0.023 |
| NR-AhR | **0.814** | 0.803 | +0.011 |
| NR-Aromatase | **0.817** | 0.791 | +0.026 |
| NR-ER | **0.743** | 0.728 | +0.015 |
| NR-ER-LBD | **0.807** | 0.791 | +0.016 |
| NR-PPAR-gamma | **0.776** | 0.726 | +0.050 |
| SR-ARE | **0.801** | 0.746 | +0.055 |
| SR-ATAD5 | **0.800** | 0.777 | +0.023 |
| SR-HSE | **0.740** | 0.690 | +0.050 |
| SR-MMP | **0.882** | 0.836 | +0.046 |
| SR-p53 | **0.788** | 0.766 | +0.022 |
| **Mean** | **0.798** | 0.768 | **+0.031** |

**Aggregate:** ROC-AUC = 0.7716 · AUPR = 0.6927 · MCC = 0.4020 · Sensitivity = 0.6125

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python run.py          # train (skip if models/ already exists)
streamlit run app.py   # launch interface
```

---

## 🗂 Pipeline

```
SMILES → Standardization → Feature Engineering (~7K features)
  [ECFP4/6 + FCFP4 + MACCS + RDKit FP + 200 descriptors + ADMET flags]
  → LightGBM + XGBoost + RF → AUPR-weighted Ensemble
  → MCC-optimal threshold (0.43) → Per-assay models (×12) → SHAP
```

---

## 📊 Key Improvements Over Baseline

- **Features**: 3 descriptors → ~7,000 (ECFP4/6, MACCS, RDKit FP, 200 descriptors)
- **Class imbalance**: Unhandled → sensitivity_boost=3.5, Platt calibration
- **Evaluation**: AUC only → AUC, AUPR, MCC, Sensitivity, Specificity, Brier
- **Ensemble**: Hardcoded → AUPR-optimized weights on validation set
- **Threshold**: Fixed 0.5 → MCC-optimal 0.43
- **Per-assay**: None → 12 dedicated classifiers with per-assay thresholds
- **Explainability**: None → SHAP TreeExplainer, grouped by feature type

---

## 📦 Datasets

- **Primary**: [Tox21](https://www.kaggle.com/datasets/epicskills/tox21-dataset) (~7,800 compounds, 12 assays)
- **Secondary**: [ZINC250k](https://www.kaggle.com/datasets/basu369victor/zinc250k)
- **Optional**: [ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/downloads)

---

> ⚠ For research purposes only. Covers Tox21 assay-based toxicity (nuclear receptors + stress pathways). Does not cover chronic toxicity, carcinogenicity, or bioaccumulation.
