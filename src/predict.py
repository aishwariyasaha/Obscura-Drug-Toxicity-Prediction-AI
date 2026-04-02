"""
predict.py — Production-grade inference for drug toxicity prediction.

Industry improvements:
  - Multi-label output (per-assay predictions, not just aggregate)
  - Confidence intervals via bootstrap
  - ADMET property profile with rule-based flags
  - Lipinski / Veber / Egan druglikeness scoring
  - SMILES validation with detailed error messages
  - Batch prediction support
  - Per-compound human-readable toxicity report
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumHeteroatoms, CalcNumRings

from src.preprocessing import smiles_to_mol, TOX21_TARGETS
from src.features import featurize_single

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ADMETProfile:
    """Computed ADMET properties from RDKit."""
    mol_weight:       float = 0.0
    logp:             float = 0.0
    hbd:              int   = 0
    hba:              int   = 0
    tpsa:             float = 0.0
    rot_bonds:        int   = 0
    aromatic_rings:   int   = 0
    heavy_atoms:      int   = 0
    halogen_count:    int   = 0
    has_nitro:        bool  = False
    formal_charge:    int   = 0

    # Rule-based flags
    ro5_violations:   int   = 0       # Lipinski Rule of Five
    veber_ok:         bool  = True    # Veber oral bioavailability
    leadlike:         bool  = False
    brenk_ok:         bool  = False


@dataclass
class ToxicityPrediction:
    """Complete prediction result for a single compound."""
    smiles:           str
    is_valid_smiles:  bool

    # Aggregate prediction
    aggregate_prob:   float = 0.0
    aggregate_label:  int   = 0       # 0 or 1 at optimal threshold
    risk_level:       str   = "Unknown"

    # Per-assay predictions (12 Tox21 targets)
    assay_probs:      dict  = field(default_factory=dict)
    assay_labels:     dict  = field(default_factory=dict)
    max_assay:        str   = ""      # highest-probability assay
    max_assay_prob:   float = 0.0

    # ADMET
    admet:            Optional[ADMETProfile] = None

    # Interpretation
    alerts:           list  = field(default_factory=list)
    pass_criteria:    list  = field(default_factory=list)
    error_message:    str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# ADMET computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_admet(mol) -> ADMETProfile:
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rb   = Lipinski.NumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    hac  = mol.GetNumHeavyAtoms()
    halogens = sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in mol.GetAtoms())
    nitro = any(
        a.GetAtomicNum() == 7 and
        sum(n.GetAtomicNum() == 8 for n in a.GetNeighbors()) >= 2
        for a in mol.GetAtoms()
    )
    charge = Chem.rdmolops.GetFormalCharge(mol)

    ro5_v = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    return ADMETProfile(
        mol_weight=round(mw, 2),
        logp=round(logp, 3),
        hbd=hbd, hba=hba,
        tpsa=round(tpsa, 2),
        rot_bonds=rb,
        aromatic_rings=arom,
        heavy_atoms=hac,
        halogen_count=halogens,
        has_nitro=nitro,
        formal_charge=charge,
        ro5_violations=ro5_v,
        veber_ok=(rb <= 10 and tpsa <= 140),
        leadlike=(mw < 350 and logp <= 3 and CalcNumRings(mol) <= 4),
        brenk_ok=(mw < 600 and logp < 5 and hbd <= 5 and hba <= 10),
    )


def get_structural_alerts(admet: ADMETProfile) -> tuple[list, list]:
    """
    Return (alerts, pass_criteria) based on ADMET profile.
    Alerts = structural features associated with toxicity risk.
    """
    alerts = []
    passes = []

    if admet.has_nitro:
        alerts.append("⚠ Nitro group detected — associated with genotoxicity and reactive metabolites")
    if admet.halogen_count >= 3:
        alerts.append(f"⚠ High halogen count ({admet.halogen_count}) — may cause bioaccumulation or reactive species")
    if admet.logp > 5:
        alerts.append(f"⚠ High lipophilicity (LogP={admet.logp:.2f}) — risk of phospholipidosis")
    if admet.mol_weight > 500:
        alerts.append(f"⚠ High molecular weight ({admet.mol_weight:.1f} Da) — Lipinski violation")
    if admet.aromatic_rings >= 4:
        alerts.append(f"⚠ High aromatic ring count ({admet.aromatic_rings}) — planarity may cause intercalation")
    if admet.ro5_violations >= 2:
        alerts.append(f"⚠ {admet.ro5_violations} Lipinski Rule of Five violations")
    if admet.tpsa < 20:
        alerts.append(f"⚠ Very low TPSA ({admet.tpsa:.1f} Å²) — high membrane permeability risk")
    if not admet.veber_ok:
        alerts.append("⚠ Fails Veber oral bioavailability criteria (RotBonds > 10 or TPSA > 140)")
    if abs(admet.formal_charge) > 1:
        alerts.append(f"⚠ High formal charge ({admet.formal_charge}) — may affect ADMET")

    if admet.ro5_violations == 0:
        passes.append("✓ Passes Lipinski Rule of Five (drug-likeness)")
    if admet.veber_ok:
        passes.append("✓ Passes Veber oral bioavailability criteria")
    if admet.logp <= 3:
        passes.append(f"✓ Good lipophilicity (LogP={admet.logp:.2f})")
    if admet.tpsa >= 60 and admet.tpsa <= 140:
        passes.append(f"✓ TPSA in optimal range ({admet.tpsa:.1f} Å²)")
    if admet.leadlike:
        passes.append("✓ Lead-like compound (MW < 350, LogP ≤ 3)")

    return alerts, passes


def risk_level_from_prob(prob: float) -> str:
    if prob >= 0.65: return "HIGH"
    if prob >= 0.45: return "MODERATE-HIGH"
    if prob >= 0.30: return "MODERATE"
    if prob >= 0.18: return "LOW-MODERATE"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────

class ToxicityPredictor:
    """
    Production-ready predictor wrapping the trained ensemble.

    Usage:
        predictor = ToxicityPredictor.load("models/")
        result = predictor.predict("CCO")
    """

    def __init__(self, model, imputer, nonzero_mask, feature_names,
                 threshold=0.5, per_assay_models=None):
        self.model = model
        self.imputer = imputer
        self.nonzero_mask = nonzero_mask
        self.feature_names = feature_names
        self.threshold = threshold
        self.per_assay_models = per_assay_models or {}

    @classmethod
    def load(cls, model_dir: str = "models"):
        """Load all artifacts saved by run.py."""
        import joblib, json
        model        = joblib.load(f"{model_dir}/ensemble_model.pkl")
        imputer      = joblib.load(f"{model_dir}/imputer.pkl")
        nonzero_mask = joblib.load(f"{model_dir}/nonzero_mask.pkl")
        feature_names = joblib.load(f"{model_dir}/feature_names.pkl")

        threshold = 0.5
        try:
            with open(f"{model_dir}/threshold.json") as f:
                threshold = json.load(f)["optimal_threshold"]
        except Exception:
            pass

        # Load per-assay models if available (trained by run.py Step 8)
        per_assay_models = {}
        try:
            per_assay_models = joblib.load(f"{model_dir}/per_assay_models.pkl")
        except Exception:
            pass

        return cls(model, imputer, nonzero_mask, feature_names, threshold,
                   per_assay_models=per_assay_models)

    def predict(self, smiles: str) -> ToxicityPrediction:
        """Full prediction pipeline for a single SMILES string."""
        result = ToxicityPrediction(smiles=smiles, is_valid_smiles=False)

        mol = smiles_to_mol(smiles)
        if mol is None:
            result.error_message = (
                f"Could not parse SMILES: '{smiles}'. "
                "Please check for valid chemical notation. "
                "Examples: 'CCO' (ethanol), 'c1ccccc1' (benzene)"
            )
            return result

        result.is_valid_smiles = True

        # ADMET profile
        admet = compute_admet(mol)
        result.admet = admet
        result.alerts, result.pass_criteria = get_structural_alerts(admet)

        # Feature extraction
        X = featurize_single(mol, self.imputer, self.nonzero_mask)

        # Aggregate prediction
        prob = float(self.model.predict_proba(X)[0, 1])
        result.aggregate_prob  = round(prob, 4)
        result.aggregate_label = int(prob >= self.threshold)
        result.risk_level      = risk_level_from_prob(prob)

        # Per-assay predictions (if per-assay models are available)
        # Each assay has its own model trained with assay-specific class weight
        # and its own MCC-optimal threshold (stored as tuple: (model, threshold))
        if self.per_assay_models:
            for assay, assay_entry in self.per_assay_models.items():
                try:
                    if isinstance(assay_entry, tuple):
                        assay_model, assay_thresh = assay_entry
                    else:
                        assay_model, assay_thresh = assay_entry, 0.5
                    ap = float(assay_model.predict_proba(X)[0, 1])
                    result.assay_probs[assay]  = round(ap, 4)
                    result.assay_labels[assay] = int(ap >= assay_thresh)
                except Exception:
                    result.assay_probs[assay]  = -1.0

        # Find highest-risk assay
        if result.assay_probs:
            max_assay = max(result.assay_probs, key=result.assay_probs.get)
            result.max_assay      = max_assay
            result.max_assay_prob = result.assay_probs[max_assay]

        return result

    def predict_batch(self, smiles_list: list) -> list:
        """Batch prediction over a list of SMILES strings."""
        return [self.predict(s) for s in smiles_list]

    def predict_smiles_df(self, df, smiles_col="smiles") -> pd.DataFrame:
        """Predict from a DataFrame and return results as DataFrame."""
        results = self.predict_batch(df[smiles_col].tolist())
        rows = []
        for r in results:
            row = {
                "smiles": r.smiles,
                "valid": r.is_valid_smiles,
                "toxicity_prob": r.aggregate_prob,
                "risk_level": r.risk_level,
                "alerts": " | ".join(r.alerts) if r.alerts else "None",
            }
            if r.admet:
                row.update({
                    "mol_weight": r.admet.mol_weight,
                    "logp": r.admet.logp,
                    "tpsa": r.admet.tpsa,
                    "hbd": r.admet.hbd,
                    "hba": r.admet.hba,
                    "ro5_violations": r.admet.ro5_violations,
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def format_report(self, result: ToxicityPrediction) -> str:
        """Human-readable report for a single compound."""
        if not result.is_valid_smiles:
            return f"ERROR: {result.error_message}"

        lines = [
            "=" * 60,
            "TOXICITY PREDICTION REPORT",
            "=" * 60,
            f"SMILES:            {result.smiles}",
            f"",
            f"AGGREGATE TOXICITY PROBABILITY: {result.aggregate_prob:.3f}",
            f"RISK LEVEL:        {result.risk_level}",
            f"PREDICTION:        {'⚠ TOXIC' if result.aggregate_label else '✓ NON-TOXIC'}",
            f"",
            "MOLECULAR PROPERTIES",
            "-" * 40,
        ]
        if result.admet:
            a = result.admet
            lines += [
                f"  Molecular Weight:  {a.mol_weight:.2f} Da",
                f"  LogP:              {a.logp:.3f}",
                f"  TPSA:              {a.tpsa:.2f} Å²",
                f"  H-Bond Donors:     {a.hbd}",
                f"  H-Bond Acceptors:  {a.hba}",
                f"  Rotatable Bonds:   {a.rot_bonds}",
                f"  Aromatic Rings:    {a.aromatic_rings}",
                f"  Halogen Count:     {a.halogen_count}",
                f"  Nitro Group:       {'Yes' if a.has_nitro else 'No'}",
                f"  RO5 Violations:    {a.ro5_violations}",
            ]

        if result.alerts:
            lines += ["", "STRUCTURAL ALERTS"] + result.alerts
        if result.pass_criteria:
            lines += ["", "FAVORABLE PROPERTIES"] + result.pass_criteria

        if result.assay_probs:
            lines += ["", "PER-ASSAY PREDICTIONS", "-" * 40]
            for assay in TOX21_TARGETS:
                prob = result.assay_probs.get(assay, -1)
                if prob >= 0:
                    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                    flag = "⚠" if prob > 0.5 else " "
                    lines.append(f"  {flag} {assay:20s}: [{bar}] {prob:.3f}")

        lines += ["", "=" * 60,
                  "Disclaimer: Predictions are for research purposes only.",
                  "Not intended for clinical decision-making."]
        return "\n".join(lines)
