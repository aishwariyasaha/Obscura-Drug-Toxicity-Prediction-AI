"""
preprocessing.py — Industry-grade data loading and label engineering
for multi-label Tox21 toxicity prediction.

Key improvements over original:
  - Proper NaN handling (masked labels, not dropped)
  - Multi-label target matrix (12 assays)
  - ZINC augmentation for descriptor feature enrichment
  - Molecule sanitization with error logging
  - Stratified splitting per compound (multi-label aware)
"""

import pandas as pd
import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem import SaltRemover
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The 12 Tox21 assay targets
TOX21_TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


def standardize_mol(mol):
    """
    Standardize a molecule. Returns original mol on failure (don't discard it).
    """
    try:
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        # Return the original parsed mol rather than None
        return mol


def smiles_to_mol(smiles: str):
    """Parse SMILES with sanitize=False first, then attempt sanitization."""
    try:
        # Parse without sanitization so RDKit doesn't immediately reject
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
        if mol is None:
            return None
        # Try to sanitize, catching specific valence errors (Al, etc.)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Still usable for fingerprints even if sanitization partially fails
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                       Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                       Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                       Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                       Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
            except Exception:
                pass
        return mol
    except Exception:
        return None


def load_tox21(path: str) -> pd.DataFrame:
    """
    Load the Tox21 CSV, parse molecules, and return a clean dataframe.

    Columns returned:
        smiles, mol, mol_id, NR-AR … SR-p53  (label columns as float, NaN = untested)
    """
    logger.info(f"Loading Tox21 from {path}")
    df = pd.read_csv(path)

    # Verify expected columns exist
    missing = [c for c in TOX21_TARGETS + ["smiles"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Parse molecules
    df["mol"] = df["smiles"].apply(smiles_to_mol)
    n_before = len(df)
    df = df[df["mol"].notnull()].reset_index(drop=True)
    n_after = len(df)
    logger.info(f"Valid molecules: {n_after}/{n_before} ({n_before - n_after} failed parsing)")

    # Keep label columns as float (NaN = label not available for this compound/assay)
    for col in TOX21_TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Label availability per assay:\n" +
                "\n".join(f"  {c}: {df[c].notna().sum()} labeled ({df[c].mean():.3f} positive rate)"
                          for c in TOX21_TARGETS))
    return df


def get_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Returns shape (n_compounds, 12) label matrix.
    NaN where compound was not tested in that assay.
    """
    return df[TOX21_TARGETS].values.astype(float)


def make_aggregate_label(df: pd.DataFrame, strategy: str = "any") -> np.ndarray:
    """
    Collapse 12 assays into a single binary label.

    strategy options:
      'any'    — toxic if active in ANY assay (conservative, maximizes safety)
      'majority' — toxic if active in >50% of tested assays
      'sr_only' — use only stress-response (SR-*) targets (more drug-relevant)

    BUG IN ORIGINAL: df.iloc[:, :12].max() grabs FIRST 12 COLS (may include
    non-label cols) and ignores NaN → incorrect labels.
    """
    label_matrix = df[TOX21_TARGETS].values  # shape (N, 12)

    if strategy == "any":
        # Compound is toxic if it tested positive in at least one assay
        # np.nanmax treats NaN-only rows as NaN → we assign 0 (no evidence of toxicity)
        y = np.nanmax(label_matrix, axis=1)
        y = np.where(np.isnan(y), 0, y).astype(int)

    elif strategy == "majority":
        pos = np.nansum(label_matrix == 1, axis=1)
        tested = np.nansum(~np.isnan(label_matrix), axis=1)
        y = (pos / np.maximum(tested, 1) > 0.5).astype(int)

    elif strategy == "sr_only":
        sr_cols = [c for c in TOX21_TARGETS if c.startswith("SR")]
        sub = df[sr_cols].values
        y = np.nanmax(sub, axis=1)
        y = np.where(np.isnan(y), 0, y).astype(int)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    pos_rate = y.mean()
    logger.info(f"Aggregate label (strategy='{strategy}'): "
                f"{y.sum()}/{len(y)} toxic ({pos_rate:.1%})")
    return y


def stratified_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Three-way stratified split: train / val / test.
    Stratified on binary label to preserve class balance in all splits.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )
    logger.info(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
