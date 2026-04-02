# test_model.py  — put this in your project root
# test_model.py
import sys
sys.path.insert(0, ".")

from src.ensemble import EnsembleModel  # ← replace "from run import EnsembleModel" # ← pickle needs this to deserialize the model
from src.predict import ToxicityPredictor

predictor = ToxicityPredictor.load("models/")

# Known toxic compounds
test_cases = [
    ("CC1=CC=C(C=C1)NC(=O)CCl", "Chloroacetanilide — known toxic"),
    ("O=C1NC(=O)c2ccccc21",     "Isatoic anhydride — moderate risk"),
]

for smiles, name in test_cases:
    result = predictor.predict(smiles)
    print(f"\n{name}")
    print(predictor.format_report(result))
