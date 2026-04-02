# fix_pickle.py
import sys
sys.path.insert(0, ".")

# Step 1: load old pickle with old class location
from run import EnsembleModel  # loads from run.py
import joblib

old_model = joblib.load("models/ensemble_model.pkl")

# Step 2: move EnsembleModel to src/ensemble.py first, then re-save
# This re-saves the pickle so it points to src.ensemble.EnsembleModel
from src.ensemble import EnsembleModel as NewEnsembleModel

new_model = NewEnsembleModel(
    models=old_model.models,
    weights=old_model.weights,
    threshold=old_model.threshold,
)

joblib.dump(new_model, "models/ensemble_model.pkl")
print("Done — pickle re-saved with src.ensemble.EnsembleModel")