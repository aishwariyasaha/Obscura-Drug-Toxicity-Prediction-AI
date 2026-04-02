# src/ensemble.py
import numpy as np
from src.train import weighted_ensemble

class EnsembleModel:
    def __init__(self, models, weights, threshold):
        self.models    = models
        self.weights   = weights
        self.threshold = threshold

    def predict_proba(self, X):
        probs = weighted_ensemble(self.models, self.weights, X)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)