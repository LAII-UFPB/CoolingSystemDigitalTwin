# model.py
from abc import ABC, abstractmethod
import numpy as np
from DigitalTwin.src.ModelMetrics import ModelMetrics

class Model(ABC):
    """Base class for all models."""

    def __init__(self):
        self.is_fitted = False
        self.metrics = ModelMetrics()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        pass

    def get_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate model with common regression metrics."""
        mae = self.metrics.mean_absolute_error(y_true, y_pred)
        rmse = self.root_mean_squared_error(y_true, y_pred)
        r2 = self.r2_score(y_true, y_pred)
        mape = self.mean_absolute_percentage_error(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    def save(self, path: str):
        """Save the model to a file."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load a model from file."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def explain(self):
        """Optional explanation method (for interpretable models)."""
        return None

    def update_online(self, X: np.ndarray, y: np.ndarray):
        """Optional online learning method for adaptive models."""
        pass
