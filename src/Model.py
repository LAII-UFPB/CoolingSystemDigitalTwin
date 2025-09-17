import os
import logging
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from src.ModelMetrics import ModelMetrics


class Model(ABC):
    """Base class for all models."""

    def __init__(self, log_dir="logs", log_level=logging.INFO):
        self.is_fitted = False
        self.metrics = ModelMetrics()

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{self.__class__.__name__}_{timestamp}.log")

        # Logger configs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, mode="w")
            fh.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def log(self, msg, level=logging.INFO):
        self.logger.log(level, msg)

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
        rmse = self.metrics.root_mean_squared_error(y_true, y_pred)
        r2 = self.metrics.r2_score(y_true, y_pred)
        mape = self.metrics.mean_absolute_percentage_error(y_true, y_pred)
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
