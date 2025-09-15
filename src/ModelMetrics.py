
import numpy as np

class ModelMetrics():
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    @staticmethod
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
    
    @staticmethod
    def absolute_percentage_error(y_true, y_pred):
        mask = y_true != 0
        return np.abs((y_true[mask] - y_pred[mask])/y_true[mask]) * 100 if np.any(mask) else 0.0

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(ModelMetrics.absolute_percentage_error(y_true, y_pred))