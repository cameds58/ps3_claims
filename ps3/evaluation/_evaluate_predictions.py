import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(y_true, y_pred, sample_weight=None):
    """
    Evaluates various metrics for a model's predictions.

    Parameters:
    - y_true: array-like, true outcome values.
    - y_pred: array-like, model predictions.
    - sample_weight: array-like, optional, weights (e.g., exposure).

    Returns:
    - metrics_df: pandas.DataFrame containing metrics as index.
    """
    # Compute metrics
    # Bias: Difference between predicted and actual exposure-weighted means
    actual_mean = np.average(y_true, weights=sample_weight)
    predicted_mean = np.average(y_pred, weights=sample_weight)
    bias = predicted_mean - actual_mean

    # Deviance
    deviance = np.average(
        (y_true - y_pred) ** 2 / sample_weight, weights=sample_weight
    ) if sample_weight is not None else np.mean((y_true - y_pred) ** 2)

    # MAE
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)

    # RMSE
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = np.sqrt(mse)

    # Gini coefficient
    def lorenz_curve(y_true, y_pred, sample_weight):
        ranking = np.argsort(y_pred)
        weighted_true = np.asarray(y_true)[ranking] * np.asarray(sample_weight)[ranking]
        cum_actual = np.cumsum(weighted_true)
        cum_actual /= cum_actual[-1]
        cum_samples = np.linspace(0, 1, len(cum_actual))
        return cum_samples, cum_actual

    def gini_coefficient(y_true, y_pred, sample_weight):
        cum_samples, cum_actual = lorenz_curve(y_true, y_pred, sample_weight)
        gini = 1 - 2 * np.trapz(cum_actual, cum_samples)
        return gini

    gini = gini_coefficient(y_true, y_pred, sample_weight)

    # Compile results into a DataFrame
    metrics = {
        "Bias": bias,
        "Deviance": deviance,
        "MAE": mae,
        "RMSE": rmse,
        "Gini Coefficient": gini,
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    return metrics_df