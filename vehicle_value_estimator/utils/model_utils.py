import time
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    models: Union[BaseEstimator, List[BaseEstimator]],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    return_model: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray], List[BaseEstimator]]]:
    """
    Evaluates the performance of machine learning models on the validation set.

    Args:
        models (list or object): A single machine learning model or a list of models to evaluate.
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.Series or np.array): Training target variable.
        y_val (pd.Series or np.array): Validation target variable.
        return_model (bool, optional): Whether to return the fitted models and predictions.

    Returns:
        df (pd.DataFrame): Performance metrics for each model.
        predictions (list): Predictions from each model (only returned if return_model=True).
        fitted_models (list): Fitted models (only returned if return_model=True).
    """
    if not isinstance(models, list):
        models = [models]

    results_list = []
    fitted_models = []
    predictions = []

    for model in models:
        start = time.time()

        fitted_model = model.fit(X_train, y_train)
        pred = fitted_model.predict(X_val)

        train_score = fitted_model.score(X_train, y_train)
        val_score = fitted_model.score(X_val, y_val)
        r2 = round(r2_score(y_val, pred), 2)
        rmse = round(np.sqrt(mean_squared_error(y_val, pred)), 2)
        mae = round(mean_absolute_error(y_val, pred), 2)

        end = time.time()
        elapsed_time = end - start

        results = {
            "Model": type(model).__name__,
            "train_score": train_score,
            "val_score": val_score,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Runtime (secs)": elapsed_time,
        }
        results_list.append(results)
        fitted_models.append(fitted_model)
        predictions.append(pred)

    df = pd.DataFrame(results_list)
    df.set_index("Model", inplace=True)
    df.sort_values("RMSE", inplace=True)

    if return_model:
        return df, predictions, fitted_models
    else:
        return df
