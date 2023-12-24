import time
from typing import List, Tuple, Union

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


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


def preprocess_for_modelling(
    X: pd.DataFrame, y: pd.Series, cat_cols: list[str], test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline, pd.DataFrame, pd.DataFrame,]:
    # Converting categorical columns to 'category' data type
    X[cat_cols] = X[cat_cols].apply(lambda x: x.astype("category"))

    # Splitting into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=1996)

    # Creating the pipeline for preprocessing
    ml_preprocessing_pipeline = Pipeline(
        [
            ("encoder", ce.LeaveOneOutEncoder(cols=cat_cols)),
            (
                "polynomial",
                ColumnTransformer(
                    transformers=[
                        ("poly", PolynomialFeatures(interaction_only=True, include_bias=False), X_train.columns)
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            # Add more preprocessing steps if needed
        ]
    ).set_output(transform="pandas")

    # Fit the preprocessing pipeline on the training data
    ml_preprocessing_pipeline.fit(X_train, y_train)

    # Apply the preprocessing pipeline to transform the data
    enc_train = ml_preprocessing_pipeline.transform(X_train)
    enc_val = ml_preprocessing_pipeline.transform(X_val)

    return (
        enc_train,
        enc_val,
        y_train,
        y_val,
        ml_preprocessing_pipeline,
        X_train,
        X_val,
    )
