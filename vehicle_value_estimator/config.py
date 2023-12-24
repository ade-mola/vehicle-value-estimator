import logging

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

TRAINING_SET = "vehicle_value_estimator/data/train.csv"
MODEL_PATH = "vehicle_value_estimator/artifacts/model.pkl"
ML_PREPROCESSOR_PATH = "vehicle_value_estimator/artifacts/ml_preprocessor.pkl"
RFECV_SELECTOR_PATH = "vehicle_value_estimator/artifacts/rfecv_selector.pkl"


car_features = pd.read_csv("vehicle_value_estimator/artifacts/car_features.csv").dropna()


def get_related_features(df: pd.DataFrame, group_column: str, target_column: str) -> dict:
    grouped_features = {}
    for value in df[group_column].unique():
        filtered_df = df[df[group_column] == value]
        grouped_features[value] = filtered_df[target_column].drop_duplicates().tolist()
    return grouped_features


STANDARD_MAKE = car_features["standard_make"].unique()
STANDARD_MODEL = get_related_features(car_features, "standard_make", "standard_model")
BODY_TYPE = get_related_features(car_features, "standard_model", "body_type")
FUEL_TYPE = get_related_features(car_features, "standard_model", "fuel_type")

VEHICLE_REGISTRATION_URL = "https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_United_Kingdom"
CAT_COLS = ["standard_make", "standard_model", "body_type", "fuel_type", "mile_yr_bin"]
