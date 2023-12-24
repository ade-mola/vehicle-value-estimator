import pickle
import warnings

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFECV

from vehicle_value_estimator.config import (
    CAT_COLS,
    ML_PREPROCESSOR_PATH,
    MODEL_PATH,
    RFECV_SELECTOR_PATH,
    TRAINING_SET,
    logging,
)
from vehicle_value_estimator.data_pipeline.cleaning import cleaning_pipeline
from vehicle_value_estimator.utils.model_utils import evaluate_model, preprocess_for_modelling


warnings.filterwarnings("ignore")


training_data = pd.read_csv(TRAINING_SET)


def log_missing_data(df: pd.DataFrame, df_name: str) -> None:
    missing = (1 - (len(df.dropna()) / len(df))) * 100
    logging.info(f"Shape of {df_name} is {df.shape}. Missing data: {round(missing, 2)}%")


logging.info("Starting data cleaning and transformation pipeline...")
log_missing_data(training_data, "Initial training data")
pipeline = cleaning_pipeline(filter_iqr=True)

X, y = pipeline.fit_transform(training_data)
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
log_missing_data(X, "Post-cleaning data")


logging.info("Starting ML training pipeline...")
enc_train, enc_val, y_train, y_val, ml_preprocessor, *_ = preprocess_for_modelling(X=X, y=y, cat_cols=CAT_COLS)

model = LGBMRegressor(
    learning_rate=0.1, max_depth=20, min_sparsity=4, num_leaves=1024, random_state=1996, n_jobs=-1, verbose=-1
)

# RFECV Selector
rfecv_selector = RFECV(estimator=LGBMRegressor(verbose=-1), step=1, cv=5, n_jobs=-1)
rfecv_selector.fit(enc_train, y_train)
enc_train_rfecv = rfecv_selector.transform(enc_train)
enc_val_rfecv = rfecv_selector.transform(enc_val)

# Evaluation metrics
eval_df = evaluate_model(model, enc_train_rfecv, enc_val_rfecv, y_train, y_val)
logging.info(f"Evaluation metrics:\n{eval_df}")

artifacts = {
    MODEL_PATH: model,
    ML_PREPROCESSOR_PATH: ml_preprocessor,
    RFECV_SELECTOR_PATH: rfecv_selector,
}

for filename, model in artifacts.items():
    with open(filename, "wb") as f:
        pickle.dump(model, f)
