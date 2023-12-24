import pickle
import warnings

import mlflow
import mlflow.lightgbm
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
from vehicle_value_estimator.utils.model_utils import eval_metrics, preprocess_for_modelling


# from pprint import pprint


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


def main() -> None:
    model = LGBMRegressor(
        learning_rate=0.1,
        max_depth=20,
        min_split_gain=4,
        num_leaves=1024,
        random_state=1996,
        n_jobs=-1,
        verbose=-1,
    )

    mlflow.lightgbm.autolog(silent=True)

    for param, value in model.get_params().items():
        mlflow.log_param(param, value)

    enc_train, enc_val, y_train, y_val, ml_preprocessor, *_ = preprocess_for_modelling(X=X, y=y, cat_cols=CAT_COLS)

    # RFECV Selector
    rfecv_selector = RFECV(estimator=LGBMRegressor(verbose=-1), step=1, cv=5, n_jobs=-1)
    rfecv_selector.fit(enc_train, y_train)
    enc_train_rfecv = rfecv_selector.transform(enc_train)
    enc_val_rfecv = rfecv_selector.transform(enc_val)

    with mlflow.start_run(nested=True):
        logging.info("Starting ML training pipeline...")
        try:
            model.fit(enc_train_rfecv, y_train, eval_set=[(enc_val_rfecv, y_val)])
            pred = model.predict(enc_val_rfecv)

            rmse, mae, r2 = eval_metrics(y_val, pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            mlflow.end_run(status="FAILED")
            return

    # run_id = mlflow.get_run(run.info.run_id).info.run_id
    run_id = mlflow.last_active_run().info.run_id
    logging.info(f"Model training completed. Logged data and model in: {run_id}")

    # for key, data in fetch_logged_data(run_id).items():
    #     print(f"\n---------- logged {key} ----------")
    #     pprint(data)

    artifacts = {
        MODEL_PATH: model,
        ML_PREPROCESSOR_PATH: ml_preprocessor,
        RFECV_SELECTOR_PATH: rfecv_selector,
    }

    for filename, model in artifacts.items():
        with open(filename, "wb") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    main()


# logging.info("Starting ML training pipeline...")
# enc_train, enc_val, y_train, y_val, ml_preprocessor, *_ = preprocess_for_modelling(X=X, y=y, cat_cols=CAT_COLS)
#
# model = LGBMRegressor(
#     learning_rate=0.1, max_depth=20, min_sparsity=4, num_leaves=1024, random_state=1996, n_jobs=-1, verbose=-1
# )
#
# # RFECV Selector
# rfecv_selector = RFECV(estimator=LGBMRegressor(verbose=-1), step=1, cv=5, n_jobs=-1)
# rfecv_selector.fit(enc_train, y_train)
# enc_train_rfecv = rfecv_selector.transform(enc_train)
# enc_val_rfecv = rfecv_selector.transform(enc_val)

# Evaluation metrics
# eval_df = evaluate_model(model, enc_train_rfecv, enc_val_rfecv, y_train, y_val)
# logging.info(f"Evaluation metrics:\n{eval_df}")
