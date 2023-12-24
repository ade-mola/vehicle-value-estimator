import pickle
import subprocess
import sys
import warnings
from typing import Any, Tuple

import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from vehicle_value_estimator.config import (
    BODY_TYPE,
    FUEL_TYPE,
    ML_PREPROCESSOR_PATH,
    MODEL_PATH,
    RFECV_SELECTOR_PATH,
    STANDARD_MAKE,
    STANDARD_MODEL,
    logging,
)
from vehicle_value_estimator.utils.preprocess_transformer import data_engineering


try:
    __import__("category_encoders")
except ImportError:
    # Incase package not installed; this installs it
    subprocess.check_call([sys.executable, "-m", "pip", "install", "category_encoders"])

warnings.simplefilter(action="default", category=InconsistentVersionWarning)


class VehicleValueEstimatorModel:
    def __init__(self, model_file_path: str, pipeline_file_path: str, selector_path: str) -> None:
        self.model = self._load(model_file_path)
        self.pipeline = self._load(pipeline_file_path)
        self.selector = self._load(selector_path)

    def _load(self, file_path: str) -> Any:
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except InconsistentVersionWarning as w:
            logging.warning(f"Original sklearn version: {w}")
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            raise

    def predict_price(self, input_data: Tuple[float, str, str, str, float, str, str]) -> float:
        df = pd.DataFrame(
            data=[input_data],
            columns=[
                "mileage",
                "standard_make",
                "standard_model",
                "vehicle_condition",
                "year_of_registration",
                "body_type",
                "fuel_type",
            ],
        )

        df = data_engineering(df, mode="prod")
        print(df)
        df = self.pipeline.transform(df)
        df = self.selector.transform(df)

        return self.model.predict(df.reshape(1, -1))[0]  # type: ignore


class VehicleValueEstimatorApp:
    def __init__(self, model_file_path: str, pipeline_file_path: str, selector_path: str) -> None:
        self.model = VehicleValueEstimatorModel(model_file_path, pipeline_file_path, selector_path)

    def show_header(self) -> None:
        st.title("Vehicle Value Estimator")
        st.markdown(
            """
        This web application is an implementation of a predictive model
        for predicting the estimated price valuation of a vehicle.

        <div style="background-color:#f15145;
                    border-radius: 25px;
                    padding:5px">
            <h2 style="color:#2c3c51;
                       text-align:center;">
                Vehicle Value Estimator App
            </h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def get_input_data(self) -> Tuple[float, str, str, str, float, str, str]:
        standard_model = ""
        body_type = ""
        fuel_type = ""

        mileage = float(st.number_input(label="Vehicle Mileage: ", min_value=0, max_value=90000))

        standard_make = str(st.selectbox(label="Choose Vehicle Brand: ", options=tuple(STANDARD_MAKE)))

        if standard_make:
            model = STANDARD_MODEL.get(standard_make, [])
            standard_model = str(st.selectbox(label="Choose Vehicle Model: ", options=tuple(model)))

            if standard_model:
                body = BODY_TYPE.get(standard_model, [])
                body_type = str(st.selectbox(label="Choose Vehicle Body Type: ", options=tuple(body)))

                fuel = FUEL_TYPE.get(standard_model, [])
                fuel_type = str(st.selectbox(label="Choose Vehicle Fuel Type: ", options=tuple(fuel)))

        vehicle_condition = str(st.radio(label="Choose Vehicle Condition: ", options=("USED", "NEW")))

        year_of_registration = float(
            st.slider(label="Choose Vehicle Registration Year: ", min_value=1933, max_value=2021, value=2017)
        )

        return mileage, standard_make, standard_model, vehicle_condition, year_of_registration, body_type, fuel_type

    def show_prediction_result(self, price: float) -> None:
        st.success(f"Estimated Price of Vehicle is: £{price:,.2f}")

    def run(self) -> None:
        self.show_header()
        data = self.get_input_data()

        if st.button("Predict"):
            price = self.model.predict_price(data)
            self.show_prediction_result(price)


if __name__ == "__main__":
    model_path = MODEL_PATH
    ml_pipe_path = ML_PREPROCESSOR_PATH
    sel_path = RFECV_SELECTOR_PATH

    app = VehicleValueEstimatorApp(model_path, ml_pipe_path, sel_path)
    app.run()
