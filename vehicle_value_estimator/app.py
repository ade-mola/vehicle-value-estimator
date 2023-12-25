import pickle
import subprocess
import sys
import time
import warnings
from typing import Any, Optional, Tuple, Union

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
        df = self.pipeline.transform(df)
        df = self.selector.transform(df)

        return self.model.predict(df.reshape(1, -1))[0]  # type: ignore


class VehicleValueEstimatorApp:
    def __init__(self, model_file_path: str, pipeline_file_path: str, selector_path: str) -> None:
        self.model = VehicleValueEstimatorModel(model_file_path, pipeline_file_path, selector_path)

    def setup_page(self) -> None:
        st.set_page_config(layout="wide", page_title="Vehicle Value Estimator App", page_icon=":car:")

    def get_input_data(self) -> Union[Tuple[float, str, str, str, float, str, str], Optional[None]]:
        standard_model = ""
        body_type = ""
        fuel_type = ""

        mileage = float(st.number_input(label="Vehicle Mileage: ", min_value=0, max_value=90000))
        standard_make = str(st.selectbox(label="Choose Vehicle Brand: ", options=tuple(STANDARD_MAKE)))

        if standard_make:
            standard_model = str(
                st.selectbox(label="Choose Vehicle Model: ", options=tuple(STANDARD_MODEL.get(standard_make, [])))
            )

            if standard_model:
                body_type = str(
                    st.selectbox(label="Choose Vehicle Body Type: ", options=tuple(BODY_TYPE.get(standard_model, [])))
                )
                fuel_type = str(
                    st.selectbox(label="Choose Vehicle Fuel Type: ", options=tuple(FUEL_TYPE.get(standard_model, [])))
                )

        vehicle_condition = str(st.radio(label="Choose Vehicle Condition: ", options=("USED", "NEW")))

        year_of_registration = float(
            st.slider(label="Choose Vehicle Registration Year: ", min_value=1933, max_value=2021, value=2017)
        )

        submitted = st.button(label="Predict")
        if submitted:
            return (
                mileage,
                standard_make,
                standard_model,
                vehicle_condition,
                year_of_registration,
                body_type,
                fuel_type,
            )
        else:
            return None

    def run(self) -> None:
        self.setup_page()
        st.header("Vehicle Value Estimator")
        st.subheader("Estimate the market value of your vehicle")

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                data = self.get_input_data()
            with col2:
                st.image(
                    "https://www.hdcarwallpapers.com/walls/2015_mercedes_amg_gt_s_uk_spec-HD.jpg",
                    use_column_width=True,
                )

                result_placeholder = st.empty()
                if data:
                    progress_text = "Operation in progress. Please wait..."
                    my_bar = st.progress(0, text=progress_text)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1, text=progress_text)
                    time.sleep(1)
                    my_bar.empty()
                    result_placeholder.empty()

                    price = self.model.predict_price(data)
                    result_placeholder.success(f"Estimated Price of Vehicle: £{price:,.2f}")

        with st.expander("More Options"):
            # Additional options can be placed here
            pass

        st.markdown("### App developed by Ademola Olokun")


if __name__ == "__main__":
    model_path = MODEL_PATH
    ml_pipe_path = ML_PREPROCESSOR_PATH
    sel_path = RFECV_SELECTOR_PATH

    app = VehicleValueEstimatorApp(model_path, ml_pipe_path, sel_path)
    app.run()
