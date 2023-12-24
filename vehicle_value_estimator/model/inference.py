import mlflow.pyfunc

from vehicle_value_estimator.config import MLFLOW_MODEL_NAME, MLFLOW_MODEL_VERSION


model = mlflow.pyfunc.load_model(model_uri=f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")
print(model)
