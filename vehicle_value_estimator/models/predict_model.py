import pandas as pd

from vehicle_value_estimator.pipeline.cleaning import cleaning_pipeline


cleaning_instance = cleaning_pipeline()

data = pd.read_csv("..data/test.csv")

X, y = cleaning_instance.fit_transform(data)
