from sklearn.pipeline import Pipeline

from vehicle_value_estimator.utils.preprocess_transformer import (
    ColumnDropper,
    FeatureEngineeringTransformer,
    FillMissingValuesTransformer,
)


def cleaning_pipeline() -> Pipeline:
    """
    Create a data cleaning pipeline for preprocessing dataset.

    Returns:
        pipeline (Pipeline): Data cleaning pipeline object.
    """

    # Define columns to fill missing values
    columns_to_fill = [
        ("fuel_type", ["standard_model", "standard_make"], False),
        ("fuel_type", "standard_model", False),
        ("body_type", ["standard_model", "standard_make"], False),
        ("body_type", "standard_model", False),
        ("standard_colour", ["standard_model", "standard_make"], False),
        ("standard_colour", "standard_model", False),
        ("mileage", ["standard_model", "year_of_registration"], True),
        ("mileage", ["fuel_type", "year_of_registration"], True),
    ]

    # Define columns to normalize
    normalise_cols = [("standard_make", 0.95), ("standard_model", 0.5)]

    steps = [
        ("fill_columns", FillMissingValuesTransformer(columns_to_fill=columns_to_fill)),
        ("transform_and_engineer", FeatureEngineeringTransformer(norm_cols=normalise_cols)),
        (
            "dropper",
            ColumnDropper(
                target="price",
                columns_to_drop=["reg_code", "year_of_registration", "crossover_car_and_van", "standard_colour"],
            ),
        ),
    ]

    pipeline = Pipeline(steps=steps)

    return pipeline
