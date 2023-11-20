import random
import warnings
from collections import Counter, namedtuple
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


warnings.filterwarnings("ignore")


def filter_data_iqr(df: pd.DataFrame, cols: List, lower_percentile: float, upper_percentile: float) -> pd.DataFrame:
    # Calculate the first and third quartiles for the columns
    quantiles: dict = {}
    for col in cols:
        q1 = df[col].quantile(lower_percentile)
        q3 = df[col].quantile(upper_percentile)
        quantiles[col] = (q1, q3)

    # Calculate the interquartile range for the columns
    iqrs: dict = {}
    for col, (q1, q3) in quantiles.items():
        iqr = q3 - q1
        iqrs[col] = iqr

    # Calculate the lower and upper bounds for the columns using the IQR
    bounds: dict = {}
    for col, iqr in iqrs.items():
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        bounds[col] = (lower_bound, upper_bound)

    # Create a boolean mask to filter the rows that are within the bounds
    mask = np.ones(df.shape[0], dtype=bool)
    for col, (lower_bound, upper_bound) in bounds.items():
        mask = mask & (df[col] >= lower_bound) & (df[col] <= upper_bound)

    # Use the mask to filter the dataframe
    filtered_df = df[mask]

    return filtered_df


def fill_col_with_aggregate(
    df: pd.DataFrame, col: str, col_by: Sequence[str], numeric: Optional[bool] = False
) -> pd.DataFrame:
    """
    Helper function to fill a column with missing values by getting
    the mode or mean of each non-missing value by one or two non-null columns.
    """

    temp_df = df[~df[col].isna()]
    AggregationFunction = namedtuple("AggregationFunction", ["func", "default"])

    # Choose the aggregation function and default value based on the value of `numeric`.
    if numeric:
        aggregation_function = AggregationFunction(func=np.mean, default=df[col].mean())
    else:
        aggregation_function = AggregationFunction(func=lambda x: stats.mode(x)[0], default=df[col].mode()[0])

    # Group by `col_by` and apply the aggregation function.
    # Update `df[col]` with the grouped values.
    if isinstance(col_by, list):
        top_col_by_car = temp_df.groupby(col_by)[col].agg(aggregation_function.func).to_frame().dropna()

        df[col] = df.apply(
            lambda x: top_col_by_car.loc[x[col_by[0]], x[col_by[1]]].item()
            if (x[col_by[0]], x[col_by[1]]) in top_col_by_car.index and pd.isnull(x[col])
            else x[col],
            axis=1,
        )
    else:
        top_col_by_car = temp_df.groupby(col_by)[col].agg(aggregation_function.func).to_frame().dropna()

        col_car_dict = dict(zip(top_col_by_car.index, top_col_by_car[col]))
        df[col] = df[col].fillna(df[col_by].map(col_car_dict))

        # Replace missing values with the default value.
        df[col] = np.where(df[col].isna(), aggregation_function.default, df[col])

    return df


def fill_year_reg(df: pd.DataFrame) -> pd.DataFrame:
    post_01 = pd.read_csv("data/post_2001.csv", dtype={"Year": np.float64, "September": "object"})
    pre_01 = pd.read_csv("data/pre_2001.csv", dtype={"Year": np.float64})

    pre_01_dict = dict(zip(pre_01["Letter"], pre_01["Year_83"]))
    pre_83_dict = dict(zip(pre_01["Letter"], pre_01["Year_63"]))
    post_01_mar_dict = dict(zip(post_01["March"], post_01["Year"]))
    post_01_sep_dict = dict(zip(post_01["September"], post_01["Year"]))

    reversed_pre_01_dict = {value: key for key, value in pre_01_dict.items()}
    reversed_pre_83_dict = {value: key for key, value in pre_83_dict.items()}
    reversed_post_01_mar_dict = {value: key for key, value in post_01_mar_dict.items()}
    reversed_post_01_sep_dict = {value: key for key, value in post_01_sep_dict.items()}

    df["year_of_registration"] = np.where(df["vehicle_condition"] == "NEW", 2021, df["year_of_registration"])

    df["reg_code"] = np.where(df["year_of_registration"] == 2021, random.choice([21, 71]), df["reg_code"])

    df["year_of_registration"] = df["year_of_registration"].fillna(df["reg_code"].map(post_01_mar_dict))
    df["year_of_registration"] = df["year_of_registration"].fillna(df["reg_code"].map(post_01_sep_dict))

    df["year_of_registration"] = df["year_of_registration"].fillna(
        df["reg_code"].map(lambda x: pre_01_dict[x] if x in pre_01_dict else np.nan)
    )

    df["year_of_registration"] = df["year_of_registration"].fillna(
        df["reg_code"].map(lambda x: pre_83_dict[x] if x in pre_83_dict else np.nan)
    )

    df.loc[df["year_of_registration"] < 1910, "year_of_registration"] = df["reg_code"].map(post_01_mar_dict)
    df["year_of_registration"] = df["year_of_registration"].fillna(df["reg_code"].map(post_01_sep_dict))

    fill_col_with_aggregate(df, "year_of_registration", ["standard_model", "standard_make"])

    year_reg_lookup = {
        ("Ferrari", "250"): 1962,
        ("AC", "Cobra"): 1989,
        ("Chevrolet", "GMC"): 2006,
        ("Mercedes-Benz", "EQV"): 2020,
        ("Lamborghini", "Jalpa"): 1985,
        ("Lamborghini", "Miura"): 1970,
        ("Ferrari", "275"): 1966,
        ("Porsche", "959"): 1990,
        ("Fiat", "600"): 1969,
        ("Maserati", "3500"): 1961,
        ("Rolls-Royce", "Silver Cloud"): 1965,
        ("Chevrolet", "Chevy"): 2008,
        ("Nissan", "Patrol"): 2006,
        ("Ferrari", "512"): 1990,
        ("Ferrari", "F40"): 1990,
        ("Lincoln", "Navigator"): 2018,
    }

    if df["year_of_registration"].isna().any():
        df.loc[df["year_of_registration"].isna(), "year_of_registration"] = df.apply(
            lambda x: year_reg_lookup[(x["standard_make"], x["standard_model"])]
            if (x["standard_make"], x["standard_model"]) in year_reg_lookup
            else x["year_of_registration"],
            axis=1,
        )

    fill_col_with_aggregate(df, "year_of_registration", "standard_make")

    df["reg_code"] = df["reg_code"].fillna(df["year_of_registration"].map(reversed_post_01_mar_dict))
    df["reg_code"] = df["reg_code"].fillna(df["year_of_registration"].map(reversed_post_01_sep_dict))

    df["reg_code"] = df["reg_code"].fillna(
        df["year_of_registration"].map(lambda x: reversed_pre_01_dict[x] if x in reversed_pre_01_dict else np.nan)
    )

    df["reg_code"] = df["reg_code"].fillna(
        df["year_of_registration"].map(lambda x: reversed_pre_83_dict[x] if x in reversed_pre_83_dict else np.nan)
    )

    fill_col_with_aggregate(df, "reg_code", ["standard_model", "standard_make"])
    df.loc[df["reg_code"].isna(), "reg_code"] = "TW"

    df["year_of_registration"] = df["year_of_registration"].astype(int)

    return df


def data_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("public_reference", inplace=True)

    df["vehicle_condition"] = np.where(df["vehicle_condition"] == "USED", 0, 1)
    df["crossover_car_and_van"] = np.where(~df["crossover_car_and_van"], 0, 1)

    df["car_age"] = 2021 - df["year_of_registration"]
    df["car_age"] = df["car_age"].astype(int)

    df["mile_per_yr"] = np.where(df["car_age"] == 0, 0, df["mileage"] / df["car_age"])

    def mile_bin(row: pd.Series) -> str:
        if row == 0:
            return "No Mileage"
        elif row < 7052:
            return "Low Mileage"
        elif row > 7052:
            return "High Mileage"
        else:
            return "Moderate Mileage"

    df["mile_yr_bin"] = df["mile_per_yr"].apply(mile_bin)

    return df


def outlier_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    df["mileage"][df["mileage"] > 200000] = 200000
    max_price = df["price"].max()
    next_max = max(val for val in df["price"].sort_values(ascending=False)[:100] if val < max_price)
    df["price"][df["price"] > next_max] = next_max
    df["price"][df["price"] < 500] = 500

    df_new = df[df["vehicle_condition"] == 1]
    df_used = df[df["vehicle_condition"] == 0]

    df_x = filter_data_iqr(df_new, ["price"], 0.2, 0.8)
    df_y = filter_data_iqr(df_used, ["mileage", "price", "car_age"], 0.2, 0.8)

    filtered_df = pd.concat([df_x, df_y])

    return filtered_df


def normalise_cols(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    """
    Helper function that helps categorise less occurring values and
    reduce the number of unique values.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    col (str): The column in the DataFrame to normalize.
    threshold (float): The threshold as a fraction to determine less occurring values.

    Returns:
    pd.DataFrame: The DataFrame with normalized column.
    """

    threshold = int(threshold * len(df[col]))
    s, i = 0, 0
    categories_list = []

    counts = Counter(df[col])

    while s < threshold and i < len(counts):
        s += counts.most_common()[i][1]
        categories_list.append(counts.most_common()[i][0])
        i += 1

    df[col] = df[col].apply(lambda x: x if x in categories_list else "Other")

    return df


class FillMissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_fill: List[Tuple[str, Sequence[str], bool]]) -> None:
        self.columns_to_fill: List[Tuple[str, Sequence[str], bool]] = columns_to_fill

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FillMissingValuesTransformer":
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X_transformed = X.copy(deep=True)

        for col, col_by, *numeric in self.columns_to_fill:
            numeric_val = numeric[0] if numeric else None
            X_transformed = fill_col_with_aggregate(X_transformed, col, col_by, numeric_val)

        X_transformed = fill_year_reg(X_transformed)

        return X_transformed


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, norm_cols: List[Tuple[str, float]]) -> None:
        self.norm_cols: List[Tuple[str, float]] = norm_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineeringTransformer":
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X_transformed = X.copy(deep=True)

        for col, threshold in self.norm_cols:
            X_transformed = normalise_cols(X_transformed, col, threshold)
        X_transformed = data_engineering(X_transformed)
        X_transformed = outlier_cutoff(X_transformed)

        return X_transformed


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, target: str, columns_to_drop: List[str]) -> None:
        self.target: str = target
        self.columns_to_drop: List[str] = columns_to_drop + [target]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return X.drop(self.columns_to_drop, axis=1), X[self.target]
