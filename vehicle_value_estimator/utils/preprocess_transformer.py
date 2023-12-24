import random
import warnings
from collections import Counter, namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from vehicle_value_estimator.config import VEHICLE_REGISTRATION_URL
from vehicle_value_estimator.utils.scraping import scrape_post_2001, scrape_pre_2001


warnings.filterwarnings("ignore")


def fill_col_with_aggregate(
    df: pd.DataFrame, col: str, col_by: Sequence[str], numeric: Optional[bool] = False
) -> pd.DataFrame:
    """
    Helper function to fill a column with missing values by getting
    the mode or mean of each non-missing value by one or two non-null columns.

    Args:
        df (DataFrame): Data.
        col (str): Column with missing values to fill.
        col_by (str or list): Column(s) to use for aggregating the values.
        numeric (bool, optional): Indicates whether 'col' is numeric. Defaults to False.

    Returns:
        df (DataFrame): Filled data.

    Notes:
        - The function uses the mode or mean as the aggregation function based on the value of `numeric`.
        - Missing values are replaced with the aggregated values or a default value.
    """

    temp_df = df[~df[col].isna()]
    AggregationFunction = namedtuple("AggregationFunction", ["func", "default"])

    # Choose the aggregation function and default value based on the value of `numeric`.
    if numeric:
        aggregation_function = AggregationFunction(func=np.mean, default=df[col].mean())
    else:
        aggregation_function = AggregationFunction(func=lambda x: pd.Series.mode(x)[0], default=df[col].mode()[0])

    # Group by `col_by` and apply the aggregation function.
    # Update `df[col]` with the grouped values.
    if isinstance(col_by, list):
        top_col_by = temp_df.groupby(col_by)[col].agg(aggregation_function.func).to_frame().dropna()

        df[col] = df.apply(
            lambda x: top_col_by.loc[x[col_by[0]], x[col_by[1]]].item()
            if (x[col_by[0]], x[col_by[1]]) in top_col_by.index and pd.isnull(x[col])
            else x[col],
            axis=1,
        )
    else:
        top_col_by = temp_df.groupby(col_by)[col].agg(aggregation_function.func).to_frame().dropna()

        col_dict = top_col_by.to_dict()[col]
        df[col] = df[col].fillna(df[col_by].map(col_dict))

    # Replace missing values with the default value.
    df[col] = np.where(df[col].isna(), aggregation_function.default, df[col])

    return df


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


def create_lookup_dictionaries(
    df: pd.DataFrame, key_col: str, value_col: str
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create and return forward and reverse lookup dictionaries."""
    forward_dict = dict(zip(df[key_col], df[value_col]))
    reverse_dict = {v: k for k, v in forward_dict.items()}
    return forward_dict, reverse_dict


def fill_year_reg(df: pd.DataFrame) -> pd.DataFrame:
    post_01 = scrape_post_2001(url=VEHICLE_REGISTRATION_URL, table_indices=[1, 2], col_count=3)
    pre_01 = scrape_pre_2001(url=VEHICLE_REGISTRATION_URL, table_indices=[4, 5], col_count=2)

    pre_01_83_dict, reversed_pre_01_83_dict = create_lookup_dictionaries(pre_01, "Letter", "Year_83")
    pre_01_63_dict, reversed_pre_01_63_dict = create_lookup_dictionaries(pre_01, "Letter", "Year_63")
    post_01_mar_dict, reversed_post_01_mar_dict = create_lookup_dictionaries(post_01, "March", "Year")
    post_01_sep_dict, reversed_post_01_sep_dict = create_lookup_dictionaries(post_01, "September", "Year")

    df["year_of_registration"] = np.where(df["vehicle_condition"] == "NEW", 2021, df["year_of_registration"])
    df["reg_code"] = np.where(df["year_of_registration"] == 2021, random.choice([21, 71]), df["reg_code"])

    for code_dict in [post_01_mar_dict, post_01_sep_dict]:
        df["year_of_registration"] = df["year_of_registration"].fillna(df["reg_code"].map(code_dict))

    df["year_of_registration"] = df["year_of_registration"].fillna(
        df["reg_code"].map(lambda x: pre_01_83_dict[x] if x in pre_01_83_dict else np.nan)
    )

    df["year_of_registration"] = df["year_of_registration"].fillna(
        df["reg_code"].map(lambda x: pre_01_63_dict[x] if x in pre_01_63_dict else np.nan)
    )

    # EDA shows data with years < 1910 are data errors, intended to be years > 2000
    df.loc[df["year_of_registration"] < 1910, "year_of_registration"] = df["reg_code"].map(post_01_mar_dict)
    df["year_of_registration"] = df["year_of_registration"].fillna(df["reg_code"].map(post_01_sep_dict))

    fill_col_with_aggregate(df, "year_of_registration", ["standard_model", "standard_make"])

    # manual lookup
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

    for year_dict in [reversed_post_01_mar_dict, reversed_post_01_sep_dict]:
        df["reg_code"] = df["reg_code"].fillna(df["year_of_registration"].map(year_dict))

    df["reg_code"] = df["reg_code"].fillna(
        df["year_of_registration"].map(
            lambda x: reversed_pre_01_83_dict[x] if x in reversed_pre_01_83_dict else np.nan
        )
    )

    df["reg_code"] = df["reg_code"].fillna(
        df["year_of_registration"].map(
            lambda x: reversed_pre_01_63_dict[x] if x in reversed_pre_01_63_dict else np.nan
        )
    )

    fill_col_with_aggregate(df, "reg_code", ["standard_model", "standard_make"])
    df.loc[df["reg_code"].isna(), "reg_code"] = "TW"

    df["year_of_registration"] = df["year_of_registration"].astype(int)

    return df


def data_engineering(df: pd.DataFrame, mode: str = "dev") -> pd.DataFrame:
    if mode == "dev":
        df.set_index("public_reference", inplace=True)
        df["crossover_car_and_van"] = np.where(~df["crossover_car_and_van"], 0, 1)

    df["vehicle_condition"] = np.where(df["vehicle_condition"] == "USED", 0, 1)

    df["car_age"] = 2021 - df["year_of_registration"]
    df["car_age"] = df["car_age"].astype(int)

    df["mile_per_yr"] = np.where(df["car_age"] == 0, 0, df["mileage"] / df["car_age"])

    def mile_bin(row: pd.Series) -> str:
        # using MOT average annual mileage per car from year 2012 to 2022;
        # our average annual mileage cutoff is 6,951
        # https://www.bymiles.co.uk/insure/magazine/mot-data-research-and-analysis/
        if row == 0:
            return "No Mileage"
        elif row < 6951:
            return "Low Mileage"
        elif row > 6951:
            return "High Mileage"
        else:
            return "Moderate Mileage"

    df["mile_yr_bin"] = df["mile_per_yr"].apply(mile_bin)

    if mode == "prod":
        df.drop(columns="year_of_registration", inplace=True)

    return df


def outlier_cutoff(df: pd.DataFrame, filter_iqr: bool = False) -> pd.DataFrame:
    # Set the mileage values greater than 200,000 to 200,000
    df["mileage"][df["mileage"] > 200000] = 200000
    # Get the maximum price from the dataset
    max_price = df["price"].max()
    # Find the next maximum price that is less than the maximum price
    next_max = max(val for val in df["price"].sort_values(ascending=False)[:100] if val < max_price)
    # Set the prices greater than the next maximum price to the next maximum price
    df["price"][df["price"] > next_max] = next_max
    # Set the prices less than 500 to 500
    df["price"][df["price"] < 500] = 500

    # we are sub-setting the data here to improve training time
    if filter_iqr:
        df_new = df[df["vehicle_condition"] == 1]
        df_used = df[df["vehicle_condition"] == 0]

        df_x = filter_data_iqr(df_new, ["price"], 0.2, 0.8)
        df_y = filter_data_iqr(df_used, ["mileage", "price", "car_age"], 0.2, 0.8)

        filtered_df = pd.concat([df_x, df_y])

        return filtered_df

    return df


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
    def __init__(self, norm_cols: List[Tuple[str, float]], filter_iqr: bool, mode: str) -> None:
        self.norm_cols: List[Tuple[str, float]] = norm_cols
        self.filter_iqr: bool = filter_iqr
        self.mode: str = mode

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineeringTransformer":
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X_transformed = X.copy(deep=True)

        for col, threshold in self.norm_cols:
            X_transformed = normalise_cols(X_transformed, col, threshold)
        X_transformed = data_engineering(X_transformed, self.mode)
        X_transformed = outlier_cutoff(X_transformed, self.filter_iqr)

        return X_transformed


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, target: str, columns_to_drop: List[str]) -> None:
        self.target: str = target
        self.columns_to_drop: List[str] = columns_to_drop + [target]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return X.drop(self.columns_to_drop, axis=1), X[self.target]
