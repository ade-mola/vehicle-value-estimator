"""Script to scrap UK vehicle registration data
'https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_United_Kingdom'
"""

import re
from typing import List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from vehicle_value_estimator.config import logging


def clean_text(text: str) -> str:
    """Remove square-bracketed words from text and strip whitespace."""
    return re.sub(r"\[\w+]", "", text).strip()


def scrape_tables(url: str, table_indices: List[int], col_count: int) -> pd.DataFrame:
    """
    Scrape specified tables from a webpage and return as a pandas DataFrame.

    Args:
        url (str): URL of the webpage to scrape.
        table_indices (List[int]): Indices of the tables to scrape.
        col_count (int): Number of columns expected in the tables.

    Returns:
        pd.DataFrame: A DataFrame containing the scraped data.
    """
    with requests.Session() as session:
        try:
            page = session.get(url, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            tables = soup.find_all("table", {"class": "wikitable"})

            if not tables:
                raise ValueError("No tables found")

            rows_list = [tables[i].find_all("tr") for i in table_indices]
            data = []
            for rows in rows_list:
                for row in rows[1:]:
                    cells = row.find_all("th") + row.find_all("td")
                    if len(cells) == col_count:
                        data.append([clean_text(cell.get_text(strip=True)) for cell in cells])

            # Create DataFrame
            headers = [clean_text(header.get_text(strip=True)) for header in rows_list[0][0].find_all("th")]
            return pd.DataFrame(data, columns=headers)

        except requests.RequestException as e:
            logging.info(f"Request failed: {e}")
        except Exception as e:
            logging.info(f"Error occurred: {e}")

    return pd.DataFrame()  # Return empty DataFrame in case of failure


def process_post_2001_row(row: pd.Series) -> pd.Series:
    year, march, september = (
        row["Year"].split("/")[0],
        row["1 March – 31 August"],
        row["1 September – 28/29 February"],
    )
    return pd.Series([year, march, september])


def scrape_post_2001(url: str, table_indices: List[int], col_count: int) -> pd.DataFrame:
    df = scrape_tables(url, table_indices, col_count)
    new_df = df.apply(process_post_2001_row, axis=1)
    new_df.columns = ["Year", "March", "September"]
    new_df["Year"] = new_df["Year"].astype(np.float64)
    new_df["March"] = new_df["March"].astype(object)
    new_df["September"] = new_df["September"].astype(object)
    return new_df


def extract_years(dates: str) -> str:
    start_year, _ = dates.split("–")
    return start_year.split()[-1]


def scrape_pre_2001(url: str, table_indices: List[int], col_count: int) -> pd.DataFrame:
    df = scrape_tables(url, table_indices, col_count)
    df["Year"] = df["Dates of issue"].apply(extract_years)
    grouped = df.groupby("Letter")["Year"].agg(["first", "last"]).reset_index()
    grouped.columns = ["Letter", "Year_63", "Year_83"]
    grouped["Year_63"] = grouped["Year_63"].astype(np.float64)
    grouped["Year_83"] = grouped["Year_83"].astype(np.float64)
    return grouped
