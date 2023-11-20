"""Module for data ingestion."""
import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from vehicle_value_estimator.config import logging


class DataIngestion:
    """
    A class for handling the ingestion of data for a vehicle value estimator.

    This class manages the process of reading a dataset,
    splitting it into training and testing sets, and saving these sets.
    """

    def __init__(self, base_dir: str = "../data") -> None:
        """
        Initializes the DataIngestion class with the base directory for data files.

        Parameters:
            base_dir (str): The base directory where data files will be stored.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.train_data_path = self.base_dir / "train.csv"
        self.test_data_path = self.base_dir / "test.csv"
        self.raw_data_path = self.base_dir / "raw_data.csv"

    def initiate_data_ingestion(self, data_path: str, test_size: float) -> tuple[Path, Path]:
        """
        Reads a dataset from a given path, performs a train-test split, and saves the data.

        Parameters:
            data_path (str): The file path of the dataset to be ingested.
            test_size (float): Proportion of the dataset to allocate to the test set.

        Returns:
            tuple[Path, Path]: Paths to the saved training and testing data files.
        """

        logging.info("Starting data ingestion process...")
        try:
            data = pd.read_csv(data_path)
            data.to_csv(self.raw_data_path, index=False, header=True)
            logging.info(f"Dataset shape is {data.shape}")
            logging.info("Dataset saved as a dataframe in 'data' folder.")

            # Set the 'public_reference' column as the index
            data_copy = data.copy(deep=True)
            data_copy = data_copy.set_index("public_reference")
            # Drop duplicate rows and reset the index
            logging.info(f"Total number of duplicates is {data_copy.duplicated().sum()}")
            logging.info("Dropping duplicates...")
            data_copy = data_copy.drop_duplicates()
            data_copy = data_copy.reset_index()
            logging.info(f"Dropped duplicates. New dataset shape is {data_copy.shape}")

            logging.info("Initiating train-test split...")
            train, test = train_test_split(data_copy, test_size=test_size, random_state=42)

            train.to_csv(self.train_data_path, index=False, header=True)
            test.to_csv(self.test_data_path, index=False, header=True)
            logging.info(f"Training dataset shape is {train.shape}; Test dataset shape is {test.shape}")
            logging.info("Training and test dataset saved in 'data' folder.")

            logging.info("Data ingestion is complete.")

            return self.train_data_path, self.test_data_path
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise


def parse_args() -> Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The namespace containing the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Data Ingestion for Vehicle Value Estimator")
    parser.add_argument("data_path", type=str, help="Path to the dataset file.")
    parser.add_argument("test_size", type=float, help="Size of the test dataset e.g. '0.2'.")
    return parser.parse_args()


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    # Create an instance of the DataIngestion class
    ingestion = DataIngestion()

    # Start data ingestion process
    try:
        train_data_path, test_data_path = ingestion.initiate_data_ingestion(args.data_path, args.test_size)
        print(f"Data ingestion complete. Training data: {train_data_path}, Test data: {test_data_path}")
    except Exception as e:
        logging.error(f"An error occurred during data ingestion: {e}")


if __name__ == "__main__":
    main()
