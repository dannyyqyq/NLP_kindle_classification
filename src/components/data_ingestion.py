import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    base_data_path: str = os.path.join("data", "all_kindle_review.csv")


class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion instance with a default configuration.

        Attributes:
            ingestion_config (DataIngestionConfig): Configuration object containing file paths for train, test, and raw data.
        """

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(self.ingestion_config.base_data_path)
            logging.info("Read the dataset as dataframe")

            # create artifact folder
            directory = os.path.dirname(self.ingestion_config.raw_data_path)
            os.makedirs(directory, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Shape of raw data: {df.shape}")
            logging.info("Data ingestion process completed successfully.")
            return df, self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e, sys)
