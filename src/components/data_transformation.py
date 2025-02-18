import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    # preprocessor_file_path: str = os.path.join("artifacts", "Preprocessing.pkl")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.stop_words = set(stopwords.words("english"))
        self.url_pattern = re.compile(
            r"(http|https|ftp|ssh)://([\w_-]+(?:\.[\w_-]+)+)([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
        )
        self.special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s-]+")
        self.lemmatizer = WordNetLemmatizer()

    def data_transformation(self, df):
        try:
            df = df[["reviewText", "rating"]].copy()

            # Convert rating: 1 if rating >=3, else 0
            df["rating"] = (df["rating"] >= 3).astype(int)

            # Convert to lowercase
            df["reviewText"] = df["reviewText"].astype(str).str.lower()

            # Remove special characters using compiled regex
            df["reviewText"] = df["reviewText"].apply(
                lambda x: self.special_chars_pattern.sub("", x)
            )

            # Remove stopwords
            df["reviewText"] = df["reviewText"].apply(
                lambda x: " ".join(
                    word for word in x.split() if word not in self.stop_words
                )
            )

            # Remove URLs
            df["reviewText"] = df["reviewText"].apply(
                lambda x: self.url_pattern.sub("", x)
            )

            # Remove HTML tags
            df["reviewText"] = df["reviewText"].apply(
                lambda x: BeautifulSoup(x, "lxml").get_text()
            )

            # Remove extra spaces
            df["reviewText"] = df["reviewText"].str.split().str.join(" ")

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def lemmatize_words(self, text):
        """Lemmatizes a given text, handling NaN values gracefully."""
        try:
            if pd.isna(text):
                return ""  # Convert NaN to an empty string to avoid errors
            words = text.split()
            return " ".join(self.lemmatizer.lemmatize(word) for word in words)
        except Exception as e:
            raise CustomException(e, sys)

    def lemmatizer_transformation(self, df):
        try:
            df["reviewText"] = df["reviewText"].apply(self.lemmatize_words)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df):
        try:
            # Apply text preprocessing
            df = self.data_transformation(df)
            df = self.lemmatizer_transformation(df)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(
                self.data_transformation_config.train_data_path,
                index=False,
            )
            logging.info(f"Train dataframe shape: {train_set.shape}")
            test_set.to_csv(self.data_transformation_config.test_data_path, index=False)
            logging.info(f"Test dataframe shape: {test_set.shape}.")
            logging.info("Data ingestion process completed successfully.")

            return (
                train_set,
                test_set,
                self.data_transformation_config.train_data_path,
                self.data_transformation_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
