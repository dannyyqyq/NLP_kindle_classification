import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.utils import save_object
from gensim.models import Word2Vec
import pandas as pd


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        # self.train_data_file_path = os.path.join("artifacts", "train.csv")
        # self.test_data_file_path = os.path.join("artifacts", "test.csv")

    @staticmethod
    def avg_word2vec(model, doc):
        """
        Compute the average word2vec vector for the words in the document (sentence).
        """
        return np.mean(
            [model.wv[word] for word in doc if word in model.wv.index_to_key], axis=0
        )

    @staticmethod
    def training_dataset_Word2Vec(df, vector_size=100):
        try:
            # Initialize Word2Vec model with default parameters
            model = Word2Vec(vector_size=vector_size, window=5)

            # Process each review directly instead of splitting into sentences
            words = [simple_preprocess(review) for review in df["reviewText"]]

            model.build_vocab(words)
            model.train(words, total_examples=len(words), epochs=model.epochs)

            # Compute average word vector per review
            X = [
                ModelTrainer.avg_word2vec(model, review_words) for review_words in words
            ]

            df_latest = pd.DataFrame(X)
            df_final = pd.concat([df.reset_index(drop=True), df_latest], axis=1)

            return df_final, model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_data, test_data, vector_size=100):
        logging.info("Starting Model Trainer...")
        try:
            # Ensure that only the reviewText and rating columns are selected
            X_train = train_data.iloc[:, 0]
            y_train = train_data.iloc[:, 1]
            X_test = test_data.iloc[:, 0]
            y_test = test_data.iloc[:, 1]

            logging.info(f"Train data shape: {train_data.shape}")
            logging.info(f"Test data shape: {test_data.shape}")
            logging.info(f"X_train: {X_train.shape}")
            logging.info(f"y_train: {y_train.shape}")
            logging.info(f"X_test: {X_test.shape}")
            logging.info(f"y_test: {y_test.shape}")

            # Word2Vec training and feature generation
            (
                train_data_with_vecs,
                word2vec_model,
            ) = ModelTrainer.training_dataset_Word2Vec(train_data)
            test_data_with_vecs, _ = ModelTrainer.training_dataset_Word2Vec(test_data)

            logging.info(f"train_data_with_vecs: {train_data_with_vecs.shape}")
            logging.info(f"test_data_with_vecs: {test_data_with_vecs.shape}")

            # Only take the number of samples that we have labels for
            X_train_avg_word2vec = train_data_with_vecs.iloc[
                : len(y_train), -vector_size:
            ].values
            X_test_avg_word2vec = test_data_with_vecs.iloc[
                : len(y_test), -vector_size:
            ].values

            logging.info(f"X_train_avg_word2vec: {X_train_avg_word2vec.shape}")
            logging.info(f"y_train shape: {y_train.shape}")

            rfc = RandomForestClassifier()
            rfc.fit(X_train_avg_word2vec, y_train)
            y_pred_rfc = rfc.predict(X_test_avg_word2vec)

            # BOW and TF-IDF
            bow = CountVectorizer()
            X_train_bow = bow.fit_transform(X_train).toarray()
            X_test_bow = bow.transform(X_test).toarray()

            tfidf = TfidfVectorizer()
            X_train_tfidf = tfidf.fit_transform(X_train).toarray()
            X_test_tfidf = tfidf.transform(X_test).toarray()

            nb_model_bow = GaussianNB().fit(X_train_bow, y_train)
            nb_model_tfidf = GaussianNB().fit(X_train_tfidf, y_train)

            y_pred_bow = nb_model_bow.predict(X_test_bow)
            y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)

            accuracy_score_bow = accuracy_score(y_test, y_pred_bow)
            accuracy_score_tfidf = accuracy_score(y_test, y_pred_tfidf)
            accuracy_score_rfc = accuracy_score(y_test, y_pred_rfc)

            logging.info(f"BOW model accuracy: {accuracy_score_bow}")
            logging.info(f"TF-IDF model accuracy: {accuracy_score_tfidf}")
            logging.info(
                f"AvgWord2Vec model accuracy using random forest: {accuracy_score_rfc}"
            )

            # Compare accuracies and save the best model
            if accuracy_score_rfc > max(accuracy_score_bow, accuracy_score_tfidf):
                logging.info(
                    "Saving Word2Vec + RandomForest model as the best model..."
                )
                best_model = rfc  # Word2Vec + RandomForest is the best
                save_object(
                    "artifacts/word2vec_model.pkl", word2vec_model
                )  # Save Word2Vec model
                save_object(
                    "artifacts/best_model.pkl", best_model
                )  # Save RandomForest model

            else:
                # Save the best GaussianNB model if it's not using Word2Vec
                logging.info("Saving best GaussianNB model...")
                best_model = (
                    nb_model_bow
                    if accuracy_score_bow > accuracy_score_tfidf
                    else nb_model_tfidf
                )
                save_object(
                    "artifacts/best_model.pkl", best_model
                )  # Save GaussianNB model

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     import pandas as pd
#     model_trainer = ModelTrainer()
#     train_path = model_trainer.train_data_file_path
#     test_path = model_trainer.test_data_file_path
#     # Read the train and test datasets
#     train_data = pd.read_csv(train_path)
#     test_data = pd.read_csv(test_path)
#     # Call the function to initiate model training
#     model_trainer.initiate_model_trainer(train_data, test_data)
