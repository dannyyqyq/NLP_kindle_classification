import streamlit as st
import pandas as pd
import os
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
import numpy as np
import nltk

# Prep downloads for docker
resources = ["stopwords", "wordnet"]
for resource in resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir="/usr/local/nltk_data")

# Paths to model artifacts
MODEL_PATH = os.path.join("artifacts", "best_model.pkl")
WORD2VEC_MODEL_PATH = os.path.join("artifacts", "word2vec_model.pkl")


# Load the model and Word2Vec model
@st.cache_resource
def load_models():
    model = load_object(MODEL_PATH)
    w2v_model = load_object(WORD2VEC_MODEL_PATH)
    return model, w2v_model


model, w2v_model = load_models()

st.title("Kindle Review Sentiment Analysis")

review_text = st.text_area("Enter your review text here:", "")

if st.button("Analyze"):
    if review_text:
        data_transformer = DataTransformation()

        df = pd.DataFrame([{"reviewText": review_text, "rating": 0}])  # Dummy rating

        try:
            df_transformed = data_transformer.data_transformation(df.copy())
            df_transformed = data_transformer.lemmatizer_transformation(df_transformed)

            # Using avgword2vec for feature conversion
            def avg_word2vec(model, doc):
                words = [word for word in doc.split() if word in model.wv.index_to_key]
                if not words:
                    return None  # Return None if no words are found
                return np.mean([model.wv[word] for word in words], axis=0)

            vector = avg_word2vec(w2v_model, df_transformed["reviewText"].iloc[0])

            if vector is not None:
                vector = vector.reshape(1, -1)  # Reshape for prediction
                prediction = model.predict(vector)

                sentiment = "Positive" if prediction[0] == 1 else "Negative"
                st.write(f"The sentiment of the review is: **{sentiment}**")
            else:
                st.write("No valid words in the review for analysis.")
        except CustomException as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please enter a review to analyze.")
