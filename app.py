import streamlit as st
import pandas as pd
import os
from src.utils import load_object
from src.components.data_transformation import (
    DataTransformation,
)  # Assuming this is where your class is
from src.exception import CustomException

# Paths to model artifacts
MODEL_PATH = os.path.join("artifacts", "best_model.pkl")
WORD2VEC_MODEL_PATH = os.path.join("artifacts", "word2vec_model.pkl")


# Load the model and Word2Vec model
@st.cache_resource  # Cache the model to speed up app loading
def load_models():
    model = load_object(MODEL_PATH)
    w2v_model = load_object(WORD2VEC_MODEL_PATH)
    return model, w2v_model


model, w2v_model = load_models()

st.title("Kindle Review Sentiment Analysis")

# Text input for user review
review_text = st.text_area("Enter your review text here:", "")

if st.button("Analyze"):
    if review_text:
        # Use DataTransformation class for preprocessing
        data_transformer = DataTransformation()

        # Create a DataFrame with the review text but without rating for now
        df = pd.DataFrame(
            [{"reviewText": review_text, "rating": 0}]
        )  # Adding a dummy rating

        try:
            # Apply data transformation
            df_transformed = data_transformer.data_transformation(df.copy())
            df_transformed = data_transformer.lemmatizer_transformation(df_transformed)

            # Convert review to features using Word2Vec (assuming 'reviewText' column after preprocessing)
            words = [
                word
                for word in df_transformed["reviewText"].iloc[0].split()
                if word in w2v_model.wv.index_to_key
            ]
            if words:
                vector = w2v_model.wv[words].mean(axis=0)
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
