# 📚 NLP Kindle Review Classification End-to-End Deployment Project
For more details, check out the [project repository on GitHub](https://github.com/dannyyqyq/NLP_kindle_classification).

## 🚀 Web Application
Experience the Kindle Review Sentiment Analysis live!  
[Live Demo: Kindle Review Sentiment Analysis](https://kindle-review-ratings-classification-avgword2vec.streamlit.app/)

## 📌 Project Overview
This project leverages Natural Language Processing (NLP) to classify Kindle book reviews into positive or negative sentiments, aiding in understanding reader feedback, achieving up to 65% accuracy using AvgWord2Vec embeddings and a Random Forest Classifier model. The project covers:

- **Data Ingestion**: Extracting and preparing review data from CSV files for analysis.
- **Data Transformation**: Preprocessing text data for model training.
- **Model Training**: Training multiple NLP models to classify review sentiments.
- **Prediction Pipeline**: Applying trained models to predict sentiments on new reviews.
- **Web Application**: A Streamlit-based interface for users to input reviews and get sentiment predictions.
- **Deployment**: Deployed on Streamlit Cloud for live access, with optional Docker containerization for local testing or custom deployment setups.
  
## 🛠 Tools and Technologies Used

### 🚀 Deployment
- **Docker**: 
  - Employed for containerization, ensuring uniform deployment across environments.
- **Streamlit**: 
  - Provides a user-friendly web interface for real-time sentiment analysis.

### 📊 Machine Learning / NLP
- **Classification Models**: 
  - Random Forest, Naive Bayes (Gaussian), with feature extraction via Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), Word2Vec, and AvgWord2Vec.
- **Evaluation Metrics**: 
  The performance of different feature extraction techniques and classification models is summarized below:

  | Feature Extraction       | Classification Model    | Accuracy Score (%) |
  |--------------------------|-------------------------|---------------------|
  | AvgWord2Vec              | Random Forest           | 64.2%                |
  | BOW       | Naive Bayes             | 54.3%                |
  | TF-IDF | Naive Bayes | 54.8%                |
