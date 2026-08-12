#!/usr/bin/env python
"""
app.py
Streamlit web application for Book Rating Prediction.
Supports single‑book input and batch CSV upload.
Uses the serialised pipeline best_pipeline.joblib.

Note: The BookRatingPreprocessor and its helper functions are defined
here so that joblib can unpickle the pipeline correctly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------
# Helper functions needed by BookRatingPreprocessor (must match training)
# --------------------------------------------------------------------


def clean_dataframe(df):
    """Apply all cleaning steps to the raw dataframe."""
    initial_rows = len(df)

    df['publication_date_parsed'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df = df.dropna(subset=['publication_date_parsed']).copy()

    df["language_code"] = (df["language_code"].str.lower().replace({
        "en-us": "eng",
        "en-gb": "eng",
        "en-ca": "eng"}))
    top_langs = ["eng", "spa", "fre", "ger", "jpn"]
    df["language_code"] = df["language_code"].apply(
    lambda x: x if x in top_langs else "other")

    df['author_count'] = df['authors'].apply(lambda x: len(str(x).split('/')))
    df['primary_author'] = df['authors'].apply(lambda x: str(x).split('/')[0].strip() if pd.notna(x) else '')

    df.loc[df["average_rating"] == 0, "average_rating"] = np.nan
    df = df.dropna(subset=["average_rating"])

    df = df[~((df["average_rating"] > 0) &
        (df["ratings_count"] == 0))]

    cap_value = df['num_pages'].quantile(0.99)
    floor_value = df['num_pages'].quantile(0.01)
    df['num_pages'] = df['num_pages'].clip(lower=floor_value, upper=cap_value)

    df[df.duplicated(subset=["title", "authors"], keep=False)].sort_values(["title", "authors"])
    df = (df.sort_values("ratings_count", ascending=False).drop_duplicates(subset=["title", "authors"], keep="first"))
   
    df['publication_date'] = df['publication_date_parsed']
    df.drop('publication_date_parsed', axis=1, inplace=True, errors='ignore')
    df.drop('authors_clean', axis=1, inplace=True, errors='ignore')
    
    return df, {'cap_value': cap_value}

def add_title_features(df):
    df['title_length'] = df['title'].astype(str).str.len()
    df['title_word_count'] = df['title'].astype(str).str.split().str.len()
    df['has_exclamation'] = df['title'].astype(str).str.contains('!').astype(int)
    return df

def add_datetime_features(df, reference_year=2026):
    df['publication_year'] = df['publication_date'].dt.year
    df['book_age'] = reference_year - df['publication_year']
    df['is_classic'] = (df['book_age'] > 50).astype(int)
    return df

def add_numeric_transforms(df):
    df['log_ratings_count'] = np.log1p(df['ratings_count'])
    df['log_text_reviews_count'] = np.log1p(df['text_reviews_count'])
    df['reviews_per_rating'] = df['text_reviews_count'] / (df['ratings_count'] + 1)
    df.drop(columns=['ratings_count','text_reviews_count'], inplace=True, errors='ignore')
    return df

def group_rare_categories(df, column, threshold=5, other_name='Other'):
    counts = df[column].value_counts()
    rare = counts[counts <= threshold].index
    return df[column].apply(lambda x: other_name if x in rare else x)

def target_encode_smooth(df, column, target, alpha=10.0):
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(['sum', 'count'])
    encoded_vals = (agg['sum'] + alpha * global_mean) / (agg['count'] + alpha)
    mapping = encoded_vals.to_dict()
    return df[column].map(mapping), mapping

class BookRatingPreprocessor(BaseEstimator, TransformerMixin):
    """Wraps cleaning and feature engineering for inference."""
    def __init__(self, reference_year=2026, target='average_rating', feature_list_path=None):
        self.reference_year = reference_year
        self.target = target
        self.feature_list_path = feature_list_path
        self._metadata = None
        self._feature_names = None

    def fit(self, X, y=None):
        df = X.copy()
        df_clean, _ = clean_dataframe(df)
        df = add_title_features(df_clean)
        df = add_datetime_features(df, self.reference_year)
        df = add_numeric_transforms(df)
        for col, thresh in [('publisher', 5), ('primary_author', 3), ('language_code', 3)]:
            df[col+'_grouped'] = group_rare_categories(df, col, threshold=thresh)
        target_encodings = {}
        for col in ['publisher_grouped', 'primary_author_grouped', 'language_code_grouped']:
            encoded, mapping = target_encode_smooth(df, col, self.target, alpha=10.0)
            te_col = col.replace('_grouped', '_te')
            df[te_col] = encoded
            target_encodings[col] = mapping
        self._metadata = {'target_encodings': target_encodings}
        if self.feature_list_path and os.path.exists(self.feature_list_path):
            with open(self.feature_list_path) as f:
                self._feature_names = json.load(f)
        else:
            self._feature_names = [c for c in df.columns if c != self.target]
        return self

    def transform(self, X):
        df = X.copy()
        df_clean, _ = clean_dataframe(df)
        df = add_title_features(df_clean)
        df = add_datetime_features(df, self.reference_year)
        df = add_numeric_transforms(df)
        for col, thresh in [('publisher', 5), ('primary_author', 3), ('language_code', 3)]:
            df[col+'_grouped'] = group_rare_categories(df, col, threshold=thresh)
        for col in ['publisher_grouped', 'primary_author_grouped', 'language_code_grouped']:
            te_col = col.replace('_grouped', '_te')
            mapping = self._metadata['target_encodings'].get(col, {})
            df[te_col] = df[col].map(mapping)
            fill_val = np.mean(list(mapping.values())) if mapping else 0
            df[te_col].fillna(fill_val, inplace=True)
        df.drop(['publisher_grouped', 'primary_author_grouped', 'language_code_grouped'], axis=1, errors='ignore')
        for col in ['authors', 'primary_author', 'publisher', 'language_code']:
            df.drop(col, axis=1, inplace=True, errors='ignore')
        df.drop(['publication_date', 'bookID', 'isbn', 'isbn13', 'title'], axis=1, inplace=True, errors='ignore')
        non_num = df.select_dtypes(include=['object','category','datetime64']).columns
        df.drop(columns=non_num, inplace=True, errors='ignore')
        return df[self._feature_names]

# ---------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------
st.set_page_config(page_title="Book Rating Predictor", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.main-header {font-size: 2.5rem; font-weight: 600; color: #1E3A8A; margin-bottom: 0.5rem;}
.sub-header {font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem;}
.prediction-card {
    background: linear-gradient(135deg, #1E3A8A, #3B82F6);
    border-radius: 12px; padding: 1.5rem; color: white; text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0;
}
.prediction-value {font-size: 3rem; font-weight: 700; margin: 0.5rem 0;}
.stButton>button {background-color: #1E3A8A; color: white; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600;}
.stButton>button:hover {background-color: #2563EB; border: none;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    path = 'models/best_pipeline.joblib'
    if not os.path.exists(path):
        st.error("Model pipeline not found. Please run run_training.py first.")
        st.stop()
    return joblib.load(path)

pipeline = load_pipeline()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/books.png", width=80)
    st.markdown("## Book Rating Predictor")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("1. Choose **Single Book** to fill in details manually.")
    st.markdown("2. Choose **Upload CSV** to predict ratings for many books at once.")

st.markdown('<div class="main-header">📚 Book Rating Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict the average rating of a book based on its characteristics.</div>', unsafe_allow_html=True)

mode = st.radio("Select input mode:", ["Single Book", "Upload CSV"], horizontal=True)

if mode == "Single Book":
    st.markdown("### Enter Book Details")
    with st.form("single_book_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Title *", placeholder="e.g. The Great Gatsby")
            authors = st.text_input("Authors * (separate multiple with '/') *", placeholder="e.g. F. Scott Fitzgerald")
            language_code = st.text_input("Language Code *", placeholder="e.g. eng, fre, ger")
            num_pages = st.number_input("Number of Pages *", min_value=0, max_value=10000, value=300, step=1)
        with col2:
            ratings_count = st.number_input("Ratings Count *", min_value=0, max_value=10000000, value=1000, step=100)
            text_reviews_count = st.number_input("Text Reviews Count *", min_value=0, max_value=1000000, value=100, step=10)
            publication_date = st.text_input("Publication Date *", placeholder="e.g. 2005 or 2005-06-15")
            publisher = st.text_input("Publisher *", placeholder="e.g. Penguin Classics")
        submitted = st.form_submit_button("Predict Rating")
        if submitted:
            if not title or not authors or not publisher or not language_code or not publication_date or not num_pages or not ratings_count or not text_reviews_count:
                st.error("Please fill all required fields marked with *.")
            else:
                input_data = pd.DataFrame([{
                    "bookID": 0, "title": title, "authors": authors, "average_rating": 1.0, # 1.0 for average rating is a placeholder value so inference isn't removed by training-time cleaning.
                    "isbn": "", "isbn13": "", "language_code": language_code,
                    "num_pages": num_pages, "ratings_count": ratings_count,
                    "text_reviews_count": text_reviews_count,
                    "publication_date": publication_date, "publisher": publisher
                }])
                with st.spinner("Predicting..."):
                    try:
                        pred = pipeline.predict(input_data)[0]
                        star = "⭐" * int(round(pred, 1))
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-label">Predicted Rating</div>
                            <div class="prediction-value">{pred:.2f}</div>
                            <div style="font-size:1.5rem;">{star}</div>
                            <div style="opacity:0.8;">out of 5.0</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

else:
    st.markdown("### Upload a CSV File")
    st.markdown("Required columns:")
    expected = ["title","authors","isbn","isbn13",
                "language_code","num_pages","ratings_count","text_reviews_count",
                "publication_date","publisher"]
    st.code(", ".join(expected))
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.columns = df_up.columns.str.strip()
            df_up['average_rating'] = 1.0
            missing = [c for c in expected if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                if st.button("Predict Ratings"):
                    with st.spinner("Predicting..."):
                        preds = pipeline.predict(df_up)
                        df_up.drop(columns=['average_rating'], inplace=True, errors='ignore')
                        df_up['predicted_rating'] = preds.round(2)
                        st.dataframe(df_up)
                        csv = df_up.to_csv(index=False)
                        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")