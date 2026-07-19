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

LANG_MAP = {
    'en-US': 'eng', 'en-GB': 'eng', 'en': 'eng',
    'fr': 'fre', 'fr-FR': 'fre',
    'de': 'ger', 'de-DE': 'ger',
    'es': 'spa', 'es-ES': 'spa',
    'it': 'ita', 'it-IT': 'ita',
    'pt': 'por', 'pt-BR': 'por', 'pt-PT': 'por',
    'nl': 'dut', 'nl-NL': 'dut',
    'ja': 'jpn', 'ja-JP': 'jpn',
    'zh': 'chi', 'zh-CN': 'chi', 'zh-TW': 'chi',
    'ru': 'rus', 'ru-RU': 'rus',
    'ar': 'ara', 'ar-SA': 'ara',
    'ko': 'kor', 'ko-KR': 'kor',
    'pl': 'pol', 'pl-PL': 'pol',
    'sv': 'swe', 'sv-SE': 'swe',
    'tr': 'tur', 'tr-TR': 'tur',
    'vi': 'vie', 'vi-VN': 'vie',
    'no': 'nor', 'nb-NO': 'nor',
    'fi': 'fin', 'fi-FI': 'fin',
    'da': 'dan', 'da-DK': 'dan',
    'el': 'gre', 'el-GR': 'gre',
    'he': 'heb', 'he-IL': 'heb',
    'hu': 'hun', 'hu-HU': 'hun',
    'cs': 'cze', 'cs-CZ': 'cze',
    'ro': 'rum', 'ro-RO': 'rum',
    'uk': 'ukr', 'uk-UA': 'ukr',
    'th': 'tha', 'th-TH': 'tha',
    'id': 'ind', 'id-ID': 'ind',
    'ms': 'may', 'ms-MY': 'may',
    'tl': 'tgl', 'tl-PH': 'tgl',
    'hr': 'hrv', 'hr-HR': 'hrv',
    'sk': 'slo', 'sk-SK': 'slo',
    'sl': 'slv', 'sl-SI': 'slv',
    'et': 'est', 'et-EE': 'est',
    'lv': 'lav', 'lv-LV': 'lav',
    'lt': 'lit', 'lt-LT': 'lit',
    'bg': 'bul', 'bg-BG': 'bul',
    'sr': 'srp', 'sr-RS': 'srp',
    'ca': 'cat', 'ca-ES': 'cat',
    'af': 'afr', 'af-ZA': 'afr',
    'bn': 'ben', 'bn-BD': 'ben',
    'fa': 'per', 'fa-IR': 'per',
    'hi': 'hin', 'hi-IN': 'hin',
    'ur': 'urd', 'ur-PK': 'urd',
    'ta': 'tam', 'ta-IN': 'tam',
    'te': 'tel', 'te-IN': 'tel',
    'mr': 'mar', 'mr-IN': 'mar',
    'gu': 'guj', 'gu-IN': 'guj',
    'ml': 'mal', 'ml-IN': 'mal',
    'kn': 'kan', 'kn-IN': 'kan',
    'pa': 'pan', 'pa-IN': 'pan',
    'or': 'ori', 'or-IN': 'ori',
    'as': 'asm', 'as-IN': 'asm',
    'ne': 'nep', 'ne-NP': 'nep',
    'si': 'sin', 'si-LK': 'sin',
    'my': 'bur', 'my-MM': 'bur',
    'km': 'khm', 'km-KH': 'khm',
    'lo': 'lao', 'lo-LA': 'lao',
    'mn': 'mon', 'mn-MN': 'mon',
    'kk': 'kaz', 'kk-KZ': 'kaz',
    'uz': 'uzb', 'uz-UZ': 'uzb',
    'tk': 'tuk', 'tk-TM': 'tuk',
    'az': 'aze', 'az-AZ': 'aze',
    'ka': 'geo', 'ka-GE': 'geo',
    'hy': 'arm', 'hy-AM': 'arm',
    'am': 'amh', 'am-ET': 'amh',
    'sw': 'swa', 'sw-KE': 'swa',
    'zu': 'zul', 'zu-ZA': 'zul',
    'xh': 'xho', 'xh-ZA': 'xho',
    'st': 'sot', 'st-ZA': 'sot',
    'tn': 'tsn', 'tn-ZA': 'tsn',
    'ts': 'tso', 'ts-ZA': 'tso',
    've': 'ven', 've-ZA': 'ven',
    'nr': 'nbl', 'nr-ZA': 'nbl',
    'ss': 'ssw', 'ss-ZA': 'ssw',
    'ny': 'nya', 'ny-MW': 'nya',
    'sn': 'sna', 'sn-ZW': 'sna',
    'mg': 'mlg', 'mg-MG': 'mlg',
    'rw': 'kin', 'rw-RW': 'kin',
    'rn': 'run', 'rn-BI': 'run',
    'sg': 'sag', 'sg-CF': 'sag',
    'ln': 'lin', 'ln-CD': 'lin',
    'lg': 'lug', 'lg-UG': 'lug',
    'wo': 'wol', 'wo-SN': 'wol',
    'bm': 'bam', 'bm-ML': 'bam',
    'ff': 'ful', 'ff-SN': 'ful',
    'ha': 'hau', 'ha-NG': 'hau',
    'yo': 'yor', 'yo-NG': 'yor',
    'ig': 'ibo', 'ig-NG': 'ibo',
    'om': 'orm', 'om-ET': 'orm',
    'ti': 'tir', 'ti-ET': 'tir',
    'so': 'som', 'so-SO': 'som',
    'aa': 'aar', 'aa-ET': 'aar',
    'ab': 'abk', 'ab-GE': 'abk',
    'av': 'ava', 'av-RU': 'ava',
    'ae': 'ave', 'ae': 'ave',
    'ak': 'aka', 'ak-GH': 'aka',
    'an': 'arg', 'an-ES': 'arg',
    'ay': 'aym', 'ay-BO': 'aym',
    'ba': 'bak', 'ba-RU': 'bak',
    'be': 'bel', 'be-BY': 'bel',
    'bh': 'bih', 'bh-IN': 'bih',
    'bi': 'bis', 'bi-VU': 'bis',
    'br': 'bre', 'br-FR': 'bre',
    'ch': 'cha', 'ch-GU': 'cha',
    'co': 'cos', 'co-FR': 'cos',
    'cr': 'cre', 'cr-CA': 'cre',
    'cu': 'chu', 'cu-RU': 'chu',
    'cv': 'chv', 'cv-RU': 'chv',
    'cy': 'wel', 'cy-GB': 'wel',
    'dz': 'dzo', 'dz-BT': 'dzo',
    'ee': 'ewe', 'ee-GH': 'ewe',
    'eo': 'epo', 'eo': 'epo',
    'fj': 'fij', 'fj-FJ': 'fij',
    'fo': 'fao', 'fo-FO': 'fao',
    'fy': 'fry', 'fy-NL': 'fry',
    'ga': 'gle', 'ga-IE': 'gle',
    'gd': 'gla', 'gd-GB': 'gla',
    'gl': 'glg', 'gl-ES': 'glg',
    'gn': 'grn', 'gn-PY': 'grn',
    'gv': 'glv', 'gv-IM': 'glv',
    'ho': 'hmo', 'ho-PG': 'hmo',
    'ht': 'hat', 'ht-HT': 'hat',
    'hz': 'her', 'hz-NA': 'her',
    'ia': 'ina', 'ia': 'ina',
    'ie': 'ile', 'ie': 'ile',
    'ik': 'ipk', 'ik-US': 'ipk',
    'io': 'ido', 'io': 'ido',
    'iu': 'iku', 'iu-CA': 'iku',
    'jv': 'jav', 'jv-ID': 'jav',
    'kg': 'kon', 'kg-CD': 'kon',
    'ki': 'kik', 'ki-KE': 'kik',
    'kj': 'kua', 'kj-NA': 'kua',
    'kl': 'kal', 'kl-GL': 'kal',
    'kr': 'kau', 'kr-NG': 'kau',
    'ks': 'kas', 'ks-IN': 'kas',
    'kv': 'kom', 'kv-RU': 'kom',
    'kw': 'cor', 'kw-GB': 'cor',
    'ky': 'kir', 'ky-KG': 'kir',
    'lb': 'ltz', 'lb-LU': 'ltz',
    'li': 'lim', 'li-NL': 'lim',
    'lu': 'lub', 'lu-CD': 'lub',
    'mh': 'mah', 'mh-MH': 'mah',
    'mi': 'mri', 'mi-NZ': 'mri',
    'mt': 'mlt', 'mt-MT': 'mlt',
    'na': 'nau', 'na-NR': 'nau',
    'nb': 'nob', 'nb-NO': 'nob',
    'nd': 'nde', 'nd-ZW': 'nde',
    'ng': 'ndo', 'ng-NA': 'ndo',
    'nl': 'nld', 'nl-NL': 'nld',
    'nn': 'nno', 'nn-NO': 'nno',
    'nv': 'nav', 'nv-US': 'nav',
    'oc': 'oci', 'oc-FR': 'oci',
    'oj': 'oji', 'oj-CA': 'oji',
    'os': 'oss', 'os-RU': 'oss',
    'pi': 'pli', 'pi': 'pli',
    'ps': 'pus', 'ps-AF': 'pus',
    'qu': 'que', 'qu-PE': 'que',
    'rm': 'roh', 'rm-CH': 'roh',
    'ro': 'ron', 'ro-RO': 'ron',
    'sa': 'san', 'sa-IN': 'san',
    'sc': 'srd', 'sc-IT': 'srd',
    'sd': 'snd', 'sd-PK': 'snd',
    'se': 'sme', 'se-NO': 'sme',
    'sm': 'smo', 'sm-WS': 'smo',
    'sq': 'sqi', 'sq-AL': 'sqi',
    'su': 'sun', 'su-ID': 'sun',
    'tg': 'tgk', 'tg-TJ': 'tgk',
    'to': 'ton', 'to-TO': 'ton',
    'tt': 'tat', 'tt-RU': 'tat',
    'tw': 'twi', 'tw-GH': 'twi',
    'ty': 'tah', 'ty-PF': 'tah',
    'ug': 'uig', 'ug-CN': 'uig',
    'vo': 'vol', 'vo': 'vol',
    'wa': 'wln', 'wa-BE': 'wln',
    'yi': 'yid', 'yi': 'yid',
    'za': 'zha', 'za-CN': 'zha',
    'zh': 'zho', 'zh-CN': 'zho',
    'zu': 'zul', 'zu-ZA': 'zul',
}

def clean_dataframe(df):
    """Apply all cleaning steps to the raw dataframe. Returns cleaned df and summary dict."""
    initial_rows = len(df)
    df['publication_date_parsed'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df = df.dropna(subset=['publication_date_parsed']).copy()
    df['language_code'] = df['language_code'].map(LANG_MAP).fillna(df['language_code'])
    df['author_count'] = df['authors'].apply(lambda x: len(str(x).split('/')))
    df['primary_author'] = df['authors'].apply(lambda x: str(x).split('/')[0].strip() if pd.notna(x) else '')
    df = df.dropna(subset=['average_rating']).copy()
    cap_value = df['num_pages'].quantile(0.99)
    df['num_pages'] = df['num_pages'].clip(upper=cap_value)
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
            authors = st.text_input("Authors * (separate multiple with '/')", placeholder="e.g. F. Scott Fitzgerald")
            language_code = st.text_input("Language Code *", placeholder="e.g. eng, fre, ger")
            num_pages = st.number_input("Number of Pages", min_value=0, max_value=10000, value=300, step=1)
        with col2:
            ratings_count = st.number_input("Ratings Count", min_value=0, max_value=10000000, value=1000, step=100)
            text_reviews_count = st.number_input("Text Reviews Count", min_value=0, max_value=1000000, value=100, step=10)
            publication_date = st.text_input("Publication Date", placeholder="e.g. 2005 or 2005-06-15")
            publisher = st.text_input("Publisher *", placeholder="e.g. Penguin Classics")
        submitted = st.form_submit_button("Predict Rating")
        if submitted:
            if not title or not authors or not publisher or not language_code:
                st.error("Please fill all required fields marked with *.")
            else:
                input_data = pd.DataFrame([{
                    "bookID": 0, "title": title, "authors": authors, "average_rating": 0.0,
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
    expected = ["bookID","title","authors","average_rating","isbn","isbn13",
                "language_code","num_pages","ratings_count","text_reviews_count",
                "publication_date","publisher"]
    st.code(", ".join(expected))
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.columns = df_up.columns.str.strip()
            missing = [c for c in expected if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                if st.button("Predict Ratings"):
                    with st.spinner("Predicting..."):
                        preds = pipeline.predict(df_up)
                        df_up['predicted_rating'] = preds.round(2)
                        st.dataframe(df_up)
                        csv = df_up.to_csv(index=False)
                        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")