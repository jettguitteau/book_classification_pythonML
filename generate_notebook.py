#!/usr/bin/env python
"""
generate_notebook.py
Creates notebooks/complete_pipeline.ipynb – a step‑by‑step, fully documented
Jupyter notebook that runs the entire Book Rating Prediction pipeline.
"""

import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

# Helper to create cells
def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

cells = []

# 1. Title and intro
cells.append(md("""# Complete Book Rating Prediction Pipeline
This notebook walks through the full machine learning workflow:
1. Load and explore the raw data
2. Clean and preprocess
3. Engineer features and select the best ones
4. Train baseline and advanced models
5. Tune hyperparameters and evaluate
6. Interpret with SHAP and feature importance
7. Select the final model and export it for deployment

Every step explains **what** we do and **why** we do it."""))

# 2. Imports
cells.append(code("""# General imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, json, joblib, warnings
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

# Advanced models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_style('whitegrid')
print("Libraries loaded successfully.")"""))
cells.append(code("%cd .."))

# 3. Data loading and inspection
cells.append(md("## 1. Load and Inspect the Raw Data"))
cells.append(md("We load the dataset exactly as the standalone training script does – using the Python engine and warning on bad lines."))
cells.append(code("""# Load the dataset
df = pd.read_csv('data/books.csv', engine='python', on_bad_lines='warn')
df.columns = df.columns.str.strip()
print(f"Shape: {df.shape}")
df.head()"""))

cells.append(code("""# Basic statistics
df.describe(include='all')"""))

cells.append(code("""# Missing values
df.isnull().sum()"""))

cells.append(code("""# Distribution of the target variable
plt.figure(figsize=(8, 5))
df['average_rating'].hist(bins=40, edgecolor='black')
plt.title('Distribution of Average Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()"""))

# 4. Data cleaning
cells.append(md("## 2. Data Cleaning"))
cells.append(md("We need to: parse dates, standardise language codes, split authors, cap outliers, and drop rows with missing targets."))
cells.append(code("""# Language code mapping (ISO 639-2)
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
print("Language map defined.")"""))

cells.append(code("""def clean_dataframe(df):
    \"\"\"Apply all cleaning steps to the raw dataframe.\"\"\"
    initial_rows = len(df)
    # 1. Parse publication_date
    df['publication_date_parsed'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df = df.dropna(subset=['publication_date_parsed']).copy()
    # 2. Standardise language codes
    df['language_code'] = df['language_code'].map(LANG_MAP).fillna(df['language_code'])
    # 3. Split authors
    df['author_count'] = df['authors'].apply(lambda x: len(str(x).split('/')))
    df['primary_author'] = df['authors'].apply(lambda x: str(x).split('/')[0].strip() if pd.notna(x) else '')
    # 4. Drop rows with missing target
    df = df.dropna(subset=['average_rating']).copy()
    # 5. Cap num_pages at 99th percentile
    cap_value = df['num_pages'].quantile(0.99)
    df['num_pages'] = df['num_pages'].clip(upper=cap_value)
    # 6. Replace old publication_date with parsed datetime
    df['publication_date'] = df['publication_date_parsed']
    df.drop('publication_date_parsed', axis=1, inplace=True, errors='ignore')
    df.drop('authors_clean', axis=1, inplace=True, errors='ignore')
    summary = {
        'initial_rows': initial_rows,
        'final_rows': len(df),
        'failed_dates': initial_rows - len(df),
        'cap_value': cap_value
    }
    return df, summary

# Apply cleaning
df_clean, clean_summary = clean_dataframe(df)
print(f"Cleaned dataset: {df_clean.shape}")
print(f"Dropped {clean_summary['failed_dates']} rows due to unparseable dates.")
df_clean.head()"""))

# 5. Feature engineering
cells.append(md("## 3. Feature Engineering"))
cells.append(md("We create new features from the title, dates, numeric columns, and categorical variables."))
cells.append(code("""# Title features
def add_title_features(df):
    df['title_length'] = df['title'].astype(str).str.len()
    df['title_word_count'] = df['title'].astype(str).str.split().str.len()
    df['has_exclamation'] = df['title'].astype(str).str.contains('!').astype(int)
    return df

# Date features (reference year = 2026)
def add_datetime_features(df, reference_year=2026):
    df['publication_year'] = df['publication_date'].dt.year
    df['book_age'] = reference_year - df['publication_year']
    df['is_classic'] = (df['book_age'] > 50).astype(int)
    return df

# Numeric transforms
def add_numeric_transforms(df):
    df['log_ratings_count'] = np.log1p(df['ratings_count'])
    df['log_text_reviews_count'] = np.log1p(df['text_reviews_count'])
    df['reviews_per_rating'] = df['text_reviews_count'] / (df['ratings_count'] + 1)
    return df

# Group rare categories
def group_rare_categories(df, column, threshold=5, other_name='Other'):
    counts = df[column].value_counts()
    rare = counts[counts <= threshold].index
    return df[column].apply(lambda x: other_name if x in rare else x)

# Smoothed target encoding
def target_encode_smooth(df, column, target, alpha=10.0):
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(['sum', 'count'])
    encoded_vals = (agg['sum'] + alpha * global_mean) / (agg['count'] + alpha)
    mapping = encoded_vals.to_dict()
    return df[column].map(mapping), mapping

print("Feature engineering functions ready.")"""))

cells.append(code("""# Apply feature engineering
df_feat = df_clean.copy()
df_feat = add_title_features(df_feat)
df_feat = add_datetime_features(df_feat, 2026)
df_feat = add_numeric_transforms(df_feat)

# Group rare categories and target encode
for col, thresh in [('publisher', 5), ('primary_author', 3), ('language_code', 3)]:
    df_feat[col+'_grouped'] = group_rare_categories(df_feat, col, threshold=thresh)

encodings = {}
for col in ['publisher_grouped', 'primary_author_grouped', 'language_code_grouped']:
    encoded, mapping = target_encode_smooth(df_feat, col, 'average_rating', alpha=10.0)
    te_col = col.replace('_grouped', '_te')
    df_feat[te_col] = encoded
    encodings[col] = mapping

# Drop original raw columns that are now encoded
cols_to_drop = ['publisher_grouped', 'primary_author_grouped', 'language_code_grouped',
                'authors', 'primary_author', 'publisher', 'language_code',
                'publication_date', 'bookID', 'isbn', 'isbn13', 'title']
df_feat.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

# Ensure numeric only (just in case)
non_num = df_feat.select_dtypes(include=['object', 'category', 'datetime64']).columns
df_feat.drop(columns=non_num, inplace=True, errors='ignore')
print(f"Engineered feature matrix shape: {df_feat.shape}")
df_feat.head()"""))

# 6. Feature selection
cells.append(md("## 4. Feature Selection"))
cells.append(md("We use mutual information and Recursive Feature Elimination (RFE) to keep the most important features."))
cells.append(code("""X = df_feat.drop(columns=['average_rating'])
y = df_feat['average_rating']

# Mutual information
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# RFE with Ridge
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rfe_estimator = Ridge(alpha=1.0)
rfe = RFE(rfe_estimator, n_features_to_select=max(len(X.columns)//2, 1))
rfe.fit(X_scaled, y)
selected = [f for f, s in zip(X.columns, rfe.support_) if s and mi_series[f] > 0.001]
print(f"Selected features ({len(selected)}): {selected}")

# Keep only selected + target
df_final = df_feat[selected + ['average_rating']]
df_final.head()"""))

# 7. Train/test split
cells.append(md("## 5. Train/Test Split"))
cells.append(code("""X = df_final[selected]
y = df_final['average_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")"""))

# 8. Baseline models
cells.append(md("## 6. Baseline Models"))
cells.append(md("We train simple models (Dummy, Linear, Ridge, Lasso) to establish a performance floor."))
cells.append(code("""def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test)
    }

def cross_val_rmse(model, X, y, cv=5):
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        return -scores.mean(), scores.std()
    except:
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores)
        return rmse.mean(), rmse.std()

baselines = {
    'Dummy': DummyRegressor(strategy='mean'),
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0, max_iter=10000)
}
base_results = []
for name, model in baselines.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
    cv_mean, cv_std = cross_val_rmse(pipe, X, y)
    metrics['Model'] = name
    metrics['CV RMSE Mean'] = cv_mean
    base_results.append(metrics)

df_base = pd.DataFrame(base_results)
df_base.to_csv('reports/baseline_model_comparison.csv', index=False)
df_base"""))

# 9. Advanced model tuning
cells.append(md("## 7. Advanced Model Tuning"))
cells.append(md("We tune Random Forest, XGBoost, LightGBM, and CatBoost using RandomizedSearchCV with 5‑fold CV."))
cells.append(code("""def build_advanced_models():
    rf = RandomForestRegressor(random_state=42)
    xgb_reg = xgb.XGBRegressor(random_state=42, verbosity=0)
    lgb_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
    cb_reg = cb.CatBoostRegressor(random_state=42, verbose=0)
    grids = {
        'RandomForest': (rf, {
            'n_estimators': [100,200,300], 'max_depth': [5,10,15,None],
            'min_samples_leaf': [1,3,5], 'min_samples_split': [2,5,10]
        }),
        'XGBoost': (xgb_reg, {
            'n_estimators': [100,200,300], 'learning_rate': [0.01,0.05,0.1],
            'max_depth': [3,5,7,10], 'subsample': [0.7,0.8,1.0]
        }),
        'LightGBM': (lgb_reg, {
            'n_estimators': [100,200,300], 'learning_rate': [0.01,0.05,0.1],
            'max_depth': [3,5,7,-1], 'subsample': [0.7,0.8,1.0]
        }),
        'CatBoost': (cb_reg, {
            'iterations': [100,200,300], 'learning_rate': [0.01,0.05,0.1],
            'depth': [4,6,8,10]
        })
    }
    return grids

advanced = build_advanced_models()
adv_results = []
best_models = {}

for name, (model, param_grid) in advanced.items():
    print(f"Tuning {name}...")
    search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5,
                                scoring='neg_root_mean_squared_error',
                                random_state=42, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    metrics = evaluate_model(best, X_train, X_test, y_train, y_test)
    cv_rmse = -search.best_score_
    metrics['Model'] = name
    metrics['CV RMSE'] = cv_rmse
    metrics['Best Params'] = str(search.best_params_)
    adv_results.append(metrics)
    best_models[name] = best
    joblib.dump(best, f'models/model_{name.replace(" ","_").lower()}.joblib')
    print(f"  {name} CV RMSE: {cv_rmse:.4f}")

df_adv = pd.DataFrame(adv_results)
df_adv.to_csv('reports/advanced_model_comparison.csv', index=False)
df_adv"""))

# 10. Deep evaluation for each model
cells.append(md("## 8. Model Evaluation & Interpretation"))
cells.append(md("For each advanced model, we generate: residual plot, predicted vs actual, error by rating band, feature importance, and SHAP summary."))
cells.append(code("""def deep_evaluation(model, X_test, y_test, model_name, selected_features, output_dir='reports/report_latex/figures'):
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # 1. Residual plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residual Plot – {model_name}')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Residual')
    plt.savefig(os.path.join(output_dir, f'{model_name}_residual.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Predicted vs Actual
    plt.figure(figsize=(8,8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Predicted vs Actual – {model_name}')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.savefig(os.path.join(output_dir, f'{model_name}_pred_vs_actual.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Error by rating band
    bins = [0,2,3,4,5]
    labels = ['0-2','2-3','3-4','4-5']
    df_eval = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    df_eval['band'] = pd.cut(df_eval['actual'], bins=bins, labels=labels)
    error_band = df_eval.groupby('band').apply(lambda x: np.sqrt(mean_squared_error(x['actual'], x['predicted'])))
    plt.figure(figsize=(8,5))
    error_band.plot(kind='bar')
    plt.title(f'RMSE by Rating Band – {model_name}')
    plt.savefig(os.path.join(output_dir, f'{model_name}_error_band.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 4. Feature importance (if tree-based)
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
        plt.figure(figsize=(10,6))
        imp.plot(kind='bar')
        plt.title(f'Feature Importance – {model_name}')
        plt.savefig(os.path.join(output_dir, f'{model_name}_importance.png'), dpi=150, bbox_inches='tight')
        plt.show()
        imp.to_json(os.path.join(output_dir, f'{model_name}_importance.json'))

    # 5. SHAP values (if shap is installed)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        np.save(os.path.join(output_dir, f'{model_name}_shap_values.npy'), shap_values)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f'SHAP Summary – {model_name}')
        plt.savefig(os.path.join(output_dir, f'{model_name}_shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")

# Run deep evaluation for each tuned model
for name in best_models.keys():
    print(f"Evaluating {name}...")
    deep_evaluation(best_models[name], X_test, y_test, name, selected)
print("All evaluation plots generated.")"""))

# 11. Ensemble and final selection
cells.append(md("## 9. Ensemble & Final Model Selection"))
cells.append(md("We try a simple average of RandomForest and LightGBM, then pick the model with the lowest test RMSE."))
cells.append(code("""rf = best_models['RandomForest']
lgbm = best_models['LightGBM']
ens_pred = (rf.predict(X_test) + lgbm.predict(X_test)) / 2
ens_rmse = np.sqrt(mean_squared_error(y_test, ens_pred))
ens_mae = mean_absolute_error(y_test, ens_pred)
ens_r2 = r2_score(y_test, ens_pred)
print(f"Ensemble Test RMSE: {ens_rmse:.4f}, R²: {ens_r2:.4f}")

# Choose final model
best_adv = min(adv_results, key=lambda x: x['Test RMSE'])
if ens_rmse < best_adv['Test RMSE']:
    final_model = type('Ensemble', (object,), {'predict': lambda self, X: (rf.predict(X)+lgbm.predict(X))/2})()
    final_name = 'Ensemble (RF+LGBM)'
else:
    final_name = best_adv['Model']
    final_model = best_models[final_name]

print(f"Final model: {final_name}")
joblib.dump(final_model, 'models/best_model.joblib')"""))

# 12. Export final pipeline
cells.append(md("## 10. Export Full Pipeline for Deployment"))
cells.append(md("We wrap the preprocessing steps and the final model into a single sklearn Pipeline and save it with joblib."))
cells.append(code("""class BookRatingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, reference_year=2026, target='average_rating', feature_list_path='models/feature_names.json'):
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
        with open(self.feature_list_path) as f:
            self._feature_names = json.load(f)
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

# Save feature names used for final model
with open('models/feature_names.json', 'w') as f:
    json.dump(selected, f)

# Fit preprocessor on the original raw data
preprocessor = BookRatingPreprocessor()
preprocessor.fit(df)

# Build and save the full pipeline
full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', final_model)])
joblib.dump(full_pipeline, 'models/best_pipeline.joblib')
print("Full pipeline saved as models/best_pipeline.joblib")"""))

# 13. Reports and summary
cells.append(md("## 11. Generate Summary Reports"))
cells.append(code("""# Write summary files
with open('reports/data_quality_summary.txt', 'w') as f:
    f.write(f"Initial rows: {clean_summary['initial_rows']}\\n")
    f.write(f"Final rows after cleaning: {clean_summary['final_rows']}\\n")
    f.write(f"Rows dropped (unparseable dates): {clean_summary['failed_dates']}\\n")
    f.write(f"num_pages capped at 99th percentile: {clean_summary['cap_value']:.1f}\\n")

with open('reports/feature_engineering_summary.txt', 'w') as f:
    f.write(f"Selected features: {', '.join(selected)}\\n")

with open('reports/final_evaluation_summary.txt', 'w') as f:
    f.write(f"Final model: {final_name}\\n")
    f.write(f"Ensemble Test RMSE: {ens_rmse:.4f}, R²: {ens_r2:.4f}\\n")
    f.write("Baseline results:\\n")
    f.write(df_base.to_string())
    f.write("\\nAdvanced results:\\n")
    f.write(df_adv.to_string())

# Correlation heatmap for the report
corr_df = df_clean[['average_rating','num_pages','ratings_count','text_reviews_count']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('reports/report_latex/figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("All reports and plots have been saved.")
print("Notebook completed. You can now use the trained model in the Streamlit app.")"""))

# Assemble the notebook
nb.cells = cells
os.makedirs('notebooks', exist_ok=True)
nbf.write(nb, 'notebooks/complete_pipeline.ipynb')
print("Successfully generated notebooks/complete_pipeline.ipynb")

