#!/usr/bin/env python
"""
run_training.py
One‑file ML pipeline for Book Rating Prediction.
Now generates all evaluation plots (residual, pred vs actual, error band,
feature importance, SHAP) for each advanced model, plus the ensemble.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, json, joblib, warnings
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Paths & setup
# ------------------------------------------------------------
RAW_DATA = 'data/books.csv'
CLEANED_DATA = 'data/books_cleaned.csv'
ENGINEERED_DATA = 'data/books_engineered.csv'
FEATURE_NAMES_JSON = 'models/feature_names.json'
BEST_MODEL = 'models/best_model.joblib'
BEST_PIPELINE = 'models/best_pipeline.joblib'
TRAINING_CONFIG = 'models/training_config.json'
REPORT_DIR = 'reports'
FIG_DIR = os.path.join(REPORT_DIR, 'report_latex', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Data Cleaning (language map, date parse, author split, outlier cap)
# ------------------------------------------------------------

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


    summary = {
        'initial_rows': initial_rows,
        'final_rows': len(df),
        'rows_removed': initial_rows - len(df),
        'cap_value': cap_value
    }
    return df, summary

# ------------------------------------------------------------
# 2. Feature Engineering
# ------------------------------------------------------------
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

def engineer_features(df, target='average_rating', reference_year=2026):
    df = df.copy()
    meta = {'generated_features': [], 'target_encodings': {}, 'rare_thresholds': {}}
    df = add_title_features(df)
    meta['generated_features'] += ['title_length', 'title_word_count', 'has_exclamation']
    df = add_datetime_features(df, reference_year)
    meta['generated_features'] += ['publication_year', 'book_age', 'is_classic']
    df = add_numeric_transforms(df)
    meta['generated_features'] += ['log_ratings_count', 'log_text_reviews_count', 'reviews_per_rating']
    for col, thresh in [('publisher', 5), ('primary_author', 3), ('language_code', 3)]:
        df[col+'_grouped'] = group_rare_categories(df, col, threshold=thresh)
        meta['rare_thresholds'][col] = thresh
    for col in ['publisher_grouped', 'primary_author_grouped', 'language_code_grouped']:
        encoded, mapping = target_encode_smooth(df, col, target, alpha=10.0)
        te_col = col.replace('_grouped', '_te')
        df[te_col] = encoded
        meta['target_encodings'][col] = mapping
        meta['generated_features'].append(te_col)
    df.drop(['publisher_grouped', 'primary_author_grouped', 'language_code_grouped'], axis=1, errors='ignore')
    for col in ['authors', 'primary_author', 'publisher', 'language_code']:
        df.drop(col, axis=1, inplace=True, errors='ignore')
    df.drop(['publication_date', 'bookID', 'isbn', 'isbn13', 'title'], axis=1, inplace=True, errors='ignore')
    non_num = df.select_dtypes(include=['object','category','datetime64']).columns
    df.drop(columns=non_num, inplace=True, errors='ignore')
    return df, meta

def select_features(df, target='average_rating', k=None, use_rfe=True):
    X = df.drop(columns=[target])
    y = df[target]
    non_num = X.select_dtypes(include=['object','category','datetime64']).columns
    X.drop(columns=non_num, inplace=True, errors='ignore')
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    if use_rfe:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        estimator = Ridge(alpha=1.0)
        n_features_to_select = k if k else max(len(X.columns)//2, 1)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X_scaled, y)
        selected = [f for f, s in zip(X.columns, rfe.support_) if s]
    else:
        selected = list(X.columns)
    selected = [f for f in selected if mi_series[f] > 0.001]
    return selected

# ------------------------------------------------------------
# 3. Model evaluation helpers
# ------------------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
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

def deep_evaluation(model, X_test, y_test, model_name, selected_features, output_dir=FIG_DIR):
    """Generate per-model evaluation plots and save them."""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Residual plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residual Plot – {model_name}')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Residual')
    plt.savefig(os.path.join(output_dir, f'{model_name}_residual.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Predicted vs Actual
    plt.figure(figsize=(8,8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Predicted vs Actual – {model_name}')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.savefig(os.path.join(output_dir, f'{model_name}_pred_vs_actual.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Error by rating band
    bins = [0,2,3,4,5]
    labels = ['0-2','2-3','3-4','4-5']
    df_eval = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    df_eval['band'] = pd.cut(df_eval['actual'], bins=bins, labels=labels)
    error_band = df_eval.groupby('band').apply(lambda x: np.sqrt(mean_squared_error(x['actual'], x['predicted'])))
    plt.figure(figsize=(8,5))
    error_band.plot(kind='bar')
    plt.title(f'RMSE by Rating Band – {model_name}')
    plt.savefig(os.path.join(output_dir, f'{model_name}_error_band.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
        plt.figure(figsize=(10,6))
        imp.plot(kind='bar')
        plt.title(f'Feature Importance – {model_name}')
        plt.savefig(os.path.join(output_dir, f'{model_name}_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        imp.to_json(os.path.join(output_dir, f'{model_name}_importance.json'))

    # SHAP (if possible)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        np.save(os.path.join(output_dir, f'{model_name}_shap_values.npy'), shap_values)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f'SHAP Summary – {model_name}')
        plt.savefig(os.path.join(output_dir, f'{model_name}_shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")

# ------------------------------------------------------------
# 4. Advanced model tuning
# ------------------------------------------------------------
def build_advanced_models():
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

# ------------------------------------------------------------
# 5. Custom preprocessor for final pipeline
# ------------------------------------------------------------
class BookRatingPreprocessor(BaseEstimator, TransformerMixin):
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

# ------------------------------------------------------------
# 6. Main pipeline
# ------------------------------------------------------------
def main():
    print("Loading raw data...")
    df_raw = pd.read_csv(RAW_DATA, engine='python', on_bad_lines='warn')
    df_raw.columns = df_raw.columns.str.strip()

    # Clean
    print("Cleaning...")
    df_clean, clean_summary = clean_dataframe(df_raw)
    print(f"Cleaned rows: {len(df_clean)} (dropped {clean_summary['rows_removed']})")

    # Data quality summary
    with open(os.path.join(REPORT_DIR, 'data_quality_summary.txt'), 'w') as f:
        f.write(f"Initial rows: {clean_summary['initial_rows']}\n")
        f.write(f"Final rows after cleaning: {clean_summary['final_rows']}\n")
        f.write(f"Rows dropped: {clean_summary['rows_removed']}\n")
        f.write(f"num_pages capped at 99th percentile: {clean_summary['cap_value']:.1f}\n")

    # Engineer features
    print("Engineering features...")
    df_eng, meta = engineer_features(df_clean)
    selected = select_features(df_eng, target='average_rating')
    print(f"Selected features ({len(selected)}): {selected}")

    with open(os.path.join(REPORT_DIR, 'feature_engineering_summary.txt'), 'w') as f:
        f.write(f"Selected features: {', '.join(selected)}\n")
        f.write(f"Generated features (all): {', '.join(meta['generated_features'])}\n")

    with open(FEATURE_NAMES_JSON, 'w') as f:
        json.dump(selected, f)

    X = df_eng[selected]
    y = df_eng['average_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baselines
    print("Training baseline models...")
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
    df_base.to_csv(os.path.join(REPORT_DIR, 'baseline_model_comparison.csv'), index=False)
    with open(os.path.join(REPORT_DIR, 'baseline_summary.txt'), 'w') as f:
        f.write(df_base.to_string())

    # Advanced tuning
    print("Tuning advanced models...")
    advanced = build_advanced_models()
    adv_results = []
    best_models = {}
    for name, (model, param_grid) in advanced.items():
        print(f"  Tuning {name}...")
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

        # Deep evaluation for each advanced model
        print(f"    Generating evaluation plots for {name}...")
        deep_evaluation(best, X_test, y_test, name, selected)

    df_adv = pd.DataFrame(adv_results)
    df_adv.to_csv(os.path.join(REPORT_DIR, 'advanced_model_comparison.csv'), index=False)
    with open(os.path.join(REPORT_DIR, 'advanced_summary.txt'), 'w') as f:
        f.write(df_adv.to_string())

    # Ensemble
    print("Evaluating ensemble (RF + LightGBM)...")
    rf = best_models['RandomForest']
    lgbm = best_models['LightGBM']
    ens_pred = (rf.predict(X_test) + lgbm.predict(X_test)) / 2
    ens_rmse = np.sqrt(mean_squared_error(y_test, ens_pred))
    ens_mae = mean_absolute_error(y_test, ens_pred)
    ens_r2 = r2_score(y_test, ens_pred)
    print(f"Ensemble Test RMSE: {ens_rmse:.4f}, R²: {ens_r2:.4f}")

    # Choose final model
    if ens_rmse < min(m['Test RMSE'] for m in adv_results):
        final_model = type('Ensemble', (object,), {'predict': lambda self, X: (rf.predict(X)+lgbm.predict(X))/2})()
        final_name = 'Ensemble (RF+LGBM)'
    else:
        best_adv = min(adv_results, key=lambda x: x['Test RMSE'])
        final_name = best_adv['Model']
        final_model = best_models[final_name]

    print(f"Final model: {final_name}")

    # Final evaluation summary
    with open(os.path.join(REPORT_DIR, 'final_evaluation_summary.txt'), 'w') as f:
        f.write(f"Final model: {final_name}\n")
        f.write(f"Ensemble Test RMSE: {ens_rmse:.4f}, R²: {ens_r2:.4f}\n")
        f.write("Baseline results:\n")
        f.write(df_base.to_string())
        f.write("\nAdvanced results:\n")
        f.write(df_adv.to_string())

    # Save final model and pipeline
    joblib.dump(final_model, BEST_MODEL)
    preprocessor = BookRatingPreprocessor(feature_list_path=FEATURE_NAMES_JSON)
    preprocessor.fit(df_raw)
    full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', final_model)])
    joblib.dump(full_pipeline, BEST_PIPELINE)
    print("Pipeline saved.")

    # Overall correlation heatmap & residual plot for final model
    corr_df = df_clean[['average_rating','num_pages','ratings_count','text_reviews_count']].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(FIG_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("All done. Run 'streamlit run app.py' to start the web app.")

if __name__ == '__main__':
    main()