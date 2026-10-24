import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nlp_utils import SentenceTransformerWrapper
import joblib
from sklearn.metrics import classification_report

def train_triagegeist():
    data_dir = 'triagegeist'
    
    print("Loading datasets...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    complaints_df = pd.read_csv(f"{data_dir}/chief_complaints.csv")
    history_df = pd.read_csv(f"{data_dir}/patient_history.csv")
    
    print("Merging datasets...")
    complaints_df = complaints_df.drop(columns=['chief_complaint_system'], errors='ignore')
    df = train_df.merge(complaints_df, on='patient_id', how='left')
    df = df.merge(history_df, on='patient_id', how='left')
    
    print("Preprocessing data...")
    # Drop rows without target
    df = df.dropna(subset=['triage_acuity'])
    
    def get_age_group(age):
        if pd.isna(age): return 'unknown'
        if age < 3: return 'baby'
        elif age < 13: return 'toddler'
        elif age < 18: return 'teen'
        elif age < 65: return 'adult'
        else: return 'senior'
        
    df['age_group'] = df['age'].apply(get_age_group)
    
    # Target (Class 1-5, mapping to 0-4 for classifier if needed, but HistGradient handles arbitrary)
    # We will map Acuity 1-2 as High Risk (1), and 3-5 as Standard Risk (0) to match the other models, 
    # or keep all 5 classes. Let's do High Risk vs Standard Risk for consistency in UI.
    y = df['triage_acuity'].apply(lambda x: 1 if x in [1.0, 2.0] else 0)
    
    # Define features
    numeric_features = [
        'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure', 'pulse_pressure', 
        'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total', 
        'pain_score', 'shock_index', 'news2_score', 'bmi', 'age', 
        'num_prior_ed_visits_12m', 'num_prior_admissions_12m', 'num_active_medications', 'num_comorbidities'
    ]
    
    categorical_features = [
        'sex', 'arrival_mode', 'arrival_season', 'mental_status_triage', 'chief_complaint_system', 'age_group'
    ]
    
    text_feature = 'chief_complaint_raw'
    
    # Find all hx_ columns
    history_cols = [col for col in history_df.columns if col.startswith('hx_')]
    
    # Fill missing text
    if text_feature not in df.columns:
        # Check actual column name in complaints
        text_cols = [c for c in complaints_df.columns if c != 'patient_id']
        if len(text_cols) > 0:
            text_feature = text_cols[0]
            
    df[text_feature] = df[text_feature].fillna("unknown")
    
    # Fill numeric/history missing for modeling (HistGradientBoosting handles NaNs naturally, but sklearn Pipeline needs careful setup. 
    # Let's explicitly impute to be safe)
    
    X = df[numeric_features + categorical_features + history_cols + [text_feature]]
    
    # We will sample 20,000 for training to keep it fast for this demo, or use all 80k if it's fast enough.
    # We will use all 80k.
    
    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    text_transformer = Pipeline(steps=[
        ('embedding', SentenceTransformerWrapper(model_name='all-MiniLM-L6-v2'))
    ])
    
    history_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('hist', history_transformer, history_cols),
            ('text', text_transformer, text_feature)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42, max_iter=100))
    ])
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training TriageGeist model...")
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("Saving model...")
    joblib.dump(model, 'triagegeist_model.pkl')
    print("Saved to triagegeist_model.pkl")

if __name__ == "__main__":
    train_triagegeist()
