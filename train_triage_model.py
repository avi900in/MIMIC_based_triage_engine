import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.calibration import CalibratedClassifierCV
from nlp_utils import SentenceTransformerWrapper
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, brier_score_loss, accuracy_score
import joblib

def load_data(data_dir):
    print("Loading data...")
    triage = pd.read_csv(f"{data_dir}/triage.csv")
    edstays = pd.read_csv(f"{data_dir}/edstays.csv")
    
    # Merge triage and edstays
    df = pd.merge(triage, edstays[['stay_id', 'disposition', 'arrival_transport']], on='stay_id', how='inner')
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Target variable: acuity is 1 (highest) to 5 (lowest). We can drop missing acuity.
    df = df.dropna(subset=['acuity'])
    
    # Let's group acuity into 2 classes for a simpler, calibrated decision tree: 
    # High Risk (Acuity 1-2) vs Standard Risk (Acuity 3-5)
    df['high_risk'] = df['acuity'].apply(lambda x: 1 if x in [1.0, 2.0] else 0)
    
    # Features
    numeric_features = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    text_features = 'chiefcomplaint'
    
    # Fill missing text
    df[text_features] = df[text_features].fillna("unknown")
    
    # Convert pain to numeric if it's not (sometimes recorded as text/strings)
    df['pain'] = pd.to_numeric(df['pain'], errors='coerce')
    
    X = df[numeric_features + [text_features]]
    y = df['high_risk']
    
    return X, y, numeric_features, text_features

def build_model(numeric_features, text_features):
    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for text data (Chief Complaint)
    text_transformer = Pipeline(steps=[
        ('embedding', SentenceTransformerWrapper(model_name='all-MiniLM-L6-v2'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('text', text_transformer, text_features)
        ])

    # Base Decision Tree classifier (Deterministic)
    # We restrict depth to keep it interpretable as a standard decision tree
    base_tree = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')

    # Calibrated Classifier to provide calibrated confidence levels
    calibrated_tree = CalibratedClassifierCV(estimator=base_tree, method='sigmoid', cv=5)

    # Full Pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', calibrated_tree)])
    return clf

def main():
    data_dir = "mimic-iv-ed-demo-2.2 synthetic/files"
    
    # 1. Load Data
    df = load_data(data_dir)
    
    # 2. Preprocess
    X, y, numeric_features, text_features = preprocess_data(df)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Build Model
    model = build_model(numeric_features, text_features)
    
    # 5. Train Model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report (High Risk = Acuity 1 or 2):")
    print(classification_report(y_test, y_pred))
    
    print(f"Brier Score Loss (Calibration metric, closer to 0 is better): {brier_score_loss(y_test, y_proba):.4f}")
    
    # 7. Save Model
    joblib.dump(model, 'calibrated_triage_tree.pkl')
    print("\nModel saved to 'calibrated_triage_tree.pkl'")

    # Extract the base tree from one of the calibrated cross-validation estimators 
    # to show the deterministic rules.
    fitted_tree = model.named_steps['classifier'].calibrated_classifiers_[0].estimator
    
    # Get feature names after transformation
    embedding_features = [f'emb_{i}' for i in range(384)] # all-MiniLM-L6-v2 output dimension
    all_feature_names = numeric_features + embedding_features
    
    tree_rules = export_text(fitted_tree, feature_names=all_feature_names)
    print("\nDeterministic Decision Tree Rules (from one CV fold):")
    print(tree_rules)

if __name__ == "__main__":
    main()
