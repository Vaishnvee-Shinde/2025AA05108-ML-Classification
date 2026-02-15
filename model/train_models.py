"""
ML Classification Models Training Script
This script trains 6 classification models on the Breast Cancer Wisconsin dataset
and saves the trained models and evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import pickle
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

def load_dataset():
    """Load Breast Cancer Wisconsin dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Select top 12 features (by variance) to meet minimum requirement
    # Using all 30 features for better performance
    feature_variances = X.var().sort_values(ascending=False)
    selected_features = feature_variances.head(12).index.tolist()
    X_selected = X[selected_features]
    
    return X_selected, y, data.target_names, selected_features

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all 6 evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def train_models(X_train, X_test, y_train, y_test):
    """Train all 6 classification models"""
    
    # Dictionary to store models and their metrics
    results = {}
    models = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = calculate_metrics(y_test, y_pred_lr, y_prob_lr)
    models['Logistic Regression'] = lr
    
    # 2. Decision Tree Classifier
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    results['Decision Tree'] = calculate_metrics(y_test, y_pred_dt, y_prob_dt)
    models['Decision Tree'] = dt
    
    # 3. K-Nearest Neighbor
    print("Training kNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_prob_knn = knn.predict_proba(X_test)[:, 1]
    results['kNN'] = calculate_metrics(y_test, y_pred_knn, y_prob_knn)
    models['kNN'] = knn
    
    # 4. Naive Bayes (Gaussian)
    print("Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    y_prob_nb = nb.predict_proba(X_test)[:, 1]
    results['Naive Bayes'] = calculate_metrics(y_test, y_pred_nb, y_prob_nb)
    models['Naive Bayes'] = nb
    
    # 5. Random Forest (Ensemble)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    results['Random Forest (Ensemble)'] = calculate_metrics(y_test, y_pred_rf, y_prob_rf)
    models['Random Forest (Ensemble)'] = rf
    
    # 6. XGBoost (Ensemble)
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost (Ensemble)'] = calculate_metrics(y_test, y_pred_xgb, y_prob_xgb)
    models['XGBoost (Ensemble)'] = xgb
    
    return models, results

def get_predictions(model, X_test, y_test):
    """Get predictions and probabilities for a model"""
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
    return y_pred, y_prob

def main():
    print("Loading dataset...")
    X, y, target_names, selected_features = load_dataset()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features used: {selected_features}")
    print(f"Target classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\nTraining models...")
    models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save models and scaler
    print("\nSaving models...")
    for name, model in models.items():
        filename = name.replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
        with open(f'model/{filename}', 'wb') as f:
            pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model/evaluation_results.csv')
    
    # Print results
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("\n" + "="*80)
    
    # Save selected features
    with open('model/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    print("\nModels and results saved successfully!")
    return models, results, X_test_scaled, y_test, target_names

if __name__ == "__main__":
    models, results, X_test, y_test, target_names = main()
