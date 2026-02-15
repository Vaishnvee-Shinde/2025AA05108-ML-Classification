"""
ML Classification Models Streamlit App
This app demonstrates 6 classification models on the Breast Cancer Wisconsin dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
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
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Breast Cancer Classification Models")
st.markdown("""
This app demonstrates **6 classification models** on the Breast Cancer Wisconsin dataset.
The dataset contains **569 samples** with **30 features** for binary classification (Malignant/Benign).
""")

# Model names mapping
MODEL_NAMES = [
    'Logistic Regression',
    'Decision Tree',
    'kNN',
    'Naive Bayes',
    'Random Forest (Ensemble)',
    'XGBoost (Ensemble)'
]

@st.cache_data
def load_default_dataset():
    """Load the default Breast Cancer Wisconsin dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.target_names

def load_uploaded_dataset(uploaded_file):
    """Load dataset from uploaded CSV file"""
    df = pd.read_csv(uploaded_file)
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y, None

def prepare_data(X, y, test_size=0.2):
    """Split and scale the data"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(model_name, X_train, y_train):
    """Train a specific model"""
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == 'kNN':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
    elif model_name == 'Random Forest (Ensemble)':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost (Ensemble)':
        model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all 6 evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

# Sidebar
st.sidebar.header("⚙️ Settings")

# Dataset selection
st.sidebar.subheader("📁 Dataset")
data_source = st.sidebar.radio("Choose data source:", 
                               ["Use Default Dataset", "Upload CSV"])

X, y, target_names = None, None, None

if data_source == "Use Default Dataset":
    X, y, target_names = load_default_dataset()
    st.sidebar.info(f"Using Breast Cancer Wisconsin Dataset\n\nFeatures: {X.shape[1]}\nSamples: {X.shape[0]}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        X, y, target_names = load_uploaded_dataset(uploaded_file)
        if target_names is None:
            target_names = ['Class 0', 'Class 1']
        st.sidebar.success(f"Loaded: {uploaded_file.name}\n\nFeatures: {X.shape[1]}\nSamples: {X.shape[0]}")
    else:
        st.info("Please upload a CSV file or use the default dataset.")
        st.stop()

# Select features
st.sidebar.subheader("🔢 Feature Selection")
all_features = X.columns.tolist()
selected_features = st.sidebar.multiselect(
    "Select features (minimum 12):",
    all_features,
    default=all_features[:12] if len(all_features) >= 12 else all_features
)

if len(selected_features) < 12:
    st.warning(f"Please select at least 12 features. Currently selected: {len(selected_features)}")
    st.stop()

X_selected = X[selected_features]

# Train-test split ratio
test_size = st.sidebar.slider("Test set size:", 0.1, 0.4, 0.2)

# Main content
try:
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X_selected, y, test_size)
    
    # Model selection
    st.subheader("🤖 Model Selection")
    selected_model = st.selectbox("Choose a classification model:", MODEL_NAMES)
    
    # Train button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner(f"Training {selected_model}..."):
            # Train the model
            model = train_model(selected_model, X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            
            # Display metrics
            st.subheader(f"📊 {selected_model} Results")
            
            # Metrics in columns
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            with col2:
                st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
            with col3:
                st.metric("Precision", f"{metrics['Precision']:.4f}")
            with col4:
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            with col5:
                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
            with col6:
                st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
            
            # Confusion Matrix
            st.subheader("🔢 Confusion Matrix")
            cm_fig = plot_confusion_matrix(y_test, y_pred, target_names)
            st.pyplot(cm_fig)
            
            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, target_names=target_names)
            st.text(report)
            
            st.success(f"✅ {selected_model} trained and evaluated successfully!")
    
    # Compare all models button
    if st.button("📈 Compare All Models", type="secondary"):
        with st.spinner("Training all models..."):
            st.subheader("📊 Model Comparison")
            
            # Results storage
            all_results = []
            
            for model_name in MODEL_NAMES:
                model = train_model(model_name, X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                metrics['Model'] = model_name
                all_results.append(metrics)
            
            # Create comparison dataframe
            results_df = pd.DataFrame(all_results)
            results_df = results_df[['Model', 'Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']]
            
            # Display as table
            st.dataframe(results_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC Score': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'MCC Score': '{:.4f}'
            }), use_container_width=True)
            
            # Bar chart comparison
            st.subheader("📊 Performance Visualization")
            metrics_to_plot = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(MODEL_NAMES))
            width = 0.12
            
            for i, metric in enumerate(metrics_to_plot):
                values = results_df[metric].values
                ax.bar(x + i*width, values, width, label=metric)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Comparison - All Metrics')
            ax.set_xticks(x + width * 2.5)
            ax.set_xticklabels([m.replace(' (Ensemble)', '\n(Ensemble)') for m in MODEL_NAMES], rotation=0)
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success("✅ All models compared successfully!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure your dataset meets the requirements (minimum 12 features, 500 samples)")

# Footer
st.markdown("---")
st.markdown("""
**ML Assignment 2 - Classification Models**
- Student Name: Vaishnvee Subhash Shinde
- BITS ID: 2025AA05108
- Email: 2025aa05108@wilp.bits-pilani.ac.in

- Dataset: Breast Cancer Wisconsin (from sklearn)
- Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost
- Metrics: Accuracy, AUC, Precision, Recall, F1 Score, MCC Score
""")
