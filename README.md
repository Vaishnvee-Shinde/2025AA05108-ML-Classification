# ML Assignment 2 - Classification Models

**Student Name:** Vaishnvee Subhash Shinde  
**BITS ID:** 2025AA05108  
**Email:** 2025aa05108@wilp.bits-pilani.ac.in

## a. Problem Statement

This assignment involves implementing multiple classification machine learning models to classify breast cancer tumors as Malignant or Benign using the Breast Cancer Wisconsin dataset. The goal is to build an interactive Streamlit web application that demonstrates these models and their performance metrics.

## b. Dataset Description

- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: UCI Machine Learning Repository / scikit-learn
- **Number of Features**: 12 (selected from 30 based on variance)
- **Number of Instances**: 569
- **Target Classes**: 2 (Malignant = 0, Benign = 1)
- **Feature Types**: All features are real-valued, computed from digitized images of cell nuclei

### Selected Features (Top 12 by Variance):
1. worst area
2. mean area
3. area error
4. worst perimeter
5. mean perimeter
6. worst texture
7. worst radius
8. mean texture
9. mean radius
10. perimeter error
11. texture error
12. radius error

### Class Distribution:
- Malignant: 212 samples (37.3%)
- Benign: 357 samples (62.7%)

## c. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9386 | 0.9927 | 0.9384 | 0.9386 | 0.9384 | 0.8676 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| kNN | 0.9561 | 0.9621 | 0.9569 | 0.9561 | 0.9558 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9706 | 0.9227 | 0.9211 | 0.9199 | 0.8299 |
| Random Forest (Ensemble) | 0.9386 | 0.9864 | 0.9390 | 0.9386 | 0.9387 | 0.8689 |
| XGBoost (Ensemble) | 0.9386 | 0.9888 | 0.9384 | 0.9386 | 0.9384 | 0.8676 |

### Observations about Model Performance

#### Logistic Regression
Logistic Regression achieves 93.86% accuracy with an excellent AUC of 0.9927. The model demonstrates strong linear separability in the feature space, making it highly effective for this classification task. The MCC score of 0.8676 indicates reliable predictions across both classes.

#### Decision Tree
Decision Tree shows the lowest performance among all models with 91.23% accuracy and AUC of 0.9157. This is expected as single decision trees are prone to overfitting. However, it still provides interpretable rules for classification and serves as a good baseline.

#### kNN
k-Nearest Neighbors achieves the highest accuracy (95.61%) and the best MCC score (0.9058). This indicates that the instance-based learning approach works well for this dataset. The high dimensional feature space with 12 selected features allows kNN to find similar neighbors effectively.

#### Naive Bayes
Gaussian Naive Bayes achieves 92.11% accuracy with a high AUC (0.9706). Despite the independence assumption being unrealistic for this dataset (many features are correlated), it still performs reasonably well. It's a fast model that works as a good baseline.

#### Random Forest (Ensemble)
Random Forest achieves 93.86% accuracy with excellent AUC (0.9864). The ensemble approach significantly improves over the single Decision Tree, reducing overfitting through bagging and feature randomness. The MCC score (0.8689) is the highest among all models.

#### XGBoost (Ensemble)
XGBoost matches Logistic Regression's performance with 93.86% accuracy and has the highest AUC (0.9888). The gradient boosting approach effectively captures complex non-linear patterns in the data. It's a powerful ensemble method that generalizes well.

### Overall Observations:
1. **Best Accuracy**: kNN achieves the highest accuracy (95.61%)
2. **Best AUC**: XGBoost has the highest AUC (0.9888)
3. **Best MCC**: Random Forest and kNN have the highest MCC scores
4. **Ensemble Methods**: Both Random Forest and XGBoost significantly outperform the Decision Tree
5. **Simple Models**: Even simple models like Logistic Regression and kNN achieve >93% accuracy
6. **All models** achieve AUC > 0.91, indicating good discriminative ability

## Deployment Instructions

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Community Cloud
1. Go to https://streamlit.io/cloud
2. Sign in using GitHub account
3. Click "New App"
4. Select your repository
5. Choose branch (usually main)
6. Select app.py
7. Click Deploy

## App Features

- ✅ Dataset upload option (CSV) - Upload test data for prediction
- ✅ Model selection dropdown - Choose from 6 classification models
- ✅ Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- ✅ Confusion matrix visualization
- ✅ Classification report
- ✅ Compare all models functionality - Visual comparison of all 6 models

## Files in Repository

```
project-folder/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   └── breast_cancer.csv  # Dataset (569 samples, 30 features + target)
└── model/
    ├── train_models.py    # Model training script
    ├── *.pkl              # Saved model files
    ├── scaler.pkl         # StandardScaler
    ├── selected_features.pkl
    └── evaluation_results.csv
```

## Requirements

- streamlit
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- xgboost
