# Binary Breast Cancer Classification using Machine Learning Models


**ML Assignment 2 – BITS WILP**

**Student Name:** Vaishnvee Subhash Shinde 

**BITS ID:** 2025AA05108

**Email:** 2025aa05108@wilp.bits-pilani.ac.in

## 🔗 Submission Links
**GitHub Repository:** [https://github.com/Vaishnvee-Shinde/2025AA05108-ML-Classification](https://github.com/)  
**Live Streamlit Application:** [https://2025aa05108-ml-classification-breast-cancer.streamlit.app/](https://2025aa05108-ml-classification-breast-cancer.streamlit.app/)

---

## 1️⃣ Problem Statement

This assignment involves implementing multiple classification machine learning models to classify breast cancer tumors as Malignant or Benign using the Breast Cancer Wisconsin dataset. The goal is to build an interactive Streamlit web application that demonstrates these models and their performance metrics.

The project demonstrates a complete end-to-end machine learning workflow including:

- Data preprocessing and feature selection
- Model training
- Model evaluation
- Comparative analysis
- Web application deployment

---

## 2️⃣ Dataset Description

- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: UCI Machine Learning Repository / scikit-learn
- **Number of Instances**: 569
- **Number of Features**: 12 (selected from 30 based on variance)
- **Target Variable**: diagnosis (Malignant/Benign)
- **Classification Type**: Binary Classification

### Target Classes
- 0 → Malignant
- 1 → Benign

### Feature Types
The dataset contains real-valued features computed from digitized images of cell nuclei.

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

### Preprocessing Steps
The following preprocessing steps were applied:

- Feature selection using Variance Threshold (Top 12 features)
- Standardization of numerical features using StandardScaler
- Stratified train-test split (80-20)

All preprocessing steps were integrated into a Scikit-learn Pipeline to ensure consistency during training and inference.

---

## 3️⃣ Models Implemented

The following six classification models were implemented and evaluated on the same dataset:

1. **Logistic Regression** - Linear classification model
2. **Decision Tree Classifier** - Tree-based model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Gaussian Naive Bayes** - Probabilistic classifier
5. **Random Forest** - Ensemble (Bagging)
6. **XGBoost** - Ensemble (Boosting)

Each model was trained using identical preprocessing and evaluated on the same test dataset to ensure fair comparison.

---

## 4️⃣ Evaluation Metrics

The following performance metrics were computed for each model:

- **Accuracy** - Overall correctness of predictions
- **Precision** - Positive predictive value
- **Recall** - Sensitivity or true positive rate
- **F1 Score** - Harmonic mean of precision and recall
- **Matthews Correlation Coefficient (MCC)** - Balanced measure for binary classification
- **Area Under ROC Curve (AUC)** - Discriminative ability

These metrics provide a comprehensive assessment of classification performance.

---

## 5️⃣ Model Performance Comparison

| Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9386 | 0.9927 | 0.9384 | 0.9386 | 0.9384 | 0.8676 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| kNN | 0.9561 | 0.9621 | 0.9569 | 0.9561 | 0.9558 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9706 | 0.9227 | 0.9211 | 0.9199 | 0.8299 |
| Random Forest | 0.9386 | 0.9864 | 0.9390 | 0.9386 | 0.9387 | 0.8689 |
| XGBoost | 0.9386 | 0.9888 | 0.9384 | 0.9386 | 0.9384 | 0.8676 |

---

## 6️⃣ Observations on Model Performance

### Logistic Regression
Logistic Regression achieves 93.86% accuracy with an excellent AUC of 0.9927. The model demonstrates strong linear separability in the feature space, making it highly effective for this classification task. The MCC score of 0.8676 indicates reliable predictions across both classes.

### Decision Tree
Decision Tree shows the lowest performance among all models with 91.23% accuracy and AUC of 0.9157. This is expected as single decision trees are prone to overfitting. However, it still provides interpretable rules for classification and serves as a good baseline.

### K-Nearest Neighbors
KNN achieves the highest accuracy (95.61%) and the best MCC score (0.9058). This indicates that the instance-based learning approach works well for this dataset. The high dimensional feature space with 12 selected features allows kNN to find similar neighbors effectively.

### Naive Bayes
Gaussian Naive Bayes achieves 92.11% accuracy with a high AUC (0.9706). Despite the independence assumption being unrealistic for this dataset (many features are correlated), it still performs reasonably well. It's a fast model that works as a good baseline.

### Random Forest
Random Forest achieves 93.86% accuracy with excellent AUC (0.9864). The ensemble approach significantly improves over the single Decision Tree, reducing overfitting through bagging and feature randomness. The MCC score (0.8689) is the highest among all models.

### XGBoost
XGBoost matches Logistic Regression's performance with 93.86% accuracy and has the highest AUC (0.9888). The gradient boosting approach effectively captures complex non-linear patterns in the data. It's a powerful ensemble method that generalizes well.

### Overall Observations:
1. **Best Accuracy**: kNN achieves the highest accuracy (95.61%)
2. **Best AUC**: XGBoost has the highest AUC (0.9888)
3. **Best MCC**: Random Forest has the highest MCC score (0.8689)
4. **Ensemble Methods**: Both Random Forest and XGBoost significantly outperform the Decision Tree
5. **Simple Models**: Even simple models like Logistic Regression and kNN achieve >93% accuracy
6. **All models** achieve AUC > 0.91, indicating good discriminative ability

| ML Model Name | 	Observation about model performance |
|------------|-------------------------------------------|
| Logistic Regression |	Achieves 93.86% accuracy with excellent AUC (0.9927). Performs well due to strong linear separability in the dataset and provides stable, balanced predictions across both classes.
| Decision Tree	 | Shows the lowest performance (91.23% accuracy, AUC 0.9157). Prone to overfitting but provides clear and interpretable classification rules, making it a useful baseline model.
| kNN	| Achieves the highest accuracy (95.61%) and strong MCC (0.9058). Instance-based learning works effectively with scaled features and captures similarity patterns well.
| Naive Bayes	| Achieves 92.11% accuracy with high AUC (0.9706). Despite unrealistic independence assumptions, it performs reasonably well and serves as a fast, simple baseline model.
| Random Forest (Ensemble)	| Achieves 93.86% accuracy with excellent AUC (0.9864). Reduces overfitting compared to Decision Tree through bagging and feature randomness, providing strong overall performance.
| XGBoost (Ensemble)	| Matches 93.86% accuracy and achieves the highest AUC (0.9888). Gradient boosting captures complex nonlinear patterns and generalizes well.


---

## 7️⃣ Streamlit Application Features

The deployed Streamlit application provides:

- ✅ Dataset upload option (CSV) - Upload test data for prediction
- ✅ Model selection dropdown - Choose from 6 classification models
- ✅ Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- ✅ Confusion matrix visualization
- ✅ Classification report
- ✅ Compare all models functionality - Visual comparison of all 6 models

The application enables interactive model evaluation in a user-friendly interface.

---

## 8️⃣ Project Structure

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

---

## 9️⃣ Technologies Used

- **Python** - Programming language
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Streamlit** - Web application framework

---

## 🔟 Conclusion

This project successfully demonstrates an end-to-end machine learning workflow from data preprocessing and feature selection to model development, evaluation, and deployment.

Among all implemented models, kNN achieved the highest accuracy (95.61%), while XGBoost achieved the highest AUC (0.9888). Ensemble methods such as Random Forest and XGBoost demonstrated superior performance due to their ability to capture complex patterns and reduce variance.

The Streamlit deployment enhances practical usability by allowing real-time model evaluation through a web interface. The application enables users to upload their own test data, select different models, and visualize performance metrics including confusion matrices and classification reports.




