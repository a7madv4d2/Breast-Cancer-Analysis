# Breast Cancer Classification Project

---

## **Overview**
This project focuses on building an efficient machine learning pipeline to classify breast cancer cases as **benign** or **malignant** using **dimensionality reduction techniques** and **supervised learning models**. The project uses the Wisconsin Breast Cancer dataset and aims to balance high accuracy with computational efficiency.

---

## **Objectives**
- Perform **dimensionality reduction** using both **PCA** (Principal Component Analysis) and **LDA** (Linear Discriminant Analysis).
- Train and evaluate multiple machine learning models for classification, including:
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
  - **Random Forest**
  - **K-Nearest Neighbors (KNN)**
  - **Gradient Boosting**
- Compare model performance based on key metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **AUC (Area Under the Curve)**
- Provide insights and recommendations based on the results.

---

## **Key Results**
- **Dimensionality Reduction**:
  - PCA reduced the dataset from 30 features to **6 principal components**, retaining 90% of the variance.
  - LDA maximized class separability, achieving high accuracy with just **1 linear discriminant**.
- **Best-Performing Model**:
  - **Gradient Boosting** achieved the highest mean accuracy (96.65%) and perfect AUC (1.00) on PCA-transformed data.
  - Logistic Regression on LDA-transformed data achieved **98% accuracy** with an AUC of 1.00.
- **Evaluation Metrics**:
  - All models demonstrated high accuracy (>95%) and low error rates, indicating robust performance and effective separability between classes.

---

## **Project Workflow**
1. **Data Preprocessing**:
   - Handled missing values and encoded the target variable.
   - Standardized the features for optimal model performance.

2. **Dimensionality Reduction**:
   - Applied **PCA** to reduce dimensionality and visualize variance contribution.
   - Implemented **LDA** to project data into a supervised lower-dimensional space.

3. **Model Training and Evaluation**:
   - Trained Logistic Regression, SVM, Random Forest, KNN, and Gradient Boosting on PCA-transformed data.
   - Used K-Fold Cross Validation for robust evaluation of model performance.
   - Evaluated Logistic Regression on LDA-transformed data for direct comparison.

4. **Performance Comparison**:
   - Summarized model performance in tables and visualized results using bar charts and ROC curves.

5. **Insights and Recommendations**:
   - Identified Gradient Boosting and LDA as the most effective approaches.
   - Highlighted potential areas for improvement through hyperparameter tuning and external validation.

---

## **Technologies Used**
- **Python**
- Libraries:
  - `pandas`, `numpy` (Data processing)
  - `matplotlib`, `seaborn` (Data visualization)
  - `scikit-learn` (Dimensionality reduction, machine learning models)
  - `xgboost` (Gradient Boosting)
  
---
