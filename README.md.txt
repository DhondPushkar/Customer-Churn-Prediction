Customer Churn Prediction using Machine Learning
📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses.
This project builds a Machine Learning model to predict whether a customer will churn based on demographic and account-related features.

The goal is to help businesses identify high-risk customers and take preventive actions.

📊 Dataset Description

The dataset contains 100,000 customer records with the following features:

Age

Gender

Tenure

MonthlyCharges

TotalCharges

Contract Type

Payment Method

Churn (Target Variable)

Target Variable:

0 → No Churn

1 → Churn

🛠 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

Joblib

🔎 Project Workflow
1️⃣ Data Preprocessing

Converted TotalCharges to numeric

Handled missing values using median

Encoded categorical variables using one-hot encoding

Dropped unnecessary column (CustomerID)

2️⃣ Exploratory Data Analysis

Visualized churn distribution

Created correlation heatmap

Identified important numerical relationships

3️⃣ Feature Engineering

One-hot encoding for categorical variables

Train-test split (80/20)

4️⃣ Model Building

Implemented and compared:

Logistic Regression

Random Forest

XGBoost

5️⃣ Handling Class Imbalance

Applied class weights using compute_class_weight

Improved recall for churn class

6️⃣ Model Evaluation

Evaluated using:

Accuracy

ROC-AUC Score

Precision, Recall, F1-score

🏆 Best Model: XGBoost

Performance after handling imbalance:

Accuracy: 74%

ROC-AUC: 0.70

Recall (Churn class): 60%

XGBoost performed better compared to Logistic Regression and Random Forest.

📈 Key Insights

MonthlyCharges is the strongest predictor of churn

Customers with month-to-month contracts are more likely to churn

Higher monthly charges increase churn probability

Longer tenure reduces churn likelihood

💾 Model Saving

The trained model is saved as:

churn_model.pkl

This allows future predictions without retraining.

🚀 Future Improvements

Hyperparameter tuning

Cross-validation

SMOTE for imbalance handling

Streamlit deployment

Confusion matrix visualization

👨‍💻 Author

Pushkar Dhond
Aspiring Data Analyst / Machine Learning Engineer