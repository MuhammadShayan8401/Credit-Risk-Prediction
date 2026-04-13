# Credit Risk Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Objective
To predict whether a loan applicant is likely to default using machine learning and provide an interactive web-based dashboard for analysis and prediction.

---

## 🌐 Live Deployment
🚀 The application is deployed on Streamlit Cloud:

👉 **[Click here to open the app](https://credit-risk-prediction-clykmamzbui8x3takdrma5.streamlit.app/)**

---

## 📊 Dataset
The dataset includes applicant financial and personal details such as:
- Income
- Loan amount
- Education
- Credit history
- Employment status
- Loan approval status

---

## ⚙️ Approach
- Data cleaning and missing value handling  
- Exploratory Data Analysis (EDA) using visualizations  
- Feature encoding using one-hot encoding  
- Model training using Logistic Regression  
- Feature scaling using StandardScaler  
- Model evaluation using accuracy and confusion matrix  
- Feature importance analysis for explainability  
- Interactive UI built using Streamlit  

---

## 📈 Features of Dashboard
- Upload dataset (CSV)
- Interactive data visualizations
- Real-time loan prediction
- Prediction probability score
- Feature importance (model explainability)
- Loan decision insights

---

## 🤖 Results
- Achieved good classification accuracy
- Confusion matrix used for evaluation
- Model explains predictions using feature importance
- Identified key factors affecting loan approval:
  - Credit History
  - Income
  - Loan Amount
  - Property Area

---

## 🧠 Model Explanation
The model uses Logistic Regression, where each feature contributes positively or negatively to loan approval probability. Feature importance helps understand **why a prediction was made**.

---

## 🛠️ Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
