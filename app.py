import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("Credit Risk Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.hist(df['LoanAmount'], bins=20)
        ax1.set_title("Loan Amount Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Education', hue='Loan_Status', data=df, ax=ax2)
        ax2.set_title("Education vs Loan Status")
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(df['ApplicantIncome'], bins=20)
    ax3.set_title("Applicant Income Distribution")
    st.pyplot(fig3)

    df = df.drop('Loan_ID', axis=1)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Loan_Status_Y', axis=1)
    y = df['Loan_Status_Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.metric("Accuracy", f"{acc:.2f}")

    st.subheader("Confusion Matrix")
    st.write(cm)

    # ---------------- FEATURE IMPORTANCE ----------------
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.coef_[0]
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Feature Importance (Model Insight)")

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.barh(feature_importance["Feature"], feature_importance["Importance"])
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)

    st.subheader("What Influences Loan Decision?")

    for _, row in feature_importance.head(5).iterrows():
        impact = "increases approval chance" if row["Importance"] > 0 else "increases rejection chance"
        st.write(f"• {row['Feature']} → {impact}")

    # ---------------- PREDICTION ----------------
    st.subheader("Make Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict"):

        input_dict = {
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Gender_Male': 1 if gender == "Male" else 0,
            'Married_Yes': 1 if married == "Yes" else 0,
            'Dependents_1': 1 if dependents == "1" else 0,
            'Dependents_2': 1 if dependents == "2" else 0,
            'Dependents_3+': 1 if dependents == "3+" else 0,
            'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
            'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
            'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
            'Property_Area_Urban': 1 if property_area == "Urban" else 0
        }

        input_df = pd.DataFrame([input_dict])

        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[X.columns]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]

        st.subheader("Result")

        if prediction[0] == 1:
            st.success(f"Loan Approved (Confidence: {probability[1]:.2f})")
        else:
            st.error(f"Loan Rejected (Confidence: {probability[0]:.2f})")