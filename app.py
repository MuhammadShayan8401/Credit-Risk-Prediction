import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Credit Risk System", layout="wide")

st.title("🏦 Credit Risk Prediction System")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    required_cols = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Loan_Status'
    ]

    df = df[required_cols]

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

    st.subheader("📊 Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if 'ApplicantIncome' in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df['ApplicantIncome'], bins=20)
            ax.set_title("Applicant Income Distribution")
            st.pyplot(fig)

    with col2:
        if 'Education' in df.columns and 'Loan_Status' in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='Education', hue='Loan_Status', data=df, ax=ax)
            ax.set_title("Education vs Loan Status")
            st.pyplot(fig)

    df_model = pd.get_dummies(df, drop_first=True)

    X = df_model.drop('Loan_Status_Y', axis=1)
    y = df_model['Loan_Status_Y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📌 Model Performance")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.write("Confusion Matrix")
    c2.write(cm)

    st.subheader("🔮 Loan Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    applicant_income = st.number_input("Applicant Income", 0)
    coapplicant_income = st.number_input("Coapplicant Income", 0)
    loan_amount = st.number_input("Loan Amount", 0)
    loan_term = st.number_input("Loan Term", 0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict Loan Status"):

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
        prob = model.predict_proba(input_scaled)[0]

        st.subheader("Result")

        if prediction[0] == 1:
            st.success(f"Loan Approved (Confidence: {prob[1]:.2f})")
        else:
            st.error(f"Loan Rejected (Confidence: {prob[0]:.2f})")