import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Canada Healthcare Analytics", layout="wide", page_icon="🏥")


# Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv('data/healthcare_dataset.csv')


df = load_data()

try:
    model = joblib.load('medical_model.pkl')
    model_loaded = True
except:
    model_loaded = False

# Header
st.title("🏥 Healthcare Billing & Patient Analytics")
st.markdown("---")

# Sidebar Inputs
st.sidebar.header("🔍 Patient Portal")
user_age = st.sidebar.slider("Patient Age", 0, 100, 45)
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
user_condition = st.sidebar.selectbox("Medical Condition",
                                      ["Cancer", "Diabetes", "Asthma", "Obesity", "Hypertension", "Arthritis"])
user_insurance = st.sidebar.selectbox("Insurance Provider",
                                      ["Blue Cross", "Cigna", "Medicare", "UnitedHealthcare", "Aetna"])

# Prediction Logic
st.sidebar.markdown("---")

if st.sidebar.button("Predict Estimated Bill"):
    if model_loaded:
        try:
            with st.spinner('Calculating...'):
                # Dictionary
                raw_data = {
                    'Age': user_age,
                    'Gender_Male': 1 if user_gender == "Male" else 0,
                    'Gender_Female': 1 if user_gender == "Female" else 0,
                    'Medical Condition_Cancer': 1 if user_condition == "Cancer" else 0,
                    'Medical Condition_Diabetes': 1 if user_condition == "Diabetes" else 0,
                    'Medical Condition_Asthma': 1 if user_condition == "Asthma" else 0,
                    'Medical Condition_Obesity': 1 if user_condition == "Obesity" else 0,
                    'Medical Condition_Hypertension': 1 if user_condition == "Hypertension" else 0,
                    'Medical Condition_Arthritis': 1 if user_condition == "Arthritis" else 0,
                    'Insurance Provider_Blue Cross': 1 if user_insurance == "Blue Cross" else 0,
                    'Insurance Provider_Cigna': 1 if user_insurance == "Cigna" else 0,
                    'Insurance Provider_Medicare': 1 if user_insurance == "Medicare" else 0,
                    'Insurance Provider_UnitedHealthcare': 1 if user_insurance == "UnitedHealthcare" else 0,
                    'Insurance Provider_Aetna': 1 if user_insurance == "Aetna" else 0
                }

                input_df = pd.DataFrame([raw_data])

                final_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

                # Predict
                prediction = model.predict(final_df)[0]

                st.sidebar.balloons()
                st.sidebar.success(f"AI Prediction: ${prediction:,.2f}")

        except Exception as e:
            st.sidebar.error("Error")
            st.sidebar.write(str(e))
    else:
        st.sidebar.error("Model file (.pkl) not found.")

# Charts & Visuals
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Cost Analysis: {user_condition}")
    cond_df = df[df['Medical Condition'] == user_condition]
    st.line_chart(cond_df.groupby('Age')['Billing Amount'].mean())

with col2:
    st.subheader("Billing Distribution by Blood Type")
    st.bar_chart(df['Blood Type'].value_counts())

# Team Credits
st.sidebar.markdown("---")
st.sidebar.write("**Project Team:**")
st.sidebar.write("- Data Engineer: MD AL Sayeed Shaikat")
st.sidebar.write("- ML Specialist: Suma Akter")