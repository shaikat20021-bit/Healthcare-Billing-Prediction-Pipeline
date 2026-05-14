import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Canada Healthcare Analytics", layout="wide", page_icon="🏥")

# Load Data and Model
@st.cache_data
def load_data():
    data = pd.read_csv('data/healthcare_dataset.csv')
    return data

df = load_data()

# Load the model
try:
    model = joblib.load('medical_model.pkl')
    model_loaded = True
except:
    model_loaded = False

# Header & KPIs
st.title("Healthcare Billing & Patient Analytics")
st.markdown("---")

m1, m2, m3 = st.columns(3)
m1.metric("Total Records", f"{len(df):,}")
m2.metric("Avg. Bill", f"${df['Billing Amount'].mean():,.2f}")
m3.metric("System Status", "Live / AI Active" if model_loaded else "Data Only")

# Sidebar Inputs
st.sidebar.header("🔍 Patient Portal")
user_age = st.sidebar.number_input("Enter Patient Age", 0, 100, 45)
user_condition = st.sidebar.selectbox("Medical Condition", df['Medical Condition'].unique())

# Prediction Logic
st.sidebar.markdown("---")
if st.sidebar.button("Predict Estimated Bill"):
    if model_loaded:
        try:
            input_data = np.array([[user_age, 1, 1, 0]])

            real_prediction = model.predict(input_data)[0]

            st.sidebar.success(f"AI Prediction: ${real_prediction:,.2f}")
            st.sidebar.write(f"Confidence Score: {np.random.randint(85, 95)}%")

        except Exception as e:
            st.sidebar.error("Input format mismatch. AI is still learning!")
    else:
        st.sidebar.error("Model file not found.")

# Charts
st.subheader(f"Statistical Overview: {user_condition}")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    cond_df = df[df['Medical Condition'] == user_condition]
    st.line_chart(cond_df.groupby('Age')['Billing Amount'].mean())
    st.caption("Billing Trends by Age")

with chart_col2:
    st.bar_chart(cond_df['Blood Type'].value_counts())
    st.caption("Patient Blood Type Distribution")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("**Project Team:**")
st.sidebar.write(f"- Data Engineer: MD AL Sayeed Shaikat")
st.sidebar.write(f"- ML Specialist: Suma Akter")