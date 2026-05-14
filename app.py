import streamlit as st
import pandas as pd

# Page Setup
st.set_page_config(page_title="Canada Healthcare App", layout="wide")

# Sidebar
st.sidebar.header("Patient Input Portal")
st.sidebar.markdown("Enter details to analyze billing patterns.")

# Input fields
user_age = st.sidebar.slider("Patient Age", 0, 100, 35)
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
user_condition = st.sidebar.selectbox("Condition", ["Cancer", "Diabetes", "Asthma", "Obesity", "Arthritis", "Hypertension"])

st.title("Healthcare Analytics & Billing Dashboard")
st.write(f"Showing analysis for a **{user_age}** year old patient with **{user_condition}**.")

# Load Data
@st.cache_data
def get_data():
    return pd.read_csv('data/healthcare_dataset.csv')

df = get_data()

# Main Dashboard Visuals
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Billing by Condition")
    avg_billing = df.groupby('Medical Condition')['Billing Amount'].mean().sort_values()
    st.bar_chart(avg_billing)

with col2:
    st.subheader("Age vs. Billing Distribution")
    st.scatter_chart(df.sample(500), x='Age', y='Billing Amount')

st.write("### Data Registry Preview")
st.dataframe(df.head(5))