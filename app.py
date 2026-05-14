import streamlit as st
import pandas as pd

# Page Setup
st.set_page_config(page_title="Canada Healthcare App", layout="wide")

# Sidebar
st.sidebar.header("🏥 Patient Input Portal")
st.sidebar.markdown("Enter details to analyze billing patterns.")

# Input fields
user_age = st.sidebar.slider("Patient Age", 0, 100, 35)
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
user_condition = st.sidebar.selectbox("Condition", ["Cancer", "Diabetes", "Asthma", "Obesity", "Arthritis", "Hypertension"])

st.title("Healthcare Analytics & Billing Dashboard")
st.write(f"Showing analysis for a **{user_age}** year old patient with **{user_condition}**.")