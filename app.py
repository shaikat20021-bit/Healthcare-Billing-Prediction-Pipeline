import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np


# --- 1. Load the Champion Model & Features ---
@st.cache_resource
def load_model():
    with open('medical_model.pkl', 'rb') as file:
        return pickle.load(file)


model_data = load_model()
model = model_data['model']
expected_features = model_data['features']


# --- 2. Load Data for SHAP and Insights ---
@st.cache_data
def load_full_data():
    try:
        return pd.read_csv('data/healthcare_dataset.csv')
    except:
        return pd.DataFrame()


df_full = load_full_data()


@st.cache_data
def get_shap_background():
    if not df_full.empty:
        features = ['Age', 'Gender', 'Medical Condition', 'Insurance Provider']
        X_sample = pd.get_dummies(df_full[features], drop_first=True).sample(100, random_state=42)
        X_sample = X_sample.reindex(columns=expected_features, fill_value=0)
        return X_sample
    else:
        return pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)


background_sample = get_shap_background()

# --- 3. App Layout ---
st.set_page_config(page_title="Medical AI Dashboard", layout="wide")

# Sidebar
st.sidebar.header("🔍 Patient Portal")
age = st.sidebar.slider("Patient Age", 18, 100, 45)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
condition = st.sidebar.selectbox("Medical Condition",
                                 ["Asthma", "Cancer", "Diabetes", "Hypertension", "Obesity", "Arthritis"], index=4)
insurance = st.sidebar.selectbox("Insurance Provider", ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"],
                                 index=3)

predict_btn = st.sidebar.button("Predict Estimated Bill")
prediction_placeholder = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.write("Project Team:")
st.sidebar.write("• Data Engineer: MD AL Sayeed Shaikat")
st.sidebar.write("• ML Specialist: Suma Akter")


# --- 4. Helper Function ---
def prepare_input(a, g, c, i):
    input_dict = {feature: 0 for feature in expected_features}
    input_dict['Age'] = a
    if f'Gender_{g}' in expected_features: input_dict[f'Gender_{g}'] = 1
    if f'Medical Condition_{c}' in expected_features: input_dict[f'Medical Condition_{c}'] = 1
    if f'Insurance Provider_{i}' in expected_features: input_dict[f'Insurance Provider_{i}'] = 1
    return pd.DataFrame([input_dict])


# --- 5. Main Logic ---
if predict_btn:
    # Prepare data
    baseline_input = prepare_input(age, gender, condition, insurance)
    prediction = model.predict(baseline_input)[0]

    # --- PHASE 1: TRUST & UNCERTAINTY UPDATE ---
    # Using the exact RMSE from the champion Linear Regression model
    model_rmse = 1504.12
    margin_of_error = 1.96 * model_rmse

    # Calculate bounds (ensuring the lower bound doesn't drop below $0)
    lower_bound = max(0, prediction - margin_of_error)
    upper_bound = prediction + margin_of_error

    # Show result in sidebar
    prediction_placeholder.success(f"AI Prediction: ${prediction:,.2f}")
    # MAIN DASHBOARD DISPLAY (Probabilistic Metric)
    st.header("🏥 Probabilistic Bill Estimate")

    col_metric, col_info = st.columns([1, 2])
    with col_metric:
        st.metric(label="Expected Base Cost", value=f"${prediction:,.2f}")

    with col_info:
        st.info(f"""
            Statistical Confidence Interval (95%)**\n
            Estimated Range: **${lower_bound:,.2f} — ${upper_bound:,.2f}**\n
            *Note: This AI model has an R-Squared of 0.957, meaning it is highly accurate, but standard medical billing variance accounts for a margin of error of ±${margin_of_error:,.2f}.*
            """)

    # --- SHAP EXPLAINER SECTION ---
    st.divider()
    st.header("🔍 Explainable AI: Why is the bill this amount?")

    explainer = shap.Explainer(model, background_sample)
    shap_values = explainer(baseline_input)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"Billing Influence: {condition} Case")
    st.pyplot(fig)

    # --- What-If Analysis ---
    st.divider()
    st.header("🔄 'What-If' Scenario Analysis")
    alt_insurance = st.selectbox("Compare with Alternative Insurance:",
                                 ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"], index=1)

    if alt_insurance != insurance:
        alt_input = prepare_input(age, gender, condition, alt_insurance)
        alt_pred = model.predict(alt_input)[0]
        diff = alt_pred - prediction

        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"Estimate with {alt_insurance}", value=f"${alt_pred:,.2f}", delta=f"${diff:,.2f}",
                      delta_color="inverse")
        with c2:
            if diff > 0:
                st.error(f"⚠️ Switching would cost **${abs(diff):,.2f} MORE.")
            else:
                st.success(f"✅ Switching would save ${abs(diff):,.2f}.")

    # --- 6. Data Insights / EDA Graphs ---
    st.divider()

    if not df_full.empty:
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader(f"Cost Analysis: {condition}")
            condition_data = df_full[df_full['Medical Condition'] == condition]
            age_cost = condition_data.groupby('Age')['Billing Amount'].mean()

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(age_cost.index, age_cost.values, color='#0055cc', linewidth=2)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig1)

        with col_chart2:
            st.subheader("Billing Distribution by Blood Type")
            blood_cost = df_full.groupby('Blood Type')['Billing Amount'].count()

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(blood_cost.index, blood_cost.values, color='#0066cc')
            plt.xticks(rotation=90)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
    else:
        st.warning("Ensure 'data/healthcare_dataset.csv' exists to generate Insight charts.")

else:
    st.info(
        "👈 Adjust patient details and click 'Predict Estimated Bill' to generate the XAI breakdown and Data Insights.")
