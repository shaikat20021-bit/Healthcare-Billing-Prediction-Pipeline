# 🏥 Healthcare Billing Prediction & Patient Analytics

An interactive Machine Learning web application built with Streamlit that predicts hospital billing amounts based on patient demographics and medical profiles. 

## 📖 Project Overview
The goal of this project is to bridge the gap between Data Engineering and Machine Learning by providing a seamless, real-time prediction portal. By inputting patient details (Age, Gender, Medical Condition, and Insurance Provider), the underlying Random Forest Machine Learning model calculates an estimated hospital bill. The app also features interactive data visualizations to analyze billing trends across different medical conditions and blood types.

## ✨ Key Features
* **Real-time ML Predictions:** Uses a trained Scikit-Learn `RandomForestRegressor` to estimate medical costs instantly.
* **Robust Feature Alignment:** Dynamically aligns Streamlit user inputs with the model's exact expected features using Pandas `reindex`, preventing one-hot encoding feature mismatches.
* **Interactive Dashboard:** A clean, user-friendly sidebar for data entry.
* **Data Visualization:** Built-in charts displaying cost analysis by medical condition and hospital admission distributions by blood type.

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Random Forest Regressor)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Model Serialization:** Joblib
* **Version Control:** Git & GitHub

## 📂 Project Structure
```text
medical_eda_project/
│
├── data/
│   └── healthcare_dataset.csv    # Raw dataset used for training and EDA
├── app.py                        # Main Streamlit web application script
├── medical_model.pkl             # Trained Machine Learning model (Joblib format)
├── predictive_model.ipynb        # Jupyter Notebook with ML training and evaluation logic
├── requirements.txt              # List of Python dependencies
└── README.md                     # Project documentation

