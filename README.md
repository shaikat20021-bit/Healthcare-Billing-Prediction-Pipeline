# Healthcare Billing Analytics and Predictive Modeling System

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-F7931E.svg)
![GitHub license](https://img.shields.io/github/license/shaikat20021-bit/medical_eda_project)

**Live Inference Endpoint:** https://medicaledaproject.streamlit.app/

## Abstract
This project presents an end-to-end Machine Learning pipeline and interactive dashboard designed to analyze and predict hospital billing costs. Utilizing a dataset of over 55,000 healthcare records, the system identifies clinical cost drivers and deploys a predictive model in a cloud-native environment. This work serves as a foundational case study in Applied Predictive Analytics, bridging the gap between exploratory data science and production-level MLOps.

## Authors & Contributions
This system was developed as a collaborative research project:
* **MD AL Sayeed Shaikat (Lead Data Engineer & MLOps)**: Architected the deployment pipeline, managed model serialization (Joblib), and engineered the dynamic feature alignment system to ensure stable production inference.
* **Suma Akter (Machine Learning Researcher)**: Led the Exploratory Data Analysis (EDA), executed feature engineering, and optimized the Random Forest Regressor for robust predictive accuracy.

## System Architecture & Methodology

### 1. Data Engineering & Preprocessing
* **Feature Alignment:** Addressed the common production challenge of one-hot encoding feature mismatch by implementing a dynamic `.reindex()` logic in the production application. This ensures that live user inputs map perfectly to the training data's feature space.
* **Pipeline Integration:** Cleaned and preprocessed raw patient demographics, medical conditions, and insurance provider data for model ingestion.

### 2. Machine Learning Model
* **Algorithm:** Scikit-Learn `RandomForestRegressor`.
* **Objective:** Predict continuous medical billing amounts based on categorical and numerical patient profiles.
* **Evaluation:** Model logic, hyperparameter considerations, and baseline metrics are documented within the core Jupyter Notebooks.

### 3. Deployment & User Interface
* **Framework:** Streamlit (deployed via Streamlit Community Cloud).
* **Functionality:** Provides a real-time, interactive UI allowing users to input patient metrics and receive immediate, data-driven cost estimations alongside clinical data visualizations.

### 4. Local Installation & Usage
To run the application locally for development or verification:
* Clone the repository: git clone [https://github.com/shaikat20021-bit/medical_eda_project.git](https://github.com/shaikat20021-bit/medical_eda_project.git)
cd medical_eda_project
* Initialize a virtual environment (Recommended): python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
* Install dependencies: pip install -r requirements.txt
* Execute the application: streamlit run app.py

### 5. Future Scope: Advanced Healthcare AI
This project establishes the baseline architecture for our ongoing research into advanced healthcare analytics. Future iterations of this work will transition from predictive modeling to prescriptive AI, specifically focusing on:
* Medical Knowledge Graphs (Neo4j): Mapping complex relationships between patient history, diagnoses, and treatment efficacy.
* Retrieval-Augmented Generation (RAG): Integrating large language models anchored by medical ontologies to mitigate hallucination in clinical decision support systems.

For inquiries regarding this research or the underlying codebase, please consult the repository authors.

## Repository Structure
```text
medical_eda_project/
├── data/                       # Contains the core healthcare_dataset.csv
├── app.py                      # Production Streamlit application and inference logic
├── medical_model.pkl           # Serialized Random Forest model weights
├── predictive_model.ipynb      # Complete ML training, validation, and evaluation pipeline
├── specialist_analysis.ipynb   # In-depth clinical trend analysis and EDA
└── requirements.txt            # Dependency configuration for cloud deployment

