cat > README.md <<'ENDOFFILE'
# Bank Customer Churn Prediction

Final Project - Machine Learning untuk prediksi customer churn

## Project Overview

Aplikasi machine learning untuk memprediksi kemungkinan customer bank akan churn (berhenti menggunakan layanan) berdasarkan data historis customer. Project ini mencakup complete ML pipeline dari data preprocessing hingga deployment dengan web interface dan REST API

## Problem Statement

Bank mengalami kerugian signifikan akibat customer churn. Biaya akuisisi customer baru 5-7x lebih mahal dibanding mempertahankan customer existing. Dengan prediksi churn yang akurat, bank dapat melakukan proactive retention campaign untuk mengurangi churn rate

## Dataset

- **Source**        : Churn_Modelling.csv
- **Total Records** : 10,000 customers
- **Features**      : 11 features (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
- **Target**        : Exited (0 = Stay, 1 = Churn)

## Tech Stack

- **Language**              : Python 3.8+
- **ML Framework**          : Scikit-learn (Random Forest)
- **Data Processing**       : Pandas, NumPy
- **Imbalanced Data**       : SMOTE (imbalanced-learn)
- **Web Interface**         : Streamlit
- **API Framework**         : FastAPI
- **Model Serialization**   : Joblib

## Project Structure
- **data/**           : Dataset
- **models/**         : Trained model & scaler
- **notebooks/**      : Exploratory analysis & training
- **src/**            : Preprocessing, model, API, Streamlit app

## How to Run
- **Install dependencies**   :pip install -r requirements.txt
- **Run Streamlit app**      :streamlit run src/streamlit_app.py
- **Run API**                :uvicorn src.api:app --reload

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://bank-churn-prediction-febriansyah.streamlit.app)
**Live Demo:** [Klik di sini untuk mencoba aplikasi](https://bank-churn-prediction-febriansyah.streamlit.app)