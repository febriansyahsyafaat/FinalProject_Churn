import streamlit as st
import pandas as pd
import numpy as np
from model import ChurnPredictor
from preprocess import create_input_from_form

# Konfigurasi Halaman
st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Bank Customer Churn Prediction")

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

try:
    predictor = load_predictor()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR INPUT ---
st.sidebar.header("Customer Information")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_card = st.sidebar.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active = st.sidebar.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# --- TOMBOL PREDIKSI ---
if st.sidebar.button("Predict Churn", type="primary"):
    input_dict = create_input_from_form(
        credit_score, gender, age, tenure, balance,
        num_products, has_card, is_active, salary, geography
    )
    
    try:
        prediction, probability = predictor.predict(input_dict)
        
        # 1. Prediction & Risk Level
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("⚠️ Customer will CHURN")
            else:
                st.success("✅ Customer will NOT CHURN")
        
        with col2:
            st.subheader("Risk Level")
            if probability < 0.3:
                st.success("🟢 Low Risk")
            elif probability < 0.6:
                st.warning("🟡 Medium Risk")
            else:
                st.error("🔴 High Risk")

        st.markdown(f"### Churn Probability: {probability:.1%}")
        
        with st.expander("View Input Summary"):
            input_df = pd.DataFrame([{
                "Credit Score": credit_score, "Gender": gender, "Age": age,
                "Tenure": tenure, "Balance": f"${balance:,.2f}",
                "Products": num_products, "Has Card": "Yes" if has_card else "No",
                "Active": "Yes" if is_active else "No", "Salary": f"${salary:,.2f}",
                "Geography": geography
            }])
            st.dataframe(input_df, use_container_width=True)

        st.write("---")

        # 2. MODEL INSIGHTS (Correction Proof)
        st.header("📊 Model Insights & Deep EDA")
        col_feat, col_perf = st.columns([2, 1])
        
        with col_feat:
            st.subheader("💡 Feature Importance Analysis")
            try:
                importances = predictor.model.feature_importances_
                # 11 Fitur Sesuai Notebook lu
                feat_names = [
                    "Credit Score", "Gender", "Age", "Tenure", "Balance", 
                    "Num Products", "Has Card", "Is Active", "Salary", 
                    "Germany", "Spain"
                ]
                feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)
                st.bar_chart(feat_df.set_index('Feature'))
                st.caption("Faktor utama dari Best Model hasil Grid Search.")
            except Exception as e:
                st.info(f"Feature importance update needed: {e}")

        with col_perf:
            st.subheader("🏆 Model Performance")
            st.info("**Optimizer:** GridSearchCV")
            st.info("**Validation:** 5-Fold Stratified CV")
            st.success("**Status:** Optimized ✅")

        # 3. DEEP BUSINESS INSIGHTS
        st.write("---")
        st.subheader("🌍 Deep Business Insights")
        ins1, ins2, ins3 = st.columns(3)
        with ins1:
            st.markdown("🏙️ **Geography**")
            st.write("Germany memiliki churn rate tertinggi (~32%).")
        with ins2:
            st.markdown("⚖️ **Balance**")
            st.write("Saldo > $100k punya risiko churn lebih tinggi.")
        with ins3:
            st.markdown("👥 **Age Segment**")
            st.write("Rentang umur 45-60 tahun paling rentan churn.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.write("Final Project - Updated with Hyperparameter Tuning")