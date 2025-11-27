import streamlit as st
import pandas as pd
from model import ChurnPredictor
from preprocess import create_input_from_form

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Bank Customer Churn Prediction")

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

try:
    predictor = load_predictor()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

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

if st.sidebar.button("Predict Churn", type="primary"):
    input_dict = create_input_from_form(
        credit_score, gender, age, tenure, balance,
        num_products, has_card, is_active, salary, geography
    )
    try:
        prediction, probability = predictor.predict(input_dict)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("Customer will CHURN")
                st.markdown(f"### Churn Probability: {probability:.1%}")
            else:
                st.success("Customer will NOT CHURN")
                st.markdown(f"### Churn Probability: {probability:.1%}")
        with col2:
            st.subheader("Risk Level")
            if probability < 0.3:
                st.success("Low Risk")
            elif probability < 0.6:
                st.warning("Medium Risk")
            else:
                st.error("High Risk")
        st.subheader("Input Summary")
        input_df = pd.DataFrame([{
            "Credit Score": credit_score,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": f"${balance:,.2f}",
            "Products": num_products,
            "Has Card": "Yes" if has_card else "No",
            "Active": "Yes" if is_active else "No",
            "Salary": f"${salary:,.2f}",
            "Geography": geography
        }])
        st.dataframe(input_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown("Final Project - Bank Churn Prediction")
