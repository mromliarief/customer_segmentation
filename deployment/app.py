from pathlib import Path
import pickle

import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "principal_component",
    "interest_component",
    "loan_amount",
    "term_months",
    "Age",
    "annual_income",
]

# Load saved model artifacts
features_path = MODEL_DIR / "features.pkl"
if features_path.exists():
    with open(features_path, "rb") as f:
        FEATURES = pickle.load(f)

with open(MODEL_DIR / "kmeans.pkl", "rb") as f:
    kmeans_seg = pickle.load(f)

with open(MODEL_DIR / "personas.pkl", "rb") as f:
    personas_seg = pickle.load(f)

SEGMENT_DESC = {
    0: "Middle Income Customers and Medium-Potential Performing Loans",
    1: "Low Income Customers and Risky-Potential Performing Loans",
    2: "High Income Customers and High-Potential Performing Loans",
}

# Optional: if you also save the scaler used during notebook training, Streamlit can use it here.
SCALER_PATH = MODEL_DIR / "scaler.pkl"
scaler = None
if SCALER_PATH.exists():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)


def predict_segment(principal_component, interest_component, loan_amount, term_months, age, annual_income):
    input_df = np.array(
        [[principal_component, interest_component, loan_amount, term_months, age, annual_income]],
        dtype=float,
    )

    # If the scaler is available, apply the same transformation used during training.
    if scaler is not None:
        input_df = scaler.transform(input_df)

    segment = int(kmeans_seg.predict(input_df)[0])
    persona = personas_seg.get(segment, "Unknown persona")
    return segment, persona


st.set_page_config(page_title="Loan Customer Segmentation", page_icon="🏦", layout="centered")
st.title("Loan Customer Segmentation Dashboard")
st.write("Enter the customer attributes below to predict the loan customer segment.")

with st.form("segment_form"):
    col1, col2 = st.columns(2)

    with col1:
        principal_component = st.number_input("Principal Component", value=0.0, step=0.1)
        interest_component = st.number_input("Interest Component", value=0.0, step=0.1)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=1000000.0, step=1000.0)

    with col2:
        term_months = st.number_input("Term Months", min_value=1, value=12, step=1)
        age = st.number_input("Age", min_value=18, value=30, step=1)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Segment")

if submitted:
    segment, persona = predict_segment(
        principal_component=principal_component,
        interest_component=interest_component,
        loan_amount=loan_amount,
        term_months=term_months,
        age=age,
        annual_income=annual_income,
    )

    st.success(f"Predicted Segment: {segment}")
    st.info(f"Persona: {persona}")
    st.write("Segment Description:")
    st.write(SEGMENT_DESC.get(segment, "Unknown segment"))

st.caption("This app expects the following saved model artifacts in the deployment/models folder:")
st.code("features.pkl\nkmeans.pkl\npersonas.pkl")
