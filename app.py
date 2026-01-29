import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model/breast_cancer_model.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("Breast Cancer Prediction System")
st.write("Educational use only — not a medical diagnostic tool.")

st.subheader("Enter Tumor Features")

radius = st.number_input("Radius Mean", min_value=0.0)
texture = st.number_input("Texture Mean", min_value=0.0)
perimeter = st.number_input("Perimeter Mean", min_value=0.0)
area = st.number_input("Area Mean", min_value=0.0)
smoothness = st.number_input("Smoothness Mean", min_value=0.0)

if st.button("Predict"):
    features = np.array([[radius, texture, perimeter, area, smoothness]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")
