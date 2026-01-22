import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model/breast_cancer_model.pkl")

st.title("Breast Cancer Prediction System")
st.write("⚠️ Educational use only. Not a medical diagnostic tool.")

st.subheader("Enter Tumor Features")

radius_mean = st.number_input("Radius Mean", min_value=0.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
area_mean = st.number_input("Area Mean", min_value=0.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[radius_mean,
                             texture_mean,
                             perimeter_mean,
                             area_mean,
                             smoothness_mean]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔴 Prediction: Malignant")
    else:
        st.success("🟢 Prediction: Benign")
 
