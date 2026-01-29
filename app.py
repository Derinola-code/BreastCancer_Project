import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open('model/breast_cancer_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['classifier']
    scaler = model_data['scaler']

st.set_page_config(page_title="Breast Cancer Predictor")

st.title("🩺 Breast Cancer Prediction System")
st.markdown("---")

st.write("### Input Tumor Features")
# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    radius = st.number_input("Radius Mean", min_value=0.0, format="%.4f")
    texture = st.number_input("Texture Mean", min_value=0.0, format="%.4f")
    perimeter = st.number_input("Perimeter Mean", min_value=0.0, format="%.4f")

with col2:
    area = st.number_input("Area Mean", min_value=0.0, format="%.4f")
    smoothness = st.number_input("Smoothness Mean", min_value=0.0, format="%.6f")

st.markdown("---")

if st.button("Run Diagnostic Prediction"):
    # 1. Organize input into a numpy array
    user_input = np.array([[radius, texture, perimeter, area, smoothness]])
    
    # 2. SCALE the input (Crucial Step!)
    user_input_scaled = scaler.transform(user_input)
    
    # 3. Predict
    prediction = model.predict(user_input_scaled)
    probability = model.predict_proba(user_input_scaled) if hasattr(model, "predict_proba") else None
    
    # 4. Show Results
    if prediction[0] == 1:
        st.error("### Result: MALIGNANT")
        st.write("The model suggests the tumor is likely cancerous.")
    else:
        st.success("### Result: BENIGN")
        st.write("The model suggests the tumor is non-cancerous.")

st.info("**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.")