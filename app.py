from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Breast Cancer Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    features = np.array(data).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    result = "Malignant" if prediction == 1 else "Benign"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
