from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("model/breast_cancer_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["radius_mean"]),
            float(request.form["texture_mean"]),
            float(request.form["perimeter_mean"]),
            float(request.form["area_mean"]),
            float(request.form["smoothness_mean"]),
        ]

        features = np.array(features).reshape(1, -1)
        result = model.predict(features)[0]

        prediction = "Malignant" if result == 1 else "Benign"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
