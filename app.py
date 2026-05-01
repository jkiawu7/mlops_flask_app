# Import libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Home route
@app.route("/")
def home():
    return "Flask ML App is running successfully!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    feature_value = data["feature"]

    input_data = pd.DataFrame([[feature_value]], columns=["feature"])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "input": feature_value,
        "prediction": prediction
    })

# Run app
if __name__ == "__main__":
    app.run(debug=True)