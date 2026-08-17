import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")

# Load model if available
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

FEATURES = [
    'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    'Temp3pm', 'RainToday', 'Year', 'Month', 'Day'
]

LABELS = {0: 'NO (No Rain Expected)', 1: 'YES (Rain Expected)'}

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    form_values = {}

    if request.method == 'POST':
        try:
            raw_input = []
            for feature in FEATURES:
                val = request.form.get(feature, 0)
                form_values[feature] = val
                try:
                    raw_input.append(float(val))
                except ValueError:
                    raw_input.append(0.0)

            if model is not None:
                input_array = np.array(raw_input).reshape(1, -1)

                # Predict Class
                pred = model.predict(input_array)[0]
                prediction = LABELS.get(pred, 'Unknown')

                # Predict Probability if model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(input_array)[0]
                    confidence = probs[pred] * 100
                    probability = f"{confidence:.2f}%"
            else:
                prediction = "Model not loaded on server."

        except Exception as e:
            print(f"Prediction Error: {e}")
            prediction = f"Error during prediction: {e}"

    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability,
        features=FEATURES,
        form_values=form_values
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')