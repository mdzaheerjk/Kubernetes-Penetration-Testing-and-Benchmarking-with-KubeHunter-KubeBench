import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

model = joblib.load(MODEL_PATH)

FEATURES = [
    'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    'Temp3pm', 'RainToday', 'Year', 'Month', 'Day'
]

LABELS = {0: 'NO', 1: 'YES'}

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            raw_input = []
            for feature in FEATURES:
                val = request.form.get(feature, 0)
                try:
                    raw_input.append(float(val))
                except ValueError:
                    raw_input.append(0.0)

            input_array = np.array(raw_input).reshape(1, -1)

            # Predict Class
            pred = model.predict(input_array)[0]
            prediction = LABELS.get(pred, 'Unknown')

            # Predict Probability (Confidence of positive class / YES)
            probs = model.predict_proba(input_array)[0]
            confidence = probs[pred] * 100  # Probability of predicted class as %
            probability = f"{confidence:.2f}%"

        except Exception as e:
            print(f"Prediction Error: {e}")

    return render_template(
        'index.html', 
        prediction=prediction, 
        probability=probability, 
        features=FEATURES
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')