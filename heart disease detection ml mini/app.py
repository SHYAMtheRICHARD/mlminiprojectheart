from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        values = [float(x) for x in request.form.values()]
        scaled_values = scaler.transform([values])
        prediction = model.predict(scaled_values)[0]
        result = "❤️ Heart Disease Detected" if prediction == 1 else "💚 No Heart Disease"
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
