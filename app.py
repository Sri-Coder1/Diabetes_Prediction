# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    status = "ready" if model else "not ready"
    return jsonify({"status": status, "message": "Diabetes Prediction API"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not trained yet. Please train the model first."}), 500
    
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodpressure']),
            float(data['skinthickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetespedigree']),
            float(data['age'])
        ]
        
        # Convert to numpy array and scale
        input_array = np.array([features])
        scaled_data = scaler.transform(input_array)
        
        # Make prediction
        probability = model.predict_proba(scaled_data)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        # Risk assessment
        if probability < 0.3:
            risk = "Low"
            message = "Low diabetes risk. Maintain healthy lifestyle!"
            color = "green"
        elif probability < 0.7:
            risk = "Medium"
            message = "Moderate diabetes risk. Monitor your health regularly."
            color = "orange"
        else:
            risk = "High"
            message = "High diabetes risk. Please consult a doctor."
            color = "red"
        
        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "probability": round(float(probability) * 100, 2),
            "risk_level": risk,
            "message": message,
            "color": color
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        # Retrain the model
        os.system('python train_model.py')
        
        # Reload the model
        global model, scaler
        model, scaler = load_model()
        
        return jsonify({"success": True, "message": "Model retrained successfully!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction Web Service...")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
