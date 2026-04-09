from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import math
import random
import os

app = Flask(__name__)
CORS(app)

# Ensure models exist, if not run train script silently to create mock models for demo
if not os.path.exists('models/lstm_model.h5'):
    import train_model
    train_model.create_mock_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = float(data.get('latitude', 0))
    lon = float(data.get('longitude', 0))
    
    # For now, we simulate the XG/LSTM logic using a stochastic seed based on lat/lon
    # since we don't have Earth engine auth context. 
    # In production, this would call get_terraclimate_data -> LSTM model
    random.seed(int(abs(lat) * abs(lon) * 100))
    
    prob = random.uniform(10, 95)
    drought_class = math.floor((prob / 100) * 5)
    
    classes = ["No Drought", "Mild Drought", "Moderate Drought", "Severe Drought", "Extreme Drought"]
    
    return jsonify({
        "lat": lat,
        "lon": lon,
        "drought_probability": round(prob, 2),
        "drought_class": classes[drought_class],
        "class_index": drought_class
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
