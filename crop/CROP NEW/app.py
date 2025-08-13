from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model/crop_recommendation.pkl')
scaler = joblib.load('model/scaler.pkl')

# Crop metadata
crop_metadata = {
    'rice': {'msp': 2369, 'season': 'Kharif', 'average_yield': 2.218494541018379, 'yield_quintals': 22.1849454101838},
    'maize': {'msp': 2249, 'season': 'Kharif', 'average_yield': 3.4272159774543587, 'yield_quintals': 34.27215977454359},
    'chickpea': {'msp': 5440, 'season': 'Kharif'},
    'kidneybeans': {'msp': 6000, 'season': 'Kharif'},
    'pigeonpeas': {'msp': 8000, 'season': 'Kharif'},
    'mothbeans': {'msp': 5000, 'season': 'Kharif'},
    'mungbean': {'msp': 8768, 'season': 'Kharif'},
    'blackgram': {'msp': 7800, 'season': 'Kharif'},
    'lentil': {'msp': 6025, 'season': 'Kharif'},
    'pomegranate': {'msp': 2500, 'season': 'Kharif'},
    'banana': {'msp': 900, 'season': 'Kharif', 'average_yield': 26.85112785404081, 'yield_quintals': 268.51127854040817},
    'mango': {'msp': 1200, 'season': 'Kharif'},
    'grapes': {'msp': 1800, 'season': 'Kharif'},
    'watermelon': {'msp': 800, 'season': 'Kharif'},
    'muskmelon': {'msp': 700, 'season': 'Kharif'},
    'apple': {'msp': 3500, 'season': 'Kharif'},
    'orange': {'msp': 1600, 'season': 'Kharif'},
    'papaya': {'msp': 600, 'season': 'Kharif'},
    'coconut': {'msp': 1000, 'season': 'Kharif', 'average_yield': 8652.000198744186, 'yield_quintals': 86520.00198744186},
    'cotton': {'msp': 7710, 'season': 'Kharif'},
    'jute': {'msp': 4750, 'season': 'Kharif', 'average_yield': 7.555392696430939, 'yield_quintals': 75.5539269643094},
    'coffee': {'msp': 15000, 'season': 'Kharif'}
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        feature_list = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        # Create a DataFrame with feature names
        features = pd.DataFrame([feature_list], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Scale the input features
        scaled_features = scaler.transform(features)

        # Get prediction probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Get top 5 predictions
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_crops = model.classes_[top5_indices]
        
        recommendations = []
        for crop in top5_crops:
            meta = crop_metadata.get(crop, {})
            profit = meta.get('yield_quintals', 0) * meta.get('msp', 0)
            recommendations.append({
                'crop': crop.capitalize(),
                'profit': f'₹{profit:,.2f}',
                'yield': f"{meta.get('yield_quintals', 'N/A')} quintals",
                'season': meta.get('season', 'N/A'),
            })

        return render_template('result.html', recommendations=recommendations)

@app.route('/predict-example')
def predict_example():
    # Example values
    feature_list = [90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536] # Example for Rice
    features = pd.DataFrame([feature_list], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Scale the input features
    scaled_features = scaler.transform(features)

    # Get prediction probabilities
    probabilities = model.predict_proba(scaled_features)[0]
    
    # Get top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_crops = model.classes_[top5_indices]
    
    recommendations = []
    for crop in top5_crops:
        meta = crop_metadata.get(crop, {})
        profit = meta.get('yield_quintals', 0) * meta.get('msp', 0)
        recommendations.append({
            'crop': crop.capitalize(),
            'profit': f'₹{profit:,.2f}',
            'yield': f"{meta.get('yield_quintals', 'N/A')} quintals",
            'season': meta.get('season', 'N/A'),
            'sustainability': 'N/A'
        })

    return render_template('result.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
