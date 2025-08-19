from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    with open('new_fertilizer_classification_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

try:
    with open('labelEncoder_01.pkl', 'rb') as f: 
        label_encoder = pickle.load(f)
    logger.info("Label Encoder  loaded successfully")
except FileNotFoundError:
    logger.error("Label encoder file not found")
    label_encoder = None
except Exception as e:
    logger.error(f"Error loading label label_encoder file: {str(e)}")
    label_encoder = None

def preprocess_input(data):
    """Preprocess input data for model prediction"""
    try:

        logger.info(f"Received data: {data}")
        for field in ['District_Name', 'Soil_color', 'Crop']:
            if not data.get(field, ''):
                raise ValueError("Missing required categorical fields")
            
        row = {
            "District_Name": data['District_Name'],
            'Soil_color': data['Soil_color'],
            'Crop': data['Crop'],
            "Nitrogen" : float(data.get("Nitrogen", 0)),
            "Phosphorus": float(data.get("Phosphorus", 0)),
            "Potassium"  : float(data.get("Potassium", 0)),
            "pH": float(data.get("pH",  0)),
            "Rainfall": float(data.get("Rainfall", 0)),
            "Temperature": float(data.get("Temperature", 25))
        }

        return pd.DataFrame([row])

    except (ValueError, TypeError) as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise ValueError(f"Invalid input  data: {str(e)}")
    
@app.route('/')
def index():
    """Serve the main HTML page"""
    districts = {"Kohlapur"}
    return render_template('index.html', districts=districts)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model is not available',
                'message': 'The machine learning model could not be loaded'
            }), 500
        
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'message' : 'Please provide input data for prediction'
            }), 400
        
        features = preprocess_input(data)

        #make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        #get fertilzier name
        fertilizer_name = label_encoder.inverse_transform([prediction])[0]

        prediction = int(prediction)
        confidence = float(np.max(prediction_proba)) * 100

        logger.info(f"Prediction made: {fertilizer_name} (confidence: {confidence:.2f}%)")

        return jsonify({
            'success': True,
            'fertilizer': str(fertilizer_name),
            'confidence': round(confidence, 2),
            'message': f'Recommended fertilizer: {fertilizer_name}'
        })

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction error',
            'model_loaded': "An error occured while making the prediction"
        }), 500

@app.route('/health', methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status' : 'healthy',
        'model_loaded': model is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)