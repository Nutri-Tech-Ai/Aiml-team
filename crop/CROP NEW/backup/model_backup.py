
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# This is a backup of the trained model.
# It is not meant to be run directly.

# Load the scaler
scaler = joblib.load('model/scaler.pkl')

# Load the model
model = joblib.load('model/crop_recommendation.pkl')

