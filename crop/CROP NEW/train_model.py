import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('backup', exist_ok=True)

# Load the dataset
df = pd.read_csv('final_crop_recommendation_complete (2).csv')

# Split the data into features (X) and target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model as a pickle file
model_path = 'model/crop_recommendation.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Save the scaler as a pickle file
scaler_path = 'model/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")


# Save a backup of the model code
backup_code = f"""
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

"""

backup_path = 'backup/model_backup.py'
with open(backup_path, 'w') as f:
    f.write(backup_code)
print(f"Model backup saved to {backup_path}")
