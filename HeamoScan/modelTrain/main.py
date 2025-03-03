import cv2
import numpy as np
import os
import joblib
import pandas as pd
from skimage.color import rgb2lab, rgb2hsv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Set upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# File paths
DATASET_PATH = "dataset_sheet.xlsx_-_data(1).csv"
MODEL_PATH = "hb_prediction_model.pkl"
SCALER_PATH = "scaler.pkl"

# Function to train the model (only if model does not exist)
def train_model():
    df = pd.read_csv(DATASET_PATH)
    
    features = [
        "R-Mean", "G-Mean", "B-Mean", "Lmean", "Amean", "Bmean",
        "Average-Hue", "Average-Saturation", "Average-Value", "R-Max"
    ]
    target = "Hb-Value"

    X = df[features]
    y = df[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save Model and Scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

# Function to extract features from image
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop central portion of the image (50% of the smallest dimension)
    h, w, _ = img.shape
    crop_size = int(min(h, w) * 0.5)
    cropped_img = img[h//2 - crop_size//2:h//2 + crop_size//2, w//2 - crop_size//2:w//2 + crop_size//2]

    # Convert to LAB and HSV color spaces
    lab = rgb2lab(cropped_img)
    hsv = rgb2hsv(cropped_img)

    # Compute Features
    features = {
        "R-Mean": np.mean(cropped_img[:, :, 0]),
        "G-Mean": np.mean(cropped_img[:, :, 1]),
        "B-Mean": np.mean(cropped_img[:, :, 2]),
        "Lmean": np.mean(lab[:, :, 0]),
        "Amean": np.mean(lab[:, :, 1]),
        "Bmean": np.mean(lab[:, :, 2]),
        "Average-Hue": np.mean(hsv[:, :, 0]) * 360,
        "Average-Saturation": np.mean(hsv[:, :, 1]) * 255,
        "Average-Value": np.mean(hsv[:, :, 2]) * 255,
        "R-Max": np.max(cropped_img[:, :, 0]),
    }

    return features

# Function to predict Hb value from extracted features
def predict_hb(features):
    # Train the model if it doesn't already exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_model()

    # Load trained model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Prepare features for prediction
    features_df = pd.DataFrame([features])
    scaled_features = scaler.transform(features_df)

    # Predict Hb value
    predicted_hb = model.predict(scaled_features)[0]
    return round(predicted_hb, 2)

# Flask Route - Homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            # Save uploaded file
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Extract features and predict Hb
            features = extract_features(image_path)
            predicted_hb = predict_hb(features)

            # Prepare result dictionary to send to template
            result = {
                'image_path': image_path,
                'features': features,
                'predicted_hb': predicted_hb
            }

    return render_template('import.html', result=result)

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True) 