from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from pyngrok import ngrok
from sklearn.impute import SimpleImputer
import math

# Initialize Flask app
app = Flask("Salary Predictor")

# Start ngrok tunnel
# public_url = ngrok.connect(5000)
# print("Public URL:", public_url)

# Load necessary resources
def load_resource(filename, resource_name):
    try:
        with open(filename, 'rb') as f:
            print(f"{resource_name} loaded successfully.")
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {resource_name} file '{filename}' not found.")
        return None

model = load_resource('model.pkl', 'Model')
imputer = load_resource('imputer.pkl', 'Imputer')
label_encoder = load_resource('label_encoder.pkl', 'Label Encoder')
feature_scaler = load_resource('feature_scaler.pkl', 'Feature Scaler')
target_scaler = load_resource('target_scaler.pkl', 'Target Scaler')

# Define the home route to serve HTML
@app.route('/')
def home():
    return render_template('index.html')

# Normalize and preprocess input data
def normalize(data):
    # Define column names and mappings
    column_names = ['Age', 'Education Level', 'Job Title', 'Years of Experience', 'Gender_Male']
    
    # Convert list to DataFrame and handle missing values
    record_df = pd.DataFrame([data], columns=column_names)

    print("Data after receive: ")
    print(record_df)

    # Check and encode "Gender_Male"
    if record_df.at[0, 'Gender_Male'] is not None:
        record_df.at[0, 'Gender_Male'] = 1 if record_df.at[0, 'Gender_Male'] == "Male" else 0

    # Check and encode "Job Title"
    job_title = record_df.at[0, 'Job Title']
    if job_title is not None and job_title in label_encoder.classes_:
        record_df.at[0, 'Job Title'] = label_encoder.transform([job_title])[0]
    else:
        record_df.at[0, 'Job Title'] = None

    # Check and encode "Education Level"
    education_mapping = {"Bachelor's": 1, "Master's": 2, 'PhD': 3}
    education_level = record_df.at[0, 'Education Level']
    if education_level is not None:
        record_df.at[0, 'Education Level'] = education_mapping.get(education_level, None)

    print("Data after normalization")
    print(record_df)

    # Impute missing values
    if imputer is not None:
        record_df = pd.DataFrame(imputer.transform(record_df), columns=column_names)
        
        # Round numeric columns to integers
        record_df = record_df.apply(lambda col: col.round() if col.dtype in ['float64', 'float32'] else col)
    else:
        print("Imputer not loaded; missing values may not be handled.")
    
    print("Data after imputing: ")
    print(record_df)

    # Scale features
    record_normalized = feature_scaler.transform(record_df)

    print("Data after scaling: ")
    print(record_normalized)


    return record_normalized.flatten().tolist()


# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be in JSON format'}), 400

    data = request.json.get('features')
    if not data:
        return jsonify({'error': 'Missing "features" in request'}), 400

    try:
        processed_data = normalize(data)
        features = np.array(processed_data).reshape(1, -1)

        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500

        # Predict and inverse transform the prediction
        salary = model.predict(features)[0].item()
        original_prediction = target_scaler.inverse_transform([[salary]])[0][0]

        return jsonify({'prediction': original_prediction})
    except Exception as e:
        print("Error in /predict route:", str(e))
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)
