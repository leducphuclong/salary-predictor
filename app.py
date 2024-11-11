import requests
import pickle
import pandas as pd

# Set the URL (use local testing URL or ngrok URL if deploying online)
url = "http://127.0.0.1:5000/predict"

# Define the record data
data = [52, "Master's", "Director", 20, "Male"]

# Define the column names
column_name = ['Age', 'Education Level', 'Job Title', 'Years of Experience', 'Gender_Male']

# Normalize the "Gender" column (convert "Male" to 1, "Female" to 0)
gender_value = 1 if data[4] == "Male" else 0
data = [data[0], data[1], data[2], data[3], gender_value]

# Create DataFrame for the record
record_df = pd.DataFrame([data], columns=column_name)

# Load the LabelEncoder for "Job Title"
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Normalize the "Job Title" column using the loaded LabelEncoder
job_title_value = record_df['Job Title'].iloc[0]  # Extract the job title value
if job_title_value in label_encoder.classes_:
    record_df.at[0, 'Job Title'] = label_encoder.transform([job_title_value])[0]
else:
    # Handle unknown job titles (not present in training)
    print(f"Unknown Job Title: {job_title_value}")
    record_df.at[0, 'Job Title'] = -1  # Placeholder value for unknown categories

# Print to verify the transformation of "Job Title"
print("Record DataFrame after 'Job Title' normalization:")
print(record_df)

# Normalize the "Education Level" column using a predefined mapping
education_mapping = {"Bachelor's": 1, "Master's": 2, 'PhD': 3}

education_level_value = record_df['Education Level'].iloc[0]  # Extract the education level value
if education_level_value in education_mapping:
    record_df.at[0, 'Education Level'] = education_mapping[education_level_value]
else:
    # Handle unknown or missing education levels
    print(f"Unknown Education Level: {education_level_value}")
    record_df.at[0, 'Education Level'] = 0  # Placeholder value for unknown education levels

# Load the saved feature scaler
with open('feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)

# Normalize numeric features in `record_df` using the feature scaler
record_normalized = feature_scaler.transform(record_df)

# Convert the normalized data back to a DataFrame for easier reading (optional)
record_normalized_df = pd.DataFrame(record_normalized, columns=record_df.columns)

# Print to verify the final transformed DataFrame
print("Final Record DataFrame with normalized fields:")
print(record_normalized_df)

# Prepare the data for sending to the API
payload = {
    'features': record_normalized.flatten().tolist()  # Flatten the array and convert to list for JSON
}

#Uncomment the lines below to send the request
response = requests.post(url, json=payload)
# Process the API response
if response.status_code == 200:
    # Get the normalized prediction from the response
    normalized_prediction = response.json()['prediction']

    # Load the target scaler to convert prediction back to original scale
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)

    # Convert the normalized prediction back to the original target scale
    original_prediction = target_scaler.inverse_transform([[normalized_prediction]])[0][0]

    print("Prediction in Original Scale:", original_prediction)
else:
    print("Error:", response.status_code, response.text)
