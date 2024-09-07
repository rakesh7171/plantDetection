#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import google.generativeai as genai  # Ensure this is the correct library
import pandas as pd
from cnn4 import load_cnn_model
from rf4 import load_rf_model

# Initialize the Google Generative AI client
def initialize_genai_client(api_key):
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Generative AI client: {e}")

# List of disease classes (restricted to Apple diseases)
disease_classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"
]

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))  # Ensure this matches your model's input size
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Predict disease from image using CNN
def predict_disease(cnn_model, image_path):
    processed_image = preprocess_image(image_path)
    predictions = cnn_model.predict(processed_image)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    disease_name = disease_classes[class_index]
    return disease_name, confidence

# Predict future diseases based on environmental factors using RF
def predict_future_disease(rf_model, environmental_data):
    # Make sure the input is a DataFrame with the correct feature names
    feature_names = ['Water_Level', 'Soil_Moisture', 'Air_Quality', 'Humidity']
    data = pd.DataFrame([environmental_data], columns=feature_names)
    prediction = rf_model.predict(data)
    return prediction[0]

# Get solutions using Google Generative AI
def get_solution_from_google_ai(disease_name):
    prompt = f"Provide a detailed solution for managing and treating the disease: {disease_name}. Include prevention and treatment measures."
    
    try:
        response = genai.generate_text(prompt=prompt, max_output_tokens=150)  # Ensure this is the correct method
        solution = response.result  # Adjust this based on actual response structure
    except Exception as e:
        print(f"Error during API call: {e}")
        solution = "Unable to fetch a solution at the moment."
    
    return solution

# Main function to take user input and make predictions
def main():
    # Set up Google Generative AI API key (replace with your actual API key)
    api_key = 'AIzaSyDFRD1WSRsDX0ihqRbnYofeNVLSQpg4vig'  # Replace with your valid API key
    initialize_genai_client(api_key)
    
    # Load models
    cnn_model_path = 'cnn_model4.keras'
    rf_model_path = 'rf_model6.pkl'
    
    cnn_model = load_cnn_model(cnn_model_path)
    rf_model = load_rf_model(rf_model_path)

    # Get user inputs
    image_path = input("Enter the path to the plant image: ")
    Water_Level = float(input("Enter temperature: "))
    Soil_Moisture = float(input("Enter soil moisture: "))
    Air_Quality = float(input("Enter air quality: "))
    Humidity = float(input("Enter humidity: "))

    # Predict disease from image
    disease_name, confidence = predict_disease(cnn_model, image_path)
    print(f"Disease Detected: {disease_name} (Confidence: {confidence*100:.2f}%)")

    # Get solution from Google AI
    solution = get_solution_from_google_ai(disease_name)
    print(f"Suggested Solution: {solution}")

    # Predict potential future disease based on environmental conditions
    environmental_data = [Water_Level, Soil_Moisture, Air_Quality, Humidity]
    future_disease = predict_future_disease(rf_model, environmental_data)
    if(future_disease=='Apple_healthy'):
        print("There is no risk of an outbreak ")
    else:
        print(f"Potential Future Disease: {future_disease}")

    # Get preventive measures from Google AI
    future_solution = get_solution_from_google_ai(future_disease)
    print(f"Preventive Measures: {future_solution}")

# Run the main function
if __name__ == '__main__':
    main()

