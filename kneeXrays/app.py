import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load the trained model (ensure you provide the correct path to your saved model)
model = load_model('knee.h5')

# Define the class names
class_names = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((75, 75))  # Resize image to match model's expected input size
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess image for InceptionV3
    return image

# Streamlit app
st.title("Knee X-ray Classification")

st.write("Upload a knee X-ray image to classify it into one of the following categories: Normal, Doubtful, Mild, Moderate, Severe.")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    # Display the prediction
    st.write(f"Prediction: {class_names[predicted_class]} ({confidence:.2f}%)")

    # Optionally, show detailed probabilities for each class
    st.write("Class Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")
