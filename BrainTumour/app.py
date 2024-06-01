import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

# Load the trained model
model = load_model("BrainTumour/brain.h5")
# Define the categories (classes) of the dataset
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Create a label binarizer
lb = LabelBinarizer()
lb.fit(labels)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the class of the tumor
def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)[0]
    predicted_label = lb.classes_[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_label, confidence

# Streamlit app
st.title('Brain Tumor Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=250)

    if st.button('Predict'):
        predicted_label, confidence = predict(image)
        st.write(f'Prediction: {predicted_label}')
        st.write(f'Confidence: {confidence:.2f}')
