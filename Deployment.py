import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Define the class labels
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Streamlit app
st.title("Brain Tumor Detection with CNN")
st.write("Upload an MRI image, and the model will predict the type of tumor.")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((150, 150))  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(image)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display the result
    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"Confidence Scores: {predictions[0]}")
