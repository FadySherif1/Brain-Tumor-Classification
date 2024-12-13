import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load the model
model = load_model('brain_tumor_model.h5')

# Class labels
class_labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Function to preprocess and predict the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Make it batch-like
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit UI
st.title("Brain Tumor Classification")

st.write("Upload an MRI scan image to classify.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(uploaded_file)
    
    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    
    # Display prediction
    st.write(f"Prediction: {predicted_class}")
