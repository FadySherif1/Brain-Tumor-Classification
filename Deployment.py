import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_modelCNN_98%.h5')

# Define the class labels
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Streamlit app
st.title("🧠Brain Tumor Detection with CNN")
st.markdown("Upload an MRI image, and the model will predict the type of tumor. Let's detect early and save lives! 🙌")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an MRI Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='📸 Uploaded Image', use_container_width=True)

    # Add a button for prediction
    if st.button("Analyze"):
        # Preprocess the image
        image = image.resize((150, 150))  # Resize to model's input size
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence_scores = predictions[0]

        # Display results
        st.markdown("### 🎯 Prediction Results")
        if predicted_class == "No Tumor":
            st.success(f"🟢 **Predicted Class**: {predicted_class}")
        else:
            st.error(f"🔴 **Predicted Class**: {predicted_class}")
        
        # Display confidence scores
        st.markdown("### 🔍 Confidence Scores")
        for label, score in zip(class_labels, confidence_scores):
            st.write(f"- {label}: {score:.2%}")

        st.markdown("---")  # Divider for clarity
        st.markdown("### 🧠 **Take Care of Your Brain Health** ❤️")
