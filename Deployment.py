import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Define the class labels
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Streamlit app
st.title("Brain Tumor Detection with CNN")
st.write("Upload MRI images to generate predictions and analyze model performance!")

# Upload multiple files for testing (Test dataset)
uploaded_files = st.file_uploader("Upload Multiple MRI Images for Testing", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Initialize lists to store true labels and predictions
true_labels = []
predicted_labels = []

if uploaded_files:
    st.write("### Uploaded Images:")
    for uploaded_file in uploaded_files:
        # Display uploaded images
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        # Preprocess the image
        image = image.resize((150, 150))  # Resize to model input size
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(image)
        predicted_class = class_labels[np.argmax(predictions)]
        predicted_labels.append(np.argmax(predictions))

        # Get the true label from the filename (assuming filenames contain labels)
        # Example: 'glioma_1.jpg' -> glioma
        for label in class_labels:
            if label.lower() in uploaded_file.name.lower():
                true_labels.append(class_labels.index(label))
                break

    # Ensure predictions and true labels are the same length
    if len(true_labels) == len(predicted_labels):
        # Display classification report
        st.write("### Classification Report:")
        report = classification_report(true_labels, predicted_labels, target_names=class_labels, output_dict=True)
        st.write(report)  # Display the report in Streamlit

        # Display confusion matrix
        st.write("### Confusion Matrix:")
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
    else:
        st.error("Unable to determine true labels for some images. Check file naming conventions.")
else:
    st.info("Upload multiple test images to evaluate model performance.")
