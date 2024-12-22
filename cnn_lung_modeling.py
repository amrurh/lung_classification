import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json
from PIL import Image

# Paths to the model and class indices
MODEL_PATH = "E:\Kuliah\Project\CNN Project\covid19_cnn_model.h5"
CLASS_INDICES_PATH = "E:\Kuliah\Project\CNN Project\class_indices.json"

# Load the model and class indices
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_cnn_model():
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Reverse the class indices to get a mapping from indices to labels
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_cnn_model()

# Define a target size for the images (width, height)
target_size = (200, 200)  # Adjust as needed

# Display class references
st.write("### Class Reference")

# List of example images for each class
class_images = {
    "Covid": r"E:\Kuliah\Project\CNN Project\covid_ex.jpeg",
    "Normal": r"E:\Kuliah\Project\CNN Project\normal_ex.jpeg",
    "Viral Pneumonia": r"E:\Kuliah\Project\CNN Project\viral_pneumonia_ex.jpeg"
}

# Create columns for each class
columns = st.columns(len(class_images))  # Create one column per class

for i, (class_name, image_path) in enumerate(class_images.items()):
    with columns[i]:
        # Open and resize the image
        img = Image.open(image_path).resize(target_size)
        st.image(img, use_container_width=True)  # Display the resized image
        st.write(f"**{class_name}**")  # Display the class name below the image


# File uploader
uploaded_file = st.file_uploader("Choose a lung scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Lung Scan", use_container_width=True)
    st.write("Classifying...")

    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((128, 128))  # Resize to match model input
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Display the result
    st.write(f"### Predicted Class: {labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
