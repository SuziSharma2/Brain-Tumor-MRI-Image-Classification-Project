import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Model mapping
model_options = {
    "Custom CNN": "../models/custom_cnn_model.h5",
    "ResNet50": "../models/resnet_model.h5",
    "MobileNetV2": "../models/mobilenet_model.h5"
}

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to predict tumor type using your chosen model.")

# Model selection dropdown
selected_model_name = st.selectbox("Select a model:", list(model_options.keys()))

# Load model only when selected
if selected_model_name:
    model_path = model_options[selected_model_name]
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.success(f"✅ Loaded {selected_model_name} model successfully.")
    else:
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

# File upload
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"🧪 Prediction: **{pred_class.upper()}**")
        st.info(f"Confidence Score: {confidence:.2f}%")
