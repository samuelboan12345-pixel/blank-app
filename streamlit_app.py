
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import io

# -----------------------------
# Load model with caching
# -----------------------------
@st.cache_resource
def load_cat_dog_model():
        model_path = os.path.join(os.path.dirname(__file__), "cat_dog_classifier.h5")
        return load_model(model_path)

model = load_cat_dog_model()

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("""
Upload images (JPEG, PNG, WebP, or AVIF) and the model will predict whether each image is a **cat** or a **dog** with a confidence score.
""")

# -----------------------------
# Image uploader (single or multiple)
# -----------------------------
uploaded_files = st.file_uploader(
        "Choose one or more images",
        type=["jpg", "jpeg", "png", "webp", "avif"],
        accept_multiple_files=True
)

if uploaded_files:
        for uploaded_file in uploaded_files:
                # Open and convert to RGB
                try:
                        img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
                except Exception as e:
                        st.error(f"Failed to open {uploaded_file.name}: {e}")
                        continue

                # Resize for model
                img_resized = img.resize((150, 150))
                img_array = np.expand_dims(np.array(img_resized), 0)

                # Prediction
                prediction = model.predict(img_array)[0][0]
                label = "Dog" if prediction >= 0.5 else "Cat"
                confidence = prediction if prediction >= 0.5 else 1 - prediction

                # Display
                st.image(img, caption=uploaded_file.name, use_container_width=True)
                st.markdown(f"**Prediction:** {label}")
                st.markdown(f"**Confidence:** {confidence:.2f}")
                st.write("---")

# -----------------------------
# Footer
st.write("Made with TensorFlow and Streamlit. Upload your own images to see predictions instantly!")
# -----------------------------

                                                                                                                                                                    