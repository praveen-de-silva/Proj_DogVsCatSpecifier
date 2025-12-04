# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os


# ------------------------------
# Load the trained model
# ------------------------------
best_model_path = "dog_cat_model_best.h5"  # update this path
model = load_model(best_model_path)

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image, and the model will predict whether it is a dog or a cat.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert('RGB')

    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((128, 128))  # use the size your model was trained on
    x = image.img_to_array(img_resized)/255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    pred = model.predict(x)
    label = "Dog" if pred[0][0] > 0.5 else "Cat"
    st.write(f"Prediction: **{label}**")
