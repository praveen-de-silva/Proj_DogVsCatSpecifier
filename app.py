import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Cat/Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ------------------------------
# Load Custom CSS
# ------------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_trained_model():
    best_model_path = "dog_cat_model_best.h5"
    return load_model(best_model_path)

model = load_trained_model()

# ------------------------------
# Initialize session state
# ------------------------------
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# ------------------------------
# Header Section
# ------------------------------
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🐾 AI Pet Classifier</h1>
    <p class="subtitle">Upload an image to identify dogs or cats</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Upload Section
# ------------------------------
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="file_uploader")

# Store uploaded file in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Hide the file uploader after upload
if st.session_state.uploaded_file is not None:
    st.markdown('<style>div[data-testid="stFileUploader"] {display: none;}</style>', unsafe_allow_html=True)

if st.session_state.uploaded_file is not None and not st.session_state.show_prediction:
    # Display the uploaded image
    img = Image.open(st.session_state.uploaded_file).convert('RGB')
    
    st.markdown('<div class="image-container-fixed">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Predict", use_container_width=True, type="primary"):
            st.session_state.show_prediction = True
            st.rerun()
    with col2:
        if st.button("✕ Cancel", use_container_width=True):
            st.session_state.show_prediction = False
            st.session_state.uploaded_file = None
            st.rerun()

elif st.session_state.uploaded_file is not None and st.session_state.show_prediction:
    # Display the uploaded image
    img = Image.open(st.session_state.uploaded_file).convert('RGB')
    
    st.markdown('<div class="image-container-fixed">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Preprocess image
    img_resized = img.resize((128, 128))
    x = image.img_to_array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    
    # Prediction
    pred = model.predict(x, verbose=0)
    confidence = float(pred[0][0])
    
    if confidence > 0.5:
        label = "Dog"
        icon = "🐕"
        confidence_percent = confidence * 100
    else:
        label = "Cat"
        icon = "🐈"
        confidence_percent = (1 - confidence) * 100
    
    # Display result
    st.markdown(f"""
    <div class="result-card">
        <div class="result-icon">{icon}</div>
        <p class="result-label">Classification Result</p>
        <h2 class="result-value">{label}</h2>
        <p class="confidence-text">Confidence: {confidence_percent:.1f}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # New prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("↻ New Prediction", use_container_width=True):
            st.session_state.show_prediction = False
            st.session_state.uploaded_file = None
            st.rerun()
else:
    st.markdown("""
    <div class="upload-placeholder">
        <p class="upload-text">📁 Drag and drop or click to upload</p>
        <p class="upload-subtext">Supports JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
