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
# Custom CSS for Modern Black & White Theme
# ------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #ffffff 0%, #a0a0a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #b0b0b0;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Upload section */
    .upload-container {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: rgba(255, 255, 255, 0.4);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Image display */
    .image-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .result-label {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 10px;
        margin-top: 1rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ffffff 0%, #808080 100%);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Icon styling */
    .icon-large {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* File uploader styling */
    .stFileUploader {
        background: transparent !important;
    }
    
    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ffffff 0%, #d0d0d0 100%);
        color: #1a1a1a;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_trained_model():
    best_model_path = "dog_cat_model_best.h5"
    return load_model(best_model_path)

model = load_trained_model()

# ------------------------------
# Header Section
# ------------------------------
st.markdown("""
<div class="main-header">
    <div class="icon-large">🐾</div>
    <h1 class="main-title">AI Pet Classifier</h1>
    <p class="subtitle">Upload an image and let AI identify whether it's a dog or a cat</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Upload Section
# ------------------------------
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if not uploaded_file:
    st.markdown("""
    <div style="color: #b0b0b0; font-size: 1.1rem;">
        <p style="margin: 0;">📸 Drag and drop or click to upload</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #808080;">Supports JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Prediction Section
# ------------------------------
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Preprocess image
    img_resized = img.resize((128, 128))
    x = image.img_to_array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    
    # Prediction with spinner
    with st.spinner('🔍 Analyzing image...'):
        pred = model.predict(x, verbose=0)
        confidence = float(pred[0][0])
        
        if confidence > 0.5:
            label = "Dog"
            icon = "🐶"
            confidence_percent = confidence * 100
        else:
            label = "Cat"
            icon = "🐱"
            confidence_percent = (1 - confidence) * 100
    
    # Display result
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size: 5rem; margin-bottom: 1rem;">{icon}</div>
        <p class="result-label">Classification Result</p>
        <h2 class="result-value">{label}</h2>
        <p style="color: #a0a0a0; margin-top: 1rem;">Confidence: {confidence_percent:.1f}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Clear button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Upload Another Image", use_container_width=True):
            st.rerun()
