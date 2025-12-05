"# 🐾 AI Pet Classifier - Dog vs Cat Specifier

An intelligent image classification application that uses deep learning to identify whether an uploaded image contains a dog or a cat. Built with TensorFlow/Keras and deployed using Streamlit.

## 📊 Data Dictionary

### Dataset Source
- **Source**: [Kaggle - Dogs vs Cats Competition](https://www.kaggle.com/competitions/dogs-vs-cats/data)
- **Dataset Type**: Binary Image Classification
- **Total Images**: 25,000 labeled images
  - Training Set: 12,500 cat images
  - Training Set: 12,500 dog images

### Data Structure
| Field | Type | Description |
|-------|------|-------------|
| Image Files | JPG/PNG | Color images of cats and dogs |
| Image Size | Variable | Resized to 128x128 pixels for model input |
| Labels | Binary | 0 = Cat, 1 = Dog |
| Channels | 3 (RGB) | Red, Green, Blue color channels |
| Pixel Values | 0-255 | Normalized to 0-1 range for training |

### Data Preprocessing
- **Image Resizing**: All images standardized to 128x128 pixels
- **Normalization**: Pixel values scaled to [0, 1] range
- **Color Format**: RGB (Red, Green, Blue)
- **Data Augmentation**: Applied during training (rotation, flip, zoom)

## 🛠️ Technical Stack

### Machine Learning & Data Science
- **Python**: 3.8+
- **TensorFlow/Keras**: Deep learning framework for model training
- **NumPy**: Numerical computations and array operations
- **PIL (Pillow)**: Image processing and manipulation

### Model Development Environment
- **Google Colab**: Cloud-based Jupyter notebook environment for model training
- **GPU Acceleration**: Leveraged Colab's GPU for faster training

### Frontend & Deployment
- **Streamlit**: Interactive web application framework
- **HTML/CSS**: Custom styling for enhanced UI/UX
- **Responsive Design**: Mobile and desktop compatible interface

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input Shape**: (128, 128, 3)
- **Output**: Binary classification (Cat/Dog)
- **Model File**: `dog_cat_model_best.h5`
- **Framework**: Keras Sequential API

### Development Tools
- **Git/GitHub**: Version control and repository hosting
- **VS Code**: Code editor
- **Virtual Environment**: Python dependency isolation

## 🚀 Features

- **Real-time Image Classification**: Upload and classify images instantly
- **Confidence Score**: Displays prediction confidence percentage
- **Error Handling**: Robust validation with user-friendly error messages
- **Interactive UI**: Modern, gradient-based dark theme design
- **Responsive Layout**: Optimized for various screen sizes

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/praveen-de-silva/Proj_DogVsCatSpecifier.git
cd Proj_DogVsCatSpecifier

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📋 Requirements

```
streamlit
tensorflow
keras
numpy
pillow
```

## 🎯 Usage

1. Launch the application using `streamlit run app.py`
2. Upload an image (JPG, JPEG, or PNG format)
3. Click the "✓ Predict" button
4. View the classification result with confidence score
5. Click "↻ New Prediction" to classify another image

## 🧠 Model Training

The model was trained on Google Colab using:
- **Dataset**: 25,000 images from Kaggle Dogs vs Cats competition
- **Training Strategy**: Binary classification with data augmentation
- **Validation Split**: 80-20 train-validation split
- **Optimization**: Adam optimizer
- **Loss Function**: Binary cross-entropy

## 📁 Project Structure

```
DogCatSpecifier/
├── app.py                    # Main Streamlit application
├── style.css                 # Custom CSS styling
├── dog_cat_model_best.h5     # Trained model weights
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 👤 Author

**Praveen De Silva**
- Machine Learning Project 01
- December 2025

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/competitions/dogs-vs-cats/data)
- Built with Streamlit and TensorFlow
- Trained on Google Colab infrastructure
" 
