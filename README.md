# Brain-Tumor-MRI-Image-Classification-Project
# 🧠 Brain Tumor MRI Image Classification

This project aims to detect and classify brain tumors from MRI scans using deep learning. A web-based interface built with Streamlit allows users to upload MRI images and get real-time predictions using models like **Custom CNN**, **MobileNetV2**, and **ResNet50**.

---

## 🚀 Project Highlights

- ✅ Built using TensorFlow & Keras
- ✅ Trained on labeled brain MRI images (4 classes)
- ✅ Compared CNN with two transfer learning models
- ✅ Deployed a user-friendly Streamlit app
- ✅ Real-time tumor prediction on uploaded images

---

## 🎯 Problem Statement

Brain tumors pose serious health threats and require early detection for better treatment outcomes. Manual diagnosis through MRI interpretation is time-consuming and subjective. This project leverages AI to assist radiologists by automating the classification of MRI brain scans into:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

---

## 🧠 Model Architectures

### 1. Custom CNN  
A Convolutional Neural Network built from scratch with 3 conv layers, max-pooling, dropout, and softmax output.

### 2. MobileNetV2 (Transfer Learning)  
A lightweight, pre-trained model optimized for mobile devices and fast inference.

### 3. ResNet50 (Transfer Learning)  
A deeper architecture with residual connections for high accuracy on complex images.

---

## 📁 Directory Structure

- brain-tumor-classification/
- │
- ├── app/ # Streamlit frontend
- │ └── streamlit_app.py
- │
- ├── models/ # Trained model files
- │ ├── custom_cnn_model.h5
- │ ├── mobilenet_model.h5
- │ └── resnet_model.h5
- │
- ├── notebooks/ # Model training notebooks
- │ └── model_training.ipynb
- │
- ├── data/ # (Optional) Local image dataset
- ├── images/ # Screenshots or sample outputs
- └── README.md

---

## 📦 Dataset Overview

- **Name:** Brain MRI Images for Brain Tumor Detection
- **Source:** [Kaggle - Brain MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Structure:**
  - `Training/` and `Testing/` folders
  - Each with subfolders: `glioma`, `meningioma`, `pituitary`, `notumor`

---

## 🖥️ How to Run Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/SuziSharma2/brain-tumor-classification.git
cd brain-tumor-classification
