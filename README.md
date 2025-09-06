# Lung Disease Detection from Chest X-rays using Deep Learning

This repository contains the code and implementation for our Final Year Project: Lung Disease Detection from Chest X-rays using Deep Learning. The project focuses on building an AI-powered diagnostic tool that can classify chest X-ray images into Normal or Pneumonia cases.

🔹 Key Features

Deep Learning Model: Built using EfficientNetB0 with transfer learning for optimal accuracy and efficiency.

High Accuracy: Achieved 94% classification accuracy on a publicly available chest X-ray dataset.

Model Evaluation: Includes confusion matrices, accuracy/loss curves, and classification reports (Precision, Recall, F1-score).

Streamlit Web App: A user-friendly web-based GUI where healthcare professionals can upload chest X-ray images and receive instant predictions with confidence scores.

Error Handling: Gracefully manages invalid or corrupted image uploads.

Deployment Ready: Can run on local desktops or be deployed on cloud platforms.

🔹 Tools & Technologies

Python 3.10+, TensorFlow, Keras, NumPy, Scikit-learn

Image Processing: OpenCV, Pillow (PIL)

Visualization: Matplotlib, Seaborn, Pandas

Deployment/UI: Streamlit

🔹 Project Workflow

Data Preprocessing: Image resizing, normalization, and augmentation.

Model Training: Transfer learning with EfficientNetB0.

Evaluation: Performance analysis using multiple metrics.

Deployment: Interactive GUI for real-time predictions.

🔹 Future Work

Extend detection to multiple lung diseases (COVID-19, Tuberculosis, Lung Cancer).

Add explainability features (heatmaps/visual reasoning
