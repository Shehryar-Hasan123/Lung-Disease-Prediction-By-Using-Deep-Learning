# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Lung Disease Detector",
    page_icon="🫁",
    layout="centered",
)

# ---------- LOAD LABELS ----------
with open("labels.json", "r") as f:
    CLASS_NAMES = list(json.load(f).keys())  # ["NORMAL", "PNEUMONIA"]

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "lung_disease_classifier.keras")
    return tf.keras.models.load_model(model_path)
model = load_model()

# ---------- PREPROCESS ----------
def preprocess_image(image: Image.Image, target_size=(224,224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype("float32")
    # Same preprocessing as EfficientNet
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------- CSS STYLING ----------
st.markdown(
    """
    <style>
    body {
        background-color: #0d0d0d;
        color: #e6e6e6;
    }
    .result-box {
        background: #111;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 0 20px rgba(0,255,204,0.2);
    }
    .prob-bar {
        height: 20px;
        border-radius: 20px;
        background-color: #222;
        margin-bottom: 12px;
        position: relative;
    }
    .prob-fill {
        height: 100%;
        border-radius: 20px;
        text-align: center;
        line-height: 20px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- UI ----------
st.title("Lung Disease Detection")
st.markdown(
    "<h4 style='text-align: center;'>Upload a chest X-ray to detect whether it is <span style='color:#00ffcc;'>Normal</span> or <span style='color:#ff4d4d;'>Pneumonia</span></h4>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Choose X-ray (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Show uploaded image
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    # Predict
    X = preprocess_image(img)
    preds = model.predict(X)

    # Sigmoid binary output
    prob_pneumonia = float(preds[0][0])
    prob_normal = 1 - prob_pneumonia
    probs = {"NORMAL": prob_normal, "PNEUMONIA": prob_pneumonia}

    # Final label
    label = max(probs, key=probs.get)
    confidence = probs[label]

    # Show result
    st.markdown(
        f"""
        <div class="result-box">
            <h2 style="color:{'#ff4d4d' if label=='PNEUMONIA' else '#00ffcc'};">
                {label} Detected
            </h2>
            <p style="font-size:18px;">Confidence: {confidence*100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability Bars
    st.subheader("Probability Distribution")
    for k, v in probs.items():
        color = "#00ffcc" if k == "NORMAL" else "#ff4d4d"
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <span style="font-size:16px;">{k}: {v*100:.2f}%</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width:{v*100}%; background: {color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Bar Chart with scores
    st.subheader("Confidence Chart")
    fig, ax = plt.subplots(facecolor="#0d0d0d")
    bars = ax.bar(
        probs.keys(), 
        [v*100 for v in probs.values()],
        color=["#00ffcc", "#ff4d4d"],
        alpha=0.9
    )
    ax.set_ylabel("Confidence (%)", color="white")
    ax.set_ylim(0, 100)
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")

    # Add percentage text above bars
    for rect, val in zip(bars, [v*100 for v in probs.values()]):
        ax.text(rect.get_x() + rect.get_width()/2, val+1, f"{val:.1f}%", 
                ha='center', color="white", fontsize=10)

    st.pyplot(fig)

else:
    st.info("👆 Upload a chest X-ray image to get started.")

