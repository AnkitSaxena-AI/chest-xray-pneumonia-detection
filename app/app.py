"""
Pneumonia Detection — Streamlit demo
Upload a chest X-ray; the app predicts NORMAL / PNEUMONIA and shows a Grad-CAM heatmap
of the regions that drove the decision.

Run:  streamlit run app/app.py
The trained model `pneumonia_efficientnet.keras` must sit next to this file (in app/).
"""
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image

IMG = 224
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(HERE, "pneumonia_efficientnet.keras"),
    os.path.join(HERE, "..", "outputs", "pneumonia_efficientnet.keras"),
]

st.set_page_config(page_title="Pneumonia X-Ray Detector", page_icon="🫁", layout="centered")


@st.cache_resource(show_spinner=True)
def load_model():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return tf.keras.models.load_model(p)
    return None


def last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def grad_cam(arr, model, last_conv):
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr[None, ...])
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    heat = tf.image.resize(heat[..., None], (IMG, IMG)).numpy().squeeze()
    return heat


def overlay(arr, heat):
    colored = cm.jet(heat)[..., :3] * 255.0
    return np.clip(0.6 * arr + 0.4 * colored, 0, 255).astype("uint8")


st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("EfficientNetB0 transfer learning + Grad-CAM explainability · educational demo, **not for medical use**")

model = load_model()
if model is None:
    st.error("Model file not found. Place `pneumonia_efficientnet.keras` in the `app/` folder "
             "(produced by the Colab training notebook).")
    st.stop()
last_conv = last_conv_layer(model)

file = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB").resize((IMG, IMG))
    arr = np.array(img, dtype="float32")
    prob = float(model.predict(arr[None, ...], verbose=0)[0, 0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    conf = prob if prob >= 0.5 else 1 - prob

    c1, c2 = st.columns(2)
    c1.image(img, caption="Input X-ray", use_column_width=True)
    c2.image(overlay(arr, grad_cam(arr, model, last_conv)),
             caption="Grad-CAM (model focus)", use_column_width=True)

    color = "#d62728" if label == "PNEUMONIA" else "#2ca02c"
    st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.progress(min(conf, 1.0))
    st.write(f"Confidence: **{conf*100:.1f}%**  ·  P(pneumonia) = {prob:.3f}")
else:
    st.info("Upload a chest X-ray image to get a prediction and a Grad-CAM heatmap.")
