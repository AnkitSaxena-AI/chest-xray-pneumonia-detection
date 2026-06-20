"""
Pneumonia Detection — Streamlit demo
Upload a chest X-ray (or pick a bundled sample); the app predicts NORMAL / PNEUMONIA
and shows a Grad-CAM heatmap of the regions that drove the decision.

Run locally:  streamlit run app/app.py
Deploy:       Streamlit Community Cloud → main file = app/app.py
"""
import os
import glob
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
SAMPLES_DIR = os.path.join(HERE, "..", "samples")

st.set_page_config(page_title="Pneumonia X-Ray Detector", page_icon="🫁", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
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
    return tf.image.resize(heat[..., None], (IMG, IMG)).numpy().squeeze()


def overlay(arr, heat):
    colored = cm.jet(heat)[..., :3] * 255.0
    return np.clip(0.6 * arr + 0.4 * colored, 0, 255).astype("uint8")


st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("EfficientNetB0 transfer learning + Grad-CAM explainability · educational demo, **not for medical use**")

model = load_model()
if model is None:
    st.error("Model file not found. Place `pneumonia_efficientnet.keras` in the `app/` folder.")
    st.stop()
last_conv = last_conv_layer(model)

samples = sorted(
    glob.glob(os.path.join(SAMPLES_DIR, "*.jpeg"))
    + glob.glob(os.path.join(SAMPLES_DIR, "*.jpg"))
    + glob.glob(os.path.join(SAMPLES_DIR, "*.png"))
)

file = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
pick = None
if samples:
    sel = st.selectbox("…or try a bundled sample X-ray", ["(none)"] + [os.path.basename(s) for s in samples])
    if sel != "(none)":
        pick = os.path.join(SAMPLES_DIR, sel)

src = file if file is not None else pick
if src is not None:
    img = Image.open(src).convert("RGB").resize((IMG, IMG))
    arr = np.array(img, dtype="float32")
    prob = float(model.predict(arr[None, ...], verbose=0)[0, 0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    conf = prob if prob >= 0.5 else 1 - prob

    c1, c2 = st.columns(2)
    c1.image(img, caption="Input X-ray", use_container_width=True)
    c2.image(overlay(arr, grad_cam(arr, model, last_conv)), caption="Grad-CAM (model focus)", use_container_width=True)

    color = "#d62728" if label == "PNEUMONIA" else "#2ca02c"
    st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.progress(min(conf, 1.0))
    st.write(f"Confidence: **{conf*100:.1f}%**  ·  P(pneumonia) = {prob:.3f}")
else:
    st.info("Upload a chest X-ray — or pick a bundled sample above.")
