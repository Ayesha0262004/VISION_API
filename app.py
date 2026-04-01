import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.express as px
from streamlit_image_comparison import image_comparison
import os
import gdown

# =========================
# 🔹 DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================
MODEL_PATH = "retina_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1diaw1WwEpbOQ9d2_uWGeFz0mDDDOcTJP"
    gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# 🔹 PAGE SETUP
# =========================
st.set_page_config(page_title="VisionAI | Precision Diagnostics",
                   layout="wide",
                   page_icon="👁️")

# Custom UI
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header {
        background: linear-gradient(90deg, #0f172a, #1e293b);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>👁️ VisionAI: Advanced OCT Analysis</h1></div>',
            unsafe_allow_html=True)

# =========================
# 🔹 LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

CLASS_NAMES = sorted(['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'GLAUCOMA', 'MH'])

# =========================
# 🔹 PREDICTION FUNCTION
# =========================
def predict(image):
    img = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    img_array = np.asarray(img).astype('float32')

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    return preds

# =========================
# 🔹 UI
# =========================
file = st.file_uploader("📤 Upload Patient OCT Image", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file)
    probs = predict(img)

    label = CLASS_NAMES[np.argmax(probs)]
    conf = np.max(probs) * 100

    # 🔹 Image comparison
    st.subheader("🖱️ Interactive Comparison Slider")

    if os.path.exists("healthy.jpg"):
        ref_img = Image.open("healthy.jpg")

        image_comparison(
            img1=ImageOps.fit(img, (1000, 450)),
            img2=ImageOps.fit(ref_img, (1000, 450)),
            label1=f"PATIENT: {label}",
            label2="HEALTHY REFERENCE",
            width=1100,
            in_memory=True
        )
    else:
        st.warning("Healthy image not found")
        st.image(img)

    # 🔹 Results
    st.markdown("---")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"""
        <div class="diagnosis-card">
            <h3 style="color:#64748b;">DETECTION RESULT</h3>
            <h1 style="color:#1e3a8a;">{label}</h1>
            <p>Confidence: <b>{conf:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = px.bar(
            x=CLASS_NAMES,
            y=probs * 100,
            color=CLASS_NAMES,
            title="AI Probability Confidence per Class"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("VisionAI 2026 | Research & Educational Use")