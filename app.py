import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.express as px
from streamlit_image_comparison import image_comparison
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="VisionAI | Precision Diagnostics", layout="wide", page_icon="👁️")

# Custom CSS for a medical aesthetic
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header { background: linear-gradient(90deg, #0f172a, #1e293b); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; }
    .diagnosis-card { background: white; padding: 25px; border-radius: 12px; border-left: 8px solid #3b82f6; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .feature-list { line-height: 1.8; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>👁️ VisionAI: Advanced OCT Analysis</h1></div>', unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_vision_model():
    return tf.keras.models.load_model('/content/drive/MyDrive/retina_model.h5', compile=False)

model = load_vision_model()
CLASS_NAMES = sorted(['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'GLAUCOMA', 'MH'])

# --- PREDICTION (HARDCODED TO 0-255) ---
def get_prediction(image, model):
    img = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    # Keeping raw pixels (0-255) as confirmed working
    img_array = np.asarray(img).astype('float32')

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)

    # Ensuring the input shape is (1, 224, 224, 3)
    return model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]

# --- MAIN UI ---
file = st.file_uploader("📤 Upload Patient OCT Image", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file)
    probs = get_prediction(img, model)
    label = CLASS_NAMES[np.argmax(probs)]
    conf = np.max(probs) * 100

    # 1. INTERACTIVE SLIDER
    st.subheader("🖱️ Interactive Comparison Slider")
    healthy_path = '/content/drive/MyDrive/healthy.jpg'

    if os.path.exists(healthy_path):
        ref_img = Image.open(healthy_path)
        image_comparison(
            img1=ImageOps.fit(img, (1000, 450)),
            img2=ImageOps.fit(ref_img, (1000, 450)),
            label1=f"PATIENT: {label}",
            label2="HEALTHY REFERENCE",
            width=1100,
            in_memory=True
        )
    else:
        st.warning("Healthy reference image not found in Drive. Using side-by-side view.")
        st.image(img, width=500)

    # 2. ANALYSIS RESULTS
    st.markdown("---")
    col_res, col_chart = st.columns([1, 1.5], gap="large")

    with col_res:
        st.markdown(f"""
            <div class="diagnosis-card">
                <h3 style="margin:0; color:#64748b;">DETECTION RESULT</h3>
                <h1 style="color:#1e3a8a; font-size: 3rem;">{label}</h1>
                <p style="font-size:1.2rem;">Match Probability: <b>{conf:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Clinical Key Features")
        with st.container():
            if label in ["DME", "DR"]:
                st.info("**What the AI sees:** Dark, fluid-filled 'cyst' pockets and retinal thickening.")
            elif label in ["AMD", "DRUSEN"]:
                st.info("**What the AI sees:** Wavy bumps (Drusen) or elevations at the RPE layer.")
            elif label == "MH":
                st.info("**What the AI sees:** A distinct anatomical break or hole in the central macula.")
            else:
                st.info("**What the AI sees:** Abnormal layer patterns or fluid accumulation characteristic of " + label)

    with col_chart:
        fig = px.bar(x=CLASS_NAMES, y=probs*100, color=CLASS_NAMES,
                     title="AI Probability Confidence per Class",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(xaxis_title="", yaxis_title="Probability (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Developed for Research & Educational Purposes | VisionAI 2026")
