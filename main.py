import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import gdown
import os

# ================== إعداد الصفحة ==================
st.set_page_config(
    page_title="PneumoAI - Chest X-Ray Diagnostic Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== تحميل الموديل ==================
@st.cache_resource
def load_model():
    model_path = "model/best_pneumonia_model.h5"
    
    url = "https://drive.google.com/uc?id=1znXRCYbXE2AoCg0AYhukNHZr61PPoRR_&confirm=t&export=download"
    
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        with st.spinner("📥 Downloading model... (30-70 ثانية)"):
            try:
                gdown.download(url, model_path, quiet=False, fuzzy=True)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                st.stop()
    
    return tf.keras.models.load_model(model_path)

model = load_model()

# ================== التصنيفات ==================
CLASS_NAMES = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}

# ================== معالجة الصورة ==================
def preprocess_image(image):
    img_array = np.array(image.convert("L"))
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================== Custom CSS ==================
st.markdown("""
    <style>
    .main { background-color: #f0f8ff; }
    h1 { color: #0277bd; text-align: center; }
    .stButton > button { background-color: #4fc3f7; color: white; border-radius: 8px; font-size: 18px; }
    .stButton > button:hover { background-color: #0288d1; }
    .result-box { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 20px 0; }
    .diagnosis { font-size: 36px; font-weight: bold; text-align: center; }
    .confidence { font-size: 24px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ================== Sidebar ==================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
    st.title("PneumoAI")
    st.markdown("**Chest X-Ray Diagnostic Tool**")
    st.info("For educational and research use only. Not a substitute for professional medical diagnosis.")

# ================== الصفحة الرئيسية ==================
st.title("🩺 PneumoAI - Chest X-Ray Analysis")
st.markdown("**Advanced AI Tool for Detecting Pneumonia Types**")

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)
    
    if st.button("Analyze Image Now"):
        with st.spinner("Processing image... This may take 10-30 seconds"):
            processed = preprocess_image(image)
            preds = model.predict(processed)[0]
            pred_class = np.argmax(preds)
            confidence = float(preds[pred_class]) * 100
            result = CLASS_NAMES[pred_class]
            
            with st.container():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                color = "green" if result == "Normal" else "red" if result == "Bacterial Pneumonia" else "orange"
                st.markdown(f'<p class="diagnosis" style="color:{color};">Diagnosis: {result}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="confidence">Confidence Level: {confidence:.2f}%</p>', unsafe_allow_html=True)
                st.subheader("Detailed Probabilities")
                for i, name in CLASS_NAMES.items():
                    prob = float(preds[i] * 100)
                    st.progress(prob / 100)
                    st.write(f"{name}: {prob:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("For educational and research purposes only.")
