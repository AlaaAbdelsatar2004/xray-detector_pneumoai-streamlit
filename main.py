import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import gdown
import os

# ================== أول أمر في الكود: إعداد الصفحة ==================
st.set_page_config(
    page_title="PneumoAI - Chest X-Ray Diagnostic Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== تحميل الموديل من Google Drive ==================
@st.cache_resource
def load_model():
    # إنشاء مجلد model لو مش موجود
    os.makedirs("model", exist_ok=True)
    
    model_path = "model/best_pneumonia_model.h5"
    
    # لو الملف مش موجود، حمله من Google Drive
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?export=download&id=1sU4IttL2dPHv9mfERgqBpM7ij0t3ih0d"
        gdown.download(url, model_path, quiet=False)
    
    return tf.keras.models.load_model(model_path)

model = load_model()

CLASS_NAMES = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}

# ================== معالجة الصورة ==================
def preprocess_image(image):
    img_array = np.array(image.convert("L"))
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Custom CSS للديزاين الطبي (سماوي + أبيض + احترافي)
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stApp {
        background-color: #f0f8ff;
    }
    h1 {
        color: #0277bd;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    .stButton > button {
        background-color: #4fc3f7;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 12px 24px;
    }
    .stButton > button:hover {
        background-color: #0288d1;
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .diagnosis {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .confidence {
        font-size: 24px;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #e3f2fd;
    }
    </style>
    """, unsafe_allow_html=True)

# ================== Sidebar (معلومات طبية سريعة) ==================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
    st.title("PneumoAI")
    st.markdown("**Chest X-Ray Diagnostic Tool**")
    st.markdown("AI-assisted analysis for pneumonia detection")
    st.info("For educational and research use only. Not a substitute for professional medical diagnosis.")
    st.markdown("---")
    st.caption("Developed by Alaa Abdelsatar")

# ================== الصفحة الرئيسية ==================
st.title("🩺 PneumoAI - Chest X-Ray Analysis")
st.markdown("**Advanced AI Tool for Detecting Pneumonia Types**")
st.markdown("Upload a chest X-ray image to receive instant classification: Normal, Bacterial Pneumonia, or Viral Pneumonia.")

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

            # عرض النتيجة في صندوق طبي أنيق
            with st.container():
                st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
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
st.caption("Important: This AI tool is designed for research and educational purposes. It is not a replacement for clinical diagnosis by a qualified radiologist or physician.")