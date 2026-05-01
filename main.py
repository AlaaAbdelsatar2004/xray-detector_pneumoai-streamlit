import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import gdown
import os

st.set_page_config(page_title="PneumoAI", page_icon="🩺", layout="wide")

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
                st.success("✅ Model downloaded!")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                st.stop()
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# تحميل الموديل
model = load_model()

CLASS_NAMES = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}

def preprocess_image(image):
    img = np.array(image.convert("L"))
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Custom CSS
st.markdown("""<style>
    .main {background-color: #f0f8ff;}
    h1 {color: #0277bd; text-align: center;}
    .result-box {background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
</style>""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("PneumoAI")
    st.info("Educational & Research Tool Only")

st.title("🩺 PneumoAI - Chest X-Ray Analysis")

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            preds = model.predict(processed)[0]
            pred_class = np.argmax(preds)
            confidence = float(preds[pred_class]) * 100
            result = CLASS_NAMES[pred_class]
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            color = "green" if result == "Normal" else "red" if result == "Bacterial Pneumonia" else "orange"
            st.markdown(f'<h2 style="color:{color}; text-align:center;">{result}</h2>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="text-align:center;">Confidence: {confidence:.2f}%</h3>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
