# predict.py
import tensorflow as tf
import numpy as np
from preprocess import preprocess_image

# تحميل الموديل مرة واحدة عند بدء البرنامج
try:
    model = tf.keras.models.load_model('model/best_pneumonia_model.h5')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

CLASS_NAMES = {
    0: "Normal",
    1: "Bacterial Pneumonia",
    2: "Viral Pneumonia"
}

def predict_pneumonia(image_path):
    """
    Predict class and confidence for a given X-ray image path.
    Returns: (result_string, confidence_percentage)
    """
    try:
        processed = preprocess_image(image_path)
        preds = model.predict(processed, verbose=0)[0]
        
        pred_class = np.argmax(preds)
        confidence = float(preds[pred_class]) * 100
        
        result = CLASS_NAMES.get(pred_class, "Unknown")
        
        # طباعة للـ debug (يمكن تعليقها لاحقاً)
        print("─" * 60)
        print(f"Image: {image_path}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probabilities → Normal: {preds[0]:.4f} | Bacterial: {preds[1]:.4f} | Viral: {preds[2]:.4f}")
        print("─" * 60)
        
        return result, confidence
    
    except Exception as e:
        print(f"Prediction failed: {e}")
        return "Error during prediction", 0.