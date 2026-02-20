# preprocess.py
import cv2
import numpy as np

IMG_SIZE = 224   # ← غيّري إذا كان الموديل متدرب على حجم مختلف

def preprocess_image(image_path):
    """
    Prepare a single X-ray image for model prediction.
    Returns: numpy array of shape (1, IMG_SIZE, IMG_SIZE, 1)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)          # (H, W, 1)
    img = np.expand_dims(img, axis=0)           # (1, H, W, 1)
    
    return img