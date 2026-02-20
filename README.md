# PneumoAI - Chest X-Ray Pneumonia Detector

**PneumoAI** is an AI-powered diagnostic tool that analyzes chest X-ray images to detect and classify pneumonia into three categories:

- Normal  
- Bacterial Pneumonia  
- Viral Pneumonia  

The project includes:  
- A desktop application (built with CustomTkinter)  
- A web application (built with Streamlit)  
- Model training script  
- Pre-trained deep learning model (based on EfficientNet architecture)

## Project Overview

This project aims to assist in the early detection of pneumonia from chest X-rays using convolutional neural networks (CNNs). The model was trained on a labeled dataset of chest X-ray images containing Normal, Bacterial, and Viral pneumonia cases.

### Final Model Performance (Validation Set)

- **Overall Accuracy**: 83.73%  
- **Validation Loss**: 0.4195  

#### Per-Class Performance

| Class               | Precision | Recall  | F1-Score | Support |
|---------------------|-----------|---------|----------|---------|
| Normal              | 95.31%    | 96.44%  | **95.87%** | 253   |
| Bacterial Pneumonia | 80.33%    | 89.81%  | **84.81%** | 432   |
| Viral Pneumonia     | 77.44%    | 60.64%  | **68.02%** | 249   |

**Key Insight**:  
- Excellent performance on **Normal** cases  
- Strong detection of **Bacterial Pneumonia**  
- Moderate performance on **Viral Pneumonia** (common challenge due to overlapping radiographic patterns)

### Confusion Matrix Highlights

- Normal: 244 correctly classified (only 9 misclassified)  
- Bacterial: 388 correctly classified (38 misclassified as Viral)  
- Viral: 151 correctly classified (92 misclassified as Bacterial)

### Technologies Used

- **Deep Learning Framework**: TensorFlow 2.15  
- **Model Architecture**: EfficientNet-based CNN  
- **Desktop GUI**: CustomTkinter  
- **Web Interface**: Streamlit  
- **Image Processing**: OpenCV, Pillow  
- **Data Handling**: NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  

### Project Structure
PneumoAI/
├── app.py                     # Streamlit web application
├── main.py                    # Desktop GUI version (CustomTkinter)
├── predict.py                 # Prediction logic
├── train_model.py             # Model training & evaluation script
├── model/
│   └── best_pneumonia_model.h5
├── requirements.txt
└── README.md

### How to Run Locally

#### Web Version (Streamlit)

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py

Desktop Version
python main.py

Live Demo
Coming soon on Streamlit Cloud
Important Medical Disclaimer
This tool is developed for educational and research purposes only.
It is NOT a certified medical device and MUST NOT be used for clinical diagnosis.
False positives/negatives are possible, particularly with Viral Pneumonia.
Always consult a qualified radiologist or physician for any medical decision.
Future Improvements

Integrate Grad-CAM heatmaps for visual explanation
Enhance Viral Pneumonia detection (class weighting / data augmentation)
Mobile-responsive design
Docker support for easier deployment

Developed with ❤️ by A'laa Abdelsatar
cairo, Egypt
February 2026
Feel free to fork, contribute, or reach out for collaborations!
Questions? → Open an issue or contact me.
