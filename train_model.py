# train_model.py
# ملف تدريب كامل + تقييم مفصل لموديل الـ Pneumonia (Normal / Bacterial / Viral)

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. إعداد المسارات ────────────────────────────────────────────────────────
# غيّري المسارات دي حسب مكان الداتا على جهازك

DATA_FOLDER = r"C:\Users\YourName\Downloads\chest-xrays-bacterial-viral-pneumonia-normal"
# أو أي مسار عندك، مثلاً: r"D:\Datasets\chest-xrays-..."

TRAIN_IMAGES_DIR = os.path.join(DATA_FOLDER, r"E:\Projects\train_images\train_images")
LABELS_CSV        = os.path.join(DATA_FOLDER, r"E:\Projects\labels_train.csv")

MODEL_SAVE_PATH = r".\model\pneumonia_model.h5"          # هيحفظ الموديل هنا
BEST_MODEL_PATH = r".\model\best_pneumonia_model.h5"     # أفضل نسخة أثناء التدريب

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 40

# ─── 2. قراءة الـ labels ────────────────────────────────────────────────────────
df = pd.read_csv(LABELS_CSV)
df['class_id'] = df['class_id'].astype(str)  # لازم string عشان categorical

print(f"عدد الصور في الـ CSV: {len(df)}")
print(df['class_id'].value_counts())

# ─── 3. Data Generators مع Augmentation ────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 15,
    width_shift_range  = 0.12,
    height_shift_range = 0.12,
    shear_range        = 0.12,
    zoom_range         = 0.12,
    horizontal_flip    = True,
    brightness_range   = [0.9, 1.1],
    validation_split   = 0.20
)

train_gen = datagen.flow_from_dataframe(
    dataframe    = df,
    directory    = TRAIN_IMAGES_DIR,
    x_col        = 'file_name',
    y_col        = 'class_id',
    target_size  = (IMG_SIZE, IMG_SIZE),
    color_mode   = 'grayscale',
    class_mode   = 'categorical',
    batch_size   = BATCH_SIZE,
    subset       = 'training',
    shuffle      = True
)

val_gen = datagen.flow_from_dataframe(
    dataframe    = df,
    directory    = TRAIN_IMAGES_DIR,
    x_col        = 'file_name',
    y_col        = 'class_id',
    target_size  = (IMG_SIZE, IMG_SIZE),
    color_mode   = 'grayscale',
    class_mode   = 'categorical',
    batch_size   = BATCH_SIZE,
    subset       = 'validation',
    shuffle      = False
)

# ─── 4. بناء الموديل ───────────────────────────────────────────────────────────
base_model = EfficientNetB0(
    include_top    = False,
    weights        = None,              # None لأن grayscale (مش RGB)
    input_shape    = (IMG_SIZE, IMG_SIZE, 1)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.35)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer = 'adam',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

# ─── 5. Callbacks ───────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor    = 'val_loss',
        patience   = 8,
        restore_best_weights = True,
        verbose    = 1
    ),
    ReduceLROnPlateau(
        monitor    = 'val_loss',
        factor     = 0.5,
        patience   = 4,
        min_lr     = 1e-6,
        verbose    = 1
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor    = 'val_accuracy',
        save_best_only = True,
        mode       = 'max',
        verbose    = 1
    )
]

# ─── 6. التدريب ────────────────────────────────────────────────────────────────
print("\nبدء التدريب...\n")

history = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = EPOCHS,
    callbacks       = callbacks,
    verbose         = 1
)

# ─── 7. حفظ النسخة النهائية ──────────────────────────────────────────────────
model.save(r".\model\pneumonia_model.h5")
print(f"\nتم حفظ الموديل النهائي في: {model.save(r".\model\pneumonia_model.h5")}")

# ─── 8. تقييم مفصل ─────────────────────────────────────────────────────────────

# أ. تقييم مباشر
val_loss, val_acc = model.evaluate(val_gen, verbose=1)
print(f"\nValidation Loss    : {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc*100:.2f} %")

# ب. Classification Report + Confusion Matrix
print("\nجاري استخراج التنبؤات على Validation set...")
y_pred_prob = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_gen.classes

class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.show()   # لو في jupyter هيظهر
# plt.savefig('confusion_matrix.png')

# ج. منحنيات التدريب
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# plt.savefig('training_curves.png')

print("\n" + "═"*80)
print(" انتهى التقييم – شوفي النتايج أعلاه ")
print("═"*80)