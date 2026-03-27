import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

from evaluate import BASE_DIR

# GPU sorunlarını önlemek için CPU modunu aktif edelim (İstersen kapatabilirsin)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "GTSRB", "Train")
IMG_SIZE = 64  # VGG16 için 64x64 ideal bir başlangıçtır

# Veri yükleme fonksiyonu (Öncekiyle aynı mantık)
def load_data():
    images, labels = [], []
    for class_id in range(43):
        p1 = os.path.join(DATA_PATH, str(class_id))
        p2 = os.path.join(DATA_PATH, format(class_id, '05d'))
        path = p1 if os.path.isdir(p1) else p2 if os.path.isdir(p2) else None
        if path:
            for img_name in os.listdir(path):
                if img_name.lower().endswith((".ppm", ".png", ".jpg")):
                    img = cv2.imread(os.path.join(path, img_name))
                    if img is not None:
                        images.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
                        labels.append(class_id)
    return np.array(images) / 255.0, np.array(labels)

X, y = load_data()
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# VGG16 MODELİ (Transfer Learning)
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False # Önceden eğitilmiş katmanları dondur

model = keras.Sequential([
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(43, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

model.save("../models/vgg16_model.h5")