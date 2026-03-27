import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 1. DOSYA YOLLARINI AYARLA
# Bu script 'src' içinde olduğu için 'data' klasörü bir üst dizindedir.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "GTSRB", "Train")

images = []
labels = []

print("Veriler yükleniyor, lütfen bekleyin...")

if not os.path.isdir(DATA_PATH):
    raise FileNotFoundError(f"Veri klasoru bulunamadi: {DATA_PATH}")

# 2. VERİ SETİNİ YÜKLE
for class_id in range(43):
    padded_class_path = os.path.join(DATA_PATH, format(class_id, '05d'))
    plain_class_path = os.path.join(DATA_PATH, str(class_id))

    if os.path.isdir(padded_class_path):
        class_path = padded_class_path
    elif os.path.isdir(plain_class_path):
        class_path = plain_class_path
    else:
        continue

    for img_name in os.listdir(class_path):
        if img_name.lower().endswith((".ppm", ".png", ".jpg", ".jpeg")):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, (32, 32))
                images.append(img)
                labels.append(class_id)

if len(images) == 0:
    raise RuntimeError("Hic gorsel yuklenemedi. DATA_PATH ve klasor yapisini kontrol edin.")

images = np.array(images, dtype=np.float32) / 255.0  # Normalizasyon
labels = np.array(labels, dtype=np.int32)

# 3. EĞİTİM VE TEST AYRIMI
x_train, x_val, y_train, y_val = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

# 4. MODELİ OLUŞTUR
model = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, (3,3), activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(43, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Eğitim başlıyor...")
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))


SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "traffic_sign_model.h5")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
model.save(SAVE_PATH)

print(f"Model başarıyla kaydedildi: {SAVE_PATH}")