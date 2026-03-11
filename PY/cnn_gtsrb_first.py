import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Veri seti yolu (OneDrive yerine kısa bir yol önerilir)
data_path = "../data/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"

images = []
labels = []

for class_id in range(43):
    class_path = os.path.join(data_path, format(class_id, '05d'))

    # Sadece geçerli dosyaları işle
    for img_name in os.listdir(class_path):
        if not img_name.lower().endswith(".ppm"):
            continue  # PPM olmayan dosyaları atla

        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print("Dosya okunamadı:", img_path)
            continue  # Boş resimleri atla

        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append(class_id)

images = np.array(images)
labels = np.array(labels)

# Normalizasyon
images = images / 255.0

# Train / validation split
x_train, x_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

print("Train shape:", x_train.shape, "Validation shape:", x_val.shape)

# CNN modeli
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(43, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Modeli eğit
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_val, y_val)
)

# Modeli kaydet
model.save("traffic_sign_model.h5")
print("Model başarıyla kaydedildi!")