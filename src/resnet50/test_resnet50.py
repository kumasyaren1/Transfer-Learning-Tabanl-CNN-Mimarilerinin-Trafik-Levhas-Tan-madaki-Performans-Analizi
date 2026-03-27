import os
import cv2
import numpy as np
from tensorflow import keras

# 1. MODELİ YÜKLE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "resnet50_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
print("Model başarıyla yüklendi!")

# 2. TRAFİK LEVHASI İSİMLERİ (Sınıf İndekslerini Anlamlı Metne Dönüştürmek İçin)
CLASSES = {
    0: 'Hız Sınırı (20km/s)', 1: 'Hız Sınırı (30km/s)', 2: 'Hız Sınırı (50km/s)',
    3: 'Hız Sınırı (60km/s)', 4: 'Hız Sınırı (70km/s)', 5: 'Hız Sınırı (80km/s)',
    6: 'Hız Sınırı Sonu (80km/s)', 7: 'Hız Sınırı (100km/s)', 8: 'Hız Sınırı (120km/s)',
    9: 'Geçme Yasağı', 10: 'Kamyonlar İçin Geçme Yasağı', 11: 'Kavşak Önceliği',
    12: 'Ana Yol', 13: 'Yol Ver', 14: 'Dur', 15: 'Giriş Yasak',
    16: 'Kamyon Yasak', 17: 'Giriş Yok (Ters Yön)', 18: 'Dikkat',
    19: 'Sola Tehlikeli Viraj', 20: 'Sağa Tehlikeli Viraj', 21: 'S Virajı',
    22: 'Engebeli Yol', 23: 'Kaygan Yol', 24: 'Sağdan Daralan Yol',
    25: 'Yol Çalışması', 26: 'Trafik Işıkları', 27: 'Yaya Geçidi',
    28: 'Okul Geçidi', 29: 'Bisiklet Geçebilir', 30: 'Kar/Buz Riski',
    31: 'Vahşi Hayvan Çıkabilir', 32: 'Hız Sınırları Sonu', 33: 'Sağa Mecburi Yön',
    34: 'Sola Mecburi Yön', 35: 'İleri Mecburi Yön', 36: 'İleri veya Sağa Mecburi',
    37: 'İleri veya Sola Mecburi', 38: 'Sağdan Gidiniz', 39: 'Soldan Gidiniz',
    40: 'Ada Etrafında Dönünüz', 41: 'Geçme Yasağı Sonu', 42: 'Kamyon Geçme Yasağı Sonu'
}

def predict_traffic_sign(image_path):
    # 3. GÖRSELİ OKU VE ÖN İŞLEMEDEN GEÇİR
    if not os.path.exists(image_path):
        print(f"HATA: Görsel bulunamadı -> {image_path}")
        return

    img = cv2.imread(image_path)
    img_display = img.copy() # Görselleştirmek için orijinal kopya
    
    # Modelin eğitildiği formata getir
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0) # Batch boyutu ekle: (1, 64, 64, 3)

    # 4. TAHMİN YAP
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    # 5. SONUCU YAZDIR
    label_name = CLASSES.get(class_id, "Bilinmeyen Sınıf")
    print("-" * 40)
    print(f"Tahmin: {label_name} (Sınıf ID: {class_id})")
    print(f"Güven Oranı: %{confidence * 100:.2f}")
    print("-" * 40)

test_image = os.path.join(BASE_DIR, "..", "data", "GTSRB", "Test", "00000.ppm")
predict_traffic_sign(test_image)