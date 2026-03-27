"""
train_mobilenet.py
------------------
MobileNetV2 Transfer Learning eğitimi.

İki aşamalı eğitim stratejisi:
    Aşama 1 — Feature Extraction : Sadece üst katmanlar eğitilir (base_model dondurulur)
    Aşama 2 — Fine-Tuning        : Base modelin son katmanları da eğitime açılır

Preprocessing Notu:
    MobileNetV2, ImageNet'te [-1, 1] aralığıyla eğitilmiştir.
    Bu yüzden mobilenet_v2.preprocess_input() kullanılır.
    evaluate.py de aynı preprocessing'i kullanmalıdır.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── GPU KONTROLÜ ──────────────────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES = "-1" satırını kaldırdık → GPU kullanılacak
# Eğer GPU bellek hatası alırsanız aşağıdaki satırı aktif edin:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU bulundu: {gpus[0].name}")
    # GPU belleğini ihtiyaç kadar kullan, baştan hepsini ayırma
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("⚠ GPU bulunamadı, CPU kullanılacak.")

# ─── YAPILANDIRMA ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..",".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "GTSRB", "Train")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIG = {
    'IMG_SIZE'         : 96,   # MobileNetV2 için önerilen minimum: 96
                                # 64 kullanıldığında ImageNet ağırlıkları 224x224
                                # için optimize edildiğinden bilgi kaybı olur.
    'BATCH_SIZE'       : 32,
    'EPOCHS_PHASE1'    : 15,   # Feature extraction epoch sayısı
    'EPOCHS_PHASE2'    : 10,   # Fine-tuning epoch sayısı
    'LEARNING_RATE_P1' : 1e-3, # Feature extraction için yüksek LR
    'LEARNING_RATE_P2' : 1e-5, # Fine-tuning için çok düşük LR (ağırlıkları korur)
    'VALIDATION_SPLIT' : 0.2,
    'NUM_CLASSES'      : 43,
    'FINE_TUNE_LAYERS' : 30,   # Base modelin son 30 katmanı eğitime açılır
}

# ─── VERİ YÜKLEME ──────────────────────────────────────────────────────────────
def load_data():
    """
    GTSRB Train klasöründen görselleri yükler.

    ÖNEMLİ: mobilenet_v2.preprocess_input() kullanılır.
    Bu fonksiyon piksel değerlerini [0,255]'ten [-1, 1]'e dönüştürür.
    evaluate.py'da da aynı dönüşüm uygulanmalıdır — tutarsızlık
    doğrudan test accuracy düşüşüne yol açar.
    """
    images, labels = [], []
    valid_ext = (".ppm", ".png", ".jpg", ".jpeg")
    img_size  = CONFIG['IMG_SIZE']

    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Train klasörü bulunamadı: {DATA_PATH}")

    for class_id in range(CONFIG['NUM_CLASSES']):
        p_plain  = os.path.join(DATA_PATH, str(class_id))
        p_padded = os.path.join(DATA_PATH, format(class_id, '05d'))
        class_path = p_plain if os.path.isdir(p_plain) else p_padded if os.path.isdir(p_padded) else None

        if class_path is None:
            continue

        for fname in os.listdir(class_path):
            if not fname.lower().endswith(valid_ext):
                continue
            img = cv2.imread(os.path.join(class_path, fname))
            if img is not None:
                # OpenCV BGR okur → RGB'ye çevir (ImageNet modelleri RGB bekler)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(class_id)

    X = np.array(images, dtype=np.float32)
    # MobileNetV2'nin beklediği preprocessing: [0,255] → [-1, 1]
    X = keras.applications.mobilenet_v2.preprocess_input(X)
    y = np.array(labels, dtype=np.int32)
    return X, y


print(f"Veriler yükleniyor ({CONFIG['IMG_SIZE']}x{CONFIG['IMG_SIZE']})...")
X, y = load_data()
print(f"Toplam örnek: {X.shape[0]}, Sınıf sayısı: {len(np.unique(y))}")

x_train, x_val, y_train, y_val = train_test_split(
    X, y,
    test_size=CONFIG['VALIDATION_SPLIT'],
    random_state=42,
    stratify=y,
)

# ─── VERİ AUGMENTATION ─────────────────────────────────────────────────────────
# Trafik levhalarında yatay çevirme (flip) kullanılmaz:
# "Dur" levhasının aynası yine "Dur"dur ama "Sola Dön" levhasının
# aynası "Sağa Dön" olur → label yanlış olur.
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Trafik levhaları için KAPALI tutulmalı
)

# ─── MODEL OLUŞTURMA ───────────────────────────────────────────────────────────
def build_model(trainable_base=False):
    base_model = keras.applications.MobileNetV2(
        input_shape=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3))
    x = base_model(inputs, training=False)  # training=False: BN katmanları inference modunda çalışır
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(CONFIG['NUM_CLASSES'], activation='softmax')(x)

    return keras.Model(inputs, outputs), base_model


# ══════════════════════════════════════════════════════════════════════════════
# AŞAMA 1: FEATURE EXTRACTION
# Base model dondurulur, sadece üst katmanlar eğitilir.
# Amaç: ImageNet ağırlıklarını koruyarak yeni sınıflar için iyi bir başlangıç noktası bulmak.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  AŞAMA 1: Feature Extraction")
print("="*55)

model, base_model = build_model(trainable_base=False)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE_P1']),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks_phase1 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3,
        min_lr=1e-7, verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "mobilenet_phase1_best.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1,
    ),
]

history_p1 = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=CONFIG['BATCH_SIZE']),
    epochs=CONFIG['EPOCHS_PHASE1'],
    validation_data=(x_val, y_val),
    callbacks=callbacks_phase1,
    steps_per_epoch=len(x_train) // CONFIG['BATCH_SIZE'],
)

best_p1_acc = max(history_p1.history['val_accuracy'])
print(f"\n  ✓ Aşama 1 tamamlandı. En iyi val_accuracy: %{best_p1_acc*100:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# AŞAMA 2: FINE-TUNING
# Base modelin son FINE_TUNE_LAYERS katmanı eğitime açılır.
# Önemli: Learning rate çok düşük tutulmalı, yoksa ImageNet ağırlıkları bozulur.
# Önemli: EarlyStopping sıfırdan başlatılır (yeni callback nesnesi).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  AŞAMA 2: Fine-Tuning")
print("="*55)

# Son FINE_TUNE_LAYERS katmanı eğitime aç
base_model.trainable = True
for layer in base_model.layers[:-CONFIG['FINE_TUNE_LAYERS']]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"  Base modelde eğitilebilir katman sayısı: {trainable_count}")

# Fine-tuning için yeniden derleme (düşük LR zorunlu)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE_P2']),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Yeni callback nesneleri — önceki aşamanın sayaçlarını taşımaz
callbacks_phase2 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=3,
        min_lr=1e-8, verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "mobilenet_phase2_best.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1,
    ),
]

history_p2 = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=CONFIG['BATCH_SIZE']),
    epochs=CONFIG['EPOCHS_PHASE2'],
    validation_data=(x_val, y_val),
    callbacks=callbacks_phase2,
    steps_per_epoch=len(x_train) // CONFIG['BATCH_SIZE'],
)

best_p2_acc = max(history_p2.history['val_accuracy'])
print(f"\n  ✓ Aşama 2 tamamlandı. En iyi val_accuracy: %{best_p2_acc*100:.2f}")

# ─── MODELİ KAYDET ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_v2_final.h5")
model.save(MODEL_PATH)
print(f"\n✅ Final model kaydedildi: {MODEL_PATH}")

# Hangi aşama daha iyiydi?
if best_p2_acc >= best_p1_acc:
    print(f"   Fine-tuning, feature extraction'ı geçti: %{best_p2_acc*100:.2f} > %{best_p1_acc*100:.2f}")
else:
    print(f"   Fine-tuning beklenen iyileşmeyi sağlamadı.")
    print(f"   Öneri: models/mobilenet_phase1_best.keras dosyasını kullanın.")