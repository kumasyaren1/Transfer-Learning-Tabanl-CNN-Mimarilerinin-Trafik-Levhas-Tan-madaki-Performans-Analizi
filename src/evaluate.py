"""
evaluate.py
-----------
Tüm eğitilmiş modelleri tek bir scriptle test eder ve karşılaştırmalı
sonuçlar üretir.

Kullanım:
    python evaluate.py                        # Tüm modelleri değerlendir
    python evaluate.py --model mobilenet      # Sadece bir modeli değerlendir

Çıktılar (outputs/ klasörüne kaydedilir):
    - Her model için confusion matrix görseli
    - Her model için sınıf bazlı classification report (CSV)
    - Tüm modelleri karşılaştıran özet tablo (CSV)
"""

import os
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ─────────────────────────────────────────────
# 1. YAPILANDIRMA
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Veri yolları
TEST_PATH  = os.path.join(BASE_DIR, "..", "data", "GTSRB", "Test")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Değerlendirilecek modeller: isim → (model dosyası, giriş boyutu)
# Yeni bir model eklemek için sadece buraya bir satır ekleyin.
# Model kaydı: isim → (model dosyası, giriş boyutu, preprocessing fonksiyonu)
# preprocessing None ise basit /255.0 normalizasyonu uygulanır.
# Transfer learning modelleri kendi preprocessing fonksiyonlarını gerektirir;
# aksi halde eğitim/test uyumsuzluğu doğrudan accuracy kaybına yol açar.
from tensorflow.keras.applications import mobilenet_v2, resnet50, vgg16

MODEL_REGISTRY = {
    "simple_cnn"  : ("traffic_sign_model.h5",      32,  None),
    "mobilenet_v2": ("mobilenet_v2_final.h5",       96,  mobilenet_v2.preprocess_input),
    "resnet50"    : ("resnet50_model.h5",            64,  resnet50.preprocess_input),
    "vgg16"       : ("vgg16_model.h5",              64,  vgg16.preprocess_input),
}

# 43 sınıfın Türkçe isimleri
CLASSES = {
    0: 'Hız Sınırı (20km/s)',        1: 'Hız Sınırı (30km/s)',
    2: 'Hız Sınırı (50km/s)',        3: 'Hız Sınırı (60km/s)',
    4: 'Hız Sınırı (70km/s)',        5: 'Hız Sınırı (80km/s)',
    6: 'Hız Sınırı Sonu (80km/s)',   7: 'Hız Sınırı (100km/s)',
    8: 'Hız Sınırı (120km/s)',       9: 'Geçme Yasağı',
    10: 'Kamyonlar İçin Geçme Yasağı', 11: 'Kavşak Önceliği',
    12: 'Ana Yol',                   13: 'Yol Ver',
    14: 'Dur',                       15: 'Giriş Yasak',
    16: 'Kamyon Yasak',              17: 'Giriş Yok (Ters Yön)',
    18: 'Dikkat',                    19: 'Sola Tehlikeli Viraj',
    20: 'Sağa Tehlikeli Viraj',      21: 'S Virajı',
    22: 'Engebeli Yol',              23: 'Kaygan Yol',
    24: 'Sağdan Daralan Yol',        25: 'Yol Çalışması',
    26: 'Trafik Işıkları',           27: 'Yaya Geçidi',
    28: 'Okul Geçidi',               29: 'Bisiklet Geçebilir',
    30: 'Kar/Buz Riski',             31: 'Vahşi Hayvan Çıkabilir',
    32: 'Hız Sınırları Sonu',        33: 'Sağa Mecburi Yön',
    34: 'Sola Mecburi Yön',          35: 'İleri Mecburi Yön',
    36: 'İleri veya Sağa Mecburi',   37: 'İleri veya Sola Mecburi',
    38: 'Sağdan Gidiniz',            39: 'Soldan Gidiniz',
    40: 'Ada Etrafında Dönünüz',     41: 'Geçme Yasağı Sonu',
    42: 'Kamyon Geçme Yasağı Sonu',
}

# ─────────────────────────────────────────────
# 2. TEST VERİSİNİ YÜKLEME
# ─────────────────────────────────────────────

def load_test_data(img_size: int):
    """
    GTSRB Test klasöründen görselleri okur, yeniden boyutlandırır
    ve normalize eder.

    Beklenen klasör yapısı (orijinal GTSRB formatı):
        data/GTSRB/Test/
            Images/
                00000.ppm
                00001.ppm
                ...
            GT-final_test.csv   ← ';' ile ayrılmış, ClassId son sütun

    Döndürür:
        images : np.ndarray  shape=(N, img_size, img_size, 3), float32 [0-1]
        labels : np.ndarray  shape=(N,), int
    """
    import pandas as pd

    csv_path    = os.path.join(TEST_PATH, "GT-final_test.csv")
    images_dir  = os.path.join(TEST_PATH, "Images")

    # Dosya varlık kontrolleri
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Etiket dosyası bulunamadı: {csv_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Görseller klasörü bulunamadı: {images_dir}")

    # CSV'yi oku — separator noktalı virgül
    df = pd.read_csv(csv_path, sep=";")

    # Beklenen sütunlar: Filename, ClassId
    if "Filename" not in df.columns or "ClassId" not in df.columns:
        raise ValueError(
            f"CSV sütunları beklenmedik: {list(df.columns)}\n"
            "'Filename' ve 'ClassId' sütunları olmalı."
        )

    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, row["Filename"])
        img = cv2.imread(img_path)

        if img is None:
            # Görsel okunamazsa sessizce atla, uyar
            print(f"  ⚠ Okunamadı, atlandı: {img_path}")
            continue

        # OpenCV BGR okur → RGB'ye çevir (transfer learning modelleri RGB bekler)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(int(row["ClassId"]))

    if len(images) == 0:
        raise RuntimeError(
            "Hiç görsel yüklenemedi.\n"
            f"CSV: {csv_path}\n"
            f"Görseller: {images_dir}"
        )

    # Ham piksel değerleri döndürülür [0, 255] float32
    # Her modelin kendi preprocessing fonksiyonu evaluate_model() içinde uygulanır
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    print(f"  ✓ Test seti yüklendi: {len(images)} görsel, {img_size}x{img_size}")
    return images, labels


# ─────────────────────────────────────────────
# 3. INFERENCE TIME ÖLÇÜMÜ
# ─────────────────────────────────────────────

def measure_inference_time(model, sample_batch: np.ndarray, n_runs: int = 50) -> float:
    """
    Modelin tek bir görsel üzerindeki ortalama çıkarım süresini ölçer.

    Neden önemli?
    Edge cihazlarda gerçek zamanlı çalışma için inference hızı,
    doğruluk kadar kritiktir. Bu metrik modelleri bu açıdan karşılaştırır.

    Parametreler:
        model       : Yüklenmiş Keras modeli
        sample_batch: Tek bir görselin batch formatı → shape (1, H, W, 3)
        n_runs      : Kararlı ortalama için kaç kez çalıştırılacak

    Döndürür:
        float: Milisaniye cinsinden ortalama inference süresi
    """
    # İlk çağrı genellikle daha yavaş olur (JIT derleme vb.), ısınma turu:
    model.predict(sample_batch, verbose=0)

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample_batch, verbose=0)
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

    return elapsed_ms


# ─────────────────────────────────────────────
# 4. CONFUSION MATRIX KAYDETME
# ─────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, model_name: str):
    """
    43x43 confusion matrix'i PNG olarak kaydeder.

    Confusion matrix ne gösterir?
    Satırlar gerçek sınıfları, sütunlar tahmin edilen sınıfları gösterir.
    Köşegen dışındaki yüksek değerler, modelin hangi levhaları
    birbirine karıştırdığını ortaya koyar.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,       # 43x43 = 1849 hücre; sayı yazmak okunaksız olur
        fmt="d",
        cmap="Blues",
        xticklabels=range(43),
        yticklabels=range(43),
        ax=ax,
        linewidths=0.3,
        linecolor="gray",
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=16, pad=15)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)

    save_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix kaydedildi: {save_path}")


# ─────────────────────────────────────────────
# 5. TEK MODEL DEĞERLENDİRME
# ─────────────────────────────────────────────

def evaluate_model(model_key: str) -> dict:
    """
    Bir modeli yükler, test seti üzerinde çalıştırır ve
    tüm metrikleri hesaplar.

    Döndürür:
        dict: Model adı, accuracy, macro F1, weighted F1,
              inference süresi, model boyutu
    """
    model_file, img_size, preprocess_fn = MODEL_REGISTRY[model_key]
    model_path = os.path.join(BASE_DIR, "..", "models", model_file)

    print(f"\n{'='*55}")
    print(f"  Model: {model_key.upper()}")
    print(f"{'='*55}")

    # Model dosyası var mı?
    if not os.path.exists(model_path):
        print(f"  ⚠ Model bulunamadı, atlanıyor: {model_path}")
        return None

    # Modeli yükle
    # compile=False → Optimizer yüklenmez; sadece ağırlıklar ve mimari yüklenir.
    # Evaluation sırasında optimizer'a ihtiyaç yoktur (sadece forward pass yapıyoruz).
    # Bu aynı zamanda Keras versiyon uyumsuzluklarını da önler.
    print("  → Model yükleniyor...")
    model = keras.models.load_model(model_path, compile=False)
    # Inference için model derlenmesine gerek yok; predict() doğrudan çalışır.

    # Model boyutu (MB)
    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    # Test verisini bu modelin beklediği boyuta göre yükle
    print("  → Test verisi yükleniyor...")
    x_test, y_test = load_test_data(img_size)

    # Model'e özgü preprocessing uygula
    # Her mimari farklı bir normalizasyon bekler:
    #   simple_cnn   → [0, 1]  (/ 255.0)
    #   mobilenet_v2 → [-1, 1] (preprocess_input)
    #   resnet50     → ImageNet mean subtraction
    #   vgg16        → ImageNet mean subtraction
    if preprocess_fn is not None:
        x_test_processed = preprocess_fn(x_test.copy())
    else:
        x_test_processed = x_test / 255.0

    # Tahminler
    print("  → Tahminler yapılıyor...")
    y_pred_probs = model.predict(x_test_processed, batch_size=32, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # ── Temel metrikler ──────────────────────
    acc = accuracy_score(y_test, y_pred)

    # Macro F1: Her sınıfa eşit ağırlık verir → dengesiz veri setinde önemli
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Weighted F1: Sınıf örnek sayısına göre ağırlıklı → genel performans
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy      : %{acc * 100:.2f}")
    print(f"  F1 (macro)    : {f1_macro:.4f}")
    print(f"  F1 (weighted) : {f1_weighted:.4f}")
    print(f"  Model Boyutu  : {model_size_mb:.2f} MB")

    # ── Sınıf bazlı rapor → CSV ──────────────
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[CLASSES[i] for i in range(43)],
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(OUTPUT_DIR, f"classification_report_{model_key}.csv")
    report_df.to_csv(report_path)
    print(f"  ✓ Sınıf raporu kaydedildi: {report_path}")

    # ── Confusion matrix görseli ─────────────
    save_confusion_matrix(y_test, y_pred, model_key)

    # ── Inference time ────────────────────────
    sample = x_test_processed[0:1]  # Tek bir görsel, batch boyutu 1
    inf_time_ms = measure_inference_time(model, sample)
    print(f"  Inference Süresi (tek görsel): {inf_time_ms:.2f} ms")

    return {
        "Model"               : model_key,
        "Accuracy (%)"        : round(acc * 100, 2),
        "F1 Macro"            : round(f1_macro, 4),
        "F1 Weighted"         : round(f1_weighted, 4),
        "Inference (ms)"      : round(inf_time_ms, 2),
        "Model Boyutu (MB)"   : round(model_size_mb, 2),
    }


# ─────────────────────────────────────────────
# 6. KARŞILAŞTIRMA TABLOSU
# ─────────────────────────────────────────────

def save_comparison_table(results: list):
    """
    Tüm modellerin sonuçlarını tek bir CSV tablosuna yazar.
    Bu tablo doğrudan bitirme raporunuza girebilir.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    save_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(save_path, index=False)

    print(f"\n{'='*55}")
    print("  KARŞILAŞTIRMA TABLOSU")
    print(f"{'='*55}")
    print(df.to_string(index=False))
    print(f"\n  ✓ Tablo kaydedildi: {save_path}")


# ─────────────────────────────────────────────
# 7. ANA AKIŞ
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Traffic Sign modeli değerlendirme aracı"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Değerlendirilecek model. Varsayılan: all",
    )
    args = parser.parse_args()

    # Hangi modeller değerlendirilecek?
    keys_to_run = (
        list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    )

    results = []
    for key in keys_to_run:
        result = evaluate_model(key)
        if result is not None:
            results.append(result)

    # Birden fazla model çalıştırıldıysa karşılaştırma tablosu oluştur
    if len(results) > 1:
        save_comparison_table(results)
    elif len(results) == 1:
        print("\nTek model değerlendirildi; karşılaştırma tablosu oluşturulmadı.")

    print("\n✅ Değerlendirme tamamlandı. Çıktılar:", OUTPUT_DIR)


if __name__ == "__main__":
    main()