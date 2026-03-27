# Trafik Levhasi Analizi (GTSRB)

Bu proje, GTSRB veri seti uzerinde birden fazla derin ogrenme modelini egitmek, test etmek ve karsilastirmak icin hazirlanmistir.

## Ozellikler

- Basit CNN, MobileNetV2, ResNet50 ve VGG16 modelleriyle calisma
- Model bazli test ve performans olcumu
- Confusion matrix ve classification report uretimi
- Coklu model karsilastirma tablosu olusturma

## Proje Yapisi

- `data/GTSRB/`: Egitim ve test verisi
- `src/train.py`: Basit CNN egitimi
- `src/test.py`: Basit CNN testi
- `src/mobilenet/train_mobilenet.py`: MobileNetV2 egitimi
- `src/mobilenet/test_mobilenet.py`: MobileNetV2 testi
- `src/models/resnet50/train_resnet50.py`: ResNet50 egitimi
- `src/models/resnet50/test_resnet50.py`: ResNet50 testi
- `src/models/vgg16/train_vgg16.py`: VGG16 egitimi
- `src/models/vgg16/test_vgg16.py`: VGG16 testi
- `src/evaluate.py`: Tum modelleri degerlendirip raporlar
- `models/`: Kaydedilmis model dosyalari
- `outputs/`: Raporlar ve confusion matrix ciktilari

## Gereksinimler

Python 3.10 (onerilir)

Bagimliliklari kurmak icin:

```bash
pip install -r req.txt
```

## Kullanim

### 1) Basit CNN Egitimi

```bash
python src/train.py
```

### 2) Basit CNN Testi

```bash
python src/test.py
```

### 3) MobileNetV2 Egitimi/Testi

```bash
python src/mobilenet/train_mobilenet.py
python src/mobilenet/test_mobilenet.py
```

### 4) Tum Modelleri Degerlendirme

```bash
python src/evaluate.py
```

Sadece tek bir modeli degerlendirmek icin:

```bash
python src/evaluate.py --model mobilenet_v2
```

Mevcut secenekler: `simple_cnn`, `mobilenet_v2`, `resnet50`, `vgg16`, `all`

## Uretilen Ciktilar

`outputs/` klasorunde su dosyalar olusur:

- `classification_report_<model>.csv`
- `confusion_matrix_<model>.png`
- `model_comparison.csv` (birden fazla model degerlendirildiginde)

## Notlar

- Veri seti klasor yapisi GTSRB formati ile uyumlu olmalidir.
- Model dosyalari `models/` klasorunde bulunmalidir.
- Transfer learning modellerinde dogru preprocessing fonksiyonlarinin kullanilmasi kritik oneme sahiptir.
