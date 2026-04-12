# CNN Batik Nusantara

Sistem klasifikasi citra cerdas berbasis **Transfer Learning (EfficientNetB0)** untuk mendeteksi **20 jenis motif batik** dari seluruh Nusantara. Proyek ini dilengkapi dengan skrip otomatis untuk instalasi lokal, pipeline pelatihan 2-fase yang modular, serta **Web App (Flask)** modern yang mendukung prediksi melalui foto upload maupun kamera secara langsung (*real-time*).

---

## 🎯 Fitur Utama

- **Transfer Learning (EfficientNetB0)**: Menggunakan model pretrained ImageNet dengan strategi pelatihan 2-fase (Feature Extraction → Fine-tuning) untuk performa optimal pada dataset kecil.
- **Unduh Dataset Otomatis**: Integrasi dengan Kaggle API untuk mengunduh dan mengekstrak dataset `hendryhb/batik-nusantara-batik-indonesia-dataset`.
- **Pipeline Modular**: Terbagi rapi ke dalam beberapa modul: `config.py` (hyperparameters), `data_loader.py` (preprocessing citra), `model.py` (arsitektur EfficientNetB0), `train.py` (pelatihan 2-fase), dan `evaluate.py` (evaluasi & visualisasi).
- **Aplikasi Web Interaktif (Flask)**: Aplikasi web *dark-mode* minimalis yang memungkinkan Anda menggunakan Webcam atau Drag-and-Drop gambar.
- **Dukungan Terminal (CLI)**: Utilitas inferensi *batch processing* langsung dari *command-line*.

---

## 📊 Performa Model

| Versi | Arsitektur | Val Accuracy | Keterangan |
|---|---|---|---|
| v1 | CNN from Scratch | <20% | Underfitting parah |
| v2 | MobileNetV2 + Transfer Learning | 56.67% | Baseline TL |
| **v3 (terkini)** | **EfficientNetB0 + Label Smoothing** | **71.25%** | Best model |

**Pelatihan terakhir (v3):**
- Dataset: 640 gambar, 20 kelas (32 gambar/kelas)
- Training split: 85% train / 15% val
- Strategi: Fase 1 (Feature Extraction) + Fase 2 (Fine-tuning layer ≥80)

---

## 📂 Struktur Direktori

```text
CNN_Batik_Nusantara/
├── app.py                   # Entry point aplikasi Flask Web
├── setup.py                 # Skrip otomatis (buat .venv & install deps)
├── requirements.txt         # Daftar dependency package
├── src/
│   ├── config.py            # Konfigurasi konstanta & hyperparameter
│   ├── data_loader.py       # Preprocessing EfficientNet & Kaggle download
│   ├── model.py             # Arsitektur EfficientNetB0 Transfer Learning
│   ├── train.py             # Pipeline pelatihan 2-fase
│   ├── evaluate.py          # Evaluasi & plot metric (accuracy, CM, report)
│   ├── predict.py           # Modul inferensi (dipakai Flask & CLI)
│   └── predict_cli.py       # Program CLI untuk prediksi di terminal
├── data/
│   └── raw/                 # Dataset Kaggle (di-download otomatis)
├── saved_models/
│   └── model_batik.keras    # Model terbaik hasil training (git-ignored)
├── outputs/                 # Plot & laporan hasil evaluasi (git-ignored)
├── templates/
│   └── index.html           # UI Utama Flask
└── static/
    ├── css/style.css        # Desain layout dan tema
    └── js/app.js            # Fungsionalitas Webcam & Fetch API
```

---

## 🚀 Persiapan dan Instalasi

### Prasyarat
- **Python 3.10 atau 3.11** (⚠️ TensorFlow belum stabil untuk Python 3.12+ di Windows)
- **Kaggle API Key**: Butuh akun Kaggle.
  - Masuk ke profil Kaggle → Settings → API → **Create New Token**
  - Letakkan file `kaggle.json` ke dalam folder root proyek (sejajar `setup.py`)

---

### 🆕 Clone Pertama Kali (Fresh Install)

```powershell
# 1. Clone repo
git clone https://github.com/<username>/CNN.git
cd CNN

# 2. Letakkan kaggle.json ke folder ini, lalu jalankan setup otomatis
python setup.py
```

Skrip `setup.py` akan secara otomatis:
1. Membuat folder `data/raw`, `saved_models`, `outputs`
2. Membuat **Virtual Environment** (`.venv`)
3. Meng-install semua library (TensorFlow, Flask, Scikit-learn, dll.)
4. Membuat file `.env`

```powershell
# 3. Aktifkan virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 4. Latih model (akan unduh dataset otomatis dari Kaggle)
python -m src.train

# 5. Jalankan Web App
python app.py
```

---

### 🔄 Pull / Sync Jika Repo Sudah Pernah Di-Clone

Jika kamu sudah pernah clone dan ingin sinkronisasi dengan perubahan terbaru:

```powershell
# 1. Simpan perubahan lokal (jika ada) agar tidak konflik
git stash

# 2. Tarik perubahan terbaru dari remote
git pull origin main

# 3. Pulihkan perubahan lokal (jika ada)
git stash pop
```

**Setelah `git pull`, lakukan langkah berikut sesuai kebutuhan:**

#### ✅ Jika ada perubahan di `requirements.txt` (dependency baru)
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

#### ✅ Jika ada perubahan di `src/model.py` atau `src/config.py` (arsitektur/hyperparameter berubah)
> ⚠️ Model lama (`saved_models/model_batik.keras`) **tidak kompatibel** dengan arsitektur baru.
> Wajib latih ulang dari awal.

```powershell
# Hapus model lama
Remove-Item saved_models\model_batik.keras    # Windows
# rm saved_models/model_batik.keras           # Linux / macOS

# Latih ulang
python -m src.train
```

#### ✅ Jika tidak ada perubahan arsitektur (hanya update UI / README / evaluate)
Tidak perlu training ulang. Langsung jalankan:
```powershell
python app.py           # Jalankan web app
python -m src.evaluate  # Re-evaluasi model yang sudah ada
```

---

## ⚙️ Cara Penggunaan

### 1. Melatih Model
```powershell
python -m src.train
```
Pipeline akan otomatis mengunduh dataset dari Kaggle (jika belum ada) dan melatih model 2 fase (Feature Extraction → Fine-tuning).

### 2. Mengevaluasi Model
```powershell
python -m src.evaluate
```
Hasil evaluasi (grafik akurasi, loss, confusion matrix, classification report) diekspor ke `outputs/`.

### 3. Menjalankan Web Aplikasi (Flask)
```powershell
python app.py
```
Buka browser → `http://localhost:5000`

### 4. Prediksi via Command-Line (CLI)
```powershell
# Satu gambar
python src/predict_cli.py path/ke/gambar.jpg

# Banyak gambar sekaligus
python src/predict_cli.py img1.jpg img2.jpg img3.jpg

# Dengan detail confidence & simpan ke JSON
python src/predict_cli.py batik.jpg --verbose --output hasil.json
```

---

## 💡 Troubleshooting

| Masalah | Solusi |
|---|---|
| `No matching distribution found` (TensorFlow) | Downgrade ke Python 3.10/3.11, atau gunakan Google Colab untuk training |
| `Kaggle 401 / 403` | Pastikan `kaggle.json` ada di root proyek sebelum `setup.py` |
| `FileNotFoundError: model_batik.keras` | Belum di-training. Jalankan `python -m src.train` terlebih dahulu |
| Akurasi rendah setelah `git pull` | Arsitektur mungkin berubah. Hapus model lama dan latih ulang (lihat bagian Sync di atas) |
| GPU tidak terdeteksi (Windows) | TF ≥2.11 tidak support GPU native Windows. Gunakan WSL2 atau TF-DirectML plugin |
