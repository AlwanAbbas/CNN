# CNN Batik Nusantara

Sistem klasifikasi citra cerdas berbasis **Convolutional Neural Network (CNN)** untuk mendeteksi berbagai jenis motif batik dari seluruh Nusantara. Proyek ini dilengkapi dengan skrip otomatis untuk instalasi lokal, pipeline pelatihan yang modular, serta **Web App (Flask)** modern yang mendukung prediksi melalui foto upload maupun kamera secara langsung (*real-time*).

---

## 🎯 Fitur Utama

- **Unduh Dataset Otomatis**: Skrip integrasi dengan Kaggle API untuk mengunduh dan mengekstrak dataset `hendryhb/batik-nusantara-batik-indonesia-dataset`.
- **Pipeline Modular**: Terbagi rapi ke dalam beberapa modul: `config.py` (hyperparameters), `data_loader.py` (pemrosesan citra dari Keras), `model.py` (Arsitektur Model CNN), `train.py` (pelatihan), dan `evaluate.py` (evaluasi loss & akurasi).
- **Aplikasi Web Interaktif (Flask)**: Aplikasi web *dark-mode* minimalis yang memungkinkan Anda menggunakan Webcam atau Drag-and-Drop gambar.
- **Dukungan Terminal (CLI)**: Utilitas inferensi *batch processing* langsung dari *command-line*.

---

## 📂 Struktur Direktori

```text
CNN_Batik_Nusantara/
├── app.py                   # Entry point aplikasi Flask Web
├── setup.py                 # Skrip otomatis (buat .venv & install deps)
├── requirements.txt         # Daftar dependency package
├── src/
│   ├── config.py            # Konfigurasi konstanta & variabel path 
│   ├── data_loader.py       # Pemrosesan ImageDataGenerator & Kaggle API
│   ├── model.py             # Definisi arsitektur model
│   ├── train.py             # Skrip melatih model
│   ├── evaluate.py          # Skrip mengevaluasi dan membuat plot metric
│   ├── predict.py           # Skrip inferensi/pemuatan model
│   └── predict_cli.py       # Program CLI untuk prediksi di terminal
├── templates/               
│   └── index.html           # UI Utama Flask
└── static/
    ├── css/style.css        # Desain layout dan tema
    └── js/app.js            # Fungsionalitas Webcam & Fetch API
```

---

## 🚀 Persiapan dan Instalasi (Local Environment)

### 1. Prasyarat
- **Python 3.10 atau 3.11** (⚠️ *Catatan: TensorFlow belum stabil untuk versi Windows pada Python 3.12+ / 3.14.*)
- **Kaggle API Key**: Anda memerlukan akun Kaggle.
  - Masuk ke profil Kaggle Anda → Settings → API → **Create New Token**.
  - Letakkan file `kaggle.json` yang terunduh ke dalam folder proyek ini (sejajar dengan file `setup.py`).

### 2. Jalankan Setup Otomatis
Buka terminal OS Anda, jadikan root direktori ini sebagai *working directory*, lalu jalankan:

```powershell
python setup.py
```

Skrip ini akan secara otomatis:
1. Membuat struktur folder data seperti `data/raw`, `saved_models`, `outputs`.
2. Membuat **Virtual Environment** (`.venv`).
3. Meng-install semua *library* pendukung termasuk TensorFlow, Flask, Scikit-learn.
4. Membuat file konfigurasi tambahan `.env`.

### 3. Aktifkan Virtual Environment
Setelah berhasil, selalu jalankan script dari dalam `.venv`.
- **Windows**:
  ```powershell
  .venv\Scripts\activate
  ```
- **Linux / MacOS**:
  ```bash
  source .venv/bin/activate
  ```

---

## ⚙️ Cara Penggunaan

### 1. Melatih Model
Jalankan file training untuk menginisiasi proses pengunduhan kelas dataset dari Kaggle (jika belum ada) dan melatih model.

```powershell
python -m src.train
```

### 2. Mengevaluasi Model
Setelah selesai dilatih, lihat performa dataset (Loss, Akurasi, Classification Report & Confusion Matrix) dengan cara:

```powershell
python -m src.evaluate
```
*Hasil evaluasi (gambar diagram) akan diekspor ke dalam folder `outputs/`.*

### 3. Menjalankan Web Aplikasi (Flask)
Untuk membuka web aplikasi dan mencoba live-webcam atau mengunggah citra buatan Anda sendiri:

```powershell
python app.py
```
*Buka browser dan akses alamat:* `http://localhost:5000`

### 4. Prediksi lewat Command-Line Interface (CLI)
Jika Anda hanya ingin mengecek suatu gambar tanpa membuka UI Web, Anda juga dapat memanfaatkan CLI bawaan proyek:

```powershell
# Memprediksi satu gambar
python src/predict_cli.py path/ke/gambar1.jpg

# Memprediksi banyak gambar sekaligus
python src/predict_cli.py img1.jpg img2.jpg

# Melihat *confidence* akurasi mendetail dan menyimpannya menjadi file JSON
python src/predict_cli.py sasirangan.jpg --verbose --output hasil_prediksi.json
```

---

## 💡 Troubleshooting

- **Error TensorFlow *(No matching distribution found)***: Itu artinya arsitektur Python lokal yang Anda jalankan terlalu canggih/baru (misal Python 3.14). Disarankan untuk men-downgrade Python, atau menggunakan *Google Colaboratory* khusus proses pelatihannya, di mana Anda tinggal mendownload model hasil perhitungannya saja.
- **Kaggle Unauthorized / 403**: Pastikan `kaggle.json` diletakkan tepat di root proyek sebelum menjalankan `setup.py`.
