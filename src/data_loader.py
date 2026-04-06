"""
data_loader.py
--------------
Modul untuk mengunduh dataset dari Kaggle, mengekstrak arsip,
dan mempersiapkan data generator untuk pelatihan dan validasi.
"""

import os
import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Impor konfigurasi dari modul config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    KAGGLE_DATASET, RAW_DIR, PROCESSED_DIR,
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE,
    VALIDATION_SPLIT, RANDOM_SEED
)


def unduh_dataset():
    """
    Mengunduh dataset dari Kaggle menggunakan Kaggle CLI,
    lalu mengekstrak file .zip ke direktori data/raw.

    Prasyarat:
        - File kaggle.json sudah dikonfigurasi di ~/.kaggle/kaggle.json
        - Library 'kaggle' sudah terinstal

    Returns:
        str: Path direktori dataset yang sudah terekstrak.
    """
    print("=" * 60)
    print("[INFO] Memeriksa dataset dari Kaggle...")
    print(f"[INFO] Dataset: {KAGGLE_DATASET}")
    print("=" * 60)

    # Buat direktori tujuan jika belum ada
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Guard: cek apakah dataset sudah ada, skip download jika sudah ada
    # Ini mencegah re-download berulang saat evaluate.py dipanggil
    # -----------------------------------------------------------------------
    direktori_terdeteksi = _temukan_direktori_dataset(RAW_DIR)
    isi_raw = os.listdir(RAW_DIR) if os.path.exists(RAW_DIR) else []
    if isi_raw:  # Sudah ada file/folder di data/raw
        print(f"[INFO] Dataset sudah ada di: {direktori_terdeteksi} — melewati download.")
        return direktori_terdeteksi

    # Perintah untuk mengunduh dataset menggunakan Kaggle API
    perintah_unduh = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", RAW_DIR,      # Simpan .zip ke folder data/raw
        "--unzip"            # Langsung ekstrak setelah unduhan selesai
    ]

    print(f"[INFO] Menjalankan perintah: {' '.join(perintah_unduh)}")

    try:
        hasil = subprocess.run(
            perintah_unduh,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[SUKSES] Output: {hasil.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Gagal mengunduh dataset: {e.stderr}")
        raise

    # Temukan direktori dataset yang sudah terekstrak
    return _temukan_direktori_dataset(RAW_DIR)


def _temukan_direktori_dataset(direktori_awal):
    """
    Fungsi internal untuk mencari direktori utama dataset
    yang berisi subdirektori kelas gambar.

    Args:
        direktori_awal (str): Direktori tempat memulai pencarian.

    Returns:
        str: Path direktori yang berisi folder-folder kelas gambar.
    """
    print(f"\n[INFO] Mencari struktur dataset di: {direktori_awal}")

    # 1. Prioritas Pertama: Cari folder dengan nama "train" yang memiliki kelas
    for root, dirs, files in os.walk(direktori_awal):
        if os.path.basename(root).lower() == "train":
            subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
            if len(subdirs) > 1:
                print(f"[INFO] Menggunakan folder 'train' dengan {len(subdirs)} kelas di: {root}")
                return root

    # 2. Prioritas Kedua: Cari folder yang memiliki subdirektori terbanyak
    max_subdirs = 0
    best_root = direktori_awal
    
    for root, dirs, files in os.walk(direktori_awal):
        subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if len(subdirs) > max_subdirs:
            max_subdirs = len(subdirs)
            best_root = root

    if max_subdirs > 1:
        print(f"[INFO] Menggunakan folder dengan kelas terbanyak ({max_subdirs} kelas) di: {best_root}")
        return best_root

    # Fallback ke direktori awal jika tidak ditemukan struktur yang sesuai
    print(f"[PERINGATAN] Struktur kelas tidak ditemukan, menggunakan: {direktori_awal}")
    return direktori_awal


def dapatkan_data_generators(direktori_dataset=None):
    """
    Membuat data generator untuk training dan validasi.

    Data dibagi 80% Training dan 20% Validation.
    Training generator dilengkapi dengan augmentasi data.
    Validation generator hanya melakukan normalisasi.

    Args:
        direktori_dataset (str, optional): Path ke direktori dataset.
            Jika None, fungsi akan mencari otomatis di RAW_DIR.

    Returns:
        tuple: (train_generator, val_generator, num_classes, class_names)
            - train_generator: Generator data training
            - val_generator: Generator data validasi
            - num_classes (int): Jumlah kelas yang terdeteksi
            - class_names (list): Daftar nama kelas
    """
    # Cari direktori dataset jika tidak disediakan
    if direktori_dataset is None:
        direktori_dataset = _temukan_direktori_dataset(RAW_DIR)

    print("\n" + "=" * 60)
    print("[INFO] Mempersiapkan Data Generator...")
    print(f"[INFO] Direktori dataset: {direktori_dataset}")
    print(f"[INFO] Ukuran gambar: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Split validasi: {VALIDATION_SPLIT * 100:.0f}%")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # ImageDataGenerator untuk Training
    # Dilengkapi augmentasi untuk meningkatkan generalisasi model
    # -----------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # Normalisasi piksel ke rentang [0, 1]
        rotation_range=20,             # Rotasi acak hingga 20 derajat
        width_shift_range=0.2,         # Geser horizontal hingga 20%
        height_shift_range=0.2,        # Geser vertikal hingga 20%
        shear_range=0.15,              # Transformasi shear
        zoom_range=0.15,               # Zoom acak hingga 15%
        horizontal_flip=True,          # Flip horizontal acak
        fill_mode='nearest',           # Isi piksel baru dengan tetangga terdekat
        validation_split=VALIDATION_SPLIT  # Proporsi data validasi
    )

    # -----------------------------------------------------------------------
    # ImageDataGenerator untuk Validasi
    # Hanya normalisasi, tanpa augmentasi agar evaluasi akurat
    # -----------------------------------------------------------------------
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,            # Normalisasi piksel ke rentang [0, 1]
        validation_split=VALIDATION_SPLIT
    )

    # -----------------------------------------------------------------------
    # Buat generator dari direktori
    # -----------------------------------------------------------------------
    train_generator = train_datagen.flow_from_directory(
        direktori_dataset,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',      # One-hot encoding untuk multi-kelas
        subset='training',             # Gunakan 80% untuk training
        seed=RANDOM_SEED,
        shuffle=True                   # Acak urutan data setiap epoch
    )

    val_generator = val_datagen.flow_from_directory(
        direktori_dataset,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',      # One-hot encoding untuk multi-kelas
        subset='validation',           # Gunakan 20% untuk validasi
        seed=RANDOM_SEED,
        shuffle=False                  # Jangan acak agar evaluasi konsisten
    )

    # Dapatkan informasi kelas
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())

    print(f"\n[INFO] Jumlah kelas terdeteksi: {num_classes}")
    print(f"[INFO] Daftar kelas: {class_names}")
    print(f"[INFO] Total sampel training: {train_generator.samples}")
    print(f"[INFO] Total sampel validasi: {val_generator.samples}")

    return train_generator, val_generator, num_classes, class_names
