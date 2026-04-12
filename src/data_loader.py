"""
data_loader.py
--------------
Modul untuk mengunduh dataset dari Kaggle dan mempersiapkan
data generator untuk pelatihan dan validasi.

Perubahan dari versi sebelumnya:
    - Preprocessing gambar menggunakan mobilenet_v2.preprocess_input
      (normalisasi ke [-1, 1]) agar kompatibel dengan bobot ImageNet.
    - Augmentasi lebih agresif untuk dataset yang sangat kecil:
      * brightness_range, channel_shift_range ditambahkan
      * vertical_flip ditambahkan (batik sering simetris)
"""

import os
import subprocess

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input  # EfficientNetB0

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    KAGGLE_DATASET, RAW_DIR, PROCESSED_DIR,
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE,
    VALIDATION_SPLIT, RANDOM_SEED
)


def unduh_dataset():
    """
    Mengunduh dataset dari Kaggle menggunakan Kaggle CLI.
    Skip download jika data sudah ada di data/raw.

    Returns:
        str: Path direktori dataset yang sudah terekstrak.
    """
    print("=" * 60)
    print("[INFO] Memeriksa dataset dari Kaggle...")
    print(f"[INFO] Dataset: {KAGGLE_DATASET}")
    print("=" * 60)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Guard: skip jika sudah ada
    direktori_terdeteksi = _temukan_direktori_dataset(RAW_DIR)
    isi_raw = os.listdir(RAW_DIR) if os.path.exists(RAW_DIR) else []
    if isi_raw:
        print(f"[INFO] Dataset sudah ada di: {direktori_terdeteksi} — melewati download.")
        return direktori_terdeteksi

    perintah_unduh = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", RAW_DIR,
        "--unzip"
    ]

    print(f"[INFO] Menjalankan: {' '.join(perintah_unduh)}")
    try:
        hasil = subprocess.run(
            perintah_unduh, check=True, capture_output=True, text=True
        )
        print(f"[SUKSES] {hasil.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Gagal mengunduh dataset: {e.stderr}")
        raise

    return _temukan_direktori_dataset(RAW_DIR)


def _temukan_direktori_dataset(direktori_awal):
    """
    Cari direktori utama dataset (berisi folder-folder kelas).

    Prioritas 1: folder bernama 'train' dengan subdirektori kelas.
    Prioritas 2: folder dengan subdirektori terbanyak.

    Args:
        direktori_awal (str): Titik awal pencarian.

    Returns:
        str: Path direktori yang berisi subfolder kelas gambar.
    """
    print(f"\n[INFO] Mencari struktur dataset di: {direktori_awal}")

    # Prioritas 1: folder 'train'
    for root, dirs, files in os.walk(direktori_awal):
        if os.path.basename(root).lower() == "train":
            subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
            if len(subdirs) > 1:
                print(f"[INFO] Folder 'train' ditemukan dengan {len(subdirs)} kelas: {root}")
                return root

    # Prioritas 2: folder dengan kelas terbanyak
    max_subdirs = 0
    best_root = direktori_awal
    for root, dirs, files in os.walk(direktori_awal):
        subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if len(subdirs) > max_subdirs:
            max_subdirs = len(subdirs)
            best_root = root

    if max_subdirs > 1:
        print(f"[INFO] Menggunakan folder dengan {max_subdirs} kelas: {best_root}")
        return best_root

    print(f"[PERINGATAN] Struktur kelas tidak ditemukan, menggunakan: {direktori_awal}")
    return direktori_awal


def dapatkan_data_generators(direktori_dataset=None):
    """
    Membuat ImageDataGenerator untuk training dan validasi.

    Preprocessing: mobilenet_v2.preprocess_input → rentang [-1, 1]
    (wajib agar kompatibel dengan bobot MobileNetV2 pretrained ImageNet).

    Augmentasi training (lebih agresif untuk dataset kecil):
        - Rotasi ±30°
        - Shift horizontal/vertikal 25%
        - Shear 20%
        - Zoom 20%
        - Flip horizontal & vertikal
        - Perubahan brightness [0.7–1.3]
        - Channel shift ±20

    Args:
        direktori_dataset (str, optional): Path direktori dataset.

    Returns:
        tuple: (train_generator, val_generator, num_classes, class_names)
    """
    if direktori_dataset is None:
        direktori_dataset = _temukan_direktori_dataset(RAW_DIR)

    print("\n" + "=" * 60)
    print("[INFO] Mempersiapkan Data Generator (MobileNetV2 preprocessing)...")
    print(f"[INFO] Direktori dataset : {direktori_dataset}")
    print(f"[INFO] Ukuran gambar     : {IMG_HEIGHT}×{IMG_WIDTH}")
    print(f"[INFO] Batch size        : {BATCH_SIZE}")
    print(f"[INFO] Split validasi    : {int(VALIDATION_SPLIT*100)}%")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Generator Training — augmentasi agresif untuk dataset kecil
    # preprocess_input MobileNetV2 menggantikan rescale=1/255
    # -----------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # EfficientNet normalisasi
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.20,
        zoom_range=0.20,
        horizontal_flip=True,
        vertical_flip=True,           # Batik sering memiliki simetri vertikal
        brightness_range=[0.8, 1.2],  # Dikurangi agar warna batik tidak terlalu distorsi
        channel_shift_range=10.0,     # Dikurangi dari 20 → 10 untuk jaga karakteristik warna
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    # -----------------------------------------------------------------------
    # Generator Validasi — hanya preprocessing, tanpa augmentasi
    # -----------------------------------------------------------------------
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # EfficientNet normalisasi
        validation_split=VALIDATION_SPLIT
    )

    # -----------------------------------------------------------------------
    # Flow dari direktori
    # -----------------------------------------------------------------------
    train_generator = train_datagen.flow_from_directory(
        direktori_dataset,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=RANDOM_SEED,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        direktori_dataset,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=RANDOM_SEED,
        shuffle=False
    )

    num_classes  = len(train_generator.class_indices)
    class_names  = list(train_generator.class_indices.keys())

    print(f"\n[INFO] Kelas terdeteksi  : {num_classes}")
    print(f"[INFO] Daftar kelas      : {class_names}")
    print(f"[INFO] Sampel training   : {train_generator.samples}")
    print(f"[INFO] Sampel validasi   : {val_generator.samples}")

    return train_generator, val_generator, num_classes, class_names
