# =============================================================================
# CNN BATIK NUSANTARA - Google Colab Notebook
# Dataset: hendryhb/cnn-batik-nusantara (Kaggle)
# Struktur Modular dengan %%writefile
# =============================================================================

# ===========================================================================
# SEL 1: PERSIAPAN INSTALASI & DIREKTORI
# Jalankan sel ini PERTAMA KALI
# ===========================================================================
# !pip install -q kaggle
# !mkdir -p src data/raw data/processed saved_models outputs

# ===========================================================================
# SEL 2: UPLOAD KAGGLE API KEY
# Upload file kaggle.json dari komputer Anda
# ===========================================================================
# from google.colab import files
# files.upload()  # Upload kaggle.json
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# ===========================================================================
# SEL 3: TULIS src/config.py
# Jalankan sel ini untuk membuat file konfigurasi
# ===========================================================================

# %%writefile src/config.py
"""
config.py
---------
File konfigurasi pusat untuk seluruh proyek CNN Batik Nusantara.
Berisi semua hyperparameter, path dataset, dan konstanta pelatihan.
"""

import os

# ---------------------------------------------------------------------------
# Hyperparameter Model
# ---------------------------------------------------------------------------
IMG_HEIGHT = 150          # Tinggi gambar setelah di-resize
IMG_WIDTH = 150           # Lebar gambar setelah di-resize
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)  # Tuple ukuran gambar
CHANNELS = 3              # Jumlah channel warna (RGB)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)  # Shape input model

BATCH_SIZE = 32           # Jumlah sampel per batch
EPOCHS = 30               # Jumlah epoch pelatihan
LEARNING_RATE = 1e-3      # Learning rate untuk optimizer Adam
DROPOUT_RATE = 0.5        # Dropout rate untuk regularisasi

VALIDATION_SPLIT = 0.2    # 20% data untuk validasi
RANDOM_SEED = 42          # Seed untuk reproduktibilitas

# ---------------------------------------------------------------------------
# Konfigurasi Dataset Kaggle
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "hendryhb/cnn-batik-nusantara"  # Identifier dataset Kaggle
DATASET_NAME = "cnn-batik-nusantara"             # Nama folder hasil ekstraksi

# ---------------------------------------------------------------------------
# Path Direktori Proyek
# ---------------------------------------------------------------------------
BASE_DIR = os.getcwd()                            # Direktori kerja saat ini
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")          # Folder file .zip dari Kaggle
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # Folder dataset terekstrak
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# Path File Output
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, "model_batik.keras")
HISTORY_SAVE_PATH = os.path.join(OUTPUTS_DIR, "training_history.pkl")
ACCURACY_PLOT_PATH = os.path.join(OUTPUTS_DIR, "plot_accuracy.png")
LOSS_PLOT_PATH = os.path.join(OUTPUTS_DIR, "plot_loss.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")


# ===========================================================================
# SEL 4: TULIS src/data_loader.py
# Jalankan sel ini untuk membuat file data loader
# ===========================================================================

# %%writefile src/data_loader.py
"""
data_loader.py
--------------
Modul untuk mengunduh dataset dari Kaggle, mengekstrak arsip,
dan mempersiapkan data generator untuk pelatihan dan validasi.
"""

import os
import zipfile
import subprocess
import tensorflow as tf
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
    lalu mengekstrak file .zip ke direktori data/processed.

    Prasyarat:
        - File kaggle.json sudah dikonfigurasi di ~/.kaggle/kaggle.json
        - Library 'kaggle' sudah terinstal

    Returns:
        str: Path direktori dataset yang sudah terekstrak.
    """
    print("=" * 60)
    print("[INFO] Memulai proses pengunduhan dataset dari Kaggle...")
    print(f"[INFO] Dataset: {KAGGLE_DATASET}")
    print("=" * 60)

    # Buat direktori tujuan jika belum ada
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

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
    # Dataset biasanya terekstrak dalam subfolder di RAW_DIR
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

    # Telusuri direktori untuk menemukan folder dengan subdirektori kelas
    for root, dirs, files in os.walk(direktori_awal):
        # Cek apakah direktori saat ini memiliki subdirektori (kelas gambar)
        subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if len(subdirs) > 1:
            print(f"[INFO] Ditemukan {len(subdirs)} kelas: {subdirs}")
            print(f"[INFO] Direktori dataset: {root}")
            return root

    # Fallback ke RAW_DIR jika tidak ditemukan struktur yang sesuai
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


# ===========================================================================
# SEL 5: TULIS src/model.py
# Jalankan sel ini untuk membuat file arsitektur CNN
# ===========================================================================

# %%writefile src/model.py
"""
model.py
--------
Modul yang mendefinisikan arsitektur Convolutional Neural Network (CNN)
untuk klasifikasi citra batik nusantara.

Arsitektur:
    3 blok Conv2D + MaxPooling2D -> Flatten -> Dense -> Dropout -> Dense (Softmax)
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization,
    Input
)


def bangun_model(input_shape, num_classes):
    """
    Membangun dan mengembalikan model CNN Sequential.

    Arsitektur Model:
        Blok 1: Conv2D(32) -> BatchNorm -> MaxPooling2D
        Blok 2: Conv2D(64) -> BatchNorm -> MaxPooling2D
        Blok 3: Conv2D(128) -> BatchNorm -> MaxPooling2D
        Head:   Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(num_classes, Softmax)

    Args:
        input_shape (tuple): Shape input gambar, misal (150, 150, 3).
        num_classes (int): Jumlah kelas output (jumlah motif batik).

    Returns:
        tf.keras.Sequential: Model CNN yang sudah dibangun (belum dikompilasi).
    """
    print("\n" + "=" * 60)
    print("[INFO] Membangun arsitektur CNN...")
    print(f"[INFO] Input shape: {input_shape}")
    print(f"[INFO] Jumlah kelas output: {num_classes}")
    print("=" * 60)

    model = Sequential(name="CNN_Batik_Nusantara")

    # -----------------------------------------------------------------------
    # Input Layer
    # -----------------------------------------------------------------------
    model.add(Input(shape=input_shape))

    # -----------------------------------------------------------------------
    # Blok Konvolusi 1
    # Ekstraksi fitur dasar (tepi, tekstur sederhana)
    # -----------------------------------------------------------------------
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv1_1'
    ))
    model.add(BatchNormalization(name='bn1'))       # Normalisasi untuk stabilitas training
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool1'
    ))

    # -----------------------------------------------------------------------
    # Blok Konvolusi 2
    # Ekstraksi fitur menengah (pola geometris batik)
    # -----------------------------------------------------------------------
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv2_1'
    ))
    model.add(BatchNormalization(name='bn2'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool2'
    ))

    # -----------------------------------------------------------------------
    # Blok Konvolusi 3
    # Ekstraksi fitur kompleks (motif batik spesifik)
    # -----------------------------------------------------------------------
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv3_1'
    ))
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv3_2'
    ))
    model.add(BatchNormalization(name='bn3'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool3'
    ))

    # -----------------------------------------------------------------------
    # Fully Connected Layers (Head Classifier)
    # -----------------------------------------------------------------------
    model.add(Flatten(name='flatten'))              # Ratakan tensor 3D ke 1D

    model.add(Dense(
        units=256,
        activation='relu',
        name='dense1'
    ))
    model.add(Dropout(
        rate=0.5,                                   # Matikan 50% neuron secara acak
        name='dropout1'
    ))

    # -----------------------------------------------------------------------
    # Output Layer
    # Jumlah neuron = jumlah kelas, aktivasi Softmax untuk probabilitas
    # -----------------------------------------------------------------------
    model.add(Dense(
        units=num_classes,
        activation='softmax',
        name='output'
    ))

    # Tampilkan ringkasan arsitektur model
    model.summary()

    return model


# Alias bahasa Inggris untuk kompatibilitas impor
build_model = bangun_model


# ===========================================================================
# SEL 6: TULIS src/train.py
# Jalankan sel ini untuk membuat skrip pelatihan
# ===========================================================================

# %%writefile src/train.py
"""
train.py
--------
Skrip utama untuk melatih model CNN Batik Nusantara.

Alur kerja:
    1. Unduh dan persiapkan dataset dari Kaggle
    2. Bangun arsitektur CNN
    3. Kompilasi model
    4. Latih model dengan data generator
    5. Simpan model dan histori pelatihan

Cara menjalankan di Colab:
    !python src/train.py
"""

import os
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

# Tambahkan direktori root ke path agar impor modul src bisa dilakukan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Impor modul-modul proyek
from src.config import (
    INPUT_SHAPE, EPOCHS, LEARNING_RATE,
    MODEL_SAVE_PATH, HISTORY_SAVE_PATH,
    SAVED_MODELS_DIR, OUTPUTS_DIR
)
from src.data_loader import unduh_dataset, dapatkan_data_generators
from src.model import bangun_model


def main():
    """
    Fungsi utama yang mengorkestrasi seluruh proses pelatihan model.
    """
    print("\n" + "=" * 60)
    print("    CNN BATIK NUSANTARA - PELATIHAN MODEL")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Langkah 1: Verifikasi GPU (opsional tapi dianjurkan di Colab)
    # -----------------------------------------------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPU terdeteksi: {len(gpus)} unit")
        for gpu in gpus:
            print(f"       - {gpu}")
    else:
        print("[PERINGATAN] Tidak ada GPU terdeteksi. Pelatihan menggunakan CPU.")

    # -----------------------------------------------------------------------
    # Langkah 2: Unduh dan Persiapkan Dataset
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 1] Mengunduh dataset dari Kaggle...")
    direktori_dataset = unduh_dataset()

    # -----------------------------------------------------------------------
    # Langkah 3: Buat Data Generator
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 2] Mempersiapkan data generator...")
    train_gen, val_gen, num_classes, class_names = dapatkan_data_generators(
        direktori_dataset
    )

    # -----------------------------------------------------------------------
    # Langkah 4: Bangun Model CNN
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 3] Membangun model CNN...")
    model = bangun_model(INPUT_SHAPE, num_classes)

    # -----------------------------------------------------------------------
    # Langkah 5: Kompilasi Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 4] Mengkompilasi model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',   # Loss untuk multi-kelas one-hot
        metrics=['accuracy']
    )
    print("[INFO] Model berhasil dikompilasi.")
    print(f"[INFO] Optimizer: Adam (lr={LEARNING_RATE})")
    print("[INFO] Loss: Categorical Crossentropy")
    print("[INFO] Metrics: Accuracy")

    # -----------------------------------------------------------------------
    # Langkah 6: Definisikan Callbacks
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 5] Mengatur callbacks pelatihan...")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    callbacks = [
        # Simpan bobot model terbaik berdasarkan val_accuracy
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Hentikan training lebih awal jika tidak ada peningkatan
        EarlyStopping(
            monitor='val_loss',
            patience=7,                    # Tunggu 7 epoch sebelum berhenti
            restore_best_weights=True,
            verbose=1
        ),
        # Kurangi learning rate jika plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                    # Kurangi LR menjadi setengahnya
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    print(f"[INFO] Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau")

    # -----------------------------------------------------------------------
    # Langkah 7: Hitung Steps per Epoch
    # -----------------------------------------------------------------------
    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    validation_steps = max(1, val_gen.samples // val_gen.batch_size)

    print(f"\n[INFO] Steps per epoch (training): {steps_per_epoch}")
    print(f"[INFO] Steps per epoch (validasi): {validation_steps}")

    # -----------------------------------------------------------------------
    # Langkah 8: Latih Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 6] Memulai pelatihan model...")
    print("=" * 60)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # -----------------------------------------------------------------------
    # Langkah 9: Simpan Histori Pelatihan
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 7] Menyimpan histori pelatihan...")
    with open(HISTORY_SAVE_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"[SUKSES] Histori pelatihan disimpan ke: {HISTORY_SAVE_PATH}")

    # -----------------------------------------------------------------------
    # Langkah 10: Simpan Nama Kelas
    # -----------------------------------------------------------------------
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    with open(class_names_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"[SUKSES] Nama kelas disimpan ke: {class_names_path}")

    # -----------------------------------------------------------------------
    # Ringkasan Hasil Pelatihan
    # -----------------------------------------------------------------------
    akurasi_akhir = history.history['accuracy'][-1]
    val_akurasi_akhir = history.history['val_accuracy'][-1]
    print("\n" + "=" * 60)
    print("    RINGKASAN HASIL PELATIHAN")
    print("=" * 60)
    print(f"[HASIL] Akurasi Training Akhir : {akurasi_akhir:.4f} ({akurasi_akhir*100:.2f}%)")
    print(f"[HASIL] Akurasi Validasi Akhir : {val_akurasi_akhir:.4f} ({val_akurasi_akhir*100:.2f}%)")
    print(f"[HASIL] Epoch terbaik model: {len(history.history['accuracy'])}")
    print(f"[HASIL] Model disimpan di: {MODEL_SAVE_PATH}")
    print("=" * 60)
    print("\n[INFO] Pelatihan selesai! Jalankan evaluate.py untuk evaluasi.")


if __name__ == "__main__":
    main()


# ===========================================================================
# SEL 7: TULIS src/evaluate.py
# Jalankan sel ini untuk membuat skrip evaluasi
# ===========================================================================

# %%writefile src/evaluate.py
"""
evaluate.py
-----------
Skrip untuk mengevaluasi model CNN Batik Nusantara yang sudah dilatih.

Fungsi:
    - Muat model dari file .keras
    - Evaluasi pada data validasi
    - Visualisasi kurva Akurasi dan Loss (Training vs Validation)
    - Tampilkan Classification Report dari scikit-learn
    - Buat dan simpan Confusion Matrix

Cara menjalankan di Colab:
    !python src/evaluate.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Gunakan backend non-interaktif untuk Colab
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Tambahkan direktori root ke path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Impor konfigurasi proyek
from src.config import (
    MODEL_SAVE_PATH, HISTORY_SAVE_PATH,
    ACCURACY_PLOT_PATH, LOSS_PLOT_PATH,
    CONFUSION_MATRIX_PATH, OUTPUTS_DIR
)
from src.data_loader import unduh_dataset, dapatkan_data_generators


def muat_model(path_model):
    """
    Memuat model Keras dari file .keras.

    Args:
        path_model (str): Path lengkap ke file model .keras.

    Returns:
        tf.keras.Model: Model yang sudah dimuat.

    Raises:
        FileNotFoundError: Jika file model tidak ditemukan.
    """
    if not os.path.exists(path_model):
        raise FileNotFoundError(
            f"[ERROR] File model tidak ditemukan: {path_model}\n"
            "Pastikan Anda sudah menjalankan train.py terlebih dahulu!"
        )

    print(f"[INFO] Memuat model dari: {path_model}")
    model = tf.keras.models.load_model(path_model)
    print("[SUKSES] Model berhasil dimuat!")
    model.summary()
    return model


def muat_histori(path_histori):
    """
    Memuat histori pelatihan dari file pickle.

    Args:
        path_histori (str): Path ke file histori .pkl.

    Returns:
        dict or None: Dictionary histori pelatihan, atau None jika tidak ada.
    """
    if not os.path.exists(path_histori):
        print(f"[PERINGATAN] File histori tidak ditemukan: {path_histori}")
        return None

    with open(path_histori, 'rb') as f:
        histori = pickle.load(f)
    print(f"[INFO] Histori pelatihan dimuat dari: {path_histori}")
    return histori


def plot_akurasi(histori, path_simpan):
    """
    Membuat dan menyimpan grafik akurasi training vs validasi.

    Args:
        histori (dict): Dictionary histori Keras berisi 'accuracy' dan 'val_accuracy'.
        path_simpan (str): Path tujuan penyimpanan grafik.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(histori['accuracy']) + 1)

    ax.plot(epochs, histori['accuracy'],
            'b-o', linewidth=2, markersize=4,
            label='Training Accuracy')
    ax.plot(epochs, histori['val_accuracy'],
            'r-o', linewidth=2, markersize=4,
            label='Validation Accuracy')

    # Tandai akurasi terbaik
    best_val_acc = max(histori['val_accuracy'])
    best_epoch = histori['val_accuracy'].index(best_val_acc) + 1
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
               label=f'Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})')

    ax.set_title('Kurva Akurasi Training vs Validasi', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Akurasi', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(path_simpan, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUKSES] Grafik akurasi disimpan ke: {path_simpan}")


def plot_loss(histori, path_simpan):
    """
    Membuat dan menyimpan grafik loss training vs validasi.

    Args:
        histori (dict): Dictionary histori Keras berisi 'loss' dan 'val_loss'.
        path_simpan (str): Path tujuan penyimpanan grafik.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(histori['loss']) + 1)

    ax.plot(epochs, histori['loss'],
            'b-o', linewidth=2, markersize=4,
            label='Training Loss')
    ax.plot(epochs, histori['val_loss'],
            'r-o', linewidth=2, markersize=4,
            label='Validation Loss')

    # Tandai loss terkecil pada data validasi
    best_val_loss = min(histori['val_loss'])
    best_epoch = histori['val_loss'].index(best_val_loss) + 1
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
               label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')

    ax.set_title('Kurva Loss Training vs Validasi', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_simpan, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUKSES] Grafik loss disimpan ke: {path_simpan}")


def evaluasi_model(model, val_generator, class_names):
    """
    Mengevaluasi model pada data validasi dan menampilkan:
    - Skor akurasi keseluruhan
    - Classification Report (Precision, Recall, F1-Score)
    - Confusion Matrix

    Args:
        model (tf.keras.Model): Model yang akan dievaluasi.
        val_generator: Data generator validasi.
        class_names (list): Daftar nama kelas.

    Returns:
        tuple: (y_true, y_pred) - Label asli dan prediksi.
    """
    print("\n" + "=" * 60)
    print("[INFO] Menjalankan evaluasi model pada data validasi...")
    print("=" * 60)

    # Evaluasi menggunakan built-in Keras evaluate
    print("\n[INFO] Evaluasi Keras (Loss & Accuracy):")
    val_loss, val_acc = model.evaluate(val_generator, verbose=1)
    print(f"\n[HASIL] Loss Validasi   : {val_loss:.4f}")
    print(f"[HASIL] Akurasi Validasi: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # Dapatkan prediksi untuk semua sampel validasi
    print("\n[INFO] Menghasilkan prediksi untuk Classification Report...")
    val_generator.reset()  # Reset generator ke posisi awal

    y_pred_proba = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)   # Indeks kelas dengan probabilitas tertinggi
    y_true = val_generator.classes              # Label asli

    # -----------------------------------------------------------------------
    # Classification Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("    CLASSIFICATION REPORT")
    print("=" * 60)
    laporan = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    print(laporan)

    # Simpan laporan ke file teks
    laporan_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(laporan_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT - CNN BATIK NUSANTARA\n")
        f.write("=" * 60 + "\n")
        f.write(laporan)
    print(f"[SUKSES] Classification Report disimpan ke: {laporan_path}")

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, path_simpan):
    """
    Membuat dan menyimpan visualisasi Confusion Matrix menggunakan Seaborn.

    Args:
        y_true (array): Label kelas asli.
        y_pred (array): Label kelas hasil prediksi.
        class_names (list): Daftar nama kelas untuk label sumbu.
        path_simpan (str): Path tujuan penyimpanan gambar.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Hitung confusion matrix dalam bentuk persentase (normalisasi baris)
    cm_persen = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Ukuran figure dinamis berdasarkan jumlah kelas
    n_kelas = len(class_names)
    ukuran = max(10, n_kelas * 0.8)
    fig, ax = plt.subplots(figsize=(ukuran, ukuran * 0.8))

    # Buat heatmap
    sns.heatmap(
        cm_persen,
        annot=True,
        fmt='.2%',              # Format persentase
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )

    ax.set_title('Confusion Matrix (Normalisasi)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Prediksi', fontsize=12, labelpad=10)
    ax.set_ylabel('Label Asli', fontsize=12, labelpad=10)

    # Rotasi label sumbu x agar mudah dibaca
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(path_simpan, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUKSES] Confusion Matrix disimpan ke: {path_simpan}")


def main():
    """
    Fungsi utama yang mengorkestrasi seluruh proses evaluasi.
    """
    print("\n" + "=" * 60)
    print("    CNN BATIK NUSANTARA - EVALUASI MODEL")
    print("=" * 60)

    # Buat direktori output jika belum ada
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Langkah 1: Muat Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 1] Memuat model yang sudah dilatih...")
    model = muat_model(MODEL_SAVE_PATH)

    # -----------------------------------------------------------------------
    # Langkah 2: Muat Histori Pelatihan
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 2] Memuat histori pelatihan...")
    histori = muat_histori(HISTORY_SAVE_PATH)

    # -----------------------------------------------------------------------
    # Langkah 3: Visualisasi Histori Pelatihan
    # -----------------------------------------------------------------------
    if histori is not None:
        print("\n[LANGKAH 3] Membuat grafik pelatihan...")
        plot_akurasi(histori, ACCURACY_PLOT_PATH)
        plot_loss(histori, LOSS_PLOT_PATH)
    else:
        print("\n[LANGKAH 3] Melewati visualisasi histori (file tidak tersedia).")

    # -----------------------------------------------------------------------
    # Langkah 4: Siapkan Data Validasi
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 4] Mempersiapkan data validasi...")

    # Coba muat nama kelas dari file yang disimpan saat training
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    if os.path.exists(class_names_path):
        with open(class_names_path, 'rb') as f:
            class_names = pickle.load(f)
        print(f"[INFO] Nama kelas dimuat dari: {class_names_path}")
    else:
        # Jika tidak ada, buat ulang generator
        print("[INFO] Membuat ulang data generator...")
        direktori_dataset = unduh_dataset()
        _, val_gen, _, class_names = dapatkan_data_generators(direktori_dataset)

    # Buat ulang val_generator untuk evaluasi
    from src.data_loader import dapatkan_data_generators
    direktori_dataset = unduh_dataset()
    _, val_gen, num_classes, class_names = dapatkan_data_generators(direktori_dataset)

    # -----------------------------------------------------------------------
    # Langkah 5: Evaluasi Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 5] Mengevaluasi model...")
    y_true, y_pred = evaluasi_model(model, val_gen, class_names)

    # -----------------------------------------------------------------------
    # Langkah 6: Buat Confusion Matrix
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 6] Membuat Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names, CONFUSION_MATRIX_PATH)

    # -----------------------------------------------------------------------
    # Ringkasan Akhir
    # -----------------------------------------------------------------------
    akurasi_akhir = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 60)
    print("    RINGKASAN EVALUASI")
    print("=" * 60)
    print(f"[HASIL] Akurasi Keseluruhan: {akurasi_akhir:.4f} ({akurasi_akhir*100:.2f}%)")
    print(f"\n[OUTPUT] File yang dihasilkan:")
    print(f"  - Grafik Akurasi : {ACCURACY_PLOT_PATH}")
    print(f"  - Grafik Loss    : {LOSS_PLOT_PATH}")
    print(f"  - Confusion Matrix: {CONFUSION_MATRIX_PATH}")
    print(f"  - Laporan Klasifikasi: {os.path.join(OUTPUTS_DIR, 'classification_report.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
