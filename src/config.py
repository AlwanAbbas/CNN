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
KAGGLE_DATASET = "hendryhb/batik-nusantara-batik-indonesia-dataset"  # Identifier dataset Kaggle
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

# ---------------------------------------------------------------------------
# Konfigurasi Flask Web App
# ---------------------------------------------------------------------------
FLASK_HOST  = '0.0.0.0'   # Dengarkan semua interface
FLASK_PORT  = 5000        # Port default Flask
FLASK_DEBUG = True        # Aktifkan debug mode (matikan di production)
FLASK_MAX_CONTENT = 16 * 1024 * 1024  # Batas upload: 16 MB
