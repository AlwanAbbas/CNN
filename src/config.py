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
IMG_HEIGHT = 224          # Tinggi gambar — standar MobileNetV2
IMG_WIDTH  = 224          # Lebar gambar  — standar MobileNetV2
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS   = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

BATCH_SIZE    = 32        # Lebih stabil dengan dataset kecil (kurangi noise)
EPOCHS        = 80        # Lebih banyak kesempatan konvergen
LEARNING_RATE = 3e-4      # Sedikit lebih besar untuk konvergen lebih cepat

# Fine-tuning (Fase 2)
FINE_TUNE_AT = 80         # Buka lebih banyak layer untuk adaptasi tekstur batik
FINE_TUNE_LR = 2e-5       # Sedikit lebih besar agar adaptasi tidak terlalu lambat

DROPOUT_RATE     = 0.5
VALIDATION_SPLIT = 0.15   # Lebih sedikit val → lebih banyak data training (547 vs 512)
RANDOM_SEED      = 42

# ---------------------------------------------------------------------------
# Konfigurasi Dataset Kaggle
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "hendryhb/batik-nusantara-batik-indonesia-dataset"
DATASET_NAME   = "cnn-batik-nusantara"

# ---------------------------------------------------------------------------
# Path Direktori Proyek
# ---------------------------------------------------------------------------
BASE_DIR       = os.getcwd()
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# Path File Output
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH      = os.path.join(SAVED_MODELS_DIR, "model_batik.keras")
HISTORY_SAVE_PATH    = os.path.join(OUTPUTS_DIR, "training_history.pkl")
ACCURACY_PLOT_PATH   = os.path.join(OUTPUTS_DIR, "plot_accuracy.png")
LOSS_PLOT_PATH       = os.path.join(OUTPUTS_DIR, "plot_loss.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

# ---------------------------------------------------------------------------
# Konfigurasi Flask Web App
# ---------------------------------------------------------------------------
FLASK_HOST        = '0.0.0.0'
FLASK_PORT        = 5000
FLASK_DEBUG       = True
FLASK_MAX_CONTENT = 16 * 1024 * 1024  # 16 MB
