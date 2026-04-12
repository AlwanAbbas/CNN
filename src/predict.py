"""
predict.py
----------
Modul inferensi model CNN Batik Nusantara.
Digunakan oleh Flask web app dan CLI terminal.

Fungsi utama:
    - muat_model_predict()  : load model sekali, cache global
    - prediksi_gambar()     : terima bytes gambar, return dict hasil prediksi
"""

import os
import sys
import io
import pickle
import numpy as np

# Tambah root proyek ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_SAVE_PATH, OUTPUTS_DIR,
    IMG_HEIGHT, IMG_WIDTH
)

# ---------------------------------------------------------------------------
# Cache global: model dan nama kelas hanya di-load sekali
# ---------------------------------------------------------------------------
_model_cache      = None
_class_names_cache = None


def muat_model_predict():
    """
    Muat model dari file .keras dan nama kelas dari pickle.
    Menggunakan cache global agar tidak reload berulang kali.

    Returns:
        tuple: (model, class_names)

    Raises:
        FileNotFoundError: Jika model atau file kelas tidak ditemukan.
    """
    global _model_cache, _class_names_cache

    if _model_cache is not None:
        return _model_cache, _class_names_cache

    # Impor TensorFlow di sini agar startup cepat
    import tensorflow as tf

    # Validasi file model
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di: {MODEL_SAVE_PATH}\n"
            "Latih model terlebih dahulu dengan: py -m src.train"
        )

    print(f"[INFO] Memuat model dari: {MODEL_SAVE_PATH}")
    _model_cache = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print("[INFO] Model berhasil dimuat.")

    # Muat nama kelas
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    if os.path.exists(class_names_path):
        with open(class_names_path, "rb") as f:
            _class_names_cache = pickle.load(f)
        print(f"[INFO] {len(_class_names_cache)} kelas dimuat.")
    else:
        # Fallback: gunakan indeks angka jika nama kelas tidak ada
        n = _model_cache.output_shape[-1]
        _class_names_cache = [f"Kelas_{i}" for i in range(n)]
        print(f"[PERINGATAN] class_names.pkl tidak ada. Menggunakan label angka ({n} kelas).")

    return _model_cache, _class_names_cache


def prediksi_gambar(img_bytes):
    """
    Klasifikasikan gambar dari bytes dan kembalikan hasil prediksi.

    Pipeline:
        bytes → PIL Image → resize (150x150) → normalize [0,1]
              → expand dims → model.predict → sort by confidence

    Args:
        img_bytes (bytes): Raw bytes gambar (JPEG/PNG).

    Returns:
        dict: {
            "kelas"       : str   — nama kelas teratas,
            "confidence"  : float — persentase confidence (0-100),
            "semua_kelas" : list of {"kelas": str, "confidence": float}
                            diurut dari tertinggi ke terendah,
            "model_path"  : str   — path model yang digunakan
        }
    """
    from PIL import Image

    model, class_names = muat_model_predict()

    from tensorflow.keras.applications.efficientnet import preprocess_input

    # Preprocessing gambar — wajib pakai EfficientNet preprocess_input agar
    # sesuai dengan normalisasi saat training
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)                      # Normalisasi EfficientNet
    arr = np.expand_dims(arr, axis=0)                # Shape: (1, H, W, 3)

    # Inferensi
    pred = model.predict(arr, verbose=0)[0]          # Shape: (num_classes,)

    # Susun hasil semua kelas
    semua = [
        {"kelas": class_names[i], "confidence": float(pred[i]) * 100}
        for i in range(len(class_names))
    ]
    # Urut dari confidence tertinggi
    semua.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "kelas"      : semua[0]["kelas"],
        "confidence" : semua[0]["confidence"],
        "semua_kelas": semua,
        "model_path" : MODEL_SAVE_PATH,
    }
