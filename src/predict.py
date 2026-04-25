"""
predict.py
----------
Modul inferensi model CNN Batik Nusantara.
(Diperbarui untuk membaca format Native Checkpoint .ckpt)
"""

import os
import sys
import io
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SAVED_MODELS_DIR, OUTPUTS_DIR, IMG_HEIGHT, IMG_WIDTH, INPUT_SHAPE
from src.model import bangun_model 

_model_cache       = None
_class_names_cache = None

def muat_model_predict():
    global _model_cache, _class_names_cache

    if _model_cache is not None:
        return _model_cache, _class_names_cache

    aman_save_path = os.path.join(SAVED_MODELS_DIR, 'bobot_model.ckpt')

    if not os.path.exists(aman_save_path + ".index"):
        raise FileNotFoundError(f"File bobot tidak ditemukan di: {aman_save_path}")

    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    if os.path.exists(class_names_path):
        with open(class_names_path, "rb") as f:
            _class_names_cache = pickle.load(f)
    else:
        raise FileNotFoundError(f"[ERROR] File kelas tidak ditemukan: {class_names_path}")

    num_classes = len(_class_names_cache)
    _model_cache = bangun_model(INPUT_SHAPE, num_classes)
    _model_cache.load_weights(aman_save_path).expect_partial()

    return _model_cache, _class_names_cache

def prediksi_gambar(img_bytes):
    from PIL import Image
    model, class_names = muat_model_predict()
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0]

    semua = [{"kelas": class_names[i], "confidence": float(pred[i]) * 100} for i in range(len(class_names))]
    semua.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "kelas"      : semua[0]["kelas"],
        "confidence" : semua[0]["confidence"],
        "semua_kelas": semua,
        "model_path" : os.path.join(SAVED_MODELS_DIR, 'bobot_model.ckpt'),
    }