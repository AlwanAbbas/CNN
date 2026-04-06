"""
app.py
------
Flask Web Application untuk klasifikasi citra batik nusantara.

Endpoint:
    GET  /              → halaman utama (index.html)
    POST /predict       → terima gambar, kembalikan JSON prediksi
    GET  /model-info    → info model dan daftar kelas
    GET  /health        → health check

Cara menjalankan:
    py app.py

Akses via browser:
    http://localhost:5000
"""

import os
import sys
import io
import base64
import json
import logging

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Tambah root ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    FLASK_MAX_CONTENT, MODEL_SAVE_PATH, OUTPUTS_DIR
)
from src.predict import prediksi_gambar, muat_model_predict

# ---------------------------------------------------------------------------
# Inisialisasi Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = FLASK_MAX_CONTENT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Format file yang diizinkan untuk upload
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_file(filename):
    """Cek apakah ekstensi file termasuk yang diizinkan."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Tampilkan halaman utama."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint prediksi gambar. Menerima dua format input:

    1. JSON body dengan gambar base64 (dari webcam):
       { "image": "data:image/jpeg;base64,..." }

    2. multipart/form-data dengan file gambar (dari upload form):
       form field: "file"

    Returns:
        JSON: {
            "sukses"     : bool,
            "kelas"      : str,
            "confidence" : float,
            "semua_kelas": [{"kelas": str, "confidence": float}, ...],
            "error"      : str (hanya jika sukses=false)
        }
    """
    try:
        img_bytes = None

        # ── Mode 1: JSON + base64 (dari webcam) ──────────────────────────────
        if request.is_json:
            data = request.get_json()
            image_data = data.get("image", "")
            if not image_data:
                return jsonify({"sukses": False, "error": "Field 'image' kosong."}), 400

            # Format: "data:image/jpeg;base64,<data>"
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            img_bytes = base64.b64decode(image_data)

        # ── Mode 2: multipart file upload ────────────────────────────────────
        elif "file" in request.files:
            f = request.files["file"]
            if f.filename == "":
                return jsonify({"sukses": False, "error": "Tidak ada file dipilih."}), 400
            if not allowed_file(f.filename):
                return jsonify({
                    "sukses": False,
                    "error": f"Format tidak didukung. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400
            img_bytes = f.read()

        else:
            return jsonify({
                "sukses": False,
                "error": "Kirim gambar via JSON (base64) atau form-data (file)."
            }), 400

        # ── Jalankan inferensi ────────────────────────────────────────────────
        hasil  = prediksi_gambar(img_bytes)
        log.info(f"Prediksi: {hasil['kelas']} ({hasil['confidence']:.2f}%)")

        return jsonify({
            "sukses"     : True,
            "kelas"      : hasil["kelas"],
            "confidence" : round(hasil["confidence"], 4),
            "semua_kelas": hasil["semua_kelas"],
        })

    except FileNotFoundError as e:
        log.error(f"Model belum ada: {e}")
        return jsonify({"sukses": False, "error": str(e)}), 503

    except Exception as e:
        log.error(f"Error prediksi: {e}", exc_info=True)
        return jsonify({"sukses": False, "error": f"Terjadi error: {str(e)}"}), 500


@app.route("/model-info")
def model_info():
    """
    Kembalikan informasi model: nama kelas, path, jumlah parameter.
    """
    try:
        model, class_names = muat_model_predict()
        total_params = model.count_params()
        return jsonify({
            "sukses"       : True,
            "jumlah_kelas" : len(class_names),
            "kelas"        : class_names,
            "model_path"   : MODEL_SAVE_PATH,
            "total_params" : total_params,
            "input_shape"  : list(model.input_shape[1:]),
        })
    except FileNotFoundError as e:
        return jsonify({"sukses": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"sukses": False, "error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    model_ada = os.path.exists(MODEL_SAVE_PATH)
    return jsonify({
        "status"   : "ok",
        "model_ada": model_ada,
        "model_path": MODEL_SAVE_PATH,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CNN BATIK NUSANTARA - WEB APP")
    print("=" * 60)
    print(f"  Model   : {MODEL_SAVE_PATH}")
    print(f"  URL     : http://localhost:{FLASK_PORT}")
    print(f"  Debug   : {FLASK_DEBUG}")
    print("=" * 60)

    # Pre-load model agar request pertama tidak lambat
    try:
        muat_model_predict()
        print("[OK] Model berhasil dimuat sebelum server start.")
    except FileNotFoundError:
        print("[??] Model belum ada. Latih dulu dengan: py -m src.train")
        print("[??] Server tetap berjalan, model akan dicoba dimuat saat predict.")

    print(f"\nAkses di: http://localhost:{FLASK_PORT}\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
