# -*- coding: utf-8 -*-
"""
setup.py
--------
Skrip setup otomatis untuk proyek CNN Batik Nusantara.

Yang dilakukan:
    1. Buat struktur direktori proyek
    2. Install library dari requirements.txt
    3. Deteksi GPU NVIDIA & Instalasi TensorFlow yang kompatibel
    4. Buat file .env dengan konfigurasi default (EfficientNetB0)
    5. Cek & konfigurasi Kaggle API key (kaggle.json)

Cara menjalankan:
    python setup.py
"""

import os
import sys
import shutil
import subprocess

# Paksa output UTF-8 agar aman di Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK  = "[OK]"
ERR = "[!!]"
WRN = "[??]"
INF = "[--]"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  SETUP - CNN BATIK NUSANTARA (AUTO-GPU)")
print("=" * 60)

# ============================================================
# Langkah 1: Buat Struktur Direktori
# ============================================================
print("\n[1/5] Membuat struktur direktori...")
dirs = ["src", "data/raw", "data/processed", "saved_models", "outputs"]
for d in dirs:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    print(f"      {OK} {d}/")
print("      Semua direktori siap.")

# ============================================================
# Langkah 2: Buat Virtual Environment (.venv)
# ============================================================
print(f"\n[2/5] Menyiapkan Virtual Environment (.venv)...")
venv_dir = os.path.join(BASE_DIR, ".venv")

if not os.path.exists(venv_dir):
    print(f"      {INF} Membuat .venv baru (proses ini mungkin butuh beberapa detik)...")
    try:
        import venv
        venv.create(venv_dir, with_pip=True)
        print(f"      {OK} .venv berhasil dibuat di: {venv_dir}")
    except Exception as e:
        print(f"      {ERR} Gagal membuat .venv: {e}")
        sys.exit(1)
else:
    print(f"      {OK} .venv sudah ada.")

if sys.platform == "win32":
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    venv_pip    = os.path.join(venv_dir, "Scripts", "pip.exe")
else:
    venv_python = os.path.join(venv_dir, "bin", "python")
    venv_pip    = os.path.join(venv_dir, "bin", "pip")

# Pastikan pip up-to-date
subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ============================================================
# Langkah 3: Deteksi GPU & Install Requirements
# ============================================================
print("\n[3/5] Menginstall library & Konfigurasi TensorFlow...")

# Auto-deteksi GPU NVIDIA
has_nvidia = False
try:
    subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    has_nvidia = True
except FileNotFoundError:
    pass

if has_nvidia:
    print(f"      {OK} GPU NVIDIA Terdeteksi di sistem Anda!")

print("      Pilih konfigurasi TensorFlow untuk `.venv` Anda:")
print("      1. TensorFlow DirectML (Sangat Disarankan - Langsung jalan tanpa perlu install CUDA manual)")
print("      2. TensorFlow NVIDIA Native (Performa maksimal, tapi butuh CUDA Toolkit 11.2 & cuDNN 8.1 terinstall di PC)")
print("      3. TensorFlow Standard (Hanya CPU)")

default_opt = "1"
pilihan_tf = input(f"      Masukkan pilihan (1/2/3) [Ketik enter untuk default: {default_opt}]: ").strip()
if not pilihan_tf:
    pilihan_tf = default_opt

req_path = os.path.join(BASE_DIR, "requirements.txt")

if not os.path.exists(req_path):
    print(f"      {ERR} requirements.txt tidak ditemukan di: {req_path}")
else:
    print(f"      {INF} Menyaring library dan memulai instalasi...")
    try:
        with open(req_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        reqs_to_install = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Buang tensorflow bawaan dari requirements.txt agar tidak bentrok
            if line.lower().startswith("tensorflow"):
                continue 
            reqs_to_install.append(line)

        # Tentukan package TensorFlow berdasarkan OS dan Pilihan
        tf_packages = []
        if pilihan_tf == "1":
            print(f"      {INF} Mode DirectML dipilih. Menginstall tensorflow-cpu==2.10.0 dan plugin DirectML...")
            tf_packages = ["tensorflow-cpu==2.10.0", "tensorflow-directml-plugin"]
        elif pilihan_tf == "2":
            print(f"      {INF} Mode NVIDIA Native dipilih. Menginstall tensorflow==2.10.0 (Support CUDA Windows)...")
            tf_packages = ["tensorflow==2.10.0"]
        else:
            print(f"      {INF} Mode CPU dipilih. Menginstall versi standar...")
            tf_packages = ["tensorflow==2.10.0"]

        full_install_list = reqs_to_install + tf_packages

        print("      " + "-" * 50)
        subprocess.run(
            [venv_python, "-m", "pip", "install"] + full_install_list,
            check=True,
            text=True
        )
            
        print("      " + "-" * 50)
        print(f"      {OK} Semua library berhasil diinstall secara otomatis!")
    except subprocess.CalledProcessError as e:
        print("      " + "-" * 50)
        print(f"      {ERR} Gagal install beberapa package. Anda bisa cek output error di atas.")

# ============================================================
# Langkah 4: Buat file .env
# ============================================================
env_path = os.path.join(BASE_DIR, ".env")
print(f"\n[4/5] Membuat file .env...")

env_content = """\
# ============================================================
# .env — Konfigurasi Lingkungan CNN Batik Nusantara
# ============================================================

# --- Flask Web App ---
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
FLASK_SECRET_KEY=batik-nusantara-secret-key-ganti-ini

# --- Model & Data ---
# Diubah ke 224x224 untuk kompatibilitas EfficientNetB0
IMG_HEIGHT=224
IMG_WIDTH=224
BATCH_SIZE=32
EPOCHS=30
LEARNING_RATE=0.001

# --- Kaggle ---
# Pastikan kaggle.json sudah ada di ~/.kaggle/kaggle.json
KAGGLE_DATASET=hendryhb/batik-nusantara-batik-indonesia-dataset

# --- Path (relatif dari root proyek) ---
DATA_DIR=data
SAVED_MODELS_DIR=saved_models
OUTPUTS_DIR=outputs
MODEL_FILENAME=model_batik.h5
"""

if os.path.exists(env_path):
    print(f"      {WRN} .env sudah ada, melewati pembuatan (tidak ditimpa).")
else:
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    print(f"      {OK} .env berhasil dibuat di: {env_path}")

env_example_path = os.path.join(BASE_DIR, ".env.example")
with open(env_example_path, "w", encoding="utf-8") as f:
    f.write(env_content)
print(f"      {OK} .env.example dibuat sebagai referensi.")

# ============================================================
# Langkah 5: Cek & Konfigurasi Kaggle API Key
# ============================================================
print("\n[5/5] Memeriksa Kaggle API key...")

kaggle_dir   = os.path.join(os.path.expanduser("~"), ".kaggle")
kaggle_path  = os.path.join(kaggle_dir, "kaggle.json")
kaggle_lokal = os.path.join(BASE_DIR, "kaggle.json")
kaggle_ok    = False

if os.path.exists(kaggle_path):
    print(f"      {OK} kaggle.json ditemukan di: {kaggle_path}")
    kaggle_ok = True
elif os.path.exists(kaggle_lokal):
    print(f"      {INF} kaggle.json ada di folder proyek. Menyalin ke ~/.kaggle/ ...")
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(kaggle_lokal, kaggle_path)
    if sys.platform != "win32":
        os.chmod(kaggle_path, 0o600)
    print(f"      {OK} Disalin ke: {kaggle_path}")
    kaggle_ok = True
else:
    print(f"      {ERR} kaggle.json TIDAK ditemukan!")
    print(f"      {INF} Letakkan kaggle.json di folder proyek ini lalu jalankan setup.py lagi.")

# ============================================================
# Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("  RINGKASAN SETUP")
print("=" * 60)

if not kaggle_ok:
    print(f"\n{ERR} Kaggle API Key belum terkonfigurasi.")
    print("     1. Kunjungi: https://www.kaggle.com/settings/account")
    print("     2. Bagian 'API' -> klik 'Create New Token'")
    print("     3. Letakkan kaggle.json di folder proyek ini")
    print("     4. Jalankan: python setup.py\n")

if kaggle_ok:
    print(f"\n{OK} Setup selesai! Virtual environment & GPU siap digunakan.")
    print("\n     Gunakan executable python dari `.venv`:")
    if sys.platform == "win32":
        print("     # Aktifkan venv:")
        print("     .venv\\Scripts\\activate")
    else:
        print("     # Aktifkan venv:")
        print("     source .venv/bin/activate")
    
    print("\n     Lalu jalankan project:")
    print("     # Latih model:")
    print("     python -m src.train")

print("\n" + "=" * 60)