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
# Gunakan backend adaptif: 'Agg' untuk file, atau 'TkAgg'/'Qt5Agg' untuk tampilan interaktif
# Di Colab otomatis menggunakan backend inline, di lokal bisa menampilkan jendela
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
        histori (dict): Dictionary histori Keras.
        path_simpan (str): Path tujuan penyimpanan grafik.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(histori['accuracy']) + 1)

    ax.plot(epochs, histori['accuracy'],
            'b-o', linewidth=2, markersize=4, label='Training Accuracy')
    ax.plot(epochs, histori['val_accuracy'],
            'r-o', linewidth=2, markersize=4, label='Validation Accuracy')

    # Tandai akurasi validasi terbaik
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
        histori (dict): Dictionary histori Keras.
        path_simpan (str): Path tujuan penyimpanan grafik.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(histori['loss']) + 1)

    ax.plot(epochs, histori['loss'],
            'b-o', linewidth=2, markersize=4, label='Training Loss')
    ax.plot(epochs, histori['val_loss'],
            'r-o', linewidth=2, markersize=4, label='Validation Loss')

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
    Mengevaluasi model pada data validasi dan menampilkan Classification Report.

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
    y_true = val_generator.classes              # Label asli (indeks integer)

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
    Membuat dan menyimpan visualisasi Confusion Matrix.

    Args:
        y_true (array): Label kelas asli.
        y_pred (array): Label kelas hasil prediksi.
        class_names (list): Daftar nama kelas untuk label sumbu.
        path_simpan (str): Path tujuan penyimpanan gambar.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalisasi confusion matrix dalam bentuk persentase (per baris)
    cm_persen = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Ukuran figure dinamis berdasarkan jumlah kelas
    n_kelas = len(class_names)
    ukuran = max(10, n_kelas * 0.8)
    fig, ax = plt.subplots(figsize=(ukuran, ukuran * 0.85))

    # Buat heatmap menggunakan seaborn
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

    ax.set_title('Confusion Matrix (Normalisasi per Baris)', fontsize=14, fontweight='bold', pad=15)
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

    # Siapkan data validasi — unduh jika belum ada
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    dir_dataset = unduh_dataset()  # Akan skip jika sudah ada (ada guard di data_loader)
    _, val_gen, _, class_names_detected = dapatkan_data_generators(dir_dataset)

    # Gunakan nama kelas dari pickle jika tersedia, fallback ke yang terdeteksi
    if os.path.exists(class_names_path):
        with open(class_names_path, 'rb') as f:
            class_names = pickle.load(f)
        print(f"[INFO] Nama kelas dimuat dari: {class_names_path}")
    else:
        class_names = class_names_detected
        print("[PERINGATAN] File nama kelas tidak ditemukan, menggunakan deteksi otomatis.")

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
    print(f"[HASIL] Akurasi Keseluruhan : {akurasi_akhir:.4f} ({akurasi_akhir*100:.2f}%)")
    print(f"\n[OUTPUT] File yang dihasilkan di folder 'outputs/':")
    print(f"  - {ACCURACY_PLOT_PATH}")
    print(f"  - {LOSS_PLOT_PATH}")
    print(f"  - {CONFUSION_MATRIX_PATH}")
    print(f"  - {os.path.join(OUTPUTS_DIR, 'classification_report.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
