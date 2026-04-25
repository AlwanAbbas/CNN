"""
evaluate.py
-----------
Skrip untuk mengevaluasi model CNN Batik Nusantara yang sudah dilatih.
(Diperbarui untuk membaca format Native Checkpoint .ckpt)
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Tambahkan direktori root ke path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    INPUT_SHAPE, HISTORY_SAVE_PATH, SAVED_MODELS_DIR,
    ACCURACY_PLOT_PATH, LOSS_PLOT_PATH,
    CONFUSION_MATRIX_PATH, OUTPUTS_DIR
)
from src.data_loader import unduh_dataset, dapatkan_data_generators
from src.model import bangun_model


def muat_model_ckpt(num_classes):
    """Membangun arsitektur dan memuat bobot .ckpt"""
    aman_save_path = os.path.join(SAVED_MODELS_DIR, 'bobot_model.ckpt')

    # TensorFlow checkpoint menyimpan metadata di file .index
    if not os.path.exists(aman_save_path + ".index"):
        raise FileNotFoundError(
            f"[ERROR] File bobot model tidak ditemukan di: {aman_save_path}\n"
            "Pastikan Anda sudah menjalankan train.py!"
        )

    print(f"[INFO] Membangun arsitektur model (Input: {INPUT_SHAPE}, Kelas: {num_classes})...")
    model = bangun_model(INPUT_SHAPE, num_classes)

    print(f"[INFO] Memuat bobot (weights) dari: {aman_save_path}")
    model.load_weights(aman_save_path).expect_partial() 
    
    # --- TAMBAHKAN BARIS INI ---
    # Model harus di-compile sebelum bisa menjalankan model.evaluate()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    # ---------------------------

    print("[SUKSES] Bobot model berhasil dimuat dan di-compile!")
    return model


def muat_histori(path_histori):
    if not os.path.exists(path_histori):
        print(f"[PERINGATAN] File histori tidak ditemukan: {path_histori}")
        return None
    with open(path_histori, 'rb') as f:
        histori = pickle.load(f)
    print(f"[INFO] Histori pelatihan dimuat dari: {path_histori}")
    return histori


def plot_akurasi(histori, path_simpan):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(histori['accuracy']) + 1)
    ax.plot(epochs, histori['accuracy'], 'b-o', linewidth=2, markersize=4, label='Training Accuracy')
    ax.plot(epochs, histori['val_accuracy'], 'r-o', linewidth=2, markersize=4, label='Validation Accuracy')
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


def plot_loss(histori, path_simpan):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(histori['loss']) + 1)
    ax.plot(epochs, histori['loss'], 'b-o', linewidth=2, markersize=4, label='Training Loss')
    ax.plot(epochs, histori['val_loss'], 'r-o', linewidth=2, markersize=4, label='Validation Loss')
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


def evaluasi_model(model, val_generator, class_names):
    print("\n" + "=" * 60)
    print("[INFO] Menjalankan evaluasi model pada data validasi...")
    print("=" * 60)

    val_loss, val_acc = model.evaluate(val_generator, verbose=1)
    print(f"\n[HASIL] Loss Validasi   : {val_loss:.4f}")
    print(f"[HASIL] Akurasi Validasi: {val_acc:.4f} ({val_acc*100:.2f}%)")

    print("\n[INFO] Menghasilkan prediksi untuk Classification Report...")
    val_generator.reset()

    y_pred_proba = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_generator.classes

    print("\n" + "=" * 60)
    print("    CLASSIFICATION REPORT")
    print("=" * 60)
    laporan = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(laporan)

    laporan_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(laporan_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT - CNN BATIK NUSANTARA\n")
        f.write("=" * 60 + "\n")
        f.write(laporan)
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, path_simpan):
    cm = confusion_matrix(y_true, y_pred)
    cm_persen = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    n_kelas = len(class_names)
    ukuran = max(10, n_kelas * 0.8)
    fig, ax = plt.subplots(figsize=(ukuran, ukuran * 0.85))

    sns.heatmap(cm_persen, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})

    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Prediksi', fontsize=12, labelpad=10)
    ax.set_ylabel('Label Asli', fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(path_simpan, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("    CNN BATIK NUSANTARA - EVALUASI MODEL")
    print("=" * 60)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Siapkan Kelas
    print("\n[LANGKAH 1] Mempersiapkan data validasi dan daftar kelas...")
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    dir_dataset = unduh_dataset() 
    _, val_gen, _, class_names_detected = dapatkan_data_generators(dir_dataset)

    if os.path.exists(class_names_path):
        with open(class_names_path, 'rb') as f:
            class_names = pickle.load(f)
    else:
        class_names = class_names_detected
        
    num_classes = len(class_names)

    # 2. Muat Arsitektur dan Bobot .ckpt
    print("\n[LANGKAH 2] Memuat model yang sudah dilatih...")
    model = muat_model_ckpt(num_classes)

    # 3. Histori
    print("\n[LANGKAH 3] Memuat histori pelatihan...")
    histori = muat_histori(HISTORY_SAVE_PATH)

    # 4. Visualisasi
    if histori is not None:
        print("\n[LANGKAH 4] Membuat grafik pelatihan...")
        plot_akurasi(histori, ACCURACY_PLOT_PATH)
        plot_loss(histori, LOSS_PLOT_PATH)
    else:
        print("\n[LANGKAH 4] Melewati visualisasi histori.")

    # 5. Evaluasi
    print("\n[LANGKAH 5] Mengevaluasi model...")
    y_true, y_pred = evaluasi_model(model, val_gen, class_names)

    # 6. CM
    print("\n[LANGKAH 6] Membuat Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names, CONFUSION_MATRIX_PATH)

    akurasi_akhir = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 60)
    print("    RINGKASAN EVALUASI")
    print("=" * 60)
    print(f"[HASIL] Akurasi Keseluruhan : {akurasi_akhir:.4f} ({akurasi_akhir*100:.2f}%)")
    print("=" * 60)

    tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
    # Mencegah error "FAILED_PRECONDITION: Python interpreter state is not initialized"
    # saat program selesai karena bug teardown tf.data.Dataset generator di TensorFlow.
    os._exit(0)