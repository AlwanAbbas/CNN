"""
train.py
--------
Skrip utama untuk melatih model CNN Batik Nusantara.

Pipeline 2 Fase (Transfer Learning — EfficientNetB0):

    Fase 1 — Feature Extraction (epoch 1 s/d PHASE1_EPOCHS):
        Base EfficientNetB0 di-freeze.
        Hanya head classifier yang dilatih.
        LR = LEARNING_RATE (3e-4).
        EarlyStopping patience=10.

    Fase 2 — Fine-tuning (dilanjutkan dari Fase 1):
        Layer EfficientNetB0 dari FINE_TUNE_AT ke atas dibuka.
        LR = FINE_TUNE_LR (2e-5) — sangat kecil agar fitur ImageNet
        tidak rusak dan hanya beradaptasi pada tekstur batik.
        EarlyStopping patience=7.

Cara menjalankan:
    python -m src.train
"""

import os
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    INPUT_SHAPE, EPOCHS, LEARNING_RATE, FINE_TUNE_LR, FINE_TUNE_AT,
    MODEL_SAVE_PATH, HISTORY_SAVE_PATH,
    SAVED_MODELS_DIR, OUTPUTS_DIR
)
from src.data_loader import unduh_dataset, dapatkan_data_generators
from src.model import bangun_model, aktifkan_fine_tuning

# Jumlah epoch untuk Fase 1 sebelum fine-tuning dimulai
PHASE1_EPOCHS = 20


def _buat_callbacks(monitor_metric='val_accuracy', patience_stop=10, patience_lr=4):
    """
    Buat daftar callbacks standar.

    Args:
        monitor_metric (str): Metrik yang dipantau EarlyStopping & ModelCheckpoint.
        patience_stop (int) : Epoch tanpa peningkatan sebelum stop.
        patience_lr (int)   : Epoch tanpa peningkatan sebelum kurangi LR.

    Returns:
        list: Daftar callback Keras.
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor=monitor_metric,
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor_metric,
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-8,
            verbose=1
        ),
    ]


def main():
    """
    Fungsi utama yang mengorkestrasi pelatihan 2 fase.
    """
    print("\n" + "=" * 60)
    print("    CNN BATIK NUSANTARA — PELATIHAN MODEL (Transfer Learning)")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Langkah 1: Verifikasi GPU
    # -----------------------------------------------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPU terdeteksi: {len(gpus)} unit")
        for gpu in gpus:
            print(f"       - {gpu}")
        # Aktifkan memory growth agar VRAM tidak langsung terpakai semua
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("[PERINGATAN] Tidak ada GPU. Pelatihan menggunakan CPU.")

    # -----------------------------------------------------------------------
    # Langkah 2: Dataset
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 1] Mempersiapkan dataset...")
    direktori_dataset = unduh_dataset()

    print("\n[LANGKAH 2] Membuat data generator...")
    train_gen, val_gen, num_classes, class_names = dapatkan_data_generators(
        direktori_dataset
    )

    steps_per_epoch  = max(1, train_gen.samples // train_gen.batch_size)
    validation_steps = max(1, val_gen.samples // val_gen.batch_size)
    print(f"[INFO] Steps/epoch training : {steps_per_epoch}")
    print(f"[INFO] Steps/epoch validasi : {validation_steps}")

    # -----------------------------------------------------------------------
    # Langkah 3: Bangun Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 3] Membangun model MobileNetV2...")
    model = bangun_model(INPUT_SHAPE, num_classes)

    # -----------------------------------------------------------------------
    # FASE 1 — Feature Extraction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("    FASE 1 — FEATURE EXTRACTION")
    print(f"    LR={LEARNING_RATE}  |  Max epoch={PHASE1_EPOCHS}")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    history_fase1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=PHASE1_EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=_buat_callbacks(patience_stop=10, patience_lr=4),
        verbose=1
    )

    akurasi_fase1 = max(history_fase1.history['val_accuracy'])
    print(f"\n[FASE 1] Val accuracy terbaik: {akurasi_fase1*100:.2f}%")

    # -----------------------------------------------------------------------
    # FASE 2 — Fine-tuning
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("    FASE 2 — FINE-TUNING")
    print(f"    LR={FINE_TUNE_LR}  |  Buka dari layer ke-{FINE_TUNE_AT}")
    print("=" * 60)

    aktifkan_fine_tuning(model, fine_tune_at=FINE_TUNE_AT)

    # Kompilasi ulang dengan LR sangat kecil + label smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    sisa_epoch = EPOCHS - PHASE1_EPOCHS
    print(f"[INFO] Melanjutkan fine-tuning untuk {sisa_epoch} epoch lagi...")

    history_fase2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=sisa_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=_buat_callbacks(patience_stop=7, patience_lr=3),
        verbose=1
    )

    akurasi_fase2 = max(history_fase2.history['val_accuracy'])
    print(f"\n[FASE 2] Val accuracy terbaik: {akurasi_fase2*100:.2f}%")

    # -----------------------------------------------------------------------
    # Gabungkan histori kedua fase
    # -----------------------------------------------------------------------
    histori_gabungan = {}
    for key in history_fase1.history:
        histori_gabungan[key] = (
            history_fase1.history[key] + history_fase2.history[key]
        )

    # -----------------------------------------------------------------------
    # Simpan histori dan nama kelas
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 7] Menyimpan histori dan metadata...")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(HISTORY_SAVE_PATH, 'wb') as f:
        pickle.dump(histori_gabungan, f)
    print(f"[SUKSES] Histori disimpan ke: {HISTORY_SAVE_PATH}")

    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    with open(class_names_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"[SUKSES] Nama kelas disimpan ke: {class_names_path}")

    # -----------------------------------------------------------------------
    # Ringkasan Akhir
    # -----------------------------------------------------------------------
    akurasi_train_akhir = histori_gabungan['accuracy'][-1]
    akurasi_val_akhir   = histori_gabungan['val_accuracy'][-1]
    total_epoch         = len(histori_gabungan['accuracy'])

    print("\n" + "=" * 60)
    print("    RINGKASAN HASIL PELATIHAN")
    print("=" * 60)
    print(f"[HASIL] Akurasi Training Akhir  : {akurasi_train_akhir*100:.2f}%")
    print(f"[HASIL] Akurasi Validasi Akhir  : {akurasi_val_akhir*100:.2f}%")
    print(f"[HASIL] Val Acc terbaik (Fase 1): {akurasi_fase1*100:.2f}%")
    print(f"[HASIL] Val Acc terbaik (Fase 2): {akurasi_fase2*100:.2f}%")
    print(f"[HASIL] Total epoch berjalan     : {total_epoch}")
    print(f"[HASIL] Model disimpan di        : {MODEL_SAVE_PATH}")
    print("=" * 60)
    print("\n[INFO] Selesai! Jalankan: python -m src.evaluate")


if __name__ == "__main__":
    main()
