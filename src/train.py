"""
train.py
--------
Skrip utama untuk melatih model CNN Batik Nusantara.
Telah diperbarui untuk kompatibilitas penuh dengan TF 2.10 GPU Native (Checkpoints).
"""

import os
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# Tambahkan direktori root ke path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    INPUT_SHAPE, EPOCHS, LEARNING_RATE, FINE_TUNE_LR, FINE_TUNE_AT,
    HISTORY_SAVE_PATH, SAVED_MODELS_DIR, OUTPUTS_DIR
)
from src.data_loader import unduh_dataset, dapatkan_data_generators
from src.model import bangun_model, aktifkan_fine_tuning

# Jumlah epoch untuk Fase 1 sebelum fine-tuning dimulai
PHASE1_EPOCHS = 20

def _buat_callbacks(monitor_metric='val_accuracy', patience_stop=10, patience_lr=4):
    """
    Buat daftar callbacks standar.
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # SECARA PAKSA GUNAKAN FORMAT NATIVE CHECKPOINT (.ckpt)
    aman_save_path = os.path.join(SAVED_MODELS_DIR, 'bobot_model.ckpt')

    return [
        ModelCheckpoint(
            filepath=aman_save_path,
            monitor=monitor_metric,
            save_best_only=True,
            save_weights_only=True,  # <--- SOLUSI ULTIMATE: Hindari error EagerTensor JSON
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
    train_gen, val_gen, num_classes, class_names = dapatkan_data_generators(direktori_dataset)

    steps_per_epoch  = max(1, train_gen.samples // train_gen.batch_size)
    validation_steps = max(1, val_gen.samples // val_gen.batch_size)
    print(f"[INFO] Steps/epoch training : {steps_per_epoch}")
    print(f"[INFO] Steps/epoch validasi : {validation_steps}")

    # -----------------------------------------------------------------------
    # Langkah 3: Bangun Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 3] Membangun model EfficientNetB0...")
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

    history_fase2_dict = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    akurasi_fase2 = 0.0

    if EPOCHS <= PHASE1_EPOCHS:
        print("[INFO] Melewati Fase 2 (Fine-tuning) dan langsung menyimpan histori.")
    else:
        aktifkan_fine_tuning(model, fine_tune_at=FINE_TUNE_AT)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        initial_epoch_fase2 = history_fase1.epoch[-1] + 1
        print(f"[INFO] Melanjutkan fine-tuning hingga epoch ke-{EPOCHS}...")

        history_fase2 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            initial_epoch=initial_epoch_fase2,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=_buat_callbacks(patience_stop=7, patience_lr=3),
            verbose=1
        )
        
        history_fase2_dict = history_fase2.history
        akurasi_fase2 = max(history_fase2_dict.get('val_accuracy', [0]))
        print(f"\n[FASE 2] Val accuracy terbaik: {akurasi_fase2*100:.2f}%")

    # -----------------------------------------------------------------------
    # Gabungkan dan simpan histori
    # -----------------------------------------------------------------------
    histori_gabungan = {}
    for key in history_fase1.history:
        histori_gabungan[key] = (
            history_fase1.history.get(key, []) + history_fase2_dict.get(key, [])
        )

    print("\n[LANGKAH 7] Menyimpan histori dan metadata...")
    with open(HISTORY_SAVE_PATH, 'wb') as f:
        pickle.dump(histori_gabungan, f)
    
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    with open(class_names_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"[SUKSES] Histori & nama kelas disimpan ke folder {OUTPUTS_DIR}/")

    # -----------------------------------------------------------------------
    # Ringkasan Akhir
    # -----------------------------------------------------------------------
    aman_save_path = os.path.join(SAVED_MODELS_DIR, 'bobot_model.ckpt')

    print("\n" + "=" * 60)
    print("    RINGKASAN HASIL PELATIHAN")
    print("=" * 60)
    print(f"[HASIL] Akurasi Training Akhir  : {histori_gabungan['accuracy'][-1]*100:.2f}%")
    print(f"[HASIL] Akurasi Validasi Akhir  : {histori_gabungan['val_accuracy'][-1]*100:.2f}%")
    print(f"[HASIL] Bobot model disimpan di : {aman_save_path}")
    print("=" * 60)
    print("\n[INFO] Selesai! Jalankan: python -m src.evaluate")

    # Bersihkan backend agar tidak muncul warning saat program tertutup
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
    # Mencegah error "FAILED_PRECONDITION: Python interpreter state is not initialized"
    # saat program selesai karena bug teardown tf.data.Dataset generator di TensorFlow.
    os._exit(0)