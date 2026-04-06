"""
train.py
--------
Skrip utama untuk melatih model CNN Batik Nusantara.

Alur kerja:
    1. Unduh dan persiapkan dataset dari Kaggle
    2. Bangun arsitektur CNN
    3. Kompilasi model
    4. Latih model dengan data generator
    5. Simpan model dan histori pelatihan

Cara menjalankan di Colab:
    !python src/train.py
"""

import os
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# Tambahkan direktori root ke path agar impor modul src bisa dilakukan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Impor modul-modul proyek
from src.config import (
    INPUT_SHAPE, EPOCHS, LEARNING_RATE,
    MODEL_SAVE_PATH, HISTORY_SAVE_PATH,
    SAVED_MODELS_DIR, OUTPUTS_DIR
)
from src.data_loader import unduh_dataset, dapatkan_data_generators
from src.model import bangun_model


def main():
    """
    Fungsi utama yang mengorkestrasi seluruh proses pelatihan model.
    """
    print("\n" + "=" * 60)
    print("    CNN BATIK NUSANTARA - PELATIHAN MODEL")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Langkah 1: Verifikasi GPU (opsional tapi dianjurkan di Colab)
    # -----------------------------------------------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPU terdeteksi: {len(gpus)} unit")
        for gpu in gpus:
            print(f"       - {gpu}")
    else:
        print("[PERINGATAN] Tidak ada GPU terdeteksi. Pelatihan menggunakan CPU.")

    # -----------------------------------------------------------------------
    # Langkah 2: Unduh dan Persiapkan Dataset
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 1] Mengunduh dataset dari Kaggle...")
    direktori_dataset = unduh_dataset()

    # -----------------------------------------------------------------------
    # Langkah 3: Buat Data Generator
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 2] Mempersiapkan data generator...")
    train_gen, val_gen, num_classes, class_names = dapatkan_data_generators(
        direktori_dataset
    )

    # -----------------------------------------------------------------------
    # Langkah 4: Bangun Model CNN
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 3] Membangun model CNN...")
    model = bangun_model(INPUT_SHAPE, num_classes)

    # -----------------------------------------------------------------------
    # Langkah 5: Kompilasi Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 4] Mengkompilasi model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',   # Loss untuk multi-kelas one-hot
        metrics=['accuracy']
    )
    print("[INFO] Model berhasil dikompilasi.")
    print(f"[INFO] Optimizer: Adam (lr={LEARNING_RATE})")
    print("[INFO] Loss: Categorical Crossentropy")
    print("[INFO] Metrics: Accuracy")

    # -----------------------------------------------------------------------
    # Langkah 6: Definisikan Callbacks
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 5] Mengatur callbacks pelatihan...")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    callbacks = [
        # Simpan bobot model terbaik berdasarkan val_accuracy
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Hentikan training lebih awal jika tidak ada peningkatan
        EarlyStopping(
            monitor='val_loss',
            patience=7,                    # Tunggu 7 epoch sebelum berhenti
            restore_best_weights=True,
            verbose=1
        ),
        # Kurangi learning rate jika plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                    # Kurangi LR menjadi setengahnya
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    print("[INFO] Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau")

    # -----------------------------------------------------------------------
    # Langkah 7: Hitung Steps per Epoch
    # -----------------------------------------------------------------------
    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    validation_steps = max(1, val_gen.samples // val_gen.batch_size)

    print(f"\n[INFO] Steps per epoch (training): {steps_per_epoch}")
    print(f"[INFO] Steps per epoch (validasi): {validation_steps}")

    # -----------------------------------------------------------------------
    # Langkah 8: Latih Model
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 6] Memulai pelatihan model...")
    print("=" * 60)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # -----------------------------------------------------------------------
    # Langkah 9: Simpan Histori Pelatihan menggunakan pickle
    # -----------------------------------------------------------------------
    print("\n[LANGKAH 7] Menyimpan histori pelatihan...")
    with open(HISTORY_SAVE_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"[SUKSES] Histori pelatihan disimpan ke: {HISTORY_SAVE_PATH}")

    # -----------------------------------------------------------------------
    # Langkah 10: Simpan Nama Kelas untuk digunakan evaluate.py
    # -----------------------------------------------------------------------
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.pkl")
    with open(class_names_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"[SUKSES] Nama kelas disimpan ke: {class_names_path}")

    # -----------------------------------------------------------------------
    # Ringkasan Hasil Pelatihan
    # -----------------------------------------------------------------------
    akurasi_akhir = history.history['accuracy'][-1]
    val_akurasi_akhir = history.history['val_accuracy'][-1]

    print("\n" + "=" * 60)
    print("    RINGKASAN HASIL PELATIHAN")
    print("=" * 60)
    print(f"[HASIL] Akurasi Training Akhir : {akurasi_akhir:.4f} ({akurasi_akhir*100:.2f}%)")
    print(f"[HASIL] Akurasi Validasi Akhir : {val_akurasi_akhir:.4f} ({val_akurasi_akhir*100:.2f}%)")
    print(f"[HASIL] Total epoch berjalan   : {len(history.history['accuracy'])}")
    print(f"[HASIL] Model disimpan di      : {MODEL_SAVE_PATH}")
    print("=" * 60)
    print("\n[INFO] Pelatihan selesai! Jalankan evaluate.py untuk evaluasi.")


if __name__ == "__main__":
    main()
