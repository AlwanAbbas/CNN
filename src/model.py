"""
model.py
--------
Modul arsitektur model untuk klasifikasi batik nusantara.

Menggunakan EfficientNetB0 pre-trained pada ImageNet sebagai feature extractor,
dengan head classifier baru yang disesuaikan untuk dataset batik.

Kenapa EfficientNetB0 vs MobileNetV2:
    - Compound scaling: depth + width + resolution diseimbangkan secara optimal
    - 5-10% lebih akurat pada dataset kecil dengan jumlah parameter serupa
    - Lebih robust pada fitur tekstur kompleks seperti pola batik

Strategi 2 Fase:
    Fase 1 - Feature Extraction:
        Base EfficientNetB0 di-freeze, hanya head yang dilatih.
        Cocok untuk dataset kecil karena fitur ImageNet sudah umum.

    Fase 2 - Fine-tuning:
        Buka layer terakhir EfficientNetB0 (dari FINE_TUNE_AT ke atas),
        latih dengan learning rate sangat kecil agar bobot pretrained
        tidak rusak dan model beradaptasi pada tekstur batik.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import EfficientNetB0


def bangun_model(input_shape, num_classes):
    """
    Membangun model Transfer Learning berbasis EfficientNetB0.

    Base EfficientNetB0 di-freeze (Fase 1: Feature Extraction).
    Gunakan aktifkan_fine_tuning() untuk Fase 2.

    Args:
        input_shape (tuple): Shape input gambar, harus (224, 224, 3).
        num_classes (int)  : Jumlah kelas output (jumlah motif batik).

    Returns:
        tf.keras.Model: Model yang sudah dibangun (belum dikompilasi).
    """
    print("\n" + "=" * 60)
    print("[INFO] Membangun model EfficientNetB0 Transfer Learning...")
    print(f"[INFO] Input shape  : {input_shape}")
    print(f"[INFO] Jumlah kelas : {num_classes}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Base Model: EfficientNetB0 pretrained pada ImageNet
    # include_top=False → buang classifier aslinya, ambil feature extractor
    # EfficientNetB0 output: (7, 7, 1280) — sama dengan MobileNetV2
    # -----------------------------------------------------------------------
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Fase 1: freeze semua layer base agar bobot ImageNet tidak berubah
    base_model.trainable = False
    print(f"[INFO] Base EfficientNetB0: {len(base_model.layers)} layer (semua di-freeze)")

    # -----------------------------------------------------------------------
    # Head Classifier
    # -----------------------------------------------------------------------
    inputs = Input(shape=input_shape)

    # Lewatkan melalui base — training=False agar BatchNorm di base
    # selalu berjalan dalam mode inference (penting saat base di-freeze)
    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D(name='gap')(x)  # (batch, 1280) dari EfficientNetB0

    x = Dense(256, name='dense1')(x)
    x = BatchNormalization(name='bn_head')(x)
    x = tf.keras.layers.Activation('relu', name='relu_head')(x)
    x = Dropout(0.5, name='dropout1')(x)

    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='EfficientNetB0_Batik')
    model.summary()

    return model


def aktifkan_fine_tuning(model, fine_tune_at=80):
    """
    Mengaktifkan fine-tuning dengan membuka layer EfficientNetB0
    dari indeks `fine_tune_at` ke atas.

    Panggil setelah Fase 1 selesai. Kompilasi ulang model dengan
    FINE_TUNE_LR sebelum melanjutkan training.

    Args:
        model (tf.keras.Model): Model yang sudah dilatih di Fase 1.
        fine_tune_at (int)    : Indeks layer mulai dibuka (default 80).

    Returns:
        int: Jumlah layer yang dibuka untuk fine-tuning.
    """
    # Temukan base_model (layer pertama yang merupakan EfficientNetB0)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Tidak menemukan base model di dalam model.")

    # Buka trainable untuk seluruh base dulu
    base_model.trainable = True

    # Freeze layer sebelum fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Pastikan BatchNorm di blok yang di-freeze tetap inference mode
    for layer in base_model.layers[:fine_tune_at]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    jumlah_bisa_latih = sum(
        1 for l in base_model.layers if l.trainable
    )
    print(f"[INFO] Fine-tuning: membuka {jumlah_bisa_latih} layer "
          f"(dari layer ke-{fine_tune_at} sampai akhir)")

    return jumlah_bisa_latih


# Alias Inggris untuk kompatibilitas
build_model = bangun_model
enable_fine_tuning = aktifkan_fine_tuning
