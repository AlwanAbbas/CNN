"""
model.py
--------
Modul yang mendefinisikan arsitektur Convolutional Neural Network (CNN)
untuk klasifikasi citra batik nusantara.

Arsitektur:
    3 blok Conv2D + MaxPooling2D -> Flatten -> Dense -> Dropout -> Dense (Softmax)
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization,
    Input
)


def bangun_model(input_shape, num_classes):
    """
    Membangun dan mengembalikan model CNN Sequential.

    Arsitektur Model:
        Blok 1: Conv2D(32) -> BatchNorm -> MaxPooling2D
        Blok 2: Conv2D(64) -> BatchNorm -> MaxPooling2D
        Blok 3: Conv2D(128) x2 -> BatchNorm -> MaxPooling2D
        Head:   Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(num_classes, Softmax)

    Args:
        input_shape (tuple): Shape input gambar, misal (150, 150, 3).
        num_classes (int): Jumlah kelas output (jumlah motif batik).

    Returns:
        tf.keras.Sequential: Model CNN yang sudah dibangun (belum dikompilasi).
    """
    print("\n" + "=" * 60)
    print("[INFO] Membangun arsitektur CNN...")
    print(f"[INFO] Input shape: {input_shape}")
    print(f"[INFO] Jumlah kelas output: {num_classes}")
    print("=" * 60)

    model = Sequential(name="CNN_Batik_Nusantara")

    # -----------------------------------------------------------------------
    # Input Layer
    # -----------------------------------------------------------------------
    model.add(Input(shape=input_shape))

    # -----------------------------------------------------------------------
    # Blok Konvolusi 1
    # Ekstraksi fitur dasar (tepi, tekstur sederhana)
    # -----------------------------------------------------------------------
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv1_1'
    ))
    model.add(BatchNormalization(name='bn1'))       # Normalisasi batch untuk stabilitas
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool1'
    ))

    # -----------------------------------------------------------------------
    # Blok Konvolusi 2
    # Ekstraksi fitur menengah (pola geometris batik)
    # -----------------------------------------------------------------------
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv2_1'
    ))
    model.add(BatchNormalization(name='bn2'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool2'
    ))

    # -----------------------------------------------------------------------
    # Blok Konvolusi 3
    # Ekstraksi fitur kompleks (motif batik spesifik)
    # Dua layer Conv2D berturutan untuk kapasitas representasi lebih tinggi
    # -----------------------------------------------------------------------
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv3_1'
    ))
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv3_2'
    ))
    model.add(BatchNormalization(name='bn3'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool3'
    ))

    # -----------------------------------------------------------------------
    # Fully Connected Layers (Head Classifier)
    # -----------------------------------------------------------------------
    model.add(Flatten(name='flatten'))              # Ratakan tensor 3D ke 1D

    model.add(Dense(
        units=256,
        activation='relu',
        name='dense1'
    ))
    model.add(Dropout(
        rate=0.5,                                   # Matikan 50% neuron secara acak
        name='dropout1'
    ))

    # -----------------------------------------------------------------------
    # Output Layer
    # Jumlah neuron = jumlah kelas, aktivasi Softmax untuk probabilitas
    # -----------------------------------------------------------------------
    model.add(Dense(
        units=num_classes,
        activation='softmax',
        name='output'
    ))

    # Tampilkan ringkasan arsitektur model
    model.summary()

    return model


# Alias bahasa Inggris untuk kompatibilitas
build_model = bangun_model
