"""
predict_cli.py
--------------
Skrip terminal (CLI) untuk mengklasifikasikan gambar batik menggunakan
model CNN yang sudah dilatih. Dapat menerima satu atau lebih path gambar.

Cara penggunaan:
    # Prediksi satu gambar
    py src/predict_cli.py gambar.jpg

    # Prediksi banyak gambar sekaligus
    py src/predict_cli.py foto1.jpg foto2.png folder/batik.jpg

    # Tampilkan top-N hasil (default: 5)
    py src/predict_cli.py gambar.jpg --top 3

    # Mode verbose (tampilkan semua kelas)
    py src/predict_cli.py gambar.jpg --verbose

    # Simpan hasil ke file JSON
    py src/predict_cli.py gambar.jpg --output hasil.json
"""

import os
import sys
import json
import argparse

# Tambah root proyek ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paksa output UTF-8 di Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.predict import prediksi_gambar


# ---------------------------------------------------------------------------
# Helper visual
# ---------------------------------------------------------------------------

def _bar(persen, lebar=30):
    """Buat progress bar ASCII sederhana."""
    isi = int(persen / 100 * lebar)
    return "[" + "#" * isi + "-" * (lebar - isi) + "]"


def tampilkan_hasil(path_gambar, hasil, top_n=5, verbose=False):
    """
    Tampilkan hasil prediksi satu gambar ke terminal.

    Args:
        path_gambar (str): Path file gambar.
        hasil (dict)     : Output dari prediksi_gambar().
        top_n (int)      : Jumlah kelas teratas yang ditampilkan.
        verbose (bool)   : Tampilkan semua kelas jika True.
    """
    LEBAR = 65
    print("\n" + "=" * LEBAR)
    print(f"  File   : {os.path.basename(path_gambar)}")
    print(f"  Path   : {path_gambar}")
    print("=" * LEBAR)

    # Hasil utama
    print(f"\n  [PREDIKSI UTAMA]")
    print(f"  Kelas      : {hasil['kelas']}")
    print(f"  Confidence : {hasil['confidence']:.2f}%  {_bar(hasil['confidence'])}")

    # Tampilkan top-N atau semua kelas
    daftar = hasil["semua_kelas"]
    batas  = len(daftar) if verbose else min(top_n, len(daftar))

    print(f"\n  [TOP {batas} PREDIKSI]")
    print(f"  {'No':<4} {'Nama Kelas':<30} {'Confidence':>10}  Bar")
    print(f"  {'-'*4} {'-'*30} {'-'*10}  {'-'*22}")

    for i, item in enumerate(daftar[:batas], 1):
        bar   = _bar(item["confidence"], lebar=20)
        nama  = item["kelas"][:29]          # Potong jika terlalu panjang
        print(f"  {i:<4} {nama:<30} {item['confidence']:>9.2f}%  {bar}")

    print("=" * LEBAR)


def proses_banyak_gambar(paths, top_n, verbose, output_json):
    """
    Proses satu atau beberapa gambar dan tampilkan hasilnya.

    Args:
        paths (list)       : Daftar path file gambar.
        top_n (int)        : Jumlah kelas teratas.
        verbose (bool)     : Mode verbose.
        output_json (str)  : Path file JSON output (opsional).
    """
    print("\n" + "=" * 65)
    print("    CNN BATIK NUSANTARA - KLASIFIKASI TERMINAL")
    print("=" * 65)
    print(f"  Jumlah gambar  : {len(paths)}")
    print(f"  Top-N prediksi : {top_n}")
    print("=" * 65)

    # Muat model sekali sebelum iterasi (lebih efisien)
    from src.predict import muat_model_predict
    muat_model_predict()

    semua_hasil = []
    untuk_json  = []

    for path in paths:
        # Validasi file
        if not os.path.exists(path):
            print(f"\n[!!] File tidak ditemukan: {path} — dilewati.")
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            print(f"\n[!!] Format tidak didukung: {path} — dilewati.")
            continue

        try:
            with open(path, "rb") as f:
                img_bytes = f.read()

            hasil = prediksi_gambar(img_bytes)
            tampilkan_hasil(path, hasil, top_n=top_n, verbose=verbose)
            semua_hasil.append((path, hasil))

            # Siapkan data untuk JSON
            untuk_json.append({
                "file"      : path,
                "kelas"     : hasil["kelas"],
                "confidence": round(hasil["confidence"], 4),
                "top_prediksi": [
                    {"kelas": k["kelas"], "confidence": round(k["confidence"], 4)}
                    for k in hasil["semua_kelas"][:top_n]
                ]
            })

        except Exception as e:
            print(f"\n[!!] Gagal memproses {path}: {e}")

    # Simpan JSON jika diminta
    if output_json and untuk_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(untuk_json, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Hasil disimpan ke: {output_json}")

    # Ringkasan jika banyak file
    if len(semua_hasil) > 1:
        print("\n" + "=" * 65)
        print("  RINGKASAN")
        print("=" * 65)
        for path, hasil in semua_hasil:
            nama = os.path.basename(path)[:35]
            print(f"  {nama:<35}  {hasil['kelas']:<25} {hasil['confidence']:.1f}%")
        print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Klasifikasi citra batik menggunakan CNN Batik Nusantara.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "gambar",
        nargs="+",
        metavar="FILE",
        help="Path ke file gambar (JPEG/PNG/BMP). Bisa lebih dari satu."
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=5,
        metavar="N",
        help="Tampilkan top-N prediksi (default: 5)."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Tampilkan semua kelas beserta confidence-nya."
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE.json",
        default=None,
        help="Simpan hasil prediksi ke file JSON."
    )

    args = parser.parse_args()
    proses_banyak_gambar(
        paths=args.gambar,
        top_n=args.top,
        verbose=args.verbose,
        output_json=args.output
    )


if __name__ == "__main__":
    main()
