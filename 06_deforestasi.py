"""Materi 6: Deforestation alerts dan atribusi driver dengan AI (Bart Slachter).

Simulasi pipeline penginderaan jauh: NDVI dari Sentinel-2, backscatter Sentinel-1,
deteksi kehilangan tutupan hutan, lalu klasifikasi penyebab (pertanian, tambang,
penebangan, kebakaran, banjir, jalan) memakai data latih berlabel.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

N = 20
SKALA_PIKSEL_M = 10
AMBANG_HUTAN = 0.60
AMBANG_HILANG = 0.55

KELAS = ["pertanian", "penebangan", "tambang", "kebakaran", "banjir", "jalan"]
SIMBOL = {
    "pertanian": "A",
    "penebangan": "L",
    "tambang": "M",
    "kebakaran": "F",
    "banjir": "B",
    "jalan": "R",
    "tidak_yakin": "?",
}

PATCHES = {
    "pertanian": [(r, c) for r in (5, 6, 7) for c in (3, 4, 5)],
    "penebangan": [(11, c) for c in range(4, 9)],
    "tambang": [(r, c) for r in (10, 11, 12) for c in (17, 18)],
    "kebakaran": [(r, c) for r in (6, 7, 8) for c in (9, 10, 11)],
    "banjir": [(r, c) for r in (15, 16) for c in (2, 3, 4)],
    "jalan": [(10, c) for c in range(10, 16)],
}

DROP_NDVI = {
    "pertanian": 0.36,
    "penebangan": 0.27,
    "tambang": 0.55,
    "kebakaran": 0.40,
    "banjir": 0.30,
    "jalan": 0.25,
}
DROP_VV = {
    "pertanian": 2.6,
    "penebangan": 1.8,
    "tambang": 5.7,
    "kebakaran": 2.0,
    "banjir": 6.2,
    "jalan": 1.5,
}

RIVER = {(17 - c // 4, c) for c in range(N)}
ROAD = {(2, c) for c in range(3, 17)} | {(r, 16) for r in range(2, 13)}
FIRE_CELLS = set(PATCHES["kebakaran"])

PROTOTIPE = {
    "pertanian": [0.36, 2.6, math.log1p(9), 0.15, 0.30, 0.20, 0.20],
    "penebangan": [0.27, 1.8, math.log1p(5), 0.45, 0.20, 0.20, 0.00],
    "tambang": [0.55, 5.7, math.log1p(6), 0.15, 0.80, 0.60, 0.00],
    "kebakaran": [0.40, 2.0, math.log1p(9), 0.15, 0.20, 0.20, 1.00],
    "banjir": [0.30, 6.2, math.log1p(6), 0.15, 1.00, 0.10, 0.00],
    "jalan": [0.25, 1.5, math.log1p(5), 0.55, 0.10, 1.00, 0.00],
}
NOISE = [0.07, 0.7, 0.25, 0.15, 0.15, 0.15, 0.08]


def jepit(nilai, bawah, atas):
    return max(bawah, min(atas, nilai))


def buat_citra(seed=17):
    rng = random.Random(seed)
    ndvi0 = [[jepit(0.74 + rng.gauss(0, 0.03), 0, 1) for _ in range(N)] for _ in range(N)]
    vv0 = [[-8.5 + rng.gauss(0, 0.4) for _ in range(N)] for _ in range(N)]

    for r, c in RIVER:
        ndvi0[r][c] = jepit(0.25 + rng.gauss(0, 0.02), 0, 1)
        vv0[r][c] = -18.0 + rng.gauss(0, 0.5)
    for r, c in ROAD:
        ndvi0[r][c] = jepit(0.38 + rng.gauss(0, 0.02), 0, 1)
        vv0[r][c] = -12.0 + rng.gauss(0, 0.4)

    ndvi1 = [baris[:] for baris in ndvi0]
    vv1 = [baris[:] for baris in vv0]
    sel = {}

    for kelas, daftar in PATCHES.items():
        for r, c in daftar:
            ndvi1[r][c] = jepit(ndvi0[r][c] - DROP_NDVI[kelas] + rng.gauss(0, 0.02), 0, 1)
            vv1[r][c] = vv0[r][c] - DROP_VV[kelas] + rng.gauss(0, 0.25)
            sel[(r, c)] = kelas
    return ndvi0, ndvi1, vv0, vv1, sel


def komponen(alert):
    """Kelompokkan piksel alert yang bertetangga menjadi satu kejadian."""
    tersisa = set(alert)
    hasil = []
    while tersisa:
        awal = tersisa.pop()
        kelompok = [awal]
        antre = [awal]
        while antre:
            r, c = antre.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    tetangga = (r + dr, c + dc)
                    if tetangga in tersisa:
                        tersisa.remove(tetangga)
                        kelompok.append(tetangga)
                        antre.append(tetangga)
        hasil.append(kelompok)
    return hasil


def dekat(kelompok, penanda):
    return 1.0 if any(
        (r + dr, c + dc) in penanda
        for r, c in kelompok
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    ) else 0.0


def fitur_kejadian(kelompok, ndvi0, ndvi1, vv0, vv1):
    baris = [r for r, _ in kelompok]
    kolom = [c for _, c in kelompok]
    luas = len(kelompok)
    lebar = max(kolom) - min(kolom) + 1
    tinggi = max(baris) - min(baris) + 1
    isi = luas / (lebar * tinggi)
    aspek = abs(lebar - tinggi) / max(lebar, tinggi)
    irreguler = 0.5 * (1 - isi) + 0.5 * aspek
    delta_ndvi = statistics.fmean(ndvi0[r][c] - ndvi1[r][c] for r, c in kelompok)
    delta_vv = statistics.fmean(vv0[r][c] - vv1[r][c] for r, c in kelompok)
    api = 1.0 if any((r, c) in FIRE_CELLS for r, c in kelompok) else 0.0
    return [
        delta_ndvi,
        delta_vv,
        math.log1p(luas),
        jepit(irreguler, 0, 1),
        dekat(kelompok, RIVER),
        dekat(kelompok, ROAD),
        api,
    ]


def standardisasi(X):
    kolom = list(zip(*X))
    rata = [statistics.fmean(k) for k in kolom]
    simpangan = [statistics.pstdev(k) or 1.0 for k in kolom]
    return rata, simpangan


def terapkan(x, rata, simpangan):
    return [(a - r) / s for a, r, s in zip(x, rata, simpangan)]


def buat_sampel(seed, per_kelas):
    rng = random.Random(seed)
    X, y = [], []
    for kelas in KELAS:
        for _ in range(per_kelas):
            fitur = [m + rng.gauss(0, s) for m, s in zip(PROTOTIPE[kelas], NOISE)]
            fitur[0] = jepit(fitur[0], 0, 1)
            fitur[1] = jepit(fitur[1], 0, 10)
            fitur[2] = jepit(fitur[2], 0, 6)
            fitur[3] = jepit(fitur[3], 0, 1)
            fitur[4] = jepit(fitur[4], 0, 1)
            fitur[5] = jepit(fitur[5], 0, 1)
            fitur[6] = jepit(fitur[6], 0, 1)
            X.append(fitur)
            y.append(kelas)
    return X, y


class Pengklasifikasi:
    def __init__(self):
        X, y = buat_sampel(seed=99, per_kelas=40)
        self.rata, self.simpangan = standardisasi(X)
        Xs = [terapkan(x, self.rata, self.simpangan) for x in X]
        self.centroid = {}
        for kelas in KELAS:
            anggota = [x for x, label in zip(Xs, y) if label == kelas]
            self.centroid[kelas] = [statistics.fmean(d) for d in zip(*anggota)]

    def klasifikasi(self, fitur):
        x = terapkan(fitur, self.rata, self.simpangan)
        jarak = {
            kelas: math.dist(x, pusat) for kelas, pusat in self.centroid.items()
        }
        kelas = min(jarak, key=jarak.get)
        return kelas, jarak[kelas]


def evaluasi(pengklasifikasi):
    X, y = buat_sampel(seed=123, per_kelas=20)
    tebak = [pengklasifikasi.klasifikasi(x)[0] for x in X]
    akurasi = sum(1 for a, b in zip(y, tebak) if a == b) / len(y)
    f1_kelas = []
    print("3. Klasifikasi driver (nearest centroid dari data berlabel)")
    print(f"   Akurasi pada data uji sintetik: {akurasi:.1%}")
    print("   Confusion matrix (baris aktual, kolom prediksi):")
    print("      " + "".join(f"{k[:4]:>6}" for k in KELAS))
    for aktual in KELAS:
        baris = [sum(1 for a, b in zip(y, tebak) if a == aktual and b == pred) for pred in KELAS]
        print(f"   {aktual[:4]:<5}" + "".join(f"{n:>6}" for n in baris))

    for kelas in KELAS:
        tp = sum(1 for a, b in zip(y, tebak) if a == kelas and b == kelas)
        fp = sum(1 for a, b in zip(y, tebak) if a != kelas and b == kelas)
        fn = sum(1 for a, b in zip(y, tebak) if a == kelas and b != kelas)
        presisi = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * presisi * recall / (presisi + recall) if presisi + recall else 0.0
        f1_kelas.append(f1)
        print(f"   {kelas:<12} presisi {presisi:.0%}  recall {recall:.0%}  F1 {f1:.0%}")
    print(f"   Macro F1: {statistics.fmean(f1_kelas):.1%}")


def peta(ndvi, klasifikasi_sel=None):
    garis = []
    for r in range(N):
        baris = []
        for c in range(N):
            if klasifikasi_sel and (r, c) in klasifikasi_sel:
                baris.append(SIMBOL[klasifikasi_sel[(r, c)]])
            elif (r, c) in RIVER:
                baris.append("~")
            elif (r, c) in ROAD:
                baris.append("=")
            elif ndvi[r][c] >= AMBANG_HUTAN:
                baris.append("^")
            else:
                baris.append(".")
        garis.append("".join(baris))
    return "\n".join(f"   {baris}" for baris in garis)


def pola_musiman():
    profil = {
        "kebakaran": [2, 2, 4, 8, 14, 22, 30, 34, 20, 8, 3, 1],
        "pertanian": [8, 9, 11, 13, 15, 17, 19, 20, 18, 15, 12, 9],
        "penebangan": [9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 12, 11],
        "tambang": [11, 12, 12, 13, 13, 14, 14, 15, 15, 14, 13, 13],
        "banjir": [5, 6, 5, 4, 3, 2, 1, 1, 2, 4, 7, 8],
    }
    print("\n5. Pola musiman (data contoh 12 bulan, indeks variasi = stdev/rata-rata)")
    for kelas, nilai in sorted(
        profil.items(), key=lambda x: -statistics.pstdev(x[1]) / statistics.fmean(x[1])
    ):
        cv = statistics.pstdev(nilai) / statistics.fmean(nilai)
        print(f"   {kelas:<12} {cv:4.2f}  {'#' * round(cv * 30)}")


def main():
    ndvi0, ndvi1, vv0, vv1, _ = buat_citra()
    alert = {
        (r, c)
        for r in range(N)
        for c in range(N)
        if ndvi0[r][c] >= AMBANG_HUTAN and ndvi1[r][c] < AMBANG_HILANG
    }
    kelompok = [k for k in komponen(alert) if len(k) >= 3]

    print("1. Input citra Sentinel-1 dan Sentinel-2 (grid 20x20, resolusi 10 m)")
    print(f"   Piksel hutan sebelum disturbance: {sum(1 for r in range(N) for c in range(N) if ndvi0[r][c] >= AMBANG_HUTAN)}")
    print(f"   Piksel alert setelah disturbance : {len(alert)}")
    print(f"   Jumlah kejadian terdeteksi      : {len(kelompok)}")

    print("\n2. Tutupan hutan sebelum disturbance (^ hutan, ~ sungai, = jalan)")
    print(peta(ndvi0))

    pengklasifikasi = Pengklasifikasi()
    evaluasi(pengklasifikasi)
    print()

    klasifikasi_sel = {}
    print("4. Peta driver pada bulan berikutnya")
    for k in sorted(kelompok, key=lambda x: (x[0][0], x[0][1])):
        fitur = fitur_kejadian(k, ndvi0, ndvi1, vv0, vv1)
        kelas, jarak = pengklasifikasi.klasifikasi(fitur)
        label = kelas if jarak <= 2.5 else "tidak_yakin"
        for sel in k:
            klasifikasi_sel[sel] = label
        luas_ha = len(k) * SKALA_PIKSEL_M * SKALA_PIKSEL_M / 10_000
        print(
            f"   {len(k):2d} piksel ({luas_ha:.2f} ha) di ({k[0][0]:2d},{k[0][1]:2d}) "
            f"-> {label:<11} jarak {jarak:.2f}"
        )

    print()
    print(peta(ndvi1, klasifikasi_sel))
    print("   A pertanian, L penebangan, M tambang, F kebakaran, B banjir, R jalan, ? tidak yakin")

    pola_musiman()
    print("\nCatatan: seluruh citra dan label di sini sintetik, hanya untuk demonstrasi alur.")


if __name__ == "__main__":
    main()
