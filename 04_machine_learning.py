"""Materi 4: Introduction to Machine Learning.

Tiga pendekatan utama tanpa library eksternal:
regresi (harga rumah), klasifikasi (spam), dan clustering (segmentasi pelanggan),
lengkap dengan pembagian data latih/uji dan metrik evaluasi.
"""

from __future__ import annotations

import math
import random
import statistics

# ---------------------------------------------------------------- utilitas


def train_test_split(X, y, rasio=0.8, seed=7):
    indeks = list(range(len(X)))
    random.Random(seed).shuffle(indeks)
    potong = int(len(X) * rasio)
    latih, uji = indeks[:potong], indeks[potong:]
    return (
        [X[i] for i in latih],
        [X[i] for i in uji],
        [y[i] for i in latih],
        [y[i] for i in uji],
    )


def standardisasi(X):
    kolom = list(zip(*X))
    rata = [statistics.fmean(k) for k in kolom]
    simpangan = [statistics.pstdev(k) or 1.0 for k in kolom]
    return rata, simpangan


def terapkan(X, rata, simpangan):
    return [[(x - r) / s for x, r, s in zip(baris, rata, simpangan)] for baris in X]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------- regresi


def buat_data_rumah(n=120, seed=11):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        luas = rng.uniform(36, 200)
        kamar = rng.randint(1, 5)
        lokasi = rng.random()
        harga = 300 + 3.5 * luas + 150 * kamar + 500 * lokasi + rng.gauss(0, 60)
        X.append([luas, kamar, lokasi])
        y.append(harga)
    return X, y


def latih_regresi(X, y, lr=0.1, epoch=3000):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(epoch):
        gw, gb = [0.0] * d, 0.0
        for xi, yi in zip(X, y):
            tebakan = b + sum(wj * xij for wj, xij in zip(w, xi))
            galat = tebakan - yi
            for j in range(d):
                gw[j] += galat * xi[j]
            gb += galat
        w = [wj - lr * g / n for wj, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def prediksi_regresi(X, w, b):
    return [b + sum(wj * xij for wj, xij in zip(w, xi)) for xi in X]


def evaluasi_regresi(y_asli, y_tebak):
    galat = [a - t for a, t in zip(y_asli, y_tebak)]
    mae = statistics.fmean(abs(g) for g in galat)
    rmse = math.sqrt(statistics.fmean(g * g for g in galat))
    rata = statistics.fmean(y_asli)
    ss_res = sum(g * g for g in galat)
    ss_tot = sum((a - rata) ** 2 for a in y_asli)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return mae, rmse, r2


def bagian_regresi():
    X, y = buat_data_rumah()
    X_tr, X_uji, y_tr, y_uji = train_test_split(X, y)
    rata, simpangan = standardisasi(X_tr)
    w, b = latih_regresi(terapkan(X_tr, rata, simpangan), [v / 100 for v in y_tr])
    tebak = [t * 100 for t in prediksi_regresi(terapkan(X_uji, rata, simpangan), w, b)]
    mae, rmse, r2 = evaluasi_regresi(y_uji, tebak)

    print("1. Regresi linear: prediksi harga rumah")
    print(f"   Data latih {len(X_tr)}, data uji {len(X_uji)}")
    print(f"   MAE {mae:.1f} juta | RMSE {rmse:.1f} juta | R2 {r2:.3f}")
    hasil = sorted(zip(tebak, y_uji), key=lambda x: -x[1])[:3]
    for pred, asli in hasil:
        print(f"   rumah termahal: aktual {asli:.0f} juta, prediksi {pred:.0f} juta")


# ---------------------------------------------------------------- klasifikasi


def buat_data_spam(n=400, seed=23):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        spam = rng.random() < 0.45
        if spam:
            tautan = rng.randint(2, 12)
            kata = rng.randint(2, 10)
            kapital = rng.uniform(0.2, 0.9)
        else:
            tautan = rng.randint(0, 3)
            kata = rng.randint(0, 2)
            kapital = rng.uniform(0.0, 0.3)
        X.append([tautan, kata, kapital])
        y.append(1 if spam else 0)
    return X, y


def latih_logistik(X, y, lr=0.4, epoch=1000):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(epoch):
        gw, gb = [0.0] * d, 0.0
        for xi, yi in zip(X, y):
            p = sigmoid(b + sum(wj * xij for wj, xij in zip(w, xi)))
            e = p - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        w = [wj - lr * g / n for wj, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def evaluasi_klasifikasi(y_asli, y_tebak):
    tp = sum(1 for a, t in zip(y_asli, y_tebak) if a == 1 and t == 1)
    fp = sum(1 for a, t in zip(y_asli, y_tebak) if a == 0 and t == 1)
    fn = sum(1 for a, t in zip(y_asli, y_tebak) if a == 1 and t == 0)
    tn = sum(1 for a, t in zip(y_asli, y_tebak) if a == 0 and t == 0)
    akurasi = (tp + tn) / len(y_asli)
    presisi = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * presisi * recall / (presisi + recall) if presisi + recall else 0.0
    return akurasi, presisi, recall, f1, (tn, fp, fn, tp)


def bagian_klasifikasi():
    X, y = buat_data_spam()
    X_tr, X_uji, y_tr, y_uji = train_test_split(X, y, rasio=0.75)
    rata, simpangan = standardisasi(X_tr)
    w, b = latih_logistik(terapkan(X_tr, rata, simpangan), y_tr)
    peluang = [
        sigmoid(b + sum(wj * xij for wj, xij in zip(w, xi)))
        for xi in terapkan(X_uji, rata, simpangan)
    ]
    tebak = [1 if p >= 0.5 else 0 for p in peluang]
    akurasi, presisi, recall, f1, (tn, fp, fn, tp) = evaluasi_klasifikasi(y_uji, tebak)

    print("\n2. Klasifikasi: deteksi email spam")
    print(f"   Akurasi {akurasi:.1%} | Presisi {presisi:.1%} | Recall {recall:.1%} | F1 {f1:.1%}")
    print("   Confusion matrix (baris aktual, kolom prediksi):")
    print(f"     ham  -> ham {tn:3d} | spam {fp:3d}")
    print(f"     spam -> ham {fn:3d} | spam {tp:3d}")


# ---------------------------------------------------------------- clustering


def jarak_kuadrat(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(X, k, iterasi=100, seed=3):
    rng = random.Random(seed)
    centroids = [list(c) for c in rng.sample(X, k)]
    label = [0] * len(X)
    for _ in range(iterasi):
        label = [min(range(k), key=lambda c: jarak_kuadrat(x, centroids[c])) for x in X]
        baru = []
        for c in range(k):
            anggota = [x for x, l in zip(X, label) if l == c]
            if anggota:
                baru.append([statistics.fmean(d) for d in zip(*anggota)])
            else:
                baru.append(centroids[c])
        if baru == centroids:
            break
        centroids = baru
    sse = sum(jarak_kuadrat(x, centroids[l]) for x, l in zip(X, label))
    return centroids, label, sse


def buat_data_pelanggan(n=150, seed=5):
    rng = random.Random(seed)
    X = []
    for _ in range(n):
        kelompok = rng.choice(["hemat", "menengah", "premium"])
        if kelompok == "hemat":
            X.append([rng.gauss(3.5, 0.8), rng.gauss(2, 0.7), rng.gauss(80, 15)])
        elif kelompok == "menengah":
            X.append([rng.gauss(8.0, 1.2), rng.gauss(7, 1.3), rng.gauss(220, 35)])
        else:
            X.append([rng.gauss(20.0, 3.0), rng.gauss(16, 2.5), rng.gauss(650, 90)])
    return X


def bagian_clustering():
    X = buat_data_pelanggan()
    rata, simpangan = standardisasi(X)
    Xs = terapkan(X, rata, simpangan)

    terbaik = None
    for seed in range(5):
        hasil = kmeans(Xs, 3, seed=seed)
        if terbaik is None or hasil[2] < terbaik[2]:
            terbaik = hasil
    centroids, label, sse = terbaik

    print("\n3. Clustering K-means: segmentasi pelanggan")
    print(f"   SSE akhir {sse:.2f} untuk 3 cluster")
    for c in range(3):
        anggota = [x for x, l in zip(X, label) if l == c]
        pusat = [statistics.fmean(d) for d in zip(*anggota)]
        print(
            f"   Cluster {c}: {len(anggota):3d} pelanggan, "
            f"pendapatan {pusat[0]:.1f} juta, "
            f"transaksi {pusat[1]:.1f}/bulan, "
            f"belanja {pusat[2]:.0f} ribu"
        )

    print("   Metode elbow (SSE tiap jumlah cluster):")
    for k in range(1, 6):
        _, _, nilai = kmeans(Xs, k, seed=0)
        print(f"     k={k} -> SSE {nilai:7.2f}")


def main():
    bagian_regresi()
    bagian_klasifikasi()
    bagian_clustering()
    print("\nCatatan: akurasi data latih yang jauh lebih tinggi dari data uji adalah tanda overfitting.")


if __name__ == "__main__":
    main()
