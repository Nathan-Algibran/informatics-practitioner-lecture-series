"""Materi 1: Make Sense of Data with Analysis and AI (Andria Arisal, BRIN).

Studi kasus PT Maju Bersama: rantai data -> informasi -> insight -> keputusan,
dengan analitik deskriptif, diagnostik, prediktif, dan preskriptif.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

PELANGGAN = [
    ("CV Sinar Jaya", "Distributor", 0.10),
    ("PT Bangun Persada", "Proyek", 0.45),
    ("Toko Berkah Abadi", "Ritel", 0.08),
    ("PT Griya Utama", "Proyek", 0.38),
    ("CV Mitra Niaga", "Distributor", 0.15),
    ("Toko Sumber Rejeki", "Ritel", 0.12),
    ("PT Karya Beton", "Proyek", 0.52),
    ("CV Anugrah Teknik", "Distributor", 0.20),
    ("Toko Makmur Jaya", "Ritel", 0.06),
    ("PT Nusa Konstruksi", "Proyek", 0.41),
    ("CV Sentosa Supply", "Distributor", 0.18),
    ("Toko Harapan Baru", "Ritel", 0.14),
]

PRODUK = [
    ("Semen 40 kg", 0.18),
    ("Besi Beton 10 mm", 0.22),
    ("Cat Tembok 20 kg", 0.27),
    ("Keramik 60x60", 0.31),
    ("Pipa PVC 3 inci", 0.24),
]

SKALA_SEGMEN = {"Distributor": 120_000_000, "Proyek": 260_000_000, "Ritel": 45_000_000}


@dataclass
class Invoice:
    id: str
    bulan: int
    pelanggan: str
    segmen: str
    produk: str
    nilai: float
    hpp: float
    dibayar: float
    telat_hari: int

    @property
    def piutang(self) -> float:
        return self.nilai - self.dibayar


def buat_dataset(seed: int = 42, jumlah_bulan: int = 24) -> list[Invoice]:
    """Meniru data sintetik PT Maju Bersama: omzet naik, margin turun, piutang naik."""
    rng = random.Random(seed)
    invoices: list[Invoice] = []
    for bulan in range(1, jumlah_bulan + 1):
        for i in range(10):
            nama, segmen, risiko = PELANGGAN[rng.randrange(len(PELANGGAN))]
            produk, margin = PRODUK[rng.randrange(len(PRODUK))]
            nilai = (
                rng.uniform(0.6, 1.8)
                * SKALA_SEGMEN[segmen]
                * (1 + 0.012 * bulan)
            )
            hpp = nilai * (1 - margin + 0.11 * bulan / jumlah_bulan)

            peluang_telat = risiko + (0.20 if segmen == "Proyek" else -0.08)
            peluang_telat = min(0.9, max(0.02, peluang_telat))
            if rng.random() < peluang_telat:
                telat = rng.randint(15, 90)
                dibayar = nilai * rng.uniform(0.0, 0.5)
            else:
                telat = 0
                dibayar = nilai

            invoices.append(
                Invoice(
                    id=f"INV{bulan:02d}{i:02d}",
                    bulan=bulan,
                    pelanggan=nama,
                    segmen=segmen,
                    produk=produk,
                    nilai=nilai,
                    hpp=hpp,
                    dibayar=dibayar,
                    telat_hari=telat,
                )
            )
    return invoices


def rupiah(nilai: float) -> str:
    return f"Rp {nilai:,.0f}".replace(",", ".")


def analisis_deskriptif(invoices: list[Invoice]) -> None:
    omzet = sum(v.nilai for v in invoices)
    hpp = sum(v.hpp for v in invoices)
    piutang = sum(v.piutang for v in invoices)
    laba = omzet - hpp

    bulan_ke = max(v.bulan for v in invoices)
    tengah = bulan_ke // 2
    awal = sum(v.nilai for v in invoices if v.bulan <= tengah)
    akhir = sum(v.nilai for v in invoices if v.bulan > tengah)

    print("1. Analitik deskriptif (apa yang terjadi?)")
    print(f"   Omzet 2 tahun       : {rupiah(omzet)}")
    print(f"   Laba kotor          : {rupiah(laba)} ({laba / omzet:.1%} dari omzet)")
    print(f"   Piutang belum bayar : {rupiah(piutang)} ({piutang / omzet:.1%} dari omzet)")
    print(f"   Pertumbuhan omzet   : semester-2 vs semester-1 = {akhir / awal - 1:+.1%}")

    for semester in (1, 2):
        baris = [v for v in invoices if (v.bulan > tengah) == (semester == 2)]
        m = (sum(v.nilai for v in baris) - sum(v.hpp for v in baris)) / sum(
            v.nilai for v in baris
        )
        print(f"   Margin semester-{semester}   : {m:.1%}")


def analisis_diagnostik(invoices: list[Invoice]) -> None:
    print("\n2. Analitik diagnostik (mengapa terjadi?)")
    print("   Segmen        Omzet              Margin   Invoice telat   Piutang")
    for segmen in ("Distributor", "Ritel", "Proyek"):
        baris = [v for v in invoices if v.segmen == segmen]
        omzet = sum(v.nilai for v in baris)
        margin = (omzet - sum(v.hpp for v in baris)) / omzet
        telat = sum(1 for v in baris if v.telat_hari > 15)
        piutang = sum(v.piutang for v in baris)
        print(
            f"   {segmen:<13} {rupiah(omzet):>18} {margin:>7.1%} "
            f"{telat / len(baris):>12.1%} {rupiah(piutang):>18}"
        )

    per_pelanggan: dict[str, float] = {}
    for v in invoices:
        per_pelanggan[v.pelanggan] = per_pelanggan.get(v.pelanggan, 0.0) + v.piutang

    print("\n   Lima pelanggan penyumbang piutang terbesar:")
    for nama, nilai in sorted(per_pelanggan.items(), key=lambda x: -x[1])[:5]:
        print(f"     {nama:<22} {rupiah(nilai):>16}")


def bangun_fitur(invoices: list[Invoice]):
    """Fitur hanya memakai riwayat bulan-bulan sebelumnya agar tidak bocor (leakage)."""
    riwayat: dict[str, tuple[float, int]] = {}
    X, y, meta = [], [], []
    for v in sorted(invoices, key=lambda x: (x.bulan, x.id)):
        total_telat, jumlah = riwayat.get(v.pelanggan, (0.0, 0))
        rata_telat = total_telat / jumlah if jumlah else 0.0
        X.append(
            [
                v.nilai / 1_000_000,
                rata_telat / 30.0,
                1.0 if v.segmen == "Proyek" else 0.0,
            ]
        )
        y.append(1 if v.telat_hari > 15 else 0)
        meta.append(v)
        riwayat[v.pelanggan] = (total_telat + v.telat_hari, jumlah + 1)
    return X, y, meta


def standardisasi(X: list[list[float]]):
    kolom = list(zip(*X))
    rata = [statistics.fmean(k) for k in kolom]
    simpangan = [statistics.pstdev(k) or 1.0 for k in kolom]
    return rata, simpangan


def terapkan(X, rata, simpangan):
    return [[(x - r) / s for x, r, s in zip(baris, rata, simpangan)] for baris in X]


def latih_logistik(X, y, epoch=800, lr=0.35):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(epoch):
        gw, gb = [0.0] * d, 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xij for wj, xij in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-z))
            e = p - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        w = [wj - lr * g / n for wj, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def analisis_prediktif(invoices: list[Invoice]) -> None:
    X, y, meta = bangun_fitur(invoices)
    potong = int(len(X) * 0.7)
    X_latih, X_uji = X[:potong], X[potong:]
    y_latih, y_uji = y[:potong], y[potong:]

    rata, simpangan = standardisasi(X_latih)
    w, b = latih_logistik(terapkan(X_latih, rata, simpangan), y_latih)

    peluang = []
    for xi in terapkan(X_uji, rata, simpangan):
        z = b + sum(wj * xij for wj, xij in zip(w, xi))
        peluang.append(1.0 / (1.0 + math.exp(-z)))

    tebak = [1 if p >= 0.5 else 0 for p in peluang]
    benar = sum(1 for t, a in zip(tebak, y_uji) if t == a)
    tp = sum(1 for t, a in zip(tebak, y_uji) if t == 1 and a == 1)
    fp = sum(1 for t, a in zip(tebak, y_uji) if t == 1 and a == 0)
    fn = sum(1 for t, a in zip(tebak, y_uji) if t == 0 and a == 1)
    presisi = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * presisi * recall / (presisi + recall) if presisi + recall else 0.0

    print("\n3. Analitik prediktif (invoice mana yang berisiko telat bayar?)")
    print(f"   Data latih {len(X_latih)} invoice, data uji {len(X_uji)} invoice")
    print(f"   Akurasi {benar / len(y_uji):.1%} | Presisi {presisi:.1%} | Recall {recall:.1%} | F1 {f1:.1%}")
    print("   Bobot fitur (nilai > 0 menambah risiko, nilai < 0 menurunkannya):")
    label = ["nilai invoice", "riwayat telat pelanggan", "segmen proyek"]
    for nama, bobot in sorted(zip(label, w), key=lambda x: -abs(x[1])):
        print(f"     {nama:<24} {bobot:+.3f}")
    print("   Tanda koefisien bisa berbeda dari korelasi sederhana karena antarfitur saling berkorelasi.")

    uji = [
        (p, m)
        for p, m in zip(peluang, meta[potong:])
        if p >= 0.5 and m.piutang > 0
    ]
    return uji


def analisis_preskriptif(uji) -> None:
    print("\n4. Analitik preskriptif (apa yang sebaiknya dilakukan?)")
    print("   Invoice yang sudah lunas tidak masuk daftar tindakan.")
    if not uji:
        print("   Tidak ada invoice berisiko pada data uji.")
        return
    print("   Invoice prioritas dan tindakan yang disarankan:")
    for p, inv in sorted(uji, key=lambda x: -x[0])[:5]:
        if inv.segmen == "Proyek":
            tindakan = "wajibkan termin muka, jadwalkan penagihan mingguan"
        elif inv.piutang > 100_000_000:
            tindakan = "batasi plafon kredit, minta jaminan tambahan"
        else:
            tindakan = "kirim pengingat otomatis, tawarkan diskon bayar cepat"
        print(
            f"     {inv.id} {inv.pelanggan:<20} peluang {p:.0%} "
            f"piutang {rupiah(inv.piutang)}"
        )
        print(f"        -> {tindakan}")


def main() -> None:
    invoices = buat_dataset()
    print("PT Maju Bersama, analisis 24 bulan (1 Juli sampai 30 Juni)\n")
    analisis_deskriptif(invoices)
    analisis_diagnostik(invoices)
    uji = analisis_prediktif(invoices)
    analisis_preskriptif(uji)


if __name__ == "__main__":
    main()
