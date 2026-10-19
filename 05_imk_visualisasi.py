"""Materi 5: IMK dan Visualisasi Data (Modul 1 sampai 6).

Empat alat kecil yang menerapkan materi: pemeriksa kontras WCAG, evaluasi
heuristik, kuesioner beban kognitif, persona dan skenario, serta dasbor ASCII
mengikuti prinsip Shneiderman.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# ---------------------------------------------------------------- kontras WCAG


def luminans(warna_hex: str) -> float:
    kode = warna_hex.lstrip("#")
    kanal = [int(kode[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linier = [
        v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in kanal
    ]
    return 0.2126 * linier[0] + 0.7152 * linier[1] + 0.0722 * linier[2]


def rasio_kontras(warna_teks: str, warna_latar: str) -> float:
    a, b = luminans(warna_teks), luminans(warna_latar)
    terang, gelap = max(a, b), min(a, b)
    return (terang + 0.05) / (gelap + 0.05)


def cek_kontras():
    print("1. Pemeriksa kontras WCAG (prinsip Perceivable)")
    pasangan = [
        ("#111111", "#ffffff", "teks utama"),
        ("#777777", "#ffffff", "teks abu-abu"),
        ("#ffffff", "#ffcc00", "tombol kuning"),
        ("#1a4d2e", "#f2f7f2", "teks hijau tua"),
    ]
    for teks, latar, nama in pasangan:
        rasio = rasio_kontras(teks, latar)
        normal = "lulus AA" if rasio >= 4.5 else "gagal AA"
        besar = "lulus" if rasio >= 3.0 else "gagal"
        print(
            f"   {nama:<18} {rasio:5.2f}:1  teks normal {normal:<9} "
            f"teks besar {besar}"
        )


# ---------------------------------------------------------------- evaluasi heuristik


@dataclass
class Temuan:
    heuristik: str
    skor: int
    masalah: str
    saran: str


def evaluasi_heuristik():
    data = [
        Temuan("Visibilitas status sistem", 2, "Tidak ada indikator proses saat grafik dimuat", "Tambahkan loading state dan progres"),
        Temuan("Kesesuaian dengan dunia nyata", 4, "Istilah teknis masih muncul di filter", "Ganti dengan bahasa pengguna"),
        Temuan("Kendali dan kebebasan pengguna", 3, "Filter tidak punya tombol reset", "Sediakan reset dan undo"),
        Temuan("Konsistensi dan standar", 2, "Warna tombol berbeda antar halaman", "Pakai satu design token"),
        Temuan("Pencegahan kesalahan", 3, "Ekspor langsung berjalan tanpa konfirmasi", "Tambahkan pratinjau dan konfirmasi"),
        Temuan("Pengenalan daripada mengingat", 2, "Pengguna harus hafal kode kategori", "Tampilkan label dan saran"),
        Temuan("Fleksibilitas dan efisiensi", 4, "Belum ada pintasan keyboard", "Tambahkan shortcut untuk pencarian"),
        Temuan("Desain minimalis", 3, "Terlalu banyak warna pada satu grafik", "Batasi palet dan hapus chartjunk"),
        Temuan("Pemulihan dari kesalahan", 2, "Pesan galat hanya kode angka", "Jelaskan penyebab dan langkah perbaikan"),
        Temuan("Bantuan dan dokumentasi", 3, "Bantuan tersembunyi di menu", "Sediakan konteks bantuan di layar"),
    ]
    rata = sum(t.skor for t in data) / len(data)
    print("\n2. Evaluasi heuristik (data contoh, skala 1 buruk sampai 5 baik)")
    print(f"   Skor rata-rata {rata:.2f}/5")
    print("   Temuan prioritas:")
    for t in sorted(data, key=lambda x: x.skor)[:5]:
        print(f"     [{t.skor}/5] {t.heuristik}: {t.masalah}")
        print(f"            saran: {t.saran}")


# ---------------------------------------------------------------- beban kognitif


def beban_kognitif():
    intrinsic = 4
    extraneous = 2
    germane = 3
    total = intrinsic + extraneous + germane
    print("\n3. Beban kognitif (skala 1 sampai 5)")
    print(f"   Intrinsic {intrinsic}, extraneous {extraneous}, germane {germane}, total {total}/15")
    if extraneous >= 3:
        print("   Beban tambahan tinggi: sederhanakan navigasi dan kurangi elemen dekoratif.")
    else:
        print("   Beban tambahan terkendali: pertahankan tampilan yang sederhana.")
    if germane <= 2:
        print("   Naikkan germane load dengan contoh nyata dan umpan balik yang menjelaskan.")


# ---------------------------------------------------------------- persona dan skenario


@dataclass
class Persona:
    nama: str
    usia: int
    peran: str
    tujuan: str
    pain_point: str
    perilaku: str


def persona_dan_skenario():
    p = Persona(
        nama="Rina",
        usia=21,
        peran="Mahasiswa yang memantau data tugas akhir",
        tujuan="Menemukan tren bulanan dalam waktu kurang dari satu menit",
        pain_point="Filter terlalu banyak dan istilah statistik membingungkan",
        perilaku="Membuka dasbor lewat ponsel, sering di jaringan lambat",
    )
    print("\n4. Persona dan skenario penggunaan (contoh)")
    print(f"   {p.nama}, {p.usia} tahun, {p.peran}")
    print(f"   Tujuan: {p.tujuan}")
    print(f"   Pain point: {p.pain_point}")
    print(f"   Perilaku: {p.perilaku}")
    print("   Skenario: buka dasbor -> lihat ringkasan -> filter kuartal -> buka detail -> ekspor.")


# ---------------------------------------------------------------- dasbor ASCII

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
PENGGUNA = [120, 135, 128, 150, 162, 158, 171, 185, 179, 190, 205, 212]


def grafik_batang(nilai, label, lebar=36):
    maks = max(nilai)
    for nama, v in zip(label, nilai):
        panjang = round(v / maks * lebar)
        print(f"   {nama:<4} {'#' * panjang:<{lebar}} {v}")


def ringkas():
    print(f"   Total {sum(PENGGUNA):,} | rata-rata {sum(PENGGUNA) / len(PENGGUNA):.1f} "
          f"| tertinggi {max(PENGGUNA)} pada {BULAN[PENGGUNA.index(max(PENGGUNA))]}")


def filter_kuartal(awal, akhir):
    print(f"   Kuartal {BULAN[awal]} sampai {BULAN[akhir]}:")
    grafik_batang(PENGGUNA[awal : akhir + 1], BULAN[awal : akhir + 1])


def detail():
    urut = sorted(zip(PENGGUNA, BULAN), reverse=True)[:3]
    print("   Tiga bulan dengan pengguna terbanyak:")
    for v, b in urut:
        print(f"     {b}: {v}")


def dasbor():
    print("\n5. Dasbor interaktif (prinsip overview first, zoom and filter, details on demand)")
    print("   Overview:")
    grafik_batang(PENGGUNA, BULAN)
    ringkas()
    print("\n   Filter kuartal 3 (Jul sampai Sep):")
    filter_kuartal(6, 8)
    print("\n   Detail sesuai permintaan:")
    detail()

    if "--interaktif" in sys.argv:
        print("\n   Mode interaktif: ketik ringkas, filter, detail, atau keluar.")
        while True:
            perintah = input("   perintah> ").strip().lower()
            if perintah in ("keluar", "exit", "q"):
                break
            if perintah == "ringkas":
                ringkas()
            elif perintah == "filter":
                filter_kuartal(6, 8)
            elif perintah == "detail":
                detail()
            else:
                print("   Perintah tidak dikenal.")


def main():
    cek_kontras()
    evaluasi_heuristik()
    beban_kognitif()
    persona_dan_skenario()
    dasbor()


if __name__ == "__main__":
    main()
