"""Materi 3: Career and Competition in Industrial Era with GIS (Widi Primadeva).

Perangkat GIS sederhana tanpa library: jarak geodesik, filter bounding box,
point-in-polygon, analisis buffer, dan skor kesesuaian lokasi (kasus Indomaret).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R_BUMI_KM = 6371.0088


@dataclass(frozen=True)
class Titik:
    nama: str
    lat: float
    lon: float
    jenis: str


@dataclass(frozen=True)
class Kandidat:
    nama: str
    lat: float
    lon: float
    populasi: int
    skor_banjir: float  # 0 aman sampai 1 rawan
    jarak_jalan_km: float


def jarak_km(a: Titik, b: Titik) -> float:
    """Formula haversine, jarak permukaan bola bumi dalam kilometer."""
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = p2 - p1
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_BUMI_KM * math.asin(math.sqrt(h))


def dalam_bbox(titik: Titik, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> bool:
    return lat_min <= titik.lat <= lat_max and lon_min <= titik.lon <= lon_max


def point_in_polygon(titik: Titik, poligon: list[tuple[float, float]]) -> bool:
    """Ray casting pada poligon (lat, lon). Titik dihitung di dalam bila
    jumlah perpotongan garis horizontal dari titik ke kanan berjumlah ganjil."""
    lat, lon = titik.lat, titik.lon
    dalam = False
    j = len(poligon) - 1
    for i in range(len(poligon)):
        lat_i, lon_i = poligon[i]
        lat_j, lon_j = poligon[j]
        if (lat_i > lat) != (lat_j > lat):
            potong = (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i) + lon_i
            if lon < potong:
                dalam = not dalam
        j = i
    return dalam


def dalam_radius(pusat: Titik, titik: Titik, radius_km: float) -> bool:
    return jarak_km(pusat, titik) <= radius_km


def terdekat(pusat: Titik, daftar: list[Titik]) -> tuple[Titik, float]:
    return min(((t, jarak_km(pusat, t)) for t in daftar), key=lambda x: x[1])


def skor_lokasi(kandidat: Kandidat, toko: list[Titik], jalan: list[Titik]) -> float:
    """Bobot: populasi 40%, akses jalan 30%, jarak pesaing 20%, risiko banjir 10%."""
    skor_populasi = min(1.0, kandidat.populasi / 120_000)
    skor_jalan = max(0.0, 1.0 - kandidat.jarak_jalan_km / 5.0)
    _, jarak_pesaing = terdekat(Titik(kandidat.nama, kandidat.lat, kandidat.lon, "kandidat"), toko)
    skor_pesaing = min(1.0, jarak_pesaing / 4.0)
    skor_aman = 1.0 - kandidat.skor_banjir
    return 0.40 * skor_populasi + 0.30 * skor_jalan + 0.20 * skor_pesaing + 0.10 * skor_aman


def peta_ascii(titik: list[Titik], poligon: list[tuple[float, float]], lebar=46, tinggi=20) -> str:
    lat_semua = [t.lat for t in titik] + [p[0] for p in poligon]
    lon_semua = [t.lon for t in titik] + [p[1] for p in poligon]
    lat_min, lat_max = min(lat_semua), max(lat_semua)
    lon_min, lon_max = min(lon_semua), max(lon_semua)
    lat_min, lat_max = lat_min - 0.01, lat_max + 0.01
    lon_min, lon_max = lon_min - 0.01, lon_max + 0.01

    kisi = [[" " for _ in range(lebar)] for _ in range(tinggi)]
    simbol = {"toko": "T", "gudang": "G", "kandidat": "?", "pelanggan": "o"}

    for baris in range(tinggi):
        lat = lat_max - (baris + 0.5) * (lat_max - lat_min) / tinggi
        for kolom in range(lebar):
            lon = lon_min + (kolom + 0.5) * (lon_max - lon_min) / lebar
            if point_in_polygon(Titik("", lat, lon, ""), poligon):
                kisi[baris][kolom] = "."

    for t in titik:
        kolom = int((t.lon - lon_min) / (lon_max - lon_min) * lebar)
        baris = int((lat_max - t.lat) / (lat_max - lat_min) * tinggi)
        kolom = min(lebar - 1, max(0, kolom))
        baris = min(tinggi - 1, max(0, baris))
        kisi[baris][kolom] = simbol.get(t.jenis, "?")

    return "\n".join("".join(baris) for baris in kisi)


def main() -> None:
    pusat = Titik("Kota Bengkulu", -3.7928, 102.2608, "pusat")

    toko = [
        Titik("Toko A", -3.7850, 102.2650, "toko"),
        Titik("Toko B", -3.8010, 102.2500, "toko"),
        Titik("Toko C", -3.7750, 102.2800, "toko"),
    ]
    gudang = [
        Titik("Gudang Utama", -3.7950, 102.2700, "gudang"),
        Titik("Gudang Timur", -3.7800, 102.2900, "gudang"),
    ]
    pelanggan = [
        Titik("Pelanggan 1", -3.7900, 102.2550, "pelanggan"),
        Titik("Pelanggan 2", -3.8050, 102.2680, "pelanggan"),
        Titik("Pelanggan 3", -3.7820, 102.2720, "pelanggan"),
        Titik("Pelanggan 4", -3.7980, 102.2820, "pelanggan"),
    ]
    kandidat = [
        Kandidat("Lokasi Sawah", -3.7880, 102.2580, 95_000, 0.15, 0.8),
        Kandidat("Lokasi Pasar", -3.7955, 102.2620, 118_000, 0.30, 0.4),
        Kandidat("Lokasi Perumahan", -3.7815, 102.2850, 72_000, 0.05, 1.6),
    ]

    area = [
        (-3.7720, 102.2450),
        (-3.7720, 102.2950),
        (-3.8120, 102.2950),
        (-3.8120, 102.2450),
    ]

    print("1. Jarak geodesik dari pusat kota (haversine)")
    for t in toko + gudang + pelanggan:
        print(f"   {t.nama:<15} {jarak_km(pusat, t):5.2f} km")

    print("\n2. Filter bounding box dan point-in-polygon")
    dekat = [t for t in toko + gudang if dalam_bbox(t, -3.8000, -3.7700, 102.2500, 102.2850)]
    print(f"   Titik di dalam bbox: {', '.join(t.nama for t in dekat)}")
    for t in toko + kandidat:
        status = "di dalam" if point_in_polygon(Titik(t.nama, t.lat, t.lon, ""), area) else "di luar"
        print(f"   {t.nama:<18} {status} area kajian")

    print("\n3. Analisis buffer 3 km dari Gudang Utama")
    for t in pelanggan:
        if dalam_radius(gudang[0], t, 3.0):
            print(f"   {t.nama} terjangkau ({jarak_km(gudang[0], t):.2f} km)")
    dekat_gudang, jarak = terdekat(pusat, gudang)
    print(f"   Gudang terdekat dari pusat: {dekat_gudang.nama} ({jarak:.2f} km)")

    print("\n4. Skor kesesuaian lokasi toko baru")
    skor = sorted(
        ((skor_lokasi(k, toko, gudang), k) for k in kandidat),
        key=lambda x: -x[0],
    )
    for nilai, k in skor:
        print(f"   {k.nama:<20} skor {nilai:.3f}")

    print("\n5. Peta ASCII (T toko, G gudang, o pelanggan, . area kajian)")
    semua = toko + gudang + pelanggan + [
        Titik(k.nama, k.lat, k.lon, "kandidat") for k in kandidat
    ]
    print(peta_ascii(semua, area))

    print(f"\nRekomendasi: bangun toko di {skor[0][1].nama}.")


if __name__ == "__main__":
    main()
