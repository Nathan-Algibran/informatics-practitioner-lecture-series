"""Materi 2: Bridging Global Infrastructure to Address Local Challenge with HPC.

Menghitung ulang gagasan inti kuliah: satuan flops, mengapa paralelisasi penting,
bagaimana speedup diukur, dan bagaimana sebuah job dikirim ke superkomputer.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time

FLOPS_PER_ITERASI_MONTE_CARLO = 5  # dua perkalian, satu penjumlahan, satu perbandingan


def monte_carlo_pi(jumlah: int, seed: int) -> int:
    """Mengembalikan jumlah titik yang jatuh di dalam seperempat lingkaran."""
    rng = random.Random(seed)
    dalam = 0
    for _ in range(jumlah):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            dalam += 1
    return dalam


def _bagian(pekerjaan: tuple[int, int]) -> int:
    jumlah, seed = pekerjaan
    return monte_carlo_pi(jumlah, seed)


def ukur_serial(total: int) -> tuple[float, float]:
    mulai = time.perf_counter()
    dalam = monte_carlo_pi(total, seed=1)
    detik = time.perf_counter() - mulai
    return 4 * dalam / total, detik


def ukur_paralel(total: int, pekerja: int) -> tuple[float, float]:
    ukuran = total // pekerja
    pekerjaan = [(ukuran, seed) for seed in range(1, pekerja + 1)]
    mulai = time.perf_counter()
    with mp.Pool(pekerja) as kolam:
        hasil = kolam.map(_bagian, pekerjaan)
    detik = time.perf_counter() - mulai
    pi = 4 * sum(hasil) / (ukuran * pekerja)
    return pi, detik


def amdahl(fraksi_paralel: float, pekerja: int) -> float:
    return 1.0 / ((1.0 - fraksi_paralel) + fraksi_paralel / pekerja)


def konversi_flops(nilai: float) -> str:
    satuan = ["flops", "kflops", "Mflops", "Gflops", "Tflops", "Pflops", "Eflops"]
    pangkat = 0
    while nilai >= 1000 and pangkat < len(satuan) - 1:
        nilai /= 1000
        pangkat += 1
    return f"{nilai:.2f} {satuan[pangkat]}"


def format_durasi(detik: float) -> str:
    if detik >= 365 * 24 * 3600:
        return f"{detik / (365 * 24 * 3600):.1f} tahun"
    if detik >= 86400:
        return f"{detik / 86400:.1f} hari"
    if detik >= 3600:
        return f"{detik / 3600:.1f} jam"
    if detik >= 60:
        return f"{detik / 60:.1f} menit"
    return f"{detik:.1f} detik"


def skrip_job(nama: str, node: int, proses: int, jam: int) -> str:
    """Skrip batch seperti yang dikirim ke sistem HPC (gaya Slurm)."""
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={nama}",
            f"#SBATCH --nodes={node}",
            f"#SBATCH --ntasks-per-node={proses}",
            f"#SBATCH --time={jam}:00:00",
            "",
            "module load python/3.12",
            "mpirun -np $((SLURM_NTASKS)) python hitung_pi.py --total 1000000000",
        ]
    )


def main() -> None:
    total = 4_000_000
    pekerja = min(4, os.cpu_count() or 1)

    print("Satuan kecepatan komputasi")
    for nilai in (1e9, 1e12, 1e15, 1e18):
        print(f"  {nilai:>22,.0f} flop/s = {konversi_flops(nilai)}")

    print("\nBenchmark Monte Carlo pi (menghitung nilai pi)")
    pi_serial, detik_serial = ukur_serial(total)
    flops = total * FLOPS_PER_ITERASI_MONTE_CARLO / detik_serial
    print(f"  Serial  : pi = {pi_serial:.5f} dalam {detik_serial:.2f} s ({konversi_flops(flops)}/s)")

    pi_paralel, detik_paralel = ukur_paralel(total, pekerja)
    speedup = detik_serial / detik_paralel
    efisiensi = speedup / pekerja
    print(f"  Paralel : pi = {pi_paralel:.5f} dalam {detik_paralel:.2f} s ({pekerja} proses)")
    print(f"  Speedup {speedup:.2f}x, efisiensi {efisiensi:.0%}")

    print("\nProyeksi hukum Amdahl (95% pekerjaan bisa diparalelkan)")
    for p in (1, 2, 4, 16, 64, 1024):
        print(f"  {p:>5} proses -> speedup {amdahl(0.95, p):>7.1f}x")

    print("\nBerapa lama 1e18 operasi dikerjakan?")
    for nama, kecepatan in (
        ("Laptop 5 Gflops", 5e9),
        ("Server 5 Tflops", 5e12),
        ("Superkomputer 1 Eflops", 1e18),
    ):
        detik = 1e18 / kecepatan
        print(f"  {nama:<26} {detik:>18,.0f} s = {format_durasi(detik):>10}")

    print("\nContoh skrip job yang dikirim ke antrean HPC")
    print(skrip_job("simulasi-pi", node=2, proses=64, jam=1))


if __name__ == "__main__":
    main()
