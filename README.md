# Implementasi Python dari 6 Materi

Enam script mandiri, satu untuk setiap file `.txt`. Semuanya memakai standard library
Python saja, jadi tidak perlu `pip install` apa pun.

| File | Sumber | Isi implementasi |
|---|---|---|
| `01_analytics_ai.py` | Resume 1, Make Sense of Data with Analysis and AI | Data sintetik PT Maju Bersama, analitik deskriptif, diagnostik, prediktif (regresi logistik), dan preskriptif |
| `02_hpc.py` | resume 2, Bridging Global Infrastructure with HPC | Konversi flops, benchmark Monte Carlo pi serial vs paralel, speedup, efisiensi, hukum Amdahl, contoh skrip job Slurm |
| `03_gis_spasial.py` | Resume 3, Career and Competition in Industrial Era with GIS | Haversine, bounding box, point-in-polygon, buffer, skor lokasi toko baru, peta ASCII |
| `04_machine_learning.py` | resume 4, Introduction to Machine Learning | Regresi harga rumah, klasifikasi spam, clustering K-means, metrik evaluasi |
| `05_imk_visualisasi.py` | resume 5, Modul IMK dan Visualisasi Data | Cek kontras WCAG, evaluasi heuristik, beban kognitif, persona, dasbor ASCII interaktif |
| `06_deforestasi.py` | resume 6, Deforestation Alerts dan Driver Classification | NDVI Sentinel-2, backscatter Sentinel-1, deteksi perubahan, klasifikasi driver, confusion matrix, pola musiman |

## Cara menjalankan

```bash
python 01_analytics_ai.py
python 02_hpc.py
python 03_gis_spasial.py
python 04_machine_learning.py
python 05_imk_visualisasi.py
python 06_deforestasi.py
```

Dasbor pada materi 5 punya mode interaktif:

```bash
python 05_imk_visualisasi.py --interaktif
```

Perintah yang tersedia: `ringkas`, `filter`, `detail`, `keluar`.

## Catatan

- Seluruh dataset dibuat secara sintetik di dalam masing-masing script agar hasilnya
  deterministik dan bisa dijalankan tanpa file tambahan.
- Angka dan kejadian pada materi 1 dan 6 adalah simulasi, bukan data nyata.
- Script memakai Python 3.10 atau lebih baru karena memakai anotasi tipe modern.
- Materi 2 memakai `multiprocessing`, sehingga harus dijalankan dengan
  `if __name__ == "__main__"` seperti pada file. Di Windows, proses tambahan akan
  dibuat ulang, jadi wajar bila speedup tidak linear.
