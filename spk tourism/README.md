# 🏝️ SPK Pemilihan Destinasi Wisata Indonesia

Sistem Pendukung Keputusan (SPK) untuk memilih destinasi wisata terbaik di Indonesia menggunakan tiga metode:

## Metode
| Metode | Deskripsi |
|--------|-----------|
| **SMART** | Menentukan bobot kriteria secara terstruktur (normalisasi 0–100) |
| **SAW** | Simple Additive Weighting — normalisasi + weighted sum |
| **TOPSIS** | Pilih alternatif terdekat ke ideal positif & terjauh dari ideal negatif |

## Kriteria
| Kriteria | Tipe | Bobot Default |
|----------|------|---------------|
| Harga Tiket | Cost ↓ | 30% |
| Rating Pengunjung | Benefit ↑ | 35% |
| Jumlah Review | Benefit ↑ | 20% |
| Waktu Kunjungan | Benefit ↑ | 15% |

## Dataset
**Indonesia Tourism Destination** — Kaggle  
https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

## Cara Menjalankan

### Di Kaggle Notebook
1. Upload `spk_tourism.ipynb` ke Kaggle
2. Tambahkan dataset `indonesia-tourism-destination`
3. Jalankan semua cell

### Streamlit App (Lokal)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy ke Streamlit Cloud
1. Push ke GitHub repo
2. Buka https://share.streamlit.io
3. Connect repo → pilih `streamlit_app.py`
4. Deploy!

> **Catatan:** Jika dataset Kaggle tidak tersedia, app otomatis menggunakan sample data realistis.

## Struktur File
```
├── spk_tourism.ipynb     # Notebook Kaggle (lengkap dengan analisis)
├── streamlit_app.py      # Aplikasi web Streamlit
├── requirements.txt      # Dependencies Python
└── README.md
```
