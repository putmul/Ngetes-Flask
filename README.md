# Modul 6 Flask

Dashboard ini menyediakan:

- ringkasan dan distribusi sentimen dari `rupiah_predicted.csv`;
- prediksi menggunakan `linear_regression_emas.pkl` berdasarkan pilihan tahun dan bulan;
- clustering teks menggunakan TF-IDF, K-Means, dan TruncatedSVD.

## Menjalankan aplikasi

Aktifkan virtual environment, lalu install dependensi:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000`.

Model `.pkl` hanya boleh dimuat jika berasal dari sumber tepercaya. Jika muncul peringatan perbedaan versi scikit-learn, gunakan versi yang sama dengan saat model dibuat atau latih ulang model menggunakan environment saat ini.

## Endpoint utama

- `GET /api/summary`
- `GET /api/sentiment-distribution`
- `POST /api/predict` dengan JSON `{"year": 2026, "month": 8}`;
  fitur model dihitung dengan `period = ((year - 2022) * 12 + month)`;
- `GET /api/gram-convert?prediction=4200.630393` untuk mengonversi harga troy ounce ke harga per gram;
- `GET /api/clusters?k=3&sentiment=all`
