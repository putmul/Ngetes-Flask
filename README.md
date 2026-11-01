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

Jika muncul peringatan perbedaan versi scikit-learn, gunakan versi yang sama dengan saat model dibuat atau latih ulang model menggunakan environment saat ini.
