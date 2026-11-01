"""Dashboard Flask untuk prediksi nilai emas dan clustering teks rupiah."""

from functools import lru_cache
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_regression_emas.pkl"
CLEAN_DATA_PATH = BASE_DIR / "rupiah_clean.csv"
PREDICTED_DATA_PATH = BASE_DIR / "rupiah_predicted.csv"
START_YEAR = 2022
MIN_PREDICTION_YEAR = 2026
MAX_PREDICTION_YEAR = 2030

app = Flask(__name__)


def load_model():
    """Muat model tepercaya satu kali saat aplikasi dimulai."""
    return joblib.load(MODEL_PATH)


model = load_model()
data = pd.read_csv(PREDICTED_DATA_PATH)
clean_data = pd.read_csv(CLEAN_DATA_PATH)

data["full_text"] = data["full_text"].fillna("").astype(str)
data["sentiment"] = data["sentiment"].fillna("unknown").astype(str).str.lower()
data["text_length"] = data["full_text"].str.len()
data["word_count"] = data["full_text"].str.split().str.len()


STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan",
    "pada", "akan", "dalam", "atau", "juga", "karena", "ada", "tidak",
    "lebih", "sudah", "sebagai", "oleh", "sebuah", "para", "dapat", "bisa",
    "saat", "seperti", "dengan", "terhadap", "hingga", "agar", "mereka",
}


def normalize_text(value):
    """Normalisasi ringan untuk teks Bahasa Indonesia."""
    value = value.lower()
    value = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", value)
    words = [word for word in value.split() if word not in STOPWORDS_ID]
    return " ".join(words)


def get_k(requested_k):
    """Validasi jumlah cluster dan membatasi nilainya agar aman."""
    try:
        k = int(requested_k)
    except (TypeError, ValueError):
        k = 3
    return max(2, min(k, min(8, len(data) - 1)))


@lru_cache(maxsize=7)
def build_clusters(k):
    """Bangun clustering dan proyeksi dua dimensi untuk nilai k tertentu."""
    texts = data["full_text"].map(normalize_text).tolist()
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = clusterer.fit_predict(matrix)

    reducer = TruncatedSVD(n_components=2, random_state=42)
    coordinates = reducer.fit_transform(matrix)

    score = None
    if len(set(labels)) > 1 and len(data) > k:
        score = round(float(silhouette_score(matrix, labels)), 4)

    feature_names = vectorizer.get_feature_names_out()
    keywords = {}
    for cluster_index, center in enumerate(clusterer.cluster_centers_):
        top_indexes = center.argsort()[::-1][:8]
        keywords[str(cluster_index)] = [feature_names[i] for i in top_indexes]

    points = []
    for index, (x_value, y_value, cluster) in enumerate(zip(
        coordinates[:, 0], coordinates[:, 1], labels
    )):
        text = data.iloc[index]["full_text"].replace("\n", " ").strip()
        points.append({
            "id": index + 1,
            "x": round(float(x_value), 6),
            "y": round(float(y_value), 6),
            "cluster": int(cluster),
            "sentiment": data.iloc[index]["sentiment"],
            "text": text[:180] + ("..." if len(text) > 180 else ""),
        })

    counts = pd.Series(labels).value_counts().sort_index()
    return {
        "points": points,
        "cluster_counts": {str(int(key)): int(value) for key, value in counts.items()},
        "keywords": keywords,
        "silhouette_score": score,
        "k": k,
    }


def filtered_points(cluster_result, sentiment):
    if not sentiment or sentiment == "all":
        return cluster_result["points"]
    return [point for point in cluster_result["points"] if point["sentiment"] == sentiment]


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")


@app.route("/clustering")
def clustering_page():
    return render_template("clustering.html")


@app.get("/api/summary")
def summary():
    sentiment_counts = data["sentiment"].value_counts().to_dict()
    return jsonify({
        "total_records": int(len(data)),
        "clean_records": int(len(clean_data)),
        "sentiment_counts": {str(key): int(value) for key, value in sentiment_counts.items()},
        "average_text_length": round(float(data["text_length"].mean()), 2),
        "average_word_count": round(float(data["word_count"].mean()), 2),
        "duplicate_texts": int(data["full_text"].duplicated().sum()),
    })


@app.get("/api/sentiment-distribution")
def sentiment_distribution():
    order = ["positive", "negative", "neutral"]
    counts = data["sentiment"].value_counts().reindex(order, fill_value=0)
    return jsonify({"labels": counts.index.tolist(), "values": counts.astype(int).tolist()})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or request.form
    raw_year = payload.get("year")
    raw_month = payload.get("month")
    if raw_year is None or raw_month is None:
        return jsonify({"error": "year dan month wajib diisi."}), 400

    try:
        year = int(raw_year)
        month = int(raw_month)
    except (TypeError, ValueError):
        return jsonify({"error": "year dan month harus berupa angka bulat."}), 400

    if not MIN_PREDICTION_YEAR <= year <= MAX_PREDICTION_YEAR:
        return jsonify({"error": f"Tahun harus berada pada {MIN_PREDICTION_YEAR}-{MAX_PREDICTION_YEAR}."}), 400
    if not 1 <= month <= 12:
        return jsonify({"error": "Bulan harus berada pada 1-12."}), 400

    period = (year - START_YEAR) * 12 + month

    try:
        prediction = np.asarray(model.predict(np.array([[period]], dtype=float))).ravel()
        prediction_value = float(prediction[0])
    except Exception as error:  # pragma: no cover - pesan aman untuk pengguna
        app.logger.exception("Gagal menjalankan model")
        return jsonify({"error": "Model tidak dapat digunakan. Periksa kompatibilitas model dan dependensi."}), 500

    return jsonify({
        "year": year,
        "month": month,
        "period": period,
        "prediction": round(prediction_value, 6),
    })

@app.get("/api/gram-convert")
def gram_convert():
    prediction_value = request.args.get("prediction", type=float)
    if prediction_value is None or not np.isfinite(prediction_value):
        return jsonify({"error": "prediction harus berupa angka yang valid."}), 400
    if prediction_value < 0:
        return jsonify({"error": "prediction tidak boleh bernilai negatif."}), 400

    troy_ounce_in_grams = 31.1034768
    price_per_gram = prediction_value / troy_ounce_in_grams
    return jsonify({
        "prediction_per_troy_ounce": round(prediction_value, 2),
        "troy_ounce_in_grams": troy_ounce_in_grams,
        "price_per_gram": round(price_per_gram, 2),
    })

@app.get("/api/clusters")
def clusters():
    k = get_k(request.args.get("k", 3))
    sentiment = request.args.get("sentiment", "all").lower()
    result = build_clusters(k)
    return jsonify({
        **result,
        "points": filtered_points(result, sentiment),
        "sentiment_filter": sentiment,
    })


if __name__ == "__main__":
    app.run(debug=True)
