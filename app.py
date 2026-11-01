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
MODEL_PATH = BASE_DIR / "model" / "linear_regression_emas.pkl"
PREDICTED_DATA_PATH = BASE_DIR / "data" / "rupiah_predicted.csv"
GOLD_HISTORY_PATH = BASE_DIR / "data" / "gold-monthly-processed.csv"
START_YEAR = 2022
MIN_PREDICTION_YEAR = 2026
MAX_PREDICTION_YEAR = 2030
TROY_OUNCE_IN_GRAMS = 31.1034768

app = Flask(__name__)

def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
data = pd.read_csv(PREDICTED_DATA_PATH)
data["full_text"] = data["full_text"].fillna("").astype(str)
data["sentiment"] = data["sentiment"].fillna("unknown").astype(str).str.lower()

gold_history_data = pd.read_csv(GOLD_HISTORY_PATH)
gold_history_data["Date"] = pd.to_datetime(
    gold_history_data["Date"], errors="coerce"
)
gold_history_data = gold_history_data[
    gold_history_data['Date'].dt.year >= 2000
]
gold_history_data["Price"] = pd.to_numeric(
    gold_history_data["Price"], errors="coerce"
)
gold_history_data = (
    gold_history_data.dropna(subset=["Date", "Price"])
    .sort_values("Date")
    .reset_index(drop=True)
)

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan",
    "pada", "akan", "dalam", "atau", "juga", "karena", "ada", "tidak",
    "lebih", "sudah", "sebagai", "oleh", "sebuah", "para", "dapat", "bisa",
    "saat", "seperti", "terhadap", "hingga", "agar", "mereka",
}

def normalize_text(value):
    value = value.lower()
    value = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", value)
    words = [word for word in value.split() if word not in STOPWORDS_ID]
    return " ".join(words)

def get_k(requested_k):
    try:
        k = int(requested_k)
    except (TypeError, ValueError):
        k = 3
    max_k = min(8, len(data) - 1)
    return max(2, min(k, max_k))

@lru_cache(maxsize=7)
def build_clusters(k):
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
    for index, (x_value, y_value, cluster) in enumerate(
        zip(coordinates[:, 0], coordinates[:, 1], labels)
    ):
        row = data.iloc[index]
        text = row["full_text"].replace("\n", " ").strip()
        points.append(
            {
                "id": index + 1,
                "x": round(float(x_value), 6),
                "y": round(float(y_value), 6),
                "cluster": int(cluster),
                "sentiment": row["sentiment"],
                "text": text[:180] + ("..." if len(text) > 180 else ""),
            }
        )

    counts = pd.Series(labels).value_counts().sort_index()
    return {
        "points": points,
        "cluster_counts": {
            str(int(key)): int(value) for key, value in counts.items()
        },
        "keywords": keywords,
        "silhouette_score": score,
        "k": k,
    }

def filtered_points(cluster_result, sentiment):
    if not sentiment or sentiment == "all":
        return cluster_result["points"]
    return [
        point
        for point in cluster_result["points"]
        if point["sentiment"] == sentiment
    ]

@app.route("/")
def dashboard(): return render_template("dashboard.html")


@app.get("/api/gold-history")
def gold_history():
    return jsonify(
        dates=gold_history_data["Date"].dt.strftime("%Y-%m-%d").tolist(),
        prices=gold_history_data["Price"].round(2).tolist(),
    )


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
        return jsonify({
            "error": (
                f"Tahun harus berada pada {MIN_PREDICTION_YEAR}-"
                f"{MAX_PREDICTION_YEAR}."
            )
        }), 400
    if not 1 <= month <= 12:
        return jsonify({"error": "Bulan harus berada pada 1-12."}), 400

    period = (year - START_YEAR) * 12 + month

    try:
        prediction = np.asarray(
            model.predict(np.array([[period]], dtype=float))
        ).ravel()
        prediction_value = float(prediction[0])
    except Exception:  # pragma: no cover - pesan aman untuk pengguna
        app.logger.exception("Gagal menjalankan model")
        return jsonify({
            "error": (
                "Model tidak dapat digunakan. "
                "Periksa kompatibilitas model dan dependensi."
            )
        }), 500

    return jsonify(
        year=year,
        month=month,
        period=period,
        prediction=round(prediction_value, 6),
    )

@app.get("/api/gram-convert")
def gram_convert():
    prediction_value = request.args.get("prediction", type=float)
    if prediction_value is None or not np.isfinite(prediction_value):
        return jsonify({"error": "prediction harus berupa angka yang valid."}), 400
    if prediction_value < 0:
        return jsonify({"error": "prediction tidak boleh bernilai negatif."}), 400

    price_per_gram = prediction_value / TROY_OUNCE_IN_GRAMS
    return jsonify(
        prediction_per_troy_ounce=round(prediction_value, 2),
        troy_ounce_in_grams=TROY_OUNCE_IN_GRAMS,
        price_per_gram=round(price_per_gram, 2),
    )

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
