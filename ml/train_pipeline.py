"""
train_pipeline.py
Pipeline lengkap: preprocessing → TF-IDF → Random Forest → LDA Topic Modeling
Jalankan: python ml/train_pipeline.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.decomposition import LatentDirichletAllocation

# Tambahkan root project ke path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing import load_and_preprocess

# ─── Konfigurasi ────────────────────────────────────────────────────────────
CONFIG = {
    # Dataset
    "data_path": ROOT / "data" / "raw" / "reviews.csv",
    "text_col": "ulasan",
    "rating_col": "rating",

    # TF-IDF
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),
    "tfidf_min_df": 3,
    "tfidf_max_df": 0.90,

    # Random Forest
    "rf_n_estimators": 200,
    "rf_max_depth": 20,
    "rf_min_samples_split": 5,
    "rf_random_state": 42,

    # LDA
    "lda_n_topics": 5,
    "lda_n_top_words": 10,
    "lda_max_iter": 20,

    # Output
    "models_dir": ROOT / "ml" / "models",
    "results_dir": ROOT / "ml" / "results",
}

# ─── Utilitas ───────────────────────────────────────────────────────────────

def ensure_dirs():
    CONFIG["models_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)


def generate_sample_data():
    """Buat dataset sampel jika file asli belum tersedia."""
    sample_reviews = [
        ("Barang cepat sampai, kualitas bagus sesuai gambar, penjual ramah", 5),
        ("Pengiriman sangat lama, lebih dari 2 minggu baru sampai", 1),
        ("Produk oke tapi packaging kurang rapi, sedikit penyok", 3),
        ("Kualitas produk tidak sesuai foto, beda banget. Minta refund", 1),
        ("Mantap banget! Harga terjangkau, kualitas premium", 5),
        ("CS tidak responsif, chat diabaikan 3 hari", 2),
        ("Produk sesuai deskripsi, pengiriman kilat. Puas banget", 5),
        ("Warna tidak sesuai pesanan, harusnya biru tapi hijau", 2),
        ("Harga murah kualitas lumayan, worth it untuk harganya", 4),
        ("Sudah belanja beberapa kali, selalu memuaskan", 5),
        ("Produk rusak saat tiba, bubble wrap tidak cukup", 1),
        ("Oke sih, biasa aja. Tidak ada yang spesial", 3),
        ("Barang original, cepat sampai, recommended seller!", 5),
        ("Ukuran tidak sesuai, lebih kecil dari yang tertera di foto", 2),
        ("Penjual fast response, barang dikemas dengan aman", 5),
        ("Kecewa bgt, barang palsu. Tolong tindak penjualnya", 1),
        ("Pengiriman standar tapi barang selamat. Overall bagus", 4),
        ("Barang sudah seminggu belum dikirim, mana pesanan saya", 1),
        ("Sangat puas! Melebihi ekspektasi, bahan berkualitas", 5),
        ("Barang tidak sesuai deskripsi, foto produk menyesatkan", 2),
        ("Oke lah, lumayan untuk harganya. Tidak terlalu bagus", 3),
        ("Respon seller cepat, barang sesuai pesanan. Recommended!", 5),
        ("Lama banget sampainya, sudah 3 minggu belum datang", 1),
        ("Kualitas bagus, harga terjangkau. Beli lagi next time", 4),
        ("Barang tidak ada di paket, sudah lapor tapi tidak ada respon", 1),
        ("Sesuai ekspektasi, tidak mengecewakan. Cukup puas", 4),
        ("Seller tidak jujur, kondisi barang jauh dari deskripsi", 2),
        ("Pengiriman super cepat 1 hari sampai. Barang aman terbungkus rapi", 5),
        ("Harga wajar, kualitas standart. Boleh lah dicoba", 3),
        ("Buruk sekali! Barang datang sudah rusak dan tidak bisa dipakai", 1),
    ] * 50  # repeat untuk simulasi dataset lebih besar

    df = pd.DataFrame(sample_reviews, columns=["review", "rating"])
    # Tambah variasi tanggal
    dates = pd.date_range("2024-06-01", periods=len(df), freq="H")
    df["date"] = dates.strftime("%Y-%m-%d")
    return df


# ─── Step 1: Load & Preprocessing ───────────────────────────────────────────

def step_preprocessing():
    print("\n" + "="*60)
    print("EDA N PREPROCESSING")
    print("="*60)

    data_path = CONFIG["data_path"]

    if not data_path.exists():
        print(f"[WARNING] Dataset tidak ditemukan di {data_path}")
        print("[INFO] Menggunakan dataset sampel untuk demo...")
        df_raw = generate_sample_data()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(data_path, index=False)
        print(f"[INFO] Dataset sampel disimpan ke {data_path}")

    # Load dataset asli
    df_raw = pd.read_csv(data_path)

    # Rename kolom dataset Kaggle → format pipeline
    df_raw = df_raw.rename(columns={
    "reviews": "review",
    "label": "sentiment"
})

    # Simpan sementara agar preprocessing tetap kompatibel
    temp_path = ROOT / "data" / "processed" / "temp_reviews.csv"
    df_raw.to_csv(temp_path, index=False)

    # Preprocessing
    df = load_and_preprocess(
    str(temp_path),
    text_col="review",
    rating_col=None,
    sentiment_col="sentiment",
    use_stemming=False
)

    # Jika sentiment belum ada, mapping dari rating
    if "sentiment" not in df.columns:
        def map_sentiment(rating):
            if rating >= 4:
                return "positif"
            elif rating == 3:
                return "netral"
            else:
                return "negatif"

        df["sentiment"] = df["rating"].apply(map_sentiment)

    # Simpan hasil preprocessing
    out_path = ROOT / "data" / "processed" / "reviews_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"[INFO] Data bersih disimpan ke: {out_path}")

    return df


# ─── Step 2: TF-IDF Vectorization ───────────────────────────────────────────

def step_tfidf(df: pd.DataFrame):
    print("\n" + "="*60)
    print("TF-IDF VECTORIZATION")
    print("="*60)

    vectorizer = TfidfVectorizer(
        max_features=CONFIG["tfidf_max_features"],
        ngram_range=CONFIG["tfidf_ngram_range"],
        min_df=CONFIG["tfidf_min_df"],
        max_df=CONFIG["tfidf_max_df"],
        sublinear_tf=True,  
    )

    X = vectorizer.fit_transform(df["clean_text"])
    print(f"[INFO] Shape matrix TF-IDF: {X.shape}")
    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Simpan vectorizer
    joblib.dump(vectorizer, CONFIG["models_dir"] / "tfidf_vectorizer.pkl")
    print(f"[INFO] TF-IDF vectorizer disimpan.")

    # Simpan top keywords per sentimen
    feature_names = vectorizer.get_feature_names_out()
    keywords_by_sentiment = {}
    for label in ["positif", "negatif", "netral"]:
        mask = (df["sentiment"] == label).to_numpy()
        if mask.sum() == 0:
            continue
        mean_tfidf = X[mask].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[-20:][::-1]
        keywords_by_sentiment[label] = [
            {"word": feature_names[i], "score": round(float(mean_tfidf[i]), 4)}
            for i in top_idx
        ]

    with open(CONFIG["results_dir"] / "keywords.json", "w", encoding="utf-8") as f:
        json.dump(keywords_by_sentiment, f, ensure_ascii=False, indent=2)
    print("[INFO] Kata kunci TF-IDF disimpan ke results/keywords.json")

    return X, vectorizer


# ─── Step 3: Random Forest Training ─────────────────────────────────────────

def step_random_forest(df: pd.DataFrame, X):
    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING")
    print("="*60)

    y = df["sentiment"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Training
    print("[INFO] Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        min_samples_split=CONFIG["rf_min_samples_split"],
        random_state=CONFIG["rf_random_state"],
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Evaluasi
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[INFO] Cross-Val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Simpan model
    joblib.dump(clf, CONFIG["models_dir"] / "random_forest.pkl")
    print("[INFO] Model Random Forest disimpan.")

    # Simpan metrics
    metrics = {
        "accuracy": round(acc, 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "classes": list(clf.classes_),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    with open(CONFIG["results_dir"] / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return clf, metrics


# ─── Step 4: LDA Topic Modeling ──────────────────────────────────────────────

def step_lda(df: pd.DataFrame):
    print("\n" + "="*60)
    print("LDA TOPIC MODELING")
    print("="*60)

    # Gunakan vectorizer khusus LDA (count-based)
    from sklearn.feature_extraction.text import CountVectorizer

    # Fokus pada ulasan negatif untuk keluhan
    df_neg = df[df["sentiment"].isin(["negatif", "netral"])].copy()
    if len(df_neg) < 10:
        df_neg = df.copy()

    count_vec = CountVectorizer(
        max_features=3000,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.85,
    )
    dtm = count_vec.fit_transform(df_neg["clean_text"])
    feature_names = count_vec.get_feature_names_out()

    # LDA
    print(f"[INFO] Training LDA dengan {CONFIG['lda_n_topics']} topik...")
    lda = LatentDirichletAllocation(
        n_components=CONFIG["lda_n_topics"],
        max_iter=CONFIG["lda_max_iter"],
        learning_method="online",
        random_state=42,
        n_jobs=-1,
    )
    lda.fit(dtm)

    # Ekstrak kata kunci per topik
    topic_names = [
        "Keterlambatan Pengiriman",
        "Kualitas Tidak Sesuai",
        "Respons CS Lambat",
        "Packaging Rusak",
        "Barang Tidak Lengkap",
    ]

    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-CONFIG["lda_n_top_words"]:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            "id": idx,
            "name": topic_names[idx] if idx < len(topic_names) else f"Topik {idx+1}",
            "keywords": top_words,
            "weight": round(float(topic.sum() / lda.components_.sum()), 4),
        })
        print(f"  Topik {idx+1} ({topic_names[idx] if idx < len(topic_names) else ''}): {', '.join(top_words[:6])}")

    # Simpan hasil
    with open(CONFIG["results_dir"] / "topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    joblib.dump(lda, CONFIG["models_dir"] / "lda_model.pkl")
    joblib.dump(count_vec, CONFIG["models_dir"] / "count_vectorizer.pkl")
    print("\n[INFO] LDA model dan topik disimpan.")
    return topics


# ─── Step 5: Generate Analytic Results ───────────────────────────────────────

def step_generate_analytics(df: pd.DataFrame, clf, vectorizer):
    print("\n" + "="*60)
    print("GENERATE ANALYTICS DATA")
    print("="*60)

    # Distribusi sentimen
    sent_dist = df["sentiment"].value_counts()
    total = len(df)

    sentiment_data = {
        "total": total,
        "positif": int(sent_dist.get("positif", 0)),
        "negatif": int(sent_dist.get("negatif", 0)),
        "netral": int(sent_dist.get("netral", 0)),
        "positif_pct": round(sent_dist.get("positif", 0) / total * 100, 1),
        "negatif_pct": round(sent_dist.get("negatif", 0) / total * 100, 1),
        "netral_pct": round(sent_dist.get("netral", 0) / total * 100, 1),
        "avg_rating": round(float(df["rating"].mean()), 2),
    }

    # Rating distribution
    rating_dist = df["rating"].value_counts().sort_index()
    sentiment_data["rating_distribution"] = {
        str(k): int(v) for k, v in rating_dist.items()
    }

    # Tren per bulan (jika kolom date ada)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly = df.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
        monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
        sentiment_data["trend"] = {
            "labels": list(monthly_pct.index),
            "positif": [round(v, 1) for v in monthly_pct.get("positif", [0]*len(monthly_pct))],
            "negatif": [round(v, 1) for v in monthly_pct.get("negatif", [0]*len(monthly_pct))],
        }

    with open(CONFIG["results_dir"] / "analytics.json", "w", encoding="utf-8") as f:
        json.dump(sentiment_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Analytics: {sentiment_data['positif_pct']}% positif, "
          f"{sentiment_data['negatif_pct']}% negatif, "
          f"{sentiment_data['netral_pct']}% netral")
    print("[INFO] Analytics data disimpan ke results/analytics.json")
    return sentiment_data


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("🚀 UMKM Dashboard — ML Training Pipeline")
    print("Dataset: Shopee Review Indonesian")
    ensure_dirs()

    # Step 1: Preprocessing
    df = step_preprocessing()

    # Step 2: TF-IDF
    X, vectorizer = step_tfidf(df)

    # Step 3: Random Forest
    clf, metrics = step_random_forest(df, X)

    # Step 4: LDA Topic Modeling
    topics = step_lda(df)

    # Step 5: Generate analytics
    analytics = step_generate_analytics(df, clf, vectorizer)

    print("\n" + "="*60)
    print("HASIL TRAINING")
    print("="*60)
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  CV Score : {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%")
    print(f"  Dataset  : {analytics['total']} ulasan")
    print(f"  Models   : {CONFIG['models_dir']}")
    print(f"  Results  : {CONFIG['results_dir']}")
    print("\nJalankan backend: cd backend && uvicorn main:app --reload")


if __name__ == "__main__":
    main()
