"""
preprocessing.py
Modul pembersihan dan normalisasi teks ulasan bahasa Indonesia.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _factory = StemmerFactory()
    stemmer = _factory.create_stemmer()
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("[WARNING] PySastrawi tidak tersedia. Stemming dinonaktifkan.")

# Download resource NLTK jika belum ada
try:
    stopwords.words("indonesian")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

STOPWORDS_ID = set(stopwords.words("indonesian"))

EXTRA_STOPWORDS = {
    "yg", "dgn", "dr", "utk", "tdk", "ga", "gak", "nggak", "udah",
    "udh", "uda", "sdh", "sdh", "bgt", "aja", "sih", "deh", "dong",
    "nya", "nih", "loh", "kah", "pun", "ku", "mu", "kita", "kami",
    "mereka", "dia", "saya", "aku", "kamu", "kau", "tp", "krn", "krna",
    "jd", "jadi", "sdg", "lg", "lagi", "msh", "masih", "jg", "juga",
    "bisa", "ada", "tidak", "dengan", "yang", "untuk", "dari", "dan",
    "atau", "ini", "itu", "di", "ke", "ya", "si", "pak", "bu", "bang",
    "kak", "min", "seller", "buyer", "shop", "toko", "produk", "item"
}

ALL_STOPWORDS = STOPWORDS_ID.union(EXTRA_STOPWORDS)

SLANG_DICT = {
    "bagus": "bagus", "bgs": "bagus", "bgus": "bagus",
    "cepet": "cepat", "cpet": "cepat",
    "jelek": "jelek", "jlek": "jelek",
    "lama": "lama", "lmbt": "lambat", "lambat": "lambat",
    "murah": "murah", "mrh": "murah",
    "mahal": "mahal", "mhal": "mahal",
    "rusak": "rusak", "rsk": "rusak",
    "puas": "puas", "puass": "puas",
    "kecewa": "kecewa", "kcw": "kecewa",
    "baguss": "bagus", "okee": "oke", "okey": "oke",
    "mantap": "mantap", "mantab": "mantap", "mantep": "mantap",
    "sesuai": "sesuai", "ssuai": "sesuai",
    "recommend": "rekomen", "rekomend": "rekomen",
    "good": "bagus", "nice": "bagus", "great": "bagus",
    "bad": "jelek", "worst": "terburuk",
    "terlambat": "terlambat", "telat": "terlambat",
}


def normalize_slang(text: str) -> str:
    """Ganti kata slang dengan padanan baku."""
    tokens = text.split()
    return " ".join(SLANG_DICT.get(t, t) for t in tokens)


def clean_text(text: str, use_stemming: bool = True) -> str:
    """
    Pipeline pembersihan teks lengkap:
    1. Lowercase
    2. Hapus URL, mention, hashtag
    3. Hapus angka dan tanda baca
    4. Normalisasi spasi
    5. Normalisasi slang
    6. Tokenisasi & hapus stopword
    7. Stemming (opsional)
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)

    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = normalize_slang(text)

    tokens = [t for t in text.split() if t not in ALL_STOPWORDS and len(t) > 1]

    # 7. Stemming
    if use_stemming and SASTRAWI_AVAILABLE:
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def rating_to_sentiment(rating) -> str:
    """Konversi rating numerik ke label sentimen."""
    rating = int(rating)
    if rating >= 4:
        return "positif"
    elif rating == 3:
        return "netral"
    else:
        return "negatif"


def load_and_preprocess(filepath: str,
                        text_col: str = "review",
                        rating_col: str = "rating",
                        sentiment_col: str = "sentiment",
                        use_stemming: bool = True) -> pd.DataFrame:
    """
    Load CSV dan preprocessing fleksibel:
    - Bisa dataset rating
    - Bisa dataset sentiment label
    """

    print(f"[INFO] Loading dataset dari: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8")

    print(f"[INFO] Total baris: {len(df)}")

    rename_map = {}

    if text_col in df.columns:
        rename_map[text_col] = "review"

    if rating_col is not None and rating_col in df.columns:
        rename_map[rating_col] = "rating"

    if sentiment_col is not None and sentiment_col in df.columns:
        rename_map[sentiment_col] = "sentiment"

    df = df.rename(columns=rename_map)


    assert "review" in df.columns, f"Kolom '{text_col}' tidak ditemukan."

    assert (
        "rating" in df.columns or "sentiment" in df.columns
    ), "Dataset harus punya kolom rating atau sentiment."

    df = df.dropna(subset=["review"])

    df["review"] = df["review"].astype(str)


    if "rating" in df.columns:

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        df = df.dropna(subset=["rating"])

        if "sentiment" not in df.columns:
            df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    else:
        df["rating"] = None

  
    print("[INFO] Memproses teks...")

    df["clean_text"] = df["review"].apply(
        lambda x: clean_text(x, use_stemming)
    )

    df = df[df["clean_text"].str.strip() != ""]

    print(f"[INFO] Selesai. Baris valid: {len(df)}")

    print(f"[INFO] Distribusi sentiment:")
    print(df["sentiment"].value_counts())

    return df[["review", "rating", "sentiment", "clean_text"]]


if __name__ == "__main__":
    contoh = [
        "Barang cepet banget nyampe, bagus bgt sesuai gambar! Recommend deh",
        "Kecewa bgt, barang gak sesuai foto. Warnanya beda & rusak pula",
        "Yaa lumayan lah, biasa aja",
    ]
    for c in contoh:
        print(f"Original : {c}")
        print(f"Cleaned  : {clean_text(c)}")
        print()
