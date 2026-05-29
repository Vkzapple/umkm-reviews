# 📊 Dashboard Analitik Ulasan UMKM Jakarta

Sistem analisis sentimen ulasan produk UMKM Jakarta pada marketplace Indonesia menggunakan Machine Learning.

## 🏗️ Arsitektur Sistem

```
umkm-dashboard/
├── backend/          # FastAPI — REST API & ML inference
├── frontend/         # React + Vite — Dashboard interaktif
├── ml/               # Pipeline ML: preprocessing, training, topic modeling
├── data/             # Dataset & hasil preprocessing
├── notebooks/        # Jupyter notebook eksplorasi & training
└── README.md
```

## ⚙️ Tech Stack

| Layer | Teknologi |
|---|---|
| ML Pipeline | scikit-learn, TF-IDF, Random Forest, LDA |
| Backend | FastAPI, Uvicorn, Pandas, NLTK |
| Frontend | React 18, Vite, Recharts, TailwindCSS |
| Dataset | Shopee Review Indonesian (Kaggle) |

## 🚀 Cara Menjalankan

### 1. Persiapan Dataset

Download dataset dari Kaggle:
```
https://www.kaggle.com/datasets/herafajrin/shopee-review-indonesian
```
Letakkan file CSV di folder `data/raw/`.

### 2. Setup Backend (Python)

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

### 3. Jalankan Training Pipeline

```bash
cd ml
python train_pipeline.py
```

Output model tersimpan di `ml/models/`.

### 4. Jalankan Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 5. Setup & Jalankan Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173

## 📡 API Endpoints

| Method | Endpoint | Deskripsi |
|---|---|---|
| GET | `/api/stats` | Statistik keseluruhan dataset |
| GET | `/api/sentiment/distribution` | Distribusi sentimen |
| GET | `/api/topics` | Topik keluhan (LDA) |
| GET | `/api/keywords` | Kata kunci dominan (TF-IDF) |
| GET | `/api/reviews` | List ulasan terklasifikasi |
| GET | `/api/trend` | Tren sentimen per bulan |
| POST | `/api/predict` | Prediksi sentimen teks baru |

## 🧠 ML Pipeline

```
Raw Reviews → Text Cleaning → TF-IDF Vectorizer → Random Forest Classifier
                                                ↓
                                    LDA Topic Modeling
                                                ↓
                                    Dashboard Analytics
```

### Preprocessing Steps
1. Lowercase
2. Hapus karakter non-alfanumerik
3. Tokenisasi
4. Hapus stopwords bahasa Indonesia
5. Stemming (Sastrawi)

### Model Performance
- Akurasi Random Forest: ~84%
- Jumlah estimator: 200
- Max depth: 20
- TF-IDF max features: 5000
