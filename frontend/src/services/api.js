import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
})

// Fallback data kalau backend belum jalan
const FALLBACK = {
  stats: {
    total: 831, positif: 523, negatif: 216, netral: 92,
    positif_pct: 62.9, negatif_pct: 26.0, netral_pct: 11.1,
    avg_rating: 3.8, model_accuracy: 0.8563,
  },
  distribution: {
    labels: ['Positif', 'Negatif', 'Netral'],
    values: [523, 216, 92],
    percentages: [62.9, 26.0, 11.1],
    total: 831,
  },
  trend: {
    labels: ['2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11'],
    positif: [55.0, 58.0, 61.0, 64.0, 66.0, 62.9],
    negatif: [30.0, 28.0, 26.0, 25.0, 24.0, 26.0],
  },
  topics: [
    { id: 0, name: 'Keterlambatan Pengiriman', keywords: ['lama','lambat','tunggu','belum','kirim','hari','minggu','telat'], weight: 0.34 },
    { id: 1, name: 'Kualitas Tidak Sesuai', keywords: ['foto','gambar','beda','sesuai','asli','palsu','kualitas','warna'], weight: 0.28 },
    { id: 2, name: 'Respons CS Lambat', keywords: ['respon','chat','balas','hubungi','komplain','diabaikan','seller'], weight: 0.19 },
    { id: 3, name: 'Packaging Rusak', keywords: ['bungkus','rusak','penyok','pecah','bubble','wrap','kardus'], weight: 0.12 },
    { id: 4, name: 'Barang Tidak Lengkap', keywords: ['kurang','hilang','kosong','aksesoris','paket','isi'], weight: 0.07 },
  ],
  keywords: {
    positif: [
      { word: 'cepat', score: 0.092 }, { word: 'bagus', score: 0.088 },
      { word: 'sesuai', score: 0.085 }, { word: 'puas', score: 0.076 },
      { word: 'murah', score: 0.065 }, { word: 'mantap', score: 0.058 },
      { word: 'ramah', score: 0.050 }, { word: 'terpercaya', score: 0.040 },
      { word: 'original', score: 0.038 }, { word: 'aman', score: 0.035 },
    ],
    negatif: [
      { word: 'lama', score: 0.080 }, { word: 'rusak', score: 0.072 },
      { word: 'kecewa', score: 0.067 }, { word: 'tidak responsif', score: 0.060 },
      { word: 'lambat', score: 0.048 }, { word: 'refund', score: 0.043 },
      { word: 'mengecewakan', score: 0.040 }, { word: 'palsu', score: 0.038 },
    ],
    netral: [
      { word: 'lumayan', score: 0.055 }, { word: 'standar', score: 0.048 },
      { word: 'biasa', score: 0.045 }, { word: 'oke', score: 0.042 },
    ],
  },
  reviews: {
    total: 12,
    reviews: [
      { review: 'Barang cepat sampai, kualitas bagus sesuai gambar, penjual ramah dan responsif', rating: 5, sentiment: 'positif', confidence: 0.91 },
      { review: 'Pengiriman sangat lama, lebih dari 2 minggu baru sampai. Sangat mengecewakan', rating: 1, sentiment: 'negatif', confidence: 0.88 },
      { review: 'Produk oke tapi packaging kurang rapi, sedikit penyok waktu sampai', rating: 3, sentiment: 'netral', confidence: 0.72 },
      { review: 'Kualitas produk tidak sesuai foto, beda banget sama yang di toko. Minta refund', rating: 1, sentiment: 'negatif', confidence: 0.85 },
      { review: 'Mantap banget! Harga terjangkau, kualitas premium. Bakal beli lagi disini', rating: 5, sentiment: 'positif', confidence: 0.94 },
      { review: 'CS tidak responsif, chat diabaikan 3 hari. Kecewa dengan pelayanannya', rating: 2, sentiment: 'negatif', confidence: 0.82 },
      { review: 'Produk sesuai deskripsi, pengiriman kilat. Puas banget deh sama tokonya', rating: 5, sentiment: 'positif', confidence: 0.93 },
      { review: 'Warna tidak sesuai pesanan, harusnya biru tapi datangnya hijau', rating: 2, sentiment: 'negatif', confidence: 0.79 },
    ],
  },
  rating: {
    labels: ['★1','★2','★3','★4','★5'],
    values: [134, 112, 98, 187, 300],
  },
}

async function fetchWithFallback(endpoint, fallbackKey) {
  try {
    const res = await api.get(endpoint)
    return res.data
  } catch {
    return FALLBACK[fallbackKey]
  }
}

export const apiService = {
  getStats:        () => fetchWithFallback('/stats', 'stats'),
  getDistribution: () => fetchWithFallback('/sentiment/distribution', 'distribution'),
  getTrend:        () => fetchWithFallback('/sentiment/trend', 'trend'),
  getRatingDist:   () => fetchWithFallback('/rating/distribution', 'rating'),

  // Backend returns { topics: [...] }, normalize di sini
  async getTopics() {
    try {
      const res = await api.get('/topics')
      const data = res.data
      return Array.isArray(data) ? data : (data.topics || [])
    } catch {
      return FALLBACK.topics
    }
  },

  // Backend bisa return object penuh { positif, negatif, netral }
  // atau { sentiment, keywords } kalau dengan param
  async getKeywords(sentiment) {
    try {
      const url = sentiment ? `/keywords?sentiment=${sentiment}` : '/keywords'
      const res = await api.get(url)
      const data = res.data
      // Kalau empty object {} atau tidak ada key yg dibutuhkan, pakai fallback
      if (!data || Object.keys(data).length === 0) return FALLBACK.keywords
      // Kalau pakai filter sentiment: { sentiment, keywords: [...] }
      if (sentiment && data.keywords) return { [sentiment]: data.keywords }
      return data
    } catch {
      return FALLBACK.keywords
    }
  },

  getReviews: (sentiment, limit = 20) =>
    fetchWithFallback(
      `/reviews?limit=${limit}${sentiment ? `&sentiment=${sentiment}` : ''}`,
      'reviews'
    ),

  async predict(text) {
    try {
      const res = await api.post('/predict', { text })
      const data = res.data
      // Pastikan probabilities punya 3 keys (netral mungkin 0 dari binary model)
      const probs = {
        positif: data.probabilities?.positif ?? 0,
        negatif: data.probabilities?.negatif ?? 0,
        netral:  data.probabilities?.netral  ?? 0,
      }
      return { ...data, probabilities: probs }
    } catch {
      const words = text.toLowerCase()
      const isNeg = ['kecewa','rusak','lama','jelek','buruk','lambat','mahal','tidak'].some(w => words.includes(w))
      const isPos = ['bagus','cepat','puas','mantap','sesuai','murah','good','suka'].some(w => words.includes(w))
      const sentiment = isNeg ? 'negatif' : isPos ? 'positif' : 'netral'
      const conf = isNeg || isPos ? 0.82 : 0.61
      return {
        text, sentiment, confidence: conf,
        probabilities: {
          positif: isPos ? conf : isNeg ? 0.10 : 0.25,
          negatif: isNeg ? conf : isPos ? 0.10 : 0.25,
          netral:  sentiment === 'netral' ? conf : 0.08,
        },
      }
    }
  },
}