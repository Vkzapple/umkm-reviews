// src/App.jsx
import { useState, useEffect, useCallback } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  LineChart, Line, Area, AreaChart,
} from 'recharts'
import {
  MessageSquare, TrendingUp, ThumbsUp, ThumbsDown,
  Minus, Star, Search, RefreshCw, Sun, Moon,
  Cpu, BarChart2, Smile, Frown, Meh, Zap,
} from 'lucide-react'
import MetricCard from './components/MetricCard'
import SentimentBadge from './components/SentimentBadge'
import { apiService } from './services/api'

// ── Color tokens ────────────────────────────────────────────────────────────
const C = {
  positif: '#10b981',
  negatif: '#ef4444',
  netral:  '#94a3b8',
}
const TOPIC_COLORS = ['#ef4444','#f97316','#3b82f6','#8b5cf6','#94a3b8']

// ── Dark mode hook ──────────────────────────────────────────────────────────
function useDarkMode() {
  const [dark, setDark] = useState(() =>
    document.documentElement.classList.contains('dark') ||
    window.matchMedia('(prefers-color-scheme: dark)').matches
  )
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])
  return [dark, setDark]
}

// ── Recharts dark-friendly tooltip ─────────────────────────────────────────
function ChartTooltip({ active, payload, label, suffix = '' }) {
  if (!active || !payload?.length) return null
  return (
    <div className="card p-3 text-sm shadow-lg min-w-28">
      {label && <div className="font-semibold text-slate-500 mb-1">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full inline-block" style={{ background: p.color }} />
          <span className="text-slate-500 dark:text-slate-400">{p.name}:</span>
          <span className="font-semibold">{p.value?.toLocaleString('id')}{suffix}</span>
        </div>
      ))}
    </div>
  )
}

// ── Topic bar ───────────────────────────────────────────────────────────────
function TopicBar({ name, weight, color, keywords }) {
  return (
    <div className="mb-5">
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">{name}</span>
        <span className="text-sm font-bold" style={{ color }}>{Math.round(weight * 100)}%</span>
      </div>
      <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2.5 mb-2">
        <div
          className="h-2.5 rounded-full transition-all duration-700"
          style={{ width: `${weight * 100}%`, background: color }}
        />
      </div>
      <div className="flex gap-1.5 flex-wrap">
        {keywords.slice(0, 5).map(k => (
          <span key={k} className="text-xs bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 rounded-lg px-2 py-0.5 border border-slate-200 dark:border-slate-600">
            {k}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Word cloud ──────────────────────────────────────────────────────────────
function WordCloud({ keywords, filter }) {
  const all = filter === 'semua'
    ? [
        ...(keywords.positif || []).map(w => ({ ...w, type: 'positif' })),
        ...(keywords.negatif || []).map(w => ({ ...w, type: 'negatif' })),
        ...(keywords.netral  || []).map(w => ({ ...w, type: 'netral'  })),
      ]
    : (keywords[filter] || []).map(w => ({ ...w, type: filter }))

  const maxScore = Math.max(...all.map(w => w.score), 0.001)

  return (
    <div className="flex flex-wrap gap-2.5 min-h-24 py-1">
      {all.map(({ word, score, type }) => {
        const size = 13 + Math.round((score / maxScore) * 10)
        const cls =
          type === 'positif' ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400 dark:border-emerald-700'
        : type === 'negatif' ? 'bg-red-50 text-red-600 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-700'
        : 'bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-700 dark:text-slate-300 dark:border-slate-600'
        return (
          <span
            key={word}
            className={`border rounded-full px-3 py-1 cursor-default hover:scale-105 transition-transform font-medium ${cls}`}
            style={{ fontSize: size }}
          >
            {word}
          </span>
        )
      })}
    </div>
  )
}

// ── Review item ─────────────────────────────────────────────────────────────
function ReviewItem({ review, rating, sentiment, confidence }) {
  const stars = Array.from({ length: 5 }, (_, i) => i < rating ? '★' : '☆').join('')
  return (
    <div className="rounded-xl p-4 bg-slate-50 dark:bg-slate-700/50 border border-slate-100 dark:border-slate-700 hover:border-brand-200 dark:hover:border-brand-700 transition-colors">
      <div className="flex items-center gap-2 mb-2.5 flex-wrap">
        <span className="text-amber-400 tracking-wider text-base">{stars}</span>
        <SentimentBadge sentiment={sentiment} />
        {confidence >= 0.85 && (
          <span className="ml-auto text-xs text-slate-400 bg-slate-100 dark:bg-slate-700 px-2 py-0.5 rounded-lg border border-slate-200 dark:border-slate-600">
            ✓ Terdeteksi otomatis
          </span>
        )}
      </div>
      <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{review}</p>
    </div>
  )
}

// ── Predict panel ───────────────────────────────────────────────────────────
function PredictPanel() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const predict = async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await apiService.predict(text)
      setResult(res)
    } finally {
      setLoading(false)
    }
  }

  const resultLabel = {
    positif: { icon: Smile, text: 'Ulasan ini terdengar positif!', color: 'text-emerald-600' },
    negatif: { icon: Frown, text: 'Ulasan ini terdengar negatif.', color: 'text-red-500' },
    netral:  { icon: Meh,   text: 'Ulasan ini terdengar netral.',  color: 'text-slate-500' },
  }

  return (
    <div className="card p-5">
      <div className="card-header">
        <Zap size={16} className="text-brand-500" />
        Coba Analisis Ulasan
      </div>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
        Ketik ulasan produk, lalu sistem akan otomatis mendeteksi apakah ulasannya positif, negatif, atau netral.
      </p>
      <textarea
        className="w-full border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700
                   rounded-xl p-3.5 text-sm resize-none focus:outline-none focus:ring-2
                   focus:ring-brand-400 focus:border-transparent transition
                   text-slate-700 dark:text-slate-200 placeholder:text-slate-400"
        rows={3}
        placeholder='Contoh: "Barangnya bagus, pengiriman cepat, puas banget!"'
        value={text}
        onChange={e => setText(e.target.value)}
      />
      <button
        className="btn-primary w-full mt-3 flex items-center justify-center gap-2"
        onClick={predict}
        disabled={loading || !text.trim()}
      >
        {loading
          ? <><RefreshCw size={15} className="animate-spin" /> Menganalisis...</>
          : <><Cpu size={15} /> Analisis Sekarang</>
        }
      </button>

      {result && (() => {
        const cfg = resultLabel[result.sentiment] || resultLabel.netral
        const Icon = cfg.icon
        return (
          <div className="mt-4 p-4 bg-slate-50 dark:bg-slate-700/60 rounded-xl border border-slate-100 dark:border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <Icon size={20} className={cfg.color} />
              <span className={`font-semibold text-sm ${cfg.color}`}>{cfg.text}</span>
            </div>
            <div className="space-y-2.5">
              {Object.entries(result.probabilities).map(([key, val]) => (
                <div key={key} className="flex items-center gap-2.5">
                  <span className="text-sm text-slate-500 dark:text-slate-400 w-16 capitalize">{key}</span>
                  <div className="flex-1 bg-slate-200 dark:bg-slate-600 rounded-full h-2">
                    <div
                      className="h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.round(val * 100)}%`, background: C[key] || '#94a3b8' }}
                    />
                  </div>
                  <span className="text-sm font-semibold w-10 text-right tabular-nums">
                    {Math.round(val * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )
      })()}
    </div>
  )
}

// ── Model info ──────────────────────────────────────────────────────────────
function ModelInfo({ accuracy }) {
  const rows = [
    { label: 'Cara kerja', value: 'Klasifikasi otomatis (Random Forest)' },
    { label: 'Pembacaan teks', value: 'TF-IDF — bobot kata penting' },
    { label: 'Pengelompokan topik', value: 'LDA — 5 kelompok keluhan' },
    { label: 'Data latih', value: 'Ulasan dari Taqiyya Ghazi' },
    { label: 'Tingkat akurasi', value: `${accuracy}% dari 100 ulasan uji` },
  ]
  return (
    <div className="card p-5 flex-1">
      <div className="card-header">
        <Cpu size={16} className="text-brand-500" />
        Cara Sistem Bekerja
      </div>
      <div className="space-y-3">
        {rows.map(({ label, value }) => (
          <div key={label} className="flex flex-col gap-0.5">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">{label}</span>
            <span className="text-sm text-slate-700 dark:text-slate-200">{value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [dark, setDark] = useDarkMode()
  const [stats, setStats] = useState(null)
  const [distribution, setDistribution] = useState(null)
  const [trend, setTrend] = useState(null)
  const [topics, setTopics] = useState([])
  const [keywords, setKeywords] = useState({})
  const [reviews, setReviews] = useState({ total: 0, reviews: [] })
  const [ratingDist, setRatingDist] = useState(null)
  const [sentFilter, setSentFilter] = useState('all')
  const [wordFilter, setWordFilter] = useState('semua')
  const [loading, setLoading] = useState(true)

  const loadData = useCallback(async (sentiment = null) => {
    const [s, d, tr, tp, kw, rv, rd] = await Promise.all([
      apiService.getStats(),
      apiService.getDistribution(),
      apiService.getTrend(),
      apiService.getTopics(),
      apiService.getKeywords(),
      apiService.getReviews(sentiment === 'all' ? null : sentiment, 15),
      apiService.getRatingDist(),
    ])
    setStats(s); setDistribution(d); setTrend(tr)
    setTopics(Array.isArray(tp) ? tp : tp.topics || [])
    setKeywords(kw); setReviews(rv); setRatingDist(rd)
    setLoading(false)
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const handleSentFilter = async (f) => {
    setSentFilter(f)
    const rv = await apiService.getReviews(f === 'all' ? null : f, 15)
    setReviews(rv)
  }

  // Chart data
  const pieData = distribution
    ? distribution.labels.map((l, i) => ({
        name: l, value: distribution.values[i], pct: distribution.percentages[i],
      }))
    : []

  const barData = ratingDist
    ? ratingDist.labels.map((l, i) => ({ name: l, value: ratingDist.values[i] }))
    : []

  const trendData = trend
    ? (trend.labels || []).map((l, i) => ({
        month: l,
        positif: trend.positif?.[i] ?? 0,
        negatif: trend.negatif?.[i] ?? 0,
      }))
    : []

  const accuracy = stats ? Math.round(stats.model_accuracy * 100) : 84

  // Axis/grid style helpers
  const axisStyle = { fontSize: 12, fill: dark ? '#94a3b8' : '#94a3b8' }
  const gridColor = dark ? '#334155' : '#f1f5f9'

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-900">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-500 text-sm font-medium">Memuat data ulasan...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-300">

      {/* ── Header ── */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-100 dark:border-slate-700 sticky top-0 z-20 shadow-sm">
        <div className="max-w-7xl mx-auto px-5 md:px-8 h-16 flex items-center justify-between">

          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-brand-600 rounded-xl flex items-center justify-center shadow-sm">
              <BarChart2 size={18} className="text-white" />
            </div>
            <div>
              <h1 className="font-bold text-base text-slate-800 dark:text-white leading-none">
                Analitik Ulasan UMKM
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">Jakarta Barat, DKI Jakarta. </p>
            </div>
          </div>

          {/* Right: live badge + dark toggle */}
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-flex items-center gap-2 text-sm text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-700 px-3 py-1.5 rounded-full font-medium">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              Sistem Aktif
            </span>

            {/* Dark mode toggle */}
            <button
              onClick={() => setDark(d => !d)}
              className="w-10 h-10 rounded-xl flex items-center justify-center border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors"
              aria-label="Toggle mode gelap/terang"
            >
              {dark
                ? <Sun size={18} className="text-amber-400" />
                : <Moon size={18} className="text-slate-500" />
              }
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-5 md:px-8 py-8 space-y-8">

        {/* ── Hero summary ── */}
        <div className="bg-gradient-to-br from-brand-600 to-brand-800 rounded-2xl p-6 md:p-8 text-white">
          <p className="text-brand-200 text-sm font-medium mb-1">Ringkasan Data</p>
          <h2 className="text-2xl md:text-3xl font-bold mb-1">
            {stats?.total?.toLocaleString('id') ?? '—'} Ulasan Dianalisis
          </h2>
          <p className="text-brand-200 text-sm mb-5">
            Dari pelanggan UMKM di marketplace Indonesia
          </p>
          <div className="grid grid-cols-3 gap-4">
            {[
              { icon: ThumbsUp,   label: 'Puas',    val: `${stats?.positif_pct ?? 0}%`,  sub: `${stats?.positif?.toLocaleString('id') ?? 0} ulasan`, color: 'bg-emerald-500/20 text-emerald-200' },
              { icon: ThumbsDown, label: 'Kecewa',  val: `${stats?.negatif_pct ?? 0}%`,  sub: `${stats?.negatif?.toLocaleString('id') ?? 0} ulasan`, color: 'bg-red-400/20 text-red-200' },
              { icon: Minus,      label: 'Biasa',   val: `${stats?.netral_pct  ?? 0}%`,  sub: `${stats?.netral?.toLocaleString('id') ?? 0} ulasan`,  color: 'bg-white/10 text-white/70' },
            ].map(({ icon: Icon, label, val, sub, color }) => (
              <div key={label} className={`rounded-xl p-3 md:p-4 ${color}`}>
                <Icon size={18} className="mb-1 opacity-80" />
                <div className="text-xl md:text-2xl font-bold">{val}</div>
                <div className="text-xs opacity-70 font-medium">{label}</div>
                <div className="text-xs opacity-50 mt-0.5">{sub}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Metric cards row ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Total Ulasan"
            value={stats?.total?.toLocaleString('id') ?? '—'}
            sub="Dikumpulkan dari hasil research Taqiyya Ghazi"
            icon={MessageSquare}
            color="text-brand-600 dark:text-brand-400"
          />
          <MetricCard
            label="Rata-rata Bintang"
            value={`${stats?.avg_rating ?? '—'} ★`}
            sub="dari skala 5 bintang"
            icon={Star}
            color="text-amber-500"
          />
          <MetricCard
            label="Ulasan Positif"
            value={`${stats?.positif_pct ?? 0}%`}
            sub={`${stats?.positif?.toLocaleString('id') ?? 0} ulasan puas`}
            icon={ThumbsUp}
            color="text-emerald-600 dark:text-emerald-400"
          />
          <MetricCard
            label="Ketepatan Sistem"
            value={`${accuracy}%`}
            sub="sistem mendeteksi dengan benar"
            icon={TrendingUp}
            color="text-brand-600 dark:text-brand-400"
          />
        </div>

        {/* ── Charts row 1: Pie + Rating ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

          {/* Donut sentimen */}
          <div className="card p-5">
            <div className="card-header">
              <MessageSquare size={15} className="text-brand-500" />
              Sebaran Perasaan Pelanggan
            </div>
            <div className="flex gap-4 mb-4 flex-wrap">
              {pieData.map(d => (
                <span key={d.name} className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                  <span className="w-3 h-3 rounded-full" style={{ background: C[d.name.toLowerCase()] }} />
                  {d.name} <strong className="text-slate-700 dark:text-slate-200">{d.pct}%</strong>
                </span>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%" cy="50%"
                  innerRadius={60} outerRadius={95}
                  dataKey="value" paddingAngle={3}
                  label={({ name, pct }) => `${name} ${pct}%`}
                  labelLine={false}
                >
                  {pieData.map(entry => (
                    <Cell key={entry.name} fill={C[entry.name.toLowerCase()] || '#94a3b8'} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Rating distribution */}
          <div className="card p-5">
            <div className="card-header">
              <Star size={15} className="text-amber-400" />
              Berapa Bintang yang Diberikan?
            </div>
            <p className="text-sm text-slate-400 mb-4">
              Mayoritas pelanggan memberi <strong className="text-slate-700 dark:text-slate-200">5 bintang</strong>
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} barSize={34}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis dataKey="name" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="value" name="Ulasan" radius={[6, 6, 0, 0]}>
                  {barData.map((_, i) => (
                    <Cell key={i} fill={['#fca5a5','#fcd34d','#cbd5e1','#86efac','#34d399'][i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ── Charts row 2: Topics + Trend ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

          {/* Topic complaints */}
          <div className="card p-5">
            <div className="card-header">
              <Frown size={15} className="text-red-400" />
              Keluhan Paling Sering Muncul
            </div>
            <p className="text-sm text-slate-400 mb-4">
              Topik yang paling banyak dikeluhkan pelanggan
            </p>
            {topics.map((t, i) => (
              <TopicBar
                key={t.id}
                name={t.name}
                weight={t.weight}
                color={TOPIC_COLORS[i % TOPIC_COLORS.length]}
                keywords={t.keywords}
              />
            ))}
          </div>

          {/* Trend chart */}
          <div className="card p-5">
            <div className="card-header">
              <TrendingUp size={15} className="text-brand-500" />
              Perubahan Ulasan dari Bulan ke Bulan
            </div>
            <div className="flex gap-5 mb-4">
              <span className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                <span className="w-4 h-1 bg-emerald-500 rounded inline-block" />
                Ulasan puas
              </span>
              <span className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                <span className="w-4 h-1 bg-red-400 rounded inline-block" style={{ borderStyle: 'dashed' }} />
                Ulasan kecewa
              </span>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="gradPos" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradNeg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis dataKey="month" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={[0, 85]} unit="%" />
                <Tooltip content={props => <ChartTooltip {...props} suffix="%" />} />
                <Area
                  type="monotone" dataKey="positif" name="Puas"
                  stroke="#10b981" strokeWidth={2.5}
                  fill="url(#gradPos)"
                  dot={{ r: 4, fill: '#10b981', strokeWidth: 0 }}
                  activeDot={{ r: 6 }}
                />
                <Area
                  type="monotone" dataKey="negatif" name="Kecewa"
                  stroke="#ef4444" strokeWidth={2} strokeDasharray="5 4"
                  fill="url(#gradNeg)"
                  dot={{ r: 4, fill: '#ef4444', strokeWidth: 0 }}
                  activeDot={{ r: 6 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ── Keywords ── */}
        <div className="card p-5">
          <div className="card-header">
            <Search size={15} className="text-brand-500" />
            Kata-kata yang Paling Sering Muncul
          </div>
          <p className="text-sm text-slate-400 mb-4">
            Kata yang ukurannya lebih besar = lebih sering disebutkan pelanggan
          </p>
          <div className="flex gap-2 mb-5 flex-wrap">
            {['semua','positif','negatif','netral'].map(f => (
              <button
                key={f}
                className={`pill ${wordFilter === f ? 'pill-active' : 'pill-inactive'}`}
                onClick={() => setWordFilter(f)}
              >
                {{ semua: 'Semua', positif: '😊 Positif', negatif: '😞 Negatif', netral: '😐 Netral' }[f]}
              </button>
            ))}
          </div>
          <WordCloud keywords={keywords} filter={wordFilter} />
        </div>

        {/* ── Reviews + Predict ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

          {/* Review list */}
          <div className="card p-5">
            <div className="card-header">
              <MessageSquare size={15} className="text-brand-500" />
              Contoh Ulasan Pelanggan
            </div>
            <div className="flex gap-2 mb-4 flex-wrap">
              {[
                ['all',     'Semua'],
                ['positif', '😊 Puas'],
                ['negatif', '😞 Kecewa'],
                ['netral',  '😐 Biasa'],
              ].map(([v, l]) => (
                <button
                  key={v}
                  className={`pill ${sentFilter === v ? 'pill-active' : 'pill-inactive'}`}
                  onClick={() => handleSentFilter(v)}
                >
                  {l}
                </button>
              ))}
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
              {(reviews.reviews || []).map((r, i) => (
                <ReviewItem key={i} {...r} />
              ))}
            </div>
          </div>

          {/* Predict + model info */}
          <div className="flex flex-col gap-5">
            <PredictPanel />
            <ModelInfo accuracy={accuracy} />
          </div>
        </div>

      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-slate-100 dark:border-slate-700 bg-white dark:bg-slate-800 mt-8 py-5">
        <div className="max-w-7xl mx-auto px-5 md:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-sm text-slate-400">
          <span>Dashboard Analitik Ulasan UMKM Jakarta · Karya Tulis Ilmiah</span>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-emerald-400 rounded-full" />
            <span>Semua data teranalisis otomatis</span>
          </div>
        </div>
      </footer>
    </div>
  )
}