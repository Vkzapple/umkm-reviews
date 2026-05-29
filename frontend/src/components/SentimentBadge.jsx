// src/components/SentimentBadge.jsx
const LABEL = {
  positif: '😊 Positif',
  negatif: '😞 Negatif',
  netral:  '😐 Netral',
}

export default function SentimentBadge({ sentiment }) {
  const cls = {
    positif: 'badge-positif',
    negatif: 'badge-negatif',
    netral:  'badge-netral',
  }[sentiment] || 'badge-netral'

  return (
    <span className={`badge ${cls}`}>
      {LABEL[sentiment] || sentiment}
    </span>
  )
}