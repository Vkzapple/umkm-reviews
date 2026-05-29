// src/components/MetricCard.jsx
export default function MetricCard({ label, value, sub, icon: Icon, color = '' }) {
  return (
    <div className="metric-card flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-400">{label}</span>
        {Icon && (
          <div className="w-9 h-9 rounded-xl bg-slate-50 dark:bg-slate-700 flex items-center justify-center">
            <Icon size={18} className={color || 'text-brand-500'} />
          </div>
        )}
      </div>
      <div>
        <div className={`text-3xl font-bold tracking-tight ${color || 'text-slate-800 dark:text-white'}`}>
          {value}
        </div>
        {sub && (
          <div className="text-sm text-slate-400 mt-1">{sub}</div>
        )}
      </div>
    </div>
  )
}