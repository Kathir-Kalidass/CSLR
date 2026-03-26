import { motion } from 'framer-motion'

export default function ConfidenceMeter({ value = 0, label = 'Confidence' }) {
  const pct = Math.min(100, Math.max(0, value * 100))
  const color = pct > 70 ? 'accent-green' : pct > 40 ? 'accent-amber' : 'accent-red'

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium text-dark-300">{label}</span>
        <span className={`text-[10px] font-bold text-${color}`}>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-dark-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full bg-${color} rounded-full`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}
