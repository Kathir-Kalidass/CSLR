import { motion } from 'framer-motion'
import { Gauge, Clock, Activity, Target } from 'lucide-react'

export default function MetricsBar({ fps = 0, latency = 0, confidence = 0, status = 'idle' }) {
  const items = [
    { icon: <Activity size={12} />, label: 'FPS', value: fps, color: 'text-accent-cyan' },
    { icon: <Clock size={12} />, label: 'Latency', value: `${latency}ms`, color: 'text-accent-amber' },
    { icon: <Target size={12} />, label: 'Confidence', value: `${(confidence * 100).toFixed(0)}%`, color: 'text-accent-green' },
    { icon: <Gauge size={12} />, label: 'Status', value: status, color: status === 'active' ? 'text-accent-green' : 'text-dark-400' },
  ]

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex items-center justify-center gap-4 flex-wrap"
    >
      {items.map(({ icon, label, value, color }, i) => (
        <motion.div
          key={label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-dark-800/50"
        >
          <span className={color}>{icon}</span>
          <span className="text-[10px] text-dark-400">{label}</span>
          <span className={`text-xs font-bold font-mono ${color}`}>{value}</span>
        </motion.div>
      ))}
    </motion.div>
  )
}
