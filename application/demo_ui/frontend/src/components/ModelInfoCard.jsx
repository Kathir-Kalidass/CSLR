import { motion } from 'framer-motion'
import { Info, Cpu, Database, Zap, Server } from 'lucide-react'

export default function ModelInfoCard({ modelInfo, systemInfo }) {
  if (!modelInfo && !systemInfo) return null

  const items = []
  if (modelInfo) {
    if (modelInfo.name) items.push({ icon: <Cpu size={12} />, label: 'Model', value: modelInfo.name })
    if (modelInfo.path) items.push({ icon: <Database size={12} />, label: 'Checkpoint', value: modelInfo.path.split('/').pop() })
    if (modelInfo.size_mb) items.push({ icon: <Server size={12} />, label: 'Size', value: `${modelInfo.size_mb} MB` })
  }
  if (systemInfo) {
    if (systemInfo.runtime_status) items.push({ icon: <Zap size={12} />, label: 'Runtime', value: systemInfo.runtime_status })
    if (systemInfo.tts_engine) items.push({ icon: <Info size={12} />, label: 'TTS', value: systemInfo.tts_engine })
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
      className="glass rounded-xl p-4 glow-border"
    >
      <h4 className="text-xs font-semibold text-dark-200 mb-3 flex items-center gap-1.5">
        <Info size={13} className="text-accent-blue" />
        System & Model Info
      </h4>
      <div className="grid grid-cols-1 gap-2">
        {items.map(({ icon, label, value }, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-dark-400">{icon}</span>
            <span className="text-[10px] text-dark-400 w-16">{label}</span>
            <span className="text-[11px] text-dark-200 font-mono truncate">{value}</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}
