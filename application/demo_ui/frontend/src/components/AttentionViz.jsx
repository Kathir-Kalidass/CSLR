import { motion } from 'framer-motion'

export default function AttentionViz({ rgb = 0.5, pose = 0.5 }) {
  const total = rgb + pose || 1
  const rgbPct = (rgb / total) * 100
  const posePct = (pose / total) * 100

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass rounded-xl p-4 glow-border"
    >
      <h4 className="text-xs font-semibold text-dark-200 mb-3">Attention Fusion</h4>

      {/* Visual bars */}
      <div className="space-y-3">
        {/* RGB */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-medium text-blue-400 flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-blue-500" />
              RGB Stream
            </span>
            <span className="text-[10px] font-mono text-blue-400">{rgbPct.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full"
              animate={{ width: `${rgbPct}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* Pose */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-medium text-emerald-400 flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-emerald-500" />
              Pose Stream
            </span>
            <span className="text-[10px] font-mono text-emerald-400">{posePct.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full"
              animate={{ width: `${posePct}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      </div>

      {/* Fusion graphic */}
      <div className="mt-3 flex items-center justify-center gap-2">
        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 2, repeat: Infinity, delay: 0 }}
          className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500/30 flex items-center justify-center"
        >
          <span className="text-[10px] font-bold text-blue-400">R</span>
        </motion.div>

        <motion.div
          animate={{ width: [20, 30, 20] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="h-0.5 bg-gradient-to-r from-blue-500/50 via-accent-cyan/80 to-emerald-500/50 rounded"
        />

        <motion.div
          animate={{ scale: [0.9, 1.15, 0.9], boxShadow: ['0 0 10px rgba(34,211,238,0.1)', '0 0 20px rgba(34,211,238,0.3)', '0 0 10px rgba(34,211,238,0.1)'] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-10 h-10 rounded-full bg-accent-cyan/15 border border-accent-cyan/30 flex items-center justify-center"
        >
          <span className="text-[10px] font-bold text-accent-cyan">⊕</span>
        </motion.div>

        <motion.div
          animate={{ width: [20, 30, 20] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
          className="h-0.5 bg-gradient-to-r from-emerald-500/50 via-accent-cyan/80 to-blue-500/50 rounded"
        />

        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 2, repeat: Infinity, delay: 0.3 }}
          className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center"
        >
          <span className="text-[10px] font-bold text-emerald-400">P</span>
        </motion.div>
      </div>
    </motion.div>
  )
}
