import { motion } from 'framer-motion'

const stageColors = {
  module1: { bg: 'bg-blue-500/15', border: 'border-blue-500/30', text: 'text-blue-400', glow: 'rgba(59,130,246,0.2)' },
  module2: { bg: 'bg-purple-500/15', border: 'border-purple-500/30', text: 'text-purple-400', glow: 'rgba(168,85,247,0.2)' },
  module3: { bg: 'bg-cyan-500/15', border: 'border-cyan-500/30', text: 'text-cyan-400', glow: 'rgba(34,211,238,0.2)' },
  module4: { bg: 'bg-amber-500/15', border: 'border-amber-500/30', text: 'text-amber-400', glow: 'rgba(251,191,36,0.2)' },
  module5: { bg: 'bg-emerald-500/15', border: 'border-emerald-500/30', text: 'text-emerald-400', glow: 'rgba(52,211,153,0.2)' },
  module6: { bg: 'bg-rose-500/15', border: 'border-rose-500/30', text: 'text-rose-400', glow: 'rgba(251,113,133,0.2)' },
  module7: { bg: 'bg-teal-500/15', border: 'border-teal-500/30', text: 'text-teal-400', glow: 'rgba(45,212,191,0.2)' },
}

const stageIcons = {
  module1: '📷', module2: '🧠', module3: '🔗', module4: '🧹',
  module5: '📝', module6: '🔊', module7: '📊',
}

export default function PipelineStage({ moduleKey, data, isActive, index }) {
  const colors = stageColors[moduleKey] || stageColors.module1
  if (!data) return null

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.08, type: 'spring', stiffness: 200 }}
      className={`relative rounded-xl border ${colors.border} ${colors.bg} p-5 transition-all ${
        isActive ? 'ring-2 ring-offset-2 ring-offset-dark-950 ring-accent-cyan/40' : ''
      }`}
      style={isActive ? { boxShadow: `0 0 25px ${colors.glow}` } : {}}
    >
      {/* Active indicator */}
      {isActive && (
        <div className="absolute -left-1 top-3 bottom-3 w-1 rounded-full bg-accent-cyan" />
      )}

      {/* Header */}
      <div className="flex items-start gap-3 mb-3">
        <span className="text-2xl">{stageIcons[moduleKey] || '⚙️'}</span>
        <div className="flex-1 min-w-0">
          <h4 className={`text-base font-bold ${colors.text}`}>
            {data.title}
          </h4>
          <p className="text-sm text-dark-400 mt-0.5 line-clamp-2">{data.note}</p>
        </div>
        {isActive && (
          <div className="w-3 h-3 bg-accent-cyan rounded-full flex-shrink-0 mt-1.5" />
        )}
      </div>

      {/* IO Flow */}
      <div className="space-y-2.5 mt-4">
        <div className="flex items-start gap-3">
          <span className="text-xs font-bold text-dark-400 w-8 flex-shrink-0 mt-0.5">IN</span>
          <span className="text-sm text-dark-200 font-mono break-all">{data.input}</span>
        </div>
        <div className="flex items-start gap-3">
          <span className="text-xs font-bold text-accent-cyan/70 w-8 flex-shrink-0 mt-0.5">FN</span>
          <span className="text-sm text-dark-300 font-mono break-all">{data.process}</span>
        </div>
        <div className="flex items-start gap-3">
          <span className="text-xs font-bold text-accent-green/70 w-8 flex-shrink-0 mt-0.5">OUT</span>
          <span className="text-sm text-dark-100 font-mono font-semibold break-all">{data.output}</span>
        </div>
      </div>

      {/* Parse logs */}
      {data.parse?.length > 0 && (
        <div className="mt-3 pt-3 border-t border-dark-700/30">
          {data.parse.map((line, i) => (
            <p
              key={i}
              className="text-xs font-mono text-dark-400 leading-relaxed truncate"
            >
              {'> '}{line}
            </p>
          ))}
        </div>
      )}
    </motion.div>
  )
}
