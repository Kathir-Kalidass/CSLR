import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle2, Loader2 } from 'lucide-react'

const steps = [
  'Initializing Vision Engine',
  'Loading Multi-Feature Attention',
  'Loading Temporal Decoder + CTC',
  'Connecting Metrics + Stream Bus',
  'Model Ready',
]

export default function InitSequence({ sequence = [], show = true }) {
  if (!show) return null
  const displaySteps = sequence.length > 0 ? sequence : steps
  const completed = sequence.length > 0

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="glass rounded-xl p-4 glow-border"
    >
      <h4 className="text-xs font-semibold text-dark-200 mb-3">Boot Sequence</h4>
      <div className="space-y-2">
        <AnimatePresence>
          {displaySteps.map((step, i) => (
            <motion.div
              key={step}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.15 }}
              className="flex items-center gap-2"
            >
              {completed ? (
                <CheckCircle2 size={12} className="text-accent-green flex-shrink-0" />
              ) : (
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
                  <Loader2 size={12} className="text-accent-cyan flex-shrink-0" />
                </motion.div>
              )}
              <span className={`text-[11px] ${completed ? 'text-dark-300' : 'text-accent-cyan'}`}>
                {step}
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
