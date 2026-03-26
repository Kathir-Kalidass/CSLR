import { motion, AnimatePresence } from 'framer-motion'
import { Hash } from 'lucide-react'

export default function GlossDisplay({ gloss = '--', confidence = 0 }) {
  const tokens = gloss === '--' ? [] : gloss.split(/\s+/).filter(Boolean)

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-xl p-4 glow-border"
    >
      <div className="flex items-center gap-2 mb-3">
        <Hash size={14} className="text-accent-purple" />
        <span className="text-xs font-semibold text-dark-200">Gloss Tokens</span>
        {confidence > 0 && (
          <span className="ml-auto text-[10px] font-mono text-accent-green">
            conf: {(confidence * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="min-h-[40px] flex flex-wrap gap-1.5">
        <AnimatePresence mode="popLayout">
          {tokens.length > 0 ? (
            tokens.map((token, i) => (
              <motion.span
                key={`${token}-${i}`}
                initial={{ opacity: 0, scale: 0.5, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.5 }}
                transition={{ delay: i * 0.05, type: 'spring', stiffness: 300, damping: 20 }}
                className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-accent-purple/15 text-accent-purple border border-accent-purple/20"
              >
                {token}
              </motion.span>
            ))
          ) : (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.5 }}
              className="text-sm text-dark-400 italic"
            >
              Waiting for sign input...
            </motion.span>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
