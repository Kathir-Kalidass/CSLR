import { motion, AnimatePresence } from 'framer-motion'
import { History, X } from 'lucide-react'
import useStore from '../store/useStore'

export default function TranscriptHistory({ history = [] }) {
  const sendControl = useStore((s) => s.sendControl)

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="glass rounded-xl p-4 glow-border"
    >
      <div className="flex items-center gap-2 mb-3">
        <History size={14} className="text-accent-amber" />
        <span className="text-xs font-semibold text-dark-200">Transcript History</span>
        <span className="text-[10px] text-dark-400">({history.length})</span>
        {history.length > 0 && (
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => sendControl('clear')}
            className="ml-auto p-1 rounded-md bg-dark-700 text-dark-400 hover:text-accent-red hover:bg-dark-600 transition-colors"
            title="Clear history"
          >
            <X size={12} />
          </motion.button>
        )}
      </div>

      <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
        <AnimatePresence>
          {history.length > 0 ? (
            history.map((text, i) => (
              <motion.div
                key={`${text}-${i}`}
                initial={{ opacity: 0, x: -15, height: 0 }}
                animate={{ opacity: 1, x: 0, height: 'auto' }}
                exit={{ opacity: 0, x: 15, height: 0 }}
                transition={{ duration: 0.25 }}
                className={`px-2.5 py-1.5 rounded-lg text-xs ${
                  i === 0
                    ? 'bg-accent-cyan/10 text-white border border-accent-cyan/20'
                    : 'bg-dark-800/50 text-dark-300'
                }`}
              >
                <span className="text-[9px] text-dark-500 mr-2">#{history.length - i}</span>
                {text}
              </motion.div>
            ))
          ) : (
            <p className="text-xs text-dark-500 italic text-center py-4">
              No transcript entries yet
            </p>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
