import { useRef, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function ConsolePanel({ logs = [] }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0
    }
  }, [logs.length])

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="glass rounded-xl overflow-hidden glow-border"
    >
      <div className="flex items-center gap-2 px-3 py-2 border-b border-dark-700/50 bg-dark-900/50">
        <div className="flex gap-1">
          <span className="w-2 h-2 rounded-full bg-accent-red/70" />
          <span className="w-2 h-2 rounded-full bg-accent-amber/70" />
          <span className="w-2 h-2 rounded-full bg-accent-green/70" />
        </div>
        <span className="text-[10px] font-mono text-dark-400">pipeline_console</span>
      </div>

      <div
        ref={scrollRef}
        className="p-3 max-h-40 overflow-y-auto font-mono text-[10px] leading-relaxed space-y-0.5"
      >
        {logs.length > 0 ? (
          logs.map((line, i) => (
            <div key={i} className={`${
              line.includes('[runtime]') ? 'text-accent-cyan' :
              line.includes('[capture]') ? 'text-blue-400' :
              line.includes('[features]') ? 'text-purple-400' :
              line.includes('[decode]') ? 'text-cyan-400' :
              line.includes('[sentence]') ? 'text-emerald-400' :
              line.includes('[voice]') ? 'text-rose-400' :
              line.includes('[insights]') ? 'text-teal-400' :
              'text-dark-300'
            }`}>
              {line}
            </div>
          ))
        ) : (
          <span className="text-dark-500 italic">Waiting for pipeline output...</span>
        )}
      </div>
    </motion.div>
  )
}
