import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle2, ClipboardList } from 'lucide-react'

function normalizeToken(text) {
  return text.toLowerCase().replace(/[^a-z0-9]/g, '')
}

export default function ReferencePrompt({ recognizedSentence = '' }) {
  const [targetSentence, setTargetSentence] = useState('I will go to school tomorrow')

  const recognizedTokens = useMemo(() => {
    return (recognizedSentence || '')
      .split(/\s+/)
      .map(normalizeToken)
      .filter(Boolean)
  }, [recognizedSentence])

  const targetTokens = useMemo(() => {
    return targetSentence
      .split(/\s+/)
      .map((raw) => ({
        raw,
        norm: normalizeToken(raw),
      }))
      .filter((t) => t.raw.trim().length > 0)
  }, [targetSentence])

  const matchedCount = targetTokens.filter((t) => t.norm && recognizedTokens.includes(t.norm)).length
  const matchRate = targetTokens.length > 0 ? matchedCount / targetTokens.length : 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-xl p-4 glow-border"
    >
      <div className="flex items-center gap-2 mb-2">
        <ClipboardList size={14} className="text-accent-cyan" />
        <span className="text-xs font-semibold text-dark-200">Demo Reference Sentence</span>
        <span className="ml-auto text-[10px] font-mono text-accent-green">
          match {(matchRate * 100).toFixed(0)}%
        </span>
      </div>

      <input
        value={targetSentence}
        onChange={(e) => setTargetSentence(e.target.value)}
        placeholder="Enter sentence to demonstrate with signing"
        className="w-full rounded-lg bg-dark-900/80 border border-dark-700 px-3 py-2 text-sm text-white outline-none focus:border-accent-cyan/60"
      />

      <div className="mt-3 flex flex-wrap gap-1.5">
        {targetTokens.map((token, idx) => {
          const matched = token.norm && recognizedTokens.includes(token.norm)
          return (
            <motion.span
              key={`${token.raw}-${idx}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] border ${
                matched
                  ? 'bg-accent-green/15 border-accent-green/30 text-accent-green'
                  : 'bg-dark-800 border-dark-700 text-dark-300'
              }`}
            >
              {matched && <CheckCircle2 size={11} />}
              {token.raw}
            </motion.span>
          )
        })}
      </div>

      <p className="mt-2 text-[10px] text-dark-400">
        Perform the sign sequence for this sentence. Matched words highlight automatically from live model output.
      </p>
    </motion.div>
  )
}
