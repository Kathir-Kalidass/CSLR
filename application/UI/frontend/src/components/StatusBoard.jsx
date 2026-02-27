import { motion } from 'framer-motion'
import { Gauge, Timer, Sparkles, AudioLines, TrendingDown, Award } from 'lucide-react'

function MetricPill({ icon: Icon, label, value, highlight }) {
  return (
    <div className={`rounded-xl border bg-white/10 px-3 py-2 text-sm text-slate-100 ${highlight ? 'border-amber-300/40' : 'border-white/20'}`}>
      <div className="flex items-center gap-2 font-semibold">
        <Icon size={16} />
        <span>{label}</span>
      </div>
      <div className="mt-1 text-lg font-bold">{value}</div>
    </div>
  )
}

export default function StatusBoard({ data }) {
  const confidencePct = Math.max(0, Math.min(100, Math.round((data.confidence || 0) * 100)))
  const wer = data.metrics?.wer
  const bleu = data.metrics?.bleu
  const werDisplay = (wer && wer > 0) ? wer.toFixed(3) : 'N/A'
  const bleuDisplay = (bleu && bleu > 0) ? bleu.toFixed(3) : 'N/A'

  return (
    <div className="space-y-4">
      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel rounded-3xl p-5"
      >
        <div className="text-sm uppercase tracking-[0.2em] text-cyan-200/90">Predicted Gloss</div>
        <div className="mt-2 text-4xl font-extrabold tracking-wide text-amber-300">{data.partial_gloss || '--'}</div>
        <div className="mt-4 border-t border-white/20 pt-3 text-2xl font-medium text-slate-100">
          Sentence: {data.final_sentence}
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel rounded-3xl p-5"
      >
        <div className="mb-2 flex items-center justify-between text-xl font-semibold">
          <span>Confidence</span>
          <span>{confidencePct}%</span>
        </div>
        <div className="h-5 rounded-full bg-black/35 p-1">
          <motion.div
            animate={{ width: `${confidencePct}%` }}
            transition={{ duration: 0.5 }}
            className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-lime-300 to-amber-300"
          />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <MetricPill icon={Gauge} label="FPS" value={data.fps} />
          <MetricPill icon={Timer} label="Latency" value={`${data.latency_ms} ms`} />
          <MetricPill icon={Sparkles} label="Accuracy" value={`${((data.metrics?.accuracy || 0) * 100).toFixed(1)}%`} />
          <MetricPill icon={AudioLines} label="Audio" value={data.audio_state || 'idle'} />
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel rounded-3xl p-5"
      >
        <div className="text-2xl font-bold text-slate-100">Transcript History</div>
        <div className="mt-3 max-h-52 space-y-2 overflow-auto">
          {(data.transcript_history?.length ? data.transcript_history : ['No transcript yet']).map((line, idx) => (
            <div key={`${line}-${idx}`} className="rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-lg">
              {line}
            </div>
          ))}
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel rounded-2xl p-4 font-mono text-xs"
      >
        <div className="mb-2 text-sm font-bold">Live Parser Console</div>
        <div className="max-h-24 space-y-1 overflow-auto text-cyan-100/90">
          {(data.parser_console?.length ? data.parser_console : ['[system] waiting']).map((line, i) => (
            <div key={`${line}-${i}`}>{line}</div>
          ))}
        </div>
      </motion.section>
    </div>
  )
}
