import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, RefreshCw, TrendingDown, Cpu, BookOpen } from 'lucide-react'

// ── Simple SVG sparkline ─────────────────────────────────────────────────────
function Sparkline({ values = [], color = '#34d399', width = 220, height = 48 }) {
  if (values.length < 2) {
    return (
      <svg width={width} height={height}>
        <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke={color} strokeWidth={1} strokeDasharray="4 4" opacity={0.4} />
      </svg>
    )
  }

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })

  return (
    <svg width={width} height={height}>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
        points={pts.join(' ')}
      />
      {/* Latest dot */}
      {(() => {
        const [lx, ly] = pts[pts.length - 1].split(',').map(Number)
        return <circle cx={lx} cy={ly} r={3} fill={color} />
      })()}
    </svg>
  )
}

// ── Stat pill ────────────────────────────────────────────────────────────────
function StatPill({ label, value, color = 'text-cyan-300' }) {
  return (
    <div className="rounded-lg bg-black/25 px-3 py-2 text-center">
      <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
      <div className="text-xs uppercase tracking-wider text-slate-400">{label}</div>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080'
const POLL_MS = 15_000

export default function TrainingMonitor() {
  const [rows, setRows] = useState([])
  const [available, setAvailable] = useState(null) // null = unknown, false = not found, true = found
  const [lastFetched, setLastFetched] = useState(null)
  const [fetchError, setFetchError] = useState(null)
  const timerRef = useRef(null)

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/training-metrics`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setRows(json.rows || [])
      setAvailable(json.available ?? false)
      setLastFetched(new Date())
      setFetchError(null)
    } catch (err) {
      setFetchError(err.message)
    }
  }

  useEffect(() => {
    fetchMetrics()
    timerRef.current = setInterval(fetchMetrics, POLL_MS)
    return () => clearInterval(timerRef.current)
  }, [])

  const latest = rows[rows.length - 1] || null
  const lossValues = rows.map((r) => r.train_loss).filter((v) => v != null && !isNaN(v))
  const werValues = rows.map((r) => r.val_wer).filter((v) => v != null && !isNaN(v))

  const totalEpochs = 80 // assumed from training command; update via VITE_TOTAL_EPOCHS if needed
  const currentEpoch = latest ? latest.epoch : 0
  const progressPct = Math.min(100, (currentEpoch / totalEpochs) * 100)

  const fmtVal = (v, decimals = 4) =>
    v == null || isNaN(v) ? '—' : Number(v).toFixed(decimals)

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-panel rounded-3xl p-5 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="text-fuchsia-300" size={22} />
          <h3 className="text-lg font-bold text-slate-100">Training Monitor</h3>
        </div>
        <button
          onClick={fetchMetrics}
          title="Refresh now"
          className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/10 hover:text-slate-200"
        >
          <RefreshCw size={15} />
        </button>
      </div>

      {/* State messages */}
      {fetchError && (
        <div className="rounded-lg bg-rose-500/20 px-3 py-2 text-xs text-rose-300">
          Backend unreachable: {fetchError}
        </div>
      )}

      {!fetchError && available === false && (
        <div className="rounded-lg bg-slate-700/40 px-3 py-2 text-xs text-slate-400">
          Training not started — <code>training_metrics.csv</code> not found.
        </div>
      )}

      {available && rows.length === 0 && (
        <div className="rounded-lg bg-slate-700/40 px-3 py-2 text-xs text-slate-400">
          CSV found but empty — training may be initialising.
        </div>
      )}

      {available && rows.length > 0 && (
        <>
          {/* Epoch progress bar */}
          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-slate-300">
              <span className="flex items-center gap-1">
                <BookOpen size={12} /> Epoch
              </span>
              <span className="font-mono font-bold text-fuchsia-300">
                {currentEpoch} / {totalEpochs}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-black/35">
              <motion.div
                animate={{ width: `${progressPct}%` }}
                transition={{ duration: 0.5 }}
                className="h-full rounded-full bg-gradient-to-r from-fuchsia-500 to-cyan-400"
              />
            </div>
          </div>

          {/* Stat pills */}
          <div className="grid grid-cols-2 gap-2">
            <StatPill label="Train Loss" value={fmtVal(latest?.train_loss)} color="text-amber-300" />
            <StatPill label="Val Loss" value={fmtVal(latest?.val_loss)} color="text-sky-300" />
            <StatPill label="Val WER" value={fmtVal(latest?.val_wer, 3)} color="text-rose-300" />
            <StatPill label="Val Exact" value={fmtVal(latest?.val_exact, 3)} color="text-emerald-300" />
          </div>

          {/* LR + GPU */}
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center gap-2 rounded-lg bg-black/25 px-3 py-2 text-xs text-slate-300">
              <TrendingDown size={13} className="text-cyan-400" />
              <span>LR</span>
              <span className="ml-auto font-mono text-cyan-300">{latest?.learning_rate?.toExponential(2) ?? '—'}</span>
            </div>
            <div className="flex items-center gap-2 rounded-lg bg-black/25 px-3 py-2 text-xs text-slate-300">
              <Cpu size={13} className="text-purple-400" />
              <span>GPU mem</span>
              <span className="ml-auto font-mono text-purple-300">
                {latest?.gpu_memory_GB != null ? `${Number(latest.gpu_memory_GB).toFixed(2)} GB` : '—'}
              </span>
            </div>
          </div>

          {/* Sparklines */}
          <div className="space-y-3">
            <div>
              <div className="mb-1 text-xs text-slate-400">Train Loss (all epochs)</div>
              <Sparkline values={lossValues} color="#fbbf24" width={230} height={44} />
            </div>
            <div>
              <div className="mb-1 text-xs text-slate-400">Val WER (all epochs)</div>
              <Sparkline values={werValues} color="#f87171" width={230} height={44} />
            </div>
          </div>

          {lastFetched && (
            <div className="text-right text-xs text-slate-500">
              Updated {lastFetched.toLocaleTimeString()}
            </div>
          )}
        </>
      )}
    </motion.div>
  )
}
