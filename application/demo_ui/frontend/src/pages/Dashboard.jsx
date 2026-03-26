import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Activity, Clock, Target, TrendingUp, BarChart2, Zap, Cpu, Eye } from 'lucide-react'
import useStore from '../store/useStore'
import useSystemInfo from '../hooks/useSystemInfo'
import StatsCard from '../components/StatsCard'
import ConfidenceMeter from '../components/ConfidenceMeter'
import ModelInfoCard from '../components/ModelInfoCard'

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.25 } },
}

function MiniChart({ data, color = '#22d3ee', height = 50 }) {
  if (!data || data.length < 2) return <div style={{ height }} className="flex items-center justify-center text-[10px] text-dark-500">No data yet</div>

  const max = Math.max(...data, 1)
  const min = Math.min(...data, 0)
  const range = max - min || 1
  const w = 200
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = height - ((v - min) / range) * (height - 6) - 3
    return `${x},${y}`
  }).join(' ')

  const fillPoints = `0,${height} ${points} ${w},${height}`

  return (
    <svg viewBox={`0 0 ${w} ${height}`} className="w-full" style={{ height }}>
      <defs>
        <linearGradient id={`grad-${color.replace('#','')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon fill={`url(#grad-${color.replace('#','')})`} points={fillPoints} />
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={points} />
    </svg>
  )
}

export default function Dashboard() {
  const payload = useStore((s) => s.payload)
  const metricsHistory = useStore((s) => s.metricsHistory)
  const systemInfo = useSystemInfo()

  const fps = payload?.fps ?? 0
  const latency = payload?.latency_ms ?? 0
  const confidence = payload?.confidence ?? 0
  const status = payload?.status ?? 'idle'
  const metrics = payload?.metrics ?? {}
  const inferenceMode = payload?.inference_mode ?? 'simulated'
  const ttsEngine = payload?.tts_engine ?? 'N/A'
  const modelInfo = payload?.model_info
  const history = payload?.transcript_history ?? []
  const attention = payload?.attention ?? { rgb: 0.5, pose: 0.5 }

  const fpsData = useMemo(() => metricsHistory.map((m) => m.fps), [metricsHistory])
  const latencyData = useMemo(() => metricsHistory.map((m) => m.latency), [metricsHistory])
  const confData = useMemo(() => metricsHistory.map((m) => m.confidence), [metricsHistory])
  const bleuData = useMemo(() => metricsHistory.map((m) => m.bleu), [metricsHistory])

  const avgFps = fpsData.length > 0 ? (fpsData.reduce((a, b) => a + b, 0) / fpsData.length).toFixed(1) : '0'
  const avgLatency = latencyData.length > 0 ? Math.round(latencyData.reduce((a, b) => a + b, 0) / latencyData.length) : 0

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="max-w-7xl mx-auto px-4 py-6 space-y-5"
    >
      {/* Header */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
        <h2 className="text-lg font-bold gradient-text">Analytics Dashboard</h2>
        <p className="text-[11px] text-dark-400 mt-0.5">
          Performance metrics, model stats, and session analytics
        </p>
      </motion.div>

      {/* Stats cards row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatsCard icon={<Activity size={14} />} label="FPS" value={fps} color="cyan" delay={0} />
        <StatsCard icon={<Clock size={14} />} label="Latency" value={latency} unit="ms" color="amber" delay={0.05} />
        <StatsCard icon={<Target size={14} />} label="Confidence" value={`${(confidence * 100).toFixed(0)}%`} color="green" delay={0.1} />
        <StatsCard icon={<TrendingUp size={14} />} label="BLEU" value={(metrics.bleu_proxy ?? 0).toFixed(3)} color="purple" delay={0.15} />
        <StatsCard icon={<BarChart2 size={14} />} label="WER" value={(metrics.wer_proxy ?? 0).toFixed(3)} color="red" delay={0.2} />
        <StatsCard icon={<Zap size={14} />} label="Mode" value={inferenceMode === 'real' ? 'Real' : 'Sim'} color="teal" delay={0.25} />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass rounded-xl p-4 glow-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-dark-200">FPS Over Time</span>
            <span className="text-[10px] font-mono text-accent-cyan">avg: {avgFps}</span>
          </div>
          <MiniChart data={fpsData} color="#22d3ee" height={60} />
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="glass rounded-xl p-4 glow-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-dark-200">Latency Over Time</span>
            <span className="text-[10px] font-mono text-accent-amber">avg: {avgLatency}ms</span>
          </div>
          <MiniChart data={latencyData} color="#fbbf24" height={60} />
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass rounded-xl p-4 glow-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-dark-200">Confidence Over Time</span>
            <span className="text-[10px] font-mono text-accent-green">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <MiniChart data={confData} color="#34d399" height={60} />
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }} className="glass rounded-xl p-4 glow-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-dark-200">BLEU Score Over Time</span>
            <span className="text-[10px] font-mono text-accent-purple">{(metrics.bleu_proxy ?? 0).toFixed(3)}</span>
          </div>
          <MiniChart data={bleuData} color="#a78bfa" height={60} />
        </motion.div>
      </div>

      {/* Bottom row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Attention / Streams */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass rounded-xl p-4 glow-border">
          <h4 className="text-xs font-semibold text-dark-200 mb-3 flex items-center gap-1.5">
            <Eye size={13} className="text-accent-cyan" />
            Stream Attention
          </h4>
          <div className="space-y-3">
            <ConfidenceMeter value={attention.rgb} label="RGB Weight" />
            <ConfidenceMeter value={attention.pose} label="Pose Weight" />
            <ConfidenceMeter value={confidence} label="Overall Confidence" />
          </div>
        </motion.div>

        {/* Model & System Info */}
        <ModelInfoCard modelInfo={modelInfo} systemInfo={systemInfo} />

        {/* Recent transcript */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }} className="glass rounded-xl p-4 glow-border">
          <h4 className="text-xs font-semibold text-dark-200 mb-3">Recent Translations</h4>
          <div className="space-y-1.5 max-h-48 overflow-y-auto">
            {history.length > 0 ? (
              history.slice(0, 10).map((text, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className={`text-[11px] px-2 py-1 rounded ${
                    i === 0 ? 'bg-accent-cyan/10 text-white' : 'text-dark-300'
                  }`}
                >
                  {text}
                </motion.div>
              ))
            ) : (
              <p className="text-xs text-dark-500 italic text-center py-4">No translations yet</p>
            )}
          </div>
        </motion.div>
      </div>

      {/* Session summary */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="glass rounded-xl p-4 glow-border"
      >
        <h4 className="text-xs font-semibold text-dark-200 mb-2">Session Summary</h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-lg font-bold text-accent-cyan">{metricsHistory.length}</p>
            <p className="text-[10px] text-dark-400">Data Points</p>
          </div>
          <div>
            <p className="text-lg font-bold text-accent-green">{history.length}</p>
            <p className="text-[10px] text-dark-400">Translations</p>
          </div>
          <div>
            <p className="text-lg font-bold text-accent-amber">{avgFps}</p>
            <p className="text-[10px] text-dark-400">Avg FPS</p>
          </div>
          <div>
            <p className="text-lg font-bold text-accent-purple">{avgLatency}ms</p>
            <p className="text-[10px] text-dark-400">Avg Latency</p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}
