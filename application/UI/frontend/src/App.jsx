import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import ControlBar from './components/ControlBar'
import StatusBoard from './components/StatusBoard'
import Module1Debug from './components/Module1Debug'
import LiveProcessingView from './components/LiveProcessingView'
import TrainingMonitor from './components/TrainingMonitor'

const DEFAULT_DATA = {
  status: 'idle',
  active_stage: 'module1',
  partial_gloss: '--',
  final_sentence: 'Press Open Camera then Start',
  audio_state: 'idle',
  confidence: 0,
  fps: 0,
  latency_ms: 0,
  metrics: { accuracy: 0, wer: 0, bleu: 0, window_size: 64, stride: 32 },
  transcript_history: [],
  control_state: { running: false, tts_enabled: true, camera_active: false },
  parser_console: [],
  module1_debug: {},
}

const MODULES = [
  { id: 'module1', name: 'Module 1: Video Preprocessing' },
  { id: 'module2', name: 'Module 2: Feature Extraction' },
  { id: 'module3', name: 'Module 3: Temporal Recognition' },
  { id: 'module4', name: 'Module 4: Translation & Output' },
]

const ACK_STATUSES = new Set([
  'started',
  'stopped',
  'camera_opened',
  'camera_closed',
  'tts_toggled',
  'module_selected',
  'cleared',
  'error',
])

export default function App() {
  const wsRef = useRef(null)
  const [data, setData] = useState(DEFAULT_DATA)
  const [selectedModule, setSelectedModule] = useState('module1')
  const [notice, setNotice] = useState('')
  const [wsStatus, setWsStatus] = useState('connecting') // 'connecting' | 'connected' | 'disconnected'

  const wsUrl = useMemo(() => {
    const fromEnv = import.meta.env.VITE_WS_URL
    if (fromEnv) return fromEnv
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://localhost:8080/ws/realtime`
  }, [])

  useEffect(() => {
    let reconnectTimer = null

    const connect = () => {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)

        if (ACK_STATUSES.has(msg.status)) {
          if (msg.message) {
            setNotice(msg.message)
            setTimeout(() => setNotice(''), 2200)
          }

          if (msg.status === 'camera_opened') {
            setData((prev) => ({
              ...prev,
              control_state: { ...prev.control_state, camera_active: true },
            }))
          } else if (msg.status === 'camera_closed') {
            setData((prev) => ({
              ...prev,
              control_state: { ...prev.control_state, camera_active: false, running: false },
            }))
          } else if (msg.status === 'started') {
            setData((prev) => ({
              ...prev,
              control_state: { ...prev.control_state, running: true },
            }))
          } else if (msg.status === 'stopped') {
            setData((prev) => ({
              ...prev,
              control_state: { ...prev.control_state, running: false },
            }))
          } else if (msg.status === 'tts_toggled') {
            setData((prev) => ({
              ...prev,
              control_state: { ...prev.control_state, tts_enabled: !!msg.enabled },
            }))
          } else if (msg.status === 'cleared') {
            setData((prev) => ({ ...prev, transcript_history: [], partial_gloss: '--' }))
          }
          return
        }

        if (msg && typeof msg === 'object' && 'partial_gloss' in msg && 'control_state' in msg) {
          setData(msg)
        }
      }

      ws.onopen = () => {
        setWsStatus('connected')
        setNotice('Connected to backend')
        setTimeout(() => setNotice(''), 1500)
      }

      ws.onclose = () => {
        setWsStatus('disconnected')
        setNotice('Disconnected. Reconnecting...')
        reconnectTimer = setTimeout(connect, 900)
      }
    }

    connect()
    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [wsUrl])

  const send = (payload) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload))
  }

  const sendControl = (action) => send({ type: action })

  const sendModuleChange = (module) => {
    setSelectedModule(module)
    send({ type: 'module_select', module })
  }

  const moduleIndex = MODULES.findIndex((m) => m.id === (data.active_stage || selectedModule))

  return (
    <div className="min-h-screen bg-space pb-32 text-slate-50">
      <div className="mx-auto max-w-7xl px-4 py-6 md:px-8">
        <motion.h1
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center font-display text-4xl font-bold tracking-wide md:text-6xl"
        >
          Real-Time ISL Translation System
        </motion.h1>

        <div className="mx-auto mt-4 flex max-w-4xl flex-wrap items-center justify-center gap-3 text-sm">
          {/* WebSocket status */}
          <span className={`flex items-center gap-2 rounded-full border px-3 py-1 ${
            wsStatus === 'connected'
              ? 'border-emerald-300/60 bg-emerald-500/20'
              : wsStatus === 'connecting'
              ? 'border-yellow-300/60 bg-yellow-500/20'
              : 'border-rose-300/60 bg-rose-500/20'
          }`}>
            <span className={`inline-block h-2 w-2 rounded-full ${
              wsStatus === 'connected' ? 'bg-emerald-400 animate-pulse' :
              wsStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' : 'bg-rose-400'
            }`} />
            WS: {wsStatus === 'connected' ? 'Connected' : wsStatus === 'connecting' ? 'Connecting…' : 'Disconnected'}
          </span>
          <span className={`rounded-full border px-3 py-1 ${data.control_state?.camera_active ? 'border-cyan-300/60 bg-cyan-500/20' : 'border-slate-300/50 bg-slate-700/30'}`}>
            Camera: {data.control_state?.camera_active ? 'Active' : 'Off'}
          </span>
          <span className={`rounded-full border px-3 py-1 ${data.control_state?.running ? 'border-emerald-300/60 bg-emerald-500/20' : 'border-slate-300/50 bg-slate-700/30'}`}>
            Pipeline: {data.control_state?.running ? 'Running' : 'Idle'}
          </span>
          <span className="rounded-full border border-fuchsia-300/60 bg-fuchsia-500/20 px-3 py-1">
            TTS: {data.control_state?.tts_enabled ? 'On' : 'Off'}
          </span>
        </div>

        {notice && (
          <div className="mx-auto mt-4 max-w-3xl rounded-xl border border-cyan-300/40 bg-cyan-500/15 px-4 py-2 text-center text-sm text-cyan-100">
            {notice}
          </div>
        )}

        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mx-auto mt-6 max-w-2xl"
        >
          <div className="rounded-2xl border border-white/20 bg-white/10 p-4">
            <label className="mb-2 block text-sm font-semibold uppercase tracking-wider text-cyan-200">
              Focus Module
            </label>
            <select
              value={selectedModule}
              onChange={(e) => sendModuleChange(e.target.value)}
              className="w-full rounded-xl border border-cyan-300/50 bg-slate-900/80 px-4 py-3 text-lg font-semibold text-slate-100 focus:border-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-300/50"
            >
              {MODULES.map((module) => (
                <option key={module.id} value={module.id}>
                  {module.name}
                </option>
              ))}
            </select>
          </div>
        </motion.div>

        <div className="mx-auto mt-4 max-w-4xl rounded-2xl border border-white/20 bg-white/10 p-3">
          <div className="h-2 rounded-full bg-black/25">
            <motion.div
              animate={{ width: `${((moduleIndex + 1) / MODULES.length) * 100}%` }}
              transition={{ duration: 0.45 }}
              className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-purple-300 to-emerald-300"
            />
          </div>
          <div className="mt-2 grid grid-cols-4 gap-2 text-center text-xs font-bold uppercase tracking-wide">
            {MODULES.map((module, idx) => (
              <div
                key={module.id}
                className={`rounded-md py-1 ${
                  selectedModule === module.id ? 'bg-cyan-300/35' : idx <= moduleIndex ? 'bg-emerald-300/25' : 'bg-white/10'
                }`}
              >
                {module.name.split(':')[0]}
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-12">
          <div className="lg:col-span-8">
            <LiveProcessingView data={data} selectedModule={selectedModule} />
          </div>
          <div className="lg:col-span-4 space-y-6">
            <Module1Debug data={data} />
            <TrainingMonitor />
          </div>
        </div>

        <div className="mt-6">
          <StatusBoard data={data} />
        </div>

        <ControlBar
          running={!!data.control_state?.running}
          ttsEnabled={!!data.control_state?.tts_enabled}
          cameraActive={!!data.control_state?.camera_active}
          onOpenCamera={() => sendControl('open_camera')}
          onCloseCamera={() => sendControl('close_camera')}
          onStart={() => sendControl('start')}
          onStop={() => sendControl('stop')}
          onClear={() => sendControl('clear')}
          onToggleTts={() => sendControl('toggle_tts')}
        />
      </div>
    </div>
  )
}
