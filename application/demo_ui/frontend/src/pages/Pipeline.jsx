import { motion } from 'framer-motion'
import {
  ArrowRight,
  ChevronRight,
  Cpu,
  Gauge,
  LayoutTemplate,
  ListChecks,
  Sparkles,
  Timer,
  Zap,
} from 'lucide-react'
import useStore from '../store/useStore'
import PipelineStage from '../components/PipelineStage'
import ConsolePanel from '../components/ConsolePanel'
import MetricsBar from '../components/MetricsBar'
import ControlBar from '../components/ControlBar'
import Camera from '../components/Camera'

const moduleLabels = {
  module1: { label: 'Capture', color: 'from-blue-500 to-blue-400', icon: '01' },
  module2: { label: 'Features', color: 'from-purple-500 to-purple-400', icon: '02' },
  module3: { label: 'Decode', color: 'from-cyan-500 to-cyan-400', icon: '03' },
  module4: { label: 'Cleanup', color: 'from-amber-500 to-amber-400', icon: '04' },
  module5: { label: 'Sentence', color: 'from-emerald-500 to-emerald-400', icon: '05' },
  module6: { label: 'Voice', color: 'from-rose-500 to-rose-400', icon: '06' },
  module7: { label: 'Insights', color: 'from-teal-500 to-teal-400', icon: '07' },
}

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.25 } },
}

export default function Pipeline() {
  const payload = useStore((s) => s.payload)
  const consoleLogs = useStore((s) => s.consoleLogs)
  const connected = useStore((s) => s.connected)
  const running = useStore((s) => s.running)
  const grandMode = useStore((s) => s.grandMode)

  const pipelineOrder = payload?.pipeline_order ?? [
    'module1', 'module2', 'module3', 'module4', 'module5', 'module6', 'module7',
  ]
  const activeStage = payload?.active_stage ?? 'module1'
  const status = payload?.status ?? 'idle'
  const fps = payload?.fps ?? 0
  const latency = payload?.latency_ms ?? 0
  const confidence = payload?.confidence ?? 0
  const gloss = payload?.partial_gloss ?? '--'
  const sentence = payload?.final_sentence ?? ''
  const runtimeStatus = payload?.runtime_status ?? 'No runtime status yet'
  const inferenceMode = payload?.inference_mode ?? 'simulated'
  const windowFrames = payload?.metrics?.window_frames ?? 0
  const stride = payload?.metrics?.stride ?? 0
  const activeWindow = payload?.timeline?.active_window ?? '--'
  const timelineWindows = payload?.timeline?.windows ?? []
  const isActive = status === 'active'
  const activeIndex = Math.max(0, pipelineOrder.findIndex((key) => key === activeStage))
  const flowProgress = pipelineOrder.length > 1
    ? Math.max(0, Math.min(100, (activeIndex / (pipelineOrder.length - 1)) * 100))
    : 0
  const activeLabel = moduleLabels[activeStage]?.label ?? activeStage
  const confidencePct = `${(confidence * 100).toFixed(0)}%`

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="max-w-[1500px] mx-auto px-4 py-5 space-y-4"
    >
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-4 sm:p-5 glow-border"
      >
        <div className="flex flex-col xl:flex-row gap-4 xl:items-center xl:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-accent-cyan/10 border border-accent-cyan/20 mb-2">
              <LayoutTemplate size={12} className="text-accent-cyan" />
              <span className="text-[11px] font-semibold tracking-wide text-accent-cyan">
                FLOW WORKSPACE
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold gradient-text">
              Real-time Pipeline Architecture
            </h2>
            <p className="text-sm text-dark-300 mt-1 max-w-2xl">
              Structured visibility across capture, feature fusion, decoding, and sentence synthesis.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="rounded-xl border border-dark-700 bg-dark-900/70 px-3 py-2">
              <p className="text-dark-400">State</p>
              <p className={`font-semibold mt-0.5 ${isActive ? 'text-accent-green' : 'text-dark-200'}`}>
                {isActive ? 'Processing' : 'Idle'}
              </p>
            </div>
            <div className="rounded-xl border border-dark-700 bg-dark-900/70 px-3 py-2">
              <p className="text-dark-400">Active Stage</p>
              <p className="font-semibold mt-0.5 text-white">{activeLabel}</p>
            </div>
            <div className="rounded-xl border border-dark-700 bg-dark-900/70 px-3 py-2">
              <p className="text-dark-400">Confidence</p>
              <p className="font-semibold mt-0.5 text-accent-green">{confidencePct}</p>
            </div>
            <div className="rounded-xl border border-dark-700 bg-dark-900/70 px-3 py-2">
              <p className="text-dark-400">Mode</p>
              <p className={`font-semibold mt-0.5 ${inferenceMode === 'real' ? 'text-accent-green' : 'text-accent-amber'}`}>
                {inferenceMode === 'real' ? 'Real Model' : 'Simulated'}
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start">
        <div className="xl:col-span-8 space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-xl p-4 glow-border"
          >
            <div className="flex items-center gap-2 mb-3">
              <Cpu size={16} className="text-accent-cyan" />
              <span className="text-sm font-semibold text-dark-100">Operations Deck</span>
            </div>
            <ControlBar />
            <div className="mt-3 pt-3 border-t border-dark-700/40">
              <MetricsBar fps={fps} latency={latency} confidence={confidence} status={status} />
            </div>
          </motion.div>

          <div className="glass rounded-xl p-3 sm:p-4 glow-border">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Sparkles size={16} className="text-accent-purple" />
                <span className="text-sm font-semibold text-dark-100">Live Capture Matrix</span>
              </div>
              <span className={`text-[11px] px-2 py-0.5 rounded-full border ${
                running ? 'text-accent-green border-accent-green/30 bg-accent-green/10' : 'text-dark-400 border-dark-700 bg-dark-900/60'
              }`}>
                {running ? 'Recognizer Running' : 'Recognizer Stopped'}
              </span>
            </div>
            <Camera />
          </div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-xl p-4 glow-border"
          >
            <div className="flex items-center gap-2 mb-3">
              <ListChecks size={16} className="text-accent-cyan" />
              <span className="text-sm font-semibold text-dark-100">Pipeline Track</span>
              <span className="ml-auto text-[11px] font-mono text-dark-400">
                Stage {activeIndex + 1}/{pipelineOrder.length}
              </span>
            </div>

            <div className="h-1.5 bg-dark-800 rounded-full overflow-hidden mb-3">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-accent-cyan via-accent-teal to-accent-purple"
                initial={{ width: 0 }}
                animate={{ width: `${flowProgress}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>

            <div className="flex items-center gap-2 overflow-x-auto pb-1">
              {pipelineOrder.map((key, i) => {
                const isCurrent = key === activeStage
                const ml = moduleLabels[key] ?? {
                  label: key,
                  color: 'from-dark-700 to-dark-600',
                  icon: '--',
                }
                return (
                  <div key={key} className="flex items-center gap-2 flex-shrink-0">
                    <div
                      className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-all border ${
                        isCurrent
                          ? `bg-gradient-to-r ${ml?.color} text-white border-white/30 shadow-lg`
                          : 'bg-dark-800/80 text-dark-300 border-dark-700'
                      }`}
                    >
                      <span className="mr-1.5 font-mono">{ml?.icon}</span>
                      {ml?.label}
                    </div>
                    {i < pipelineOrder.length - 1 && (
                      <ChevronRight size={16} className={isCurrent ? 'text-accent-cyan' : 'text-dark-600'} />
                    )}
                  </div>
                )
              })}
            </div>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {pipelineOrder.map((key, i) => (
              <div key={key} className="relative">
                <PipelineStage
                  moduleKey={key}
                  data={payload?.[key]}
                  isActive={isActive && key === activeStage}
                  index={i}
                />
                {i < pipelineOrder.length - 1 && (
                  <div className="hidden 2xl:flex absolute -right-3 top-1/2 -translate-y-1/2 z-10">
                    <ArrowRight
                      size={16}
                      className={isActive && key === activeStage ? 'text-accent-cyan' : 'text-dark-700'}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="xl:col-span-4 space-y-4 xl:sticky xl:top-20">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className={`glass rounded-xl p-5 ${isActive ? 'glow-border-active' : 'glow-border'}`}
          >
            <div className="flex items-center gap-2 mb-4">
              <Zap size={18} className={isActive ? 'text-accent-cyan' : 'text-dark-500'} />
              <span className="text-sm font-semibold text-dark-200">Pipeline Output</span>
              <span
                className={`ml-auto text-sm font-mono px-3 py-1 rounded-md ${
                  isActive ? 'bg-accent-green/15 text-accent-green' : 'bg-dark-800 text-dark-400'
                }`}
              >
                {isActive ? 'PROCESSING' : 'IDLE'}
              </span>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-[11px] text-dark-400 uppercase tracking-wider font-semibold">
                  Gloss Output
                </p>
                <p className={`text-2xl font-mono mt-1 font-bold ${isActive && gloss !== '--' ? 'text-accent-purple' : 'text-dark-500'}`}>
                  {gloss}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-dark-400 uppercase tracking-wider font-semibold">
                  Sentence
                </p>
                <p className={`text-base font-medium mt-1 leading-relaxed ${
                  isActive && sentence && sentence !== 'Analyzing live sign window...'
                    ? 'text-white'
                    : 'text-dark-500'
                }`}>
                  {sentence || 'Waiting...'}
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-xl p-4 glow-border"
          >
            <div className="flex items-center gap-2 mb-3">
              <Gauge size={15} className="text-accent-amber" />
              <span className="text-sm font-semibold text-dark-100">Execution Context</span>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between border-b border-dark-700/30 pb-2">
                <span className="text-dark-400">WebSocket</span>
                <span className={connected ? 'text-accent-green' : 'text-accent-red'}>
                  {connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-dark-700/30 pb-2">
                <span className="text-dark-400">Runtime</span>
                <span className="text-dark-200 text-right pl-2">{runtimeStatus}</span>
              </div>
              <div className="flex items-center justify-between border-b border-dark-700/30 pb-2">
                <span className="text-dark-400">Window</span>
                <span className="font-mono text-dark-200">{activeWindow}</span>
              </div>
              <div className="flex items-center justify-between border-b border-dark-700/30 pb-2">
                <span className="text-dark-400">Frames / Stride</span>
                <span className="font-mono text-dark-200">{windowFrames} / {stride}</span>
              </div>
              <div className="flex items-center justify-between border-b border-dark-700/30 pb-2">
                <span className="text-dark-400">Grand Mode</span>
                <span className={grandMode ? 'text-accent-purple' : 'text-dark-300'}>
                  {grandMode ? 'Enabled' : 'Disabled'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-dark-400 flex items-center gap-1">
                  <Timer size={12} />
                  Latency
                </span>
                <span className="font-mono text-accent-amber">{latency}ms</span>
              </div>
            </div>

            {timelineWindows.length > 0 && (
              <div className="mt-3 pt-3 border-t border-dark-700/30">
                <p className="text-[11px] text-dark-400 mb-2">Window Timeline</p>
                <div className="space-y-1.5">
                  {timelineWindows.map((window) => (
                    <div
                      key={window}
                      className={`px-2 py-1 rounded-md text-[11px] font-mono ${
                        window === activeWindow ? 'bg-accent-cyan/15 text-accent-cyan' : 'bg-dark-900/70 text-dark-300'
                      }`}
                    >
                      {window}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>

          <div className="glass rounded-xl p-3 glow-border">
            <div className="flex items-center gap-2 px-1 pb-2">
              <span className="text-sm font-semibold text-dark-100">Parser Console</span>
              <span className="ml-auto text-[10px] text-dark-500 font-mono">
                {consoleLogs.length} lines
              </span>
            </div>
            <ConsolePanel logs={consoleLogs} />
          </div>
        </div>
      </div>
    </motion.div>
  )
}
