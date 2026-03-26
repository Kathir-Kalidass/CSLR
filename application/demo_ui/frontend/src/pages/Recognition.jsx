import { motion } from 'framer-motion'
import useStore from '../store/useStore'
import useSystemInfo from '../hooks/useSystemInfo'
import Camera from '../components/Camera'
import GlossDisplay from '../components/GlossDisplay'
import SentenceOutput from '../components/SentenceOutput'
import ControlBar from '../components/ControlBar'
import MetricsBar from '../components/MetricsBar'
import AttentionViz from '../components/AttentionViz'
import AudioWave from '../components/AudioWave'
import TranscriptHistory from '../components/TranscriptHistory'
import ConfidenceMeter from '../components/ConfidenceMeter'
import ConsolePanel from '../components/ConsolePanel'
import InitSequence from '../components/InitSequence'
import ModelInfoCard from '../components/ModelInfoCard'
import ReferencePrompt from '../components/ReferencePrompt'

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4, staggerChildren: 0.06 } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.25 } },
}

export default function Recognition() {
  const payload = useStore((s) => s.payload)
  const consoleLogs = useStore((s) => s.consoleLogs)
  const systemInfo = useSystemInfo()

  const status = payload?.status ?? 'idle'
  const confidence = payload?.confidence ?? 0
  const attention = payload?.attention ?? { rgb: 0.5, pose: 0.5 }
  const audioWave = payload?.audio_wave ?? []
  const audioState = payload?.audio_state ?? 'idle'
  const history = payload?.transcript_history ?? []
  const fps = payload?.fps ?? 0
  const latency = payload?.latency_ms ?? 0
  const gloss = payload?.partial_gloss ?? '--'
  const sentence = payload?.final_sentence ?? ''
  const initSeq = payload?.init_sequence ?? []
  const modelInfo = payload?.model_info
  const metrics = payload?.metrics ?? {}

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="max-w-[1400px] mx-auto px-4 py-5 space-y-4"
    >
      {/* Title */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
        <h2 className="text-lg font-bold gradient-text">Live Sign Recognition</h2>
        <p className="text-[11px] text-dark-400 mt-0.5">
          Real-time Indian Sign Language recognition from webcam
        </p>
      </motion.div>

      {/* Controls + Metrics */}
      <ControlBar />
      <MetricsBar fps={fps} latency={latency} confidence={confidence} status={status} />

      {/* ─── Camera: full-width 3-panel (Live | RGB | Pose) ─── */}
      <Camera />

      {/* Reference prompt */}
      <ReferencePrompt recognizedSentence={sentence} />

      {/* ─── Recognition output: prominent full-width ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <GlossDisplay gloss={gloss} confidence={confidence} />
        <SentenceOutput sentence={sentence} audioState={audioState} />
      </div>

      {/* ─── Details grid ─── */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {/* Attention */}
        <AttentionViz rgb={attention.rgb} pose={attention.pose} />

        {/* Confidence meters */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-xl p-4 glow-border space-y-3"
        >
          <ConfidenceMeter value={confidence} label="Model Confidence" />
          <ConfidenceMeter value={metrics.bleu_proxy ?? 0} label="BLEU Score" />
          <ConfidenceMeter value={1 - (metrics.wer_proxy ?? 0)} label="Accuracy (1-WER)" />
        </motion.div>

        {/* Audio */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-xl p-3 glow-border"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-medium text-dark-300">Audio Waveform</span>
            <span
              className={`text-[10px] font-bold ${
                audioState === 'speaking' ? 'text-accent-teal' : 'text-dark-500'
              }`}
            >
              {audioState}
            </span>
          </div>
          <AudioWave wave={audioWave} speaking={audioState === 'speaking'} />
        </motion.div>

        {/* Timeline */}
        {payload?.timeline && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-xl p-4 glow-border"
          >
            <h4 className="text-xs font-semibold text-dark-200 mb-2">Sliding Windows</h4>
            <div className="space-y-1.5">
              {payload.timeline.windows?.map((win, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className={`flex items-center gap-2 px-2 py-1 rounded-md text-[10px] font-mono ${
                    win === payload.timeline.active_window
                      ? 'bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/20'
                      : 'bg-dark-800/50 text-dark-400'
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      win === payload.timeline.active_window
                        ? 'bg-accent-cyan animate-pulse'
                        : 'bg-dark-600'
                    }`}
                  />
                  {win}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </div>

      {/* ─── Bottom row: History + Model + Init + Console ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <TranscriptHistory history={history} />
        <div className="space-y-4">
          <ModelInfoCard modelInfo={modelInfo} systemInfo={systemInfo} />
          <InitSequence sequence={initSeq} show={true} />
        </div>
        <ConsolePanel logs={consoleLogs} />
      </div>
    </motion.div>
  )
}
