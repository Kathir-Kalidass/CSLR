import { motion } from 'framer-motion'
import { Play, Square, Trash2, Volume2, VolumeX, Camera, CameraOff } from 'lucide-react'

function ControlButton({ label, icon: Icon, className, onClick, disabled }) {
  return (
    <motion.button
      whileTap={{ scale: 0.95 }}
      whileHover={{ y: -2, scale: 1.02 }}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-2xl border px-4 py-2.5 text-sm font-semibold tracking-wide transition disabled:cursor-not-allowed disabled:opacity-50 md:px-5 md:py-3 md:text-base ${className}`}
    >
      <Icon size={18} />
      {label}
    </motion.button>
  )
}

export default function ControlBar({
  running,
  ttsEnabled,
  cameraActive,
  onOpenCamera,
  onCloseCamera,
  onStart,
  onStop,
  onClear,
  onToggleTts,
}) {
  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 border-t border-white/15 bg-slate-950/65 px-3 py-3 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-center gap-3">
        <ControlButton
          label="Open Camera"
          icon={Camera}
          onClick={onOpenCamera}
          disabled={cameraActive}
          className="border-cyan-300/70 bg-cyan-500/35 text-cyan-50"
        />
        <ControlButton
          label="Close Camera"
          icon={CameraOff}
          onClick={onCloseCamera}
          disabled={!cameraActive}
          className="border-slate-300/70 bg-slate-500/35 text-slate-100"
        />
        <ControlButton
          label="Start"
          icon={Play}
          onClick={onStart}
          disabled={running || !cameraActive}
          className="border-emerald-300/70 bg-emerald-500/35 text-emerald-50"
        />
        <ControlButton
          label="Stop"
          icon={Square}
          onClick={onStop}
          disabled={!running}
          className="border-rose-300/70 bg-rose-500/35 text-rose-50"
        />
        <ControlButton
          label="Clear"
          icon={Trash2}
          onClick={onClear}
          disabled={!cameraActive && !running}
          className="border-sky-300/70 bg-sky-500/35 text-sky-50"
        />
        <ControlButton
          label={ttsEnabled ? 'TTS ON' : 'TTS OFF'}
          icon={ttsEnabled ? Volume2 : VolumeX}
          onClick={onToggleTts}
          className="border-fuchsia-300/70 bg-fuchsia-500/35 text-fuchsia-50"
        />
      </div>
    </div>
  )
}
