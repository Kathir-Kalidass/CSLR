import { motion } from 'framer-motion'
import { Play, Square, RotateCcw, Maximize2 } from 'lucide-react'
import useStore from '../store/useStore'

export default function ControlBar() {
  const running = useStore((s) => s.running)
  const sendControl = useStore((s) => s.sendControl)
  const grandMode = useStore((s) => s.grandMode)
  const setGrandMode = useStore((s) => s.setGrandMode)
  const connected = useStore((s) => s.connected)
  const cameraActive = useStore((s) => s.cameraActive)

  const toggleRun = () => {
    if (!cameraActive && !running) return

    if (running) {
      sendControl('stop')
    } else {
      sendControl('start')
    }
  }

  const clearAll = () => {
    sendControl('clear')
  }

  const toggleGrand = () => {
    const next = !grandMode
    sendControl('set_grand_mode', { value: next })
    setGrandMode(next)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center justify-center gap-3"
    >
      {/* Start / Stop */}
      <motion.button
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        disabled={!connected || (!cameraActive && !running)}
        onClick={toggleRun}
        className={`flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-semibold transition-all disabled:opacity-40 ${
          running
            ? 'bg-accent-red/20 text-accent-red border border-accent-red/30 hover:bg-accent-red/30'
            : 'bg-accent-green/20 text-accent-green border border-accent-green/30 hover:bg-accent-green/30'
        }`}
      >
        {running ? <Square size={16} /> : <Play size={16} />}
        {running ? 'Stop' : cameraActive ? 'Start Recognition' : 'Open Camera First'}
      </motion.button>

      {/* Clear */}
      <motion.button
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        onClick={clearAll}
        className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-dark-800 text-dark-300 border border-dark-600 hover:text-white hover:border-dark-500 transition-all"
      >
        <RotateCcw size={14} />
        Clear
      </motion.button>

      {/* Grand Mode */}
      <motion.button
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        onClick={toggleGrand}
        className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium border transition-all ${
          grandMode
            ? 'bg-accent-purple/20 text-accent-purple border-accent-purple/30'
            : 'bg-dark-800 text-dark-400 border-dark-600 hover:text-white'
        }`}
      >
        <Maximize2 size={14} />
        Grand
      </motion.button>
    </motion.div>
  )
}
