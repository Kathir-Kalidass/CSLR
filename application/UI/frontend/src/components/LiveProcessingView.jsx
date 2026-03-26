import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

export default function LiveProcessingView({ data, selectedModule }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!data.video_frame || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      
      // Overlay processing information
      ctx.font = '16px monospace'
      ctx.fillStyle = 'rgba(0, 255, 255, 0.9)'
      ctx.fillText(`Module: ${selectedModule}`, 10, 25)
      ctx.fillText(`FPS: ${data.fps || 0}`, 10, 50)
      ctx.fillText(`Latency: ${data.latency_ms || 0}ms`, 10, 75)
    }
    img.src = `data:image/jpeg;base64,${data.video_frame}`
  }, [data.video_frame, data.fps, data.latency_ms, selectedModule])

  const moduleDescriptions = {
    module1: 'Capturing video, detecting motion, extracting pose landmarks, cropping ROI',
    module2: 'Extracting RGB features (ResNet18) and Pose features (MLP), fusing with attention',
    module3: 'Processing temporal sequence with BiLSTM, performing CTC decoding',
    module4: 'Translating glosses to English, correcting grammar, generating speech'
  }

  return (
    <div className="space-y-4">
      {/* Live Camera Feed */}
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative overflow-hidden rounded-2xl border-2 border-cyan-300/40 bg-black/60 shadow-2xl"
      >
        <div className="absolute left-3 top-3 z-10 rounded-lg bg-red-600 px-3 py-1 text-xs font-bold uppercase tracking-wider shadow-lg">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 animate-pulse rounded-full bg-white" />
            LIVE CAMERA
          </div>
        </div>

        <canvas
          ref={canvasRef}
          className="w-full"
          style={{ aspectRatio: '16/9', objectFit: 'contain' }}
        />

        {/* Module Badge */}
        <div className="absolute bottom-3 right-3 rounded-lg bg-cyan-600 px-4 py-2 font-bold uppercase tracking-wider shadow-lg">
          {selectedModule}
        </div>
      </motion.div>

      {/* Processing Stage Info */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-purple-300/30 bg-purple-900/20 p-4"
      >
        <h3 className="mb-2 font-semibold text-purple-200">Active Processing:</h3>
        <p className="text-sm text-slate-300">
          {moduleDescriptions[selectedModule] || 'Processing...'}
        </p>
      </motion.div>

      {/* Real-time Outputs */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Gloss Output */}
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="rounded-xl border border-emerald-300/30 bg-emerald-900/20 p-4"
        >
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-emerald-200">
            Detected Glosses
          </h3>
          <div className="min-h-[60px] rounded-lg bg-black/40 p-3">
            <code className="text-lg font-bold text-emerald-300">
              {data.partial_gloss || '--'}
            </code>
          </div>
        </motion.div>

        {/* Sentence Output */}
        <motion.div
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          className="rounded-xl border border-cyan-300/30 bg-cyan-900/20 p-4"
        >
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-cyan-200">
            English Translation
          </h3>
          <div className="min-h-[60px] rounded-lg bg-black/40 p-3">
            <p className="text-lg leading-relaxed text-slate-100">
              {data.final_sentence || 'Waiting for sign input...'}
            </p>
          </div>
        </motion.div>
      </div>

      {/* Confidence & Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-white/20 bg-white/10 p-4"
      >
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-cyan-300">
              {((data.confidence || 0) * 100).toFixed(0)}%
            </div>
            <div className="text-xs uppercase tracking-wide text-slate-400">Confidence</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-emerald-300">{data.fps || 0}</div>
            <div className="text-xs uppercase tracking-wide text-slate-400">FPS</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-300">{data.latency_ms || 0}ms</div>
            <div className="text-xs uppercase tracking-wide text-slate-400">Latency</div>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-black/30">
          <motion.div
            animate={{ width: `${(data.confidence || 0) * 100}%` }}
            transition={{ duration: 0.3 }}
            className={`h-full ${
              (data.confidence || 0) > 0.8
                ? 'bg-emerald-400'
                : (data.confidence || 0) > 0.5
                ? 'bg-yellow-400'
                : 'bg-red-400'
            }`}
          />
        </div>
      </motion.div>

      {/* Parser Console */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-slate-600/50 bg-slate-900/60 p-4"
      >
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-400">
          Processing Log
        </h3>
        <div className="max-h-40 overflow-y-auto rounded-lg bg-black/60 p-3 font-mono text-xs">
          {(data.parser_console || []).map((line, idx) => (
            <div key={idx} className="text-green-400">
              {line}
            </div>
          ))}
          {(!data.parser_console || data.parser_console.length === 0) && (
            <div className="text-slate-500">No logs yet...</div>
          )}
        </div>
      </motion.div>

      {/* Transcript History */}
      {data.transcript_history && data.transcript_history.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-xl border border-blue-300/30 bg-blue-900/20 p-4"
        >
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-blue-200">
            Translation History
          </h3>
          <div className="space-y-2">
            {data.transcript_history.slice(-5).map((sentence, idx) => (
              <div
                key={idx}
                className="rounded-lg bg-black/40 px-3 py-2 text-sm text-slate-200"
              >
                {sentence}
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}
