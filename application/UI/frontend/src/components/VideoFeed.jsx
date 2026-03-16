import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

export default function VideoFeed({ onStats }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [cameraActive, setCameraActive] = useState(false)

  useEffect(() => {
    let timer = null
    if (cameraActive) {
      timer = setInterval(() => {
        const track = streamRef.current?.getVideoTracks?.()[0]
        const s = track?.getSettings?.() || {}
        const res = s.width && s.height ? `${s.width}x${s.height}` : 'unknown'
        onStats?.({ camera_active: true, frame_hint: Date.now() % 100000, resolution: res })
      }, 350)
    }
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [cameraActive, onStats])

  const toggleCamera = async () => {
    if (cameraActive && streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      if (videoRef.current) videoRef.current.srcObject = null
      setCameraActive(false)
      onStats?.({ camera_active: false, frame_hint: 0, resolution: 'unknown' })
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setCameraActive(true)
    } catch {
      setCameraActive(false)
    }
  }

  return (
    <div className="rounded-[28px] border border-cyan-200/60 bg-slate-900/45 p-3 backdrop-blur-xl">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-xl font-semibold text-cyan-100">Live Webcam Feed</h2>
        <button
          onClick={toggleCamera}
          className="rounded-xl border border-cyan-200/60 bg-cyan-500/25 px-3 py-1.5 text-sm font-semibold text-cyan-50"
        >
          {cameraActive ? 'Stop Camera' : 'Start Camera'}
        </button>
      </div>
      <motion.div
        animate={cameraActive ? { boxShadow: ['0 0 0 rgba(0,0,0,0)', '0 0 35px rgba(56,189,248,0.5)', '0 0 0 rgba(0,0,0,0)'] } : {}}
        transition={{ duration: 2.2, repeat: Infinity }}
        className="relative overflow-hidden rounded-2xl border border-cyan-300/60 bg-slate-900"
      >
        <video ref={videoRef} autoPlay playsInline muted className="aspect-video w-full object-cover" />
      </motion.div>
    </div>
  )
}
