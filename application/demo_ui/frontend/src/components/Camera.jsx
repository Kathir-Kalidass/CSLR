import { motion } from 'framer-motion'
import { Camera as CamIcon, CameraOff, Video, Activity } from 'lucide-react'
import useCamera from '../hooks/useCamera'
import useStore from '../store/useStore'

export default function Camera() {
  const {
    videoRef,
    canvasRef,
    overlayRef,
    rgbPreviewRef,
    posePreviewRef,
    startCamera,
    stopCamera,
    cameraActive,
    cameraError,
  } = useCamera()
  const running = useStore((s) => s.running)
  const payload = useStore((s) => s.payload)
  const bufferFill = payload?.module1?.note?.match(/(\d+)\/(\d+)/)

  const panelBase =
    'relative aspect-video rounded-xl overflow-hidden glass glow-border'
  const labelBase =
    'absolute top-2 left-2 z-10 flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-mono font-bold backdrop-blur-sm'

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-2"
    >
      {/* Header row */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              cameraActive && running
                ? 'bg-accent-green animate-pulse'
                : cameraActive
                  ? 'bg-accent-amber'
                  : 'bg-dark-500'
            }`}
          />
          <span className="text-xs font-semibold text-dark-200">
            Camera System
          </span>
          {cameraActive && running && (
            <span className="text-[10px] text-accent-green font-mono">
              STREAMING
            </span>
          )}
        </div>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={cameraActive ? stopCamera : startCamera}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
            cameraActive
              ? 'bg-accent-red/15 text-accent-red hover:bg-accent-red/25'
              : 'bg-accent-cyan/15 text-accent-cyan hover:bg-accent-cyan/25'
          }`}
        >
          {cameraActive ? (
            <CameraOff size={14} />
          ) : (
            <CamIcon size={14} />
          )}
          {cameraActive ? 'Stop' : 'Open Camera'}
        </motion.button>
      </div>

      {cameraError && (
        <div className="px-3 py-1.5 rounded-lg text-[11px] bg-accent-red/10 border border-accent-red/20 text-accent-red">
          {cameraError}
        </div>
      )}

      {/* ─── Three large panels ─── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Panel 1: Live Camera */}
        <div className={panelBase}>
          <div className={`${labelBase} bg-dark-900/70 text-accent-cyan`}>
            <Video size={12} />
            LIVE
          </div>

          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            playsInline
            muted
            style={{ display: cameraActive ? 'block' : 'none' }}
          />
          <canvas
            ref={overlayRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ display: cameraActive ? 'block' : 'none' }}
          />

          {/* Placeholder */}
          {!cameraActive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-dark-900">
              <motion.div
                animate={{ opacity: [0.3, 0.6, 0.3] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <CamIcon size={36} className="text-dark-500" />
              </motion.div>
              <p className="text-[11px] text-dark-400">
                Click &quot;Open Camera&quot; to start
              </p>
            </div>
          )}

          {/* Scan line */}
          {cameraActive && running && (
            <div className="absolute inset-0 scan-line pointer-events-none" />
          )}

          {/* Corner brackets */}
          {cameraActive && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-2 left-2 w-5 h-5 border-t-2 border-l-2 border-accent-cyan/50 rounded-tl" />
              <div className="absolute top-2 right-2 w-5 h-5 border-t-2 border-r-2 border-accent-cyan/50 rounded-tr" />
              <div className="absolute bottom-2 left-2 w-5 h-5 border-b-2 border-l-2 border-accent-cyan/50 rounded-bl" />
              <div className="absolute bottom-2 right-2 w-5 h-5 border-b-2 border-r-2 border-accent-cyan/50 rounded-br" />
            </div>
          )}

          {/* Buffer fill */}
          {bufferFill && (
            <div className="absolute bottom-2 left-2 right-2">
              <div className="flex items-center gap-2 text-[10px] text-dark-300 font-mono">
                <span>Buffer</span>
                <div className="flex-1 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-accent-cyan rounded-full"
                    initial={{ width: 0 }}
                    animate={{
                      width: `${(parseInt(bufferFill[1]) / parseInt(bufferFill[2])) * 100}%`,
                    }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <span>
                  {bufferFill[1]}/{bufferFill[2]}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Panel 2: RGB Stream */}
        <div className={panelBase}>
          <div className={`${labelBase} bg-dark-900/70 text-accent-cyan`}>
            <Video size={12} />
            RGB
          </div>
          <canvas
            ref={rgbPreviewRef}
            className="w-full h-full bg-dark-900"
          />
          {!cameraActive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-dark-900">
              <Video size={30} className="text-dark-600" />
              <p className="text-[10px] text-dark-500">RGB Stream</p>
            </div>
          )}
        </div>

        {/* Panel 3: Pose / Motion Stream */}
        <div className={panelBase}>
          <div className={`${labelBase} bg-dark-900/70 text-accent-purple`}>
            <Activity size={12} />
            POSE
          </div>
          <canvas
            ref={posePreviewRef}
            className="w-full h-full bg-dark-900"
          />
          {!cameraActive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-dark-900">
              <Activity size={30} className="text-dark-600" />
              <p className="text-[10px] text-dark-500">Pose · Motion</p>
            </div>
          )}
        </div>
      </div>

      {/* Hidden capture canvas */}
      <canvas ref={canvasRef} className="hidden" />
    </motion.div>
  )
}
