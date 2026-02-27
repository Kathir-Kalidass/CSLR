import { motion } from 'framer-motion'
import { Camera, Zap, Activity, Target, Hand, User } from 'lucide-react'

function DebugMetric({ label, value, color = 'cyan' }) {
  const colorClasses = {
    cyan: 'text-cyan-300',
    emerald: 'text-emerald-300',
    amber: 'text-amber-300',
    rose: 'text-rose-300',
    purple: 'text-purple-300',
  }
  
  return (
    <div className="flex items-center justify-between rounded-lg bg-black/25 px-3 py-2">
      <span className="text-sm text-slate-300">{label}</span>
      <span className={`font-mono font-bold ${colorClasses[color]}`}>{value}</span>
    </div>
  )
}

function BufferProgress({ current, max }) {
  const percentage = max > 0 ? (current / max) * 100 : 0
  
  return (
    <div className="mt-2">
      <div className="mb-1 flex items-center justify-between text-xs text-slate-300">
        <span>Frame Buffer</span>
        <span className="font-mono font-bold">
          {current} / {max}
        </span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-black/35">
        <motion.div
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.3 }}
          className="h-full bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400"
          style={{
            boxShadow: percentage > 80 ? '0 0 12px rgba(56, 189, 248, 0.6)' : 'none'
          }}
        />
      </div>
    </div>
  )
}

export default function Module1Debug({ data }) {
  const debug = data?.module1_debug || {}
  const {
    buffer_fill = 0,
    buffer_capacity = 64,
    frames_kept = 0,
    frames_discarded = 0,
    motion_score = 0,
    roi_detected = false,
    pose_detected = false,
  } = debug

  const keepRate = (frames_kept + frames_discarded) > 0
    ? ((frames_kept / (frames_kept + frames_discarded)) * 100).toFixed(1)
    : 0

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-panel rounded-3xl p-5"
    >
      <div className="mb-4 flex items-center gap-2">
        <Activity className="text-cyan-300" size={24} />
        <h3 className="text-xl font-bold text-slate-100">Module 1 Debug</h3>
      </div>

      <div className="space-y-2">
        <DebugMetric 
          label="Motion Score" 
          value={motion_score.toFixed(2)} 
          color="amber" 
        />
        <DebugMetric 
          label="Frames Kept" 
          value={frames_kept} 
          color="emerald" 
        />
        <DebugMetric 
          label="Frames Discarded" 
          value={frames_discarded} 
          color="rose" 
        />
        <DebugMetric 
          label="Keep Rate" 
          value={`${keepRate}%`} 
          color="cyan" 
        />
        
        <div className="mt-3 flex gap-4">
          <div className={`flex items-center gap-2 rounded-lg px-3 py-2 ${roi_detected ? 'bg-emerald-500/25' : 'bg-slate-700/25'}`}>
            <Target size={16} className={roi_detected ? 'text-emerald-300' : 'text-slate-500'} />
            <span className="text-xs font-semibold">ROI</span>
          </div>
          <div className={`flex items-center gap-2 rounded-lg px-3 py-2 ${pose_detected ? 'bg-emerald-500/25' : 'bg-slate-700/25'}`}>
            <User size={16} className={pose_detected ? 'text-emerald-300' : 'text-slate-500'} />
            <span className="text-xs font-semibold">Pose</span>
          </div>
        </div>

        <BufferProgress current={buffer_fill} max={buffer_capacity} />
      </div>

      <div className="mt-4 rounded-lg border border-cyan-300/30 bg-cyan-500/10 p-3">
        <div className="mb-1 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-cyan-200">
          <Zap size={14} />
          <span>Optimization</span>
        </div>
        <div className="space-y-1 text-xs text-slate-300">
          <div>✓ Motion-based filtering</div>
          <div>✓ Temporal subsampling (1/2)</div>
          <div>✓ ROI cropping enabled</div>
          <div>✓ YOLOv8 pose (17 kpts, 34-dim)</div>
        </div>
      </div>
    </motion.div>
  )
}
