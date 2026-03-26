import { motion } from 'framer-motion'

export default function AudioWave({ wave = [], speaking = false }) {
  const bars = wave.length > 0 ? wave : Array(24).fill(0)

  return (
    <div className="flex items-end justify-center gap-0.5 h-8">
      {bars.map((val, i) => (
        <motion.div
          key={i}
          animate={{
            height: speaking ? Math.max(2, val * 28) : 2,
            backgroundColor: speaking
              ? `rgba(45, 212, 191, ${0.4 + val * 0.6})`
              : 'rgba(100, 116, 139, 0.3)',
          }}
          transition={{ duration: 0.15, ease: 'easeOut' }}
          className="w-1 rounded-full"
          style={{ minHeight: 2 }}
        />
      ))}
    </div>
  )
}
