import { motion } from 'framer-motion'

export default function StatsCard({ icon, label, value, unit = '', color = 'cyan', delay = 0 }) {
  const colorMap = {
    cyan: 'text-accent-cyan bg-accent-cyan/10 border-accent-cyan/20',
    green: 'text-accent-green bg-accent-green/10 border-accent-green/20',
    amber: 'text-accent-amber bg-accent-amber/10 border-accent-amber/20',
    purple: 'text-accent-purple bg-accent-purple/10 border-accent-purple/20',
    red: 'text-accent-red bg-accent-red/10 border-accent-red/20',
    teal: 'text-accent-teal bg-accent-teal/10 border-accent-teal/20',
    blue: 'text-accent-blue bg-accent-blue/10 border-accent-blue/20',
  }

  const classes = colorMap[color] || colorMap.cyan

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: 'spring', stiffness: 200 }}
      whileHover={{ scale: 1.03, y: -2 }}
      className={`glass rounded-xl p-3 border ${classes.split(' ').slice(2).join(' ')} card-hover`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        {icon && <span className={classes.split(' ')[0]}>{icon}</span>}
        <span className="text-[10px] font-medium text-dark-400 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`text-xl font-bold ${classes.split(' ')[0]}`}>
          {value}
        </span>
        {unit && <span className="text-[10px] text-dark-400">{unit}</span>}
      </div>
    </motion.div>
  )
}
