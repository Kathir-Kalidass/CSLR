import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Video, GitBranch, BarChart3, Wifi, WifiOff, Zap } from 'lucide-react'
import useStore from '../store/useStore'

const navItems = [
  { to: '/live', icon: Video, label: 'Live Recognition' },
  { to: '/flow', icon: GitBranch, label: 'Pipeline Flow' },
  { to: '/insights', icon: BarChart3, label: 'Dashboard' },
]

export default function Navbar() {
  const connected = useStore((s) => s.connected)
  const payload = useStore((s) => s.payload)
  const location = useLocation()
  const inferenceMode = payload?.inference_mode ?? 'simulated'

  return (
    <motion.nav
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="fixed top-0 left-0 right-0 z-50 glass-strong"
    >
      <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <span className="text-2xl">🤟</span>
          <div>
            <h1 className="text-sm font-bold gradient-text leading-tight">ISL CSLR</h1>
            <p className="text-[10px] text-dark-400 leading-tight">
              Continuous Sign Language Recognition
            </p>
          </div>
        </div>

        {/* Nav Links */}
        <div className="flex items-center gap-1">
          {navItems.map(({ to, icon: Icon, label }) => {
            const isActive = location.pathname === to
            return (
              <NavLink key={to} to={to}>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-accent-cyan/15 text-accent-cyan glow-border-active'
                      : 'text-dark-300 hover:text-white hover:bg-dark-800/50'
                  }`}
                >
                  <Icon size={14} />
                  <span className="hidden sm:inline">{label}</span>
                </motion.div>
              </NavLink>
            )
          })}
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-3">
          {/* Inference Mode Badge */}
          <motion.div
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${
              inferenceMode === 'real'
                ? 'bg-accent-green/15 text-accent-green border border-accent-green/30'
                : 'bg-accent-amber/15 text-accent-amber border border-accent-amber/30'
            }`}
          >
            <Zap size={10} />
            {inferenceMode === 'real' ? 'REAL MODEL' : 'SIMULATED'}
          </motion.div>

          {/* Connection Status */}
          <div className={`flex items-center gap-1 text-[10px] font-medium ${
            connected ? 'text-accent-green' : 'text-accent-red'
          }`}>
            {connected ? <Wifi size={12} /> : <WifiOff size={12} />}
            <span className="hidden sm:inline">{connected ? 'Connected' : 'Offline'}</span>
          </div>
        </div>
      </div>
    </motion.nav>
  )
}
