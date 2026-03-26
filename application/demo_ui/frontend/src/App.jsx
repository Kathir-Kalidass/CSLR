import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import ParticleBackground from './components/ParticleBackground'
import Recognition from './pages/Recognition'
import Pipeline from './pages/Pipeline'
import Dashboard from './pages/Dashboard'
import useWebSocket from './hooks/useWebSocket'

export default function App() {
  useWebSocket()

  return (
    <div className="min-h-screen bg-dark-950 text-white noise-overlay relative">
      <ParticleBackground />
      <Navbar />
      <main className="relative z-10 pt-16">
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/live" element={<Recognition />} />
            <Route path="/flow" element={<Pipeline />} />
            <Route path="/insights" element={<Dashboard />} />
            <Route path="*" element={<Navigate to="/live" replace />} />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  )
}
