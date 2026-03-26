import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { MessageSquare, Volume2, VolumeX } from 'lucide-react'
import useStore from '../store/useStore'

export default function SentenceOutput({ sentence = '', audioState = 'idle' }) {
  const ttsEnabled = useStore((s) => s.ttsEnabled)
  const sendControl = useStore((s) => s.sendControl)
  const isPlaceholder = !sentence || sentence === '--' || sentence === 'Analyzing live sign window...'

  const lastSpokenRef = useRef('')
  const activeAudioRef = useRef(null)

  const handleTTS = () => {
    sendControl('toggle_tts')
    useStore.getState().setTtsEnabled(!ttsEnabled)
  }

  const speakNow = async (textOverride = '') => {
    const text = (textOverride || sentence || '').trim()
    if (!text || text === '--' || text === 'Analyzing live sign window...') return
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang: 'en', slow: false }),
      })
      if (!res.ok) return
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      activeAudioRef.current = audio
      audio.play()
      audio.onended = () => {
        URL.revokeObjectURL(url)
        if (activeAudioRef.current === audio) activeAudioRef.current = null
      }
    } catch {
      // TTS unavailable
    }
  }

  useEffect(() => {
    const next = (sentence || '').trim()
    if (!ttsEnabled || !next || isPlaceholder) return
    if (next === lastSpokenRef.current) return
    lastSpokenRef.current = next
    speakNow(next)
  }, [sentence, isPlaceholder, ttsEnabled])

  useEffect(() => {
    if (ttsEnabled) return
    if (activeAudioRef.current) {
      activeAudioRef.current.pause()
      activeAudioRef.current.currentTime = 0
      activeAudioRef.current = null
    }
  }, [ttsEnabled])

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="glass rounded-xl p-4 glow-border"
    >
      <div className="flex items-center gap-2 mb-3">
        <MessageSquare size={14} className="text-accent-teal" />
        <span className="text-xs font-semibold text-dark-200">Sentence Output</span>

        <div className="ml-auto flex items-center gap-1.5">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={handleTTS}
            className={`p-1 rounded-md transition-colors ${
              ttsEnabled
                ? 'bg-accent-teal/15 text-accent-teal'
                : 'bg-dark-700 text-dark-400'
            }`}
            title={ttsEnabled ? 'Disable TTS' : 'Enable TTS'}
          >
            {ttsEnabled ? <Volume2 size={12} /> : <VolumeX size={12} />}
          </motion.button>

          {!isPlaceholder && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => speakNow(sentence)}
              className="px-2 py-0.5 rounded-md bg-accent-teal/15 text-accent-teal text-[10px] font-medium hover:bg-accent-teal/25 transition-colors"
            >
              Speak
            </motion.button>
          )}
        </div>
      </div>

      <div className="min-h-[32px] flex items-center">
        <motion.p
          key={sentence}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className={`text-sm leading-relaxed ${
            isPlaceholder ? 'text-dark-400 italic' : 'text-white font-medium typing-cursor'
          }`}
        >
          {sentence || 'Open camera and start recognition...'}
        </motion.p>
      </div>

      {audioState === 'speaking' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-2 flex items-center gap-1.5"
        >
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              animate={{ height: [4, 10 + i * 1.3, 4] }}
              transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.08 }}
              className="w-1 bg-accent-teal rounded-full"
            />
          ))}
          <span className="text-[10px] text-accent-teal ml-1">Speaking...</span>
        </motion.div>
      )}
    </motion.div>
  )
}
