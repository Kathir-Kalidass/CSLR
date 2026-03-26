import { useEffect, useRef } from 'react'
import useStore from '../store/useStore'

export default function useWebSocket() {
  const setConnected = useStore((s) => s.setConnected)
  const setPayload = useStore((s) => s.setPayload)
  const setRunning = useStore((s) => s.setRunning)
  const setTtsEnabled = useStore((s) => s.setTtsEnabled)
  const setGrandMode = useStore((s) => s.setGrandMode)
  const pushMetrics = useStore((s) => s.pushMetrics)
  const pushConsoleLogs = useStore((s) => s.pushConsoleLogs)
  const wsRef = useStore((s) => s.wsRef)
  const reconnectTimer = useRef(null)

  useEffect(() => {
    let alive = true

    function connect() {
      if (!alive) return
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const explicitWsUrl = import.meta.env.VITE_WS_URL
      const ws = new WebSocket(explicitWsUrl || `${protocol}://${window.location.host}/ws/demo`)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current)
          reconnectTimer.current = null
        }
      }

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data)
          setPayload(data)

          const control = data?.control_state ?? {}
          if (typeof control.running === 'boolean') setRunning(control.running)
          if (typeof control.tts_enabled === 'boolean') setTtsEnabled(control.tts_enabled)
          if (typeof control.grand_mode === 'boolean') setGrandMode(control.grand_mode)

          // Push metrics for dashboard charts
          if (data.status === 'active') {
            pushMetrics({
              fps: data.fps ?? 0,
              latency: data.latency_ms ?? 0,
              confidence: data.confidence ?? 0,
              wer: data.metrics?.wer_proxy ?? 0,
              bleu: data.metrics?.bleu_proxy ?? 0,
            })
          }

          // Push console logs
          if (data.parser_console?.length) pushConsoleLogs(data.parser_console)
        } catch { /* ignore parse errors */ }
      }

      ws.onclose = () => {
        setConnected(false)
        wsRef.current = null
        if (alive) {
          reconnectTimer.current = setTimeout(connect, 2000)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      alive = false
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [pushConsoleLogs, pushMetrics, setConnected, setGrandMode, setPayload, setRunning, setTtsEnabled, wsRef])
}
