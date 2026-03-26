import { create } from 'zustand'

const useStore = create((set, get) => ({
  // Connection
  connected: false,
  setConnected: (v) => set({ connected: v }),

  // WebSocket payload data
  payload: null,
  setPayload: (p) => set({ payload: p }),

  // Controls
  running: false,
  ttsEnabled: true,
  cameraActive: false,
  grandMode: false,

  setRunning: (v) => set({ running: v }),
  setTtsEnabled: (v) => set({ ttsEnabled: v }),
  setCameraActive: (v) => set({ cameraActive: v }),
  setGrandMode: (v) => set({ grandMode: v }),

  // Camera
  cameraStream: null,
  setCameraStream: (s) => set({ cameraStream: s }),

  // System info
  systemInfo: null,
  setSystemInfo: (info) => set({ systemInfo: info }),

  // WebSocket ref
  wsRef: { current: null },

  // Derived selectors
  get status() { return get().payload?.status ?? 'idle' },
  get inferenceMode() { return get().payload?.inference_mode ?? 'simulated' },

  // Actions that send WS messages
  sendControl: (action, extra = {}) => {
    const ws = get().wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'control', action, ...extra }))
    }
  },

  sendFrame: (base64, seq) => {
    const ws = get().wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'client_video_frame',
        image_jpeg_base64: base64,
        frame_seq: seq,
      }))
    }
  },

  sendVideoStats: (stats) => {
    const ws = get().wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'client_video_stats', ...stats }))
    }
  },

  // Metrics history for charts
  metricsHistory: [],
  pushMetrics: (m) => set((s) => ({
    metricsHistory: [...s.metricsHistory.slice(-59), {
      ...m,
      t: Date.now(),
    }],
  })),

  // Console log
  consoleLogs: [],
  pushConsoleLogs: (lines) => set({
    consoleLogs: lines.slice(-80),
  }),
}))

export default useStore
