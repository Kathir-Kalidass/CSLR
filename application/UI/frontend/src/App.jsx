import { useEffect, useRef, useState } from 'react'

const BACKEND_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const initialResult = {
  gloss: [],
  sentence: '',
  confidence: 0,
  fps: 0,
  processing_time: 0,
}

export default function App() {
  const cameraVideoRef = useRef(null)
  const canvasRef = useRef(null)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [message, setMessage] = useState('Ready.')
  const [uploadFile, setUploadFile] = useState(null)
  const [uploadPreview, setUploadPreview] = useState('')
  const [uploadResult, setUploadResult] = useState(initialResult)
  const [cameraResult, setCameraResult] = useState(initialResult)
  const [uploadRunning, setUploadRunning] = useState(false)
  const [cameraRunning, setCameraRunning] = useState(false)
  const [cameraStream, setCameraStream] = useState(null)

  useEffect(() => {
    let active = true
    fetch(`${BACKEND_BASE}/health`)
      .then((res) => {
        if (!res.ok) throw new Error('Backend unavailable')
        if (active) setBackendStatus('connected')
      })
      .catch(() => {
        if (active) {
          setBackendStatus('unavailable')
          setMessage('Start the backend on port 8000 to run inference.')
        }
      })
    return () => {
      active = false
      if (uploadPreview) URL.revokeObjectURL(uploadPreview)
    }
  }, [uploadPreview])

  const speak = (text) => {
    if (!text || !window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'en-US'
    utterance.rate = 1.02
    utterance.pitch = 1
    window.speechSynthesis.speak(utterance)
  }

  const metricSource = cameraResult.sentence ? cameraResult : uploadResult

  const onSelectVideo = (event) => {
    const file = event.target.files?.[0]
    setUploadFile(file || null)
    setUploadResult(initialResult)
    if (uploadPreview) URL.revokeObjectURL(uploadPreview)
    if (file) {
      const nextUrl = URL.createObjectURL(file)
      setUploadPreview(nextUrl)
      setMessage(`Selected ${file.name}. Ready to generate caption.`)
    } else {
      setUploadPreview('')
    }
  }

  const runUploadInference = async () => {
    if (!uploadFile) return
    setUploadRunning(true)
    setMessage('Running video inference...')
    try {
      const formData = new FormData()
      formData.append('file', uploadFile)
      const response = await fetch(`${BACKEND_BASE}/api/v1/inference/video`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) throw new Error(await response.text())
      const result = await response.json()
      setUploadResult(result)
      setMessage('Video caption generated.')
      if (result.sentence) speak(result.sentence)
    } catch (error) {
      setMessage(`Video inference failed: ${error.message}`)
    } finally {
      setUploadRunning(false)
    }
  }

  const openCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      })
      setCameraStream(stream)
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = stream
      }
      setMessage('Camera opened. Use Generate From Camera when ready.')
    } catch {
      setMessage('Camera permission was denied or unavailable.')
    }
  }

  const closeCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop())
    }
    setCameraStream(null)
    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = null
    }
    setMessage('Camera closed.')
  }

  const captureFrames = async () => {
    const video = cameraVideoRef.current
    const canvas = canvasRef.current
    const width = video?.videoWidth || 640
    const height = video?.videoHeight || 360
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    const frames = []

    for (let i = 0; i < 24; i += 1) {
      ctx.drawImage(video, 0, 0, width, height)
      frames.push(canvas.toDataURL('image/jpeg', 0.8))
      await new Promise((resolve) => setTimeout(resolve, 80))
    }
    return frames
  }

  const runCameraInference = async () => {
    if (!cameraStream) return
    setCameraRunning(true)
    setMessage('Capturing camera frames and generating caption...')
    try {
      const frames = await captureFrames()
      const response = await fetch(`${BACKEND_BASE}/api/v1/inference/frames`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames, fps: 12 }),
      })
      if (!response.ok) throw new Error(await response.text())
      const result = await response.json()
      setCameraResult(result)
      setMessage('Camera caption generated.')
      if (result.sentence) speak(result.sentence)
    } catch (error) {
      setMessage(`Camera inference failed: ${error.message}`)
    } finally {
      setCameraRunning(false)
    }
  }

  return (
    <div className="app-shell">
      <main className="page">
        <header className="hero-card">
          <p className="eyebrow">Indian Sign Language Demo</p>
          <h1>Video or camera in, English caption and voice out</h1>
          <p className="hero-copy">
            Using <code>application/backend/checkpoints/isign_pose_only_npy/best.pt</code> through the backend inference API.
          </p>
          <div className="pill-row">
            <span className="pill">Model: pose_only_npy/best.pt</span>
            <span className="pill">Backend: {backendStatus}</span>
            <span className="pill">Voice: {'speechSynthesis' in window ? 'browser TTS' : 'not supported'}</span>
          </div>
        </header>

        <section className="two-col">
          <article className="panel">
            <div className="panel-head">
              <div>
                <h2>Upload Video</h2>
                <p>Upload a video and generate gloss, caption, and English speech.</p>
              </div>
              <label className="primary-btn">
                <input type="file" accept="video/*" hidden onChange={onSelectVideo} />
                Choose Video
              </label>
            </div>

            <div className="preview-shell">
              {uploadPreview ? (
                <video src={uploadPreview} controls playsInline className="preview-video" />
              ) : (
                <div className="placeholder">No video selected yet</div>
              )}
            </div>

            <div className="button-row">
              <button className="primary-btn" onClick={runUploadInference} disabled={!uploadFile || uploadRunning}>
                {uploadRunning ? 'Generating...' : 'Generate Caption'}
              </button>
              <button className="secondary-btn" onClick={() => speak(uploadResult.sentence)} disabled={!uploadResult.sentence}>
                Play Voice
              </button>
            </div>
          </article>

          <article className="panel">
            <div className="panel-head">
              <div>
                <h2>Use Camera</h2>
                <p>Allow camera access, capture live frames, then generate caption and voice.</p>
              </div>
            </div>

            <div className="preview-shell">
              <video ref={cameraVideoRef} autoPlay muted playsInline className="preview-video" />
              <canvas ref={canvasRef} className="hidden" />
            </div>

            <div className="button-row">
              <button className="primary-btn" onClick={openCamera} disabled={!!cameraStream}>Open Camera</button>
              <button className="primary-btn" onClick={runCameraInference} disabled={!cameraStream || cameraRunning}>
                {cameraRunning ? 'Generating...' : 'Generate From Camera'}
              </button>
              <button className="secondary-btn" onClick={closeCamera} disabled={!cameraStream}>Close Camera</button>
              <button className="secondary-btn" onClick={() => speak(cameraResult.sentence)} disabled={!cameraResult.sentence}>
                Play Voice
              </button>
            </div>
          </article>
        </section>

        <section className="two-col results">
          <article className="result-card">
            <p className="result-label">Upload Result</p>
            <div className="result-box">
              <p className="mini-label">Gloss Tokens</p>
              <p className="result-text">{uploadResult.gloss?.join(' ') || 'Waiting for uploaded video.'}</p>
            </div>
            <div className="result-box">
              <p className="mini-label">Caption</p>
              <p className="caption-text">{uploadResult.sentence || 'Generated English text will appear here.'}</p>
            </div>
          </article>

          <article className="result-card">
            <p className="result-label">Camera Result</p>
            <div className="result-box">
              <p className="mini-label">Gloss Tokens</p>
              <p className="result-text">{cameraResult.gloss?.join(' ') || 'Waiting for camera input.'}</p>
            </div>
            <div className="result-box">
              <p className="mini-label">Caption</p>
              <p className="caption-text">{cameraResult.sentence || 'Generated English text will appear here.'}</p>
            </div>
          </article>
        </section>

        <section className="footer-card">
          <div className="pill-row">
            <span className="pill">Latency: {((metricSource.processing_time || 0) * 1000).toFixed(0)} ms</span>
            <span className="pill">Confidence: {((metricSource.confidence || 0) * 100).toFixed(1)}%</span>
            <span className="pill">FPS: {(metricSource.fps || 0).toFixed(1)}</span>
          </div>
          <p className="footer-copy">{message}</p>
        </section>
      </main>
    </div>
  )
}
