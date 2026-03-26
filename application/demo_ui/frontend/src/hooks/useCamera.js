import { useEffect, useRef, useCallback, useState } from 'react'
import useStore from '../store/useStore'

/* ─── COCO-17 skeleton connections ─── */
const SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],       // face
  [5, 6],                                  // shoulders
  [5, 7], [7, 9],                          // left arm
  [6, 8], [8, 10],                         // right arm
  [5, 11], [6, 12],                        // torso
  [11, 12],                                // hips
  [11, 13], [13, 15],                      // left leg
  [12, 14], [14, 16],                      // right leg
]

const HAND_INDICES = [9, 10]  // wrists — highlight for sign language
const BODY_COLORS = { joint: '#22d3ee', bone: 'rgba(34,211,238,0.6)', hand: '#a855f7' }
const POSE_COLORS = { joint: '#c084fc', bone: 'rgba(168,85,247,0.6)', hand: '#f472b6' }

/* ─── MediaPipe Hand 21-landmark connections ─── */
const HAND_SKELETON = [
  // Thumb
  [0,1],[1,2],[2,3],[3,4],
  // Index
  [0,5],[5,6],[6,7],[7,8],
  // Middle
  [0,9],[9,10],[10,11],[11,12],
  // Ring
  [0,13],[13,14],[14,15],[15,16],
  // Pinky
  [0,17],[17,18],[18,19],[19,20],
  // Palm connections
  [5,9],[9,13],[13,17],
]
const FINGERTIP_INDICES = [4, 8, 12, 16, 20]
const HAND_COLORS = {
  Left:  { bone: 'rgba(251,146,60,0.8)',  joint: '#fb923c', tip: '#f97316' },
  Right: { bone: 'rgba(244,114,182,0.8)', joint: '#f472b6', tip: '#ec4899' },
  Unknown: { bone: 'rgba(250,204,21,0.8)', joint: '#facc15', tip: '#eab308' },
}
const POSE_HAND_COLORS = {
  Left:  { bone: 'rgba(251,191,36,0.8)',  joint: '#fbbf24', tip: '#f59e0b' },
  Right: { bone: 'rgba(236,72,153,0.8)',  joint: '#ec4899', tip: '#db2777' },
  Unknown: { bone: 'rgba(163,230,53,0.8)', joint: '#a3e635', tip: '#84cc16' },
}

/* ─── Face indices in COCO-17: 0=nose 1=Leye 2=Reye 3=Lear 4=Rear ─── */
const FACE_INDICES = [0, 1, 2, 3, 4]

/* ========== TOGGLE: set to false to disable face mask ========== */
const FACE_MASK_ENABLED = false
/* =============================================================== */

/**
 * Compute face center from COCO-17 landmarks.
 * Returns {cx, cy, r} in pixel space, or null if no face visible.
 */
function computeFaceRegion(landmarks, width, height) {
  if (!FACE_MASK_ENABLED) return null
  if (!landmarks || landmarks.length < 5) return null
  let sumX = 0, sumY = 0, count = 0
  for (const i of FACE_INDICES) {
    const [x, y] = landmarks[i]
    if (x === 0 && y === 0) continue
    sumX += x * width
    sumY += y * height
    count++
  }
  if (count < 2) return null
  const cx = sumX / count
  const cy = sumY / count
  let maxDist = 0
  for (const i of FACE_INDICES) {
    const [x, y] = landmarks[i]
    if (x === 0 && y === 0) continue
    const dx = x * width - cx
    const dy = y * height - cy
    maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy))
  }
  const r = Math.max(maxDist * 1.5, 18)
  return { cx, cy, r }
}

/**
 * Draw a cartoon man emoji over the face — no circle, no background.
 */
function drawFaceMask(ctx, face) {
  if (!face) return
  const { cx, cy, r } = face
  ctx.save()
  const emojiSize = Math.max(20, r * 2.0 | 0)
  ctx.font = `${emojiSize}px sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('\u{1F468}', cx, cy)
  ctx.restore()
}

/**
 * Draw hand/finger landmarks for one hand.
 * hand: { landmarks: [[x,y],...], label: "Left"|"Right" }
 */
function drawHandLandmarks(ctx, hand, width, height, colorSet, dotSize = 3) {
  const pts = hand?.landmarks
  if (!pts || pts.length < 21) return
  const colors = colorSet[hand.label] || colorSet.Unknown

  // Draw bones
  ctx.lineWidth = 2
  for (const [a, b] of HAND_SKELETON) {
    const [ax, ay] = pts[a]
    const [bx, by] = pts[b]
    if (ax === 0 && ay === 0) continue
    if (bx === 0 && by === 0) continue
    ctx.strokeStyle = colors.bone
    ctx.beginPath()
    ctx.moveTo(ax * width, ay * height)
    ctx.lineTo(bx * width, by * height)
    ctx.stroke()
  }

  // Draw joints
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i]
    if (x === 0 && y === 0) continue
    const isTip = FINGERTIP_INDICES.includes(i)
    ctx.fillStyle = isTip ? colors.tip : colors.joint
    ctx.beginPath()
    ctx.arc(x * width, y * height, isTip ? dotSize + 2 : dotSize, 0, Math.PI * 2)
    ctx.fill()
    // Glow ring on fingertips
    if (isTip) {
      ctx.strokeStyle = colors.tip
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(x * width, y * height, dotSize + 4, 0, Math.PI * 2)
      ctx.stroke()
    }
  }

  // Label
  const wrist = pts[0]
  if (wrist[0] > 0 || wrist[1] > 0) {
    ctx.fillStyle = colors.joint
    ctx.font = 'bold 9px monospace'
    ctx.fillText(hand.label === 'Left' ? 'L' : hand.label === 'Right' ? 'R' : '?', wrist[0] * width - 3, wrist[1] * height - 8)
  }
}

/**
 * Draw all detected hands on a canvas.
 */
function drawAllHands(ctx, hands, width, height, colorSet, dotSize = 3) {
  if (!hands || !Array.isArray(hands)) return
  for (const hand of hands) {
    drawHandLandmarks(ctx, hand, width, height, colorSet, dotSize)
  }
}

/**
 * Draw 17-keypoint skeleton on a canvas.
 * landmarks: [[x,y], ...] in 0-1 space
 */
function drawLandmarks(ctx, landmarks, width, height, colors, dotSize = 4) {
  if (!landmarks || landmarks.length < 17) return

  // Draw bones
  ctx.lineWidth = 2.5
  for (const [a, b] of SKELETON) {
    const [ax, ay] = landmarks[a]
    const [bx, by] = landmarks[b]
    if (ax === 0 && ay === 0) continue
    if (bx === 0 && by === 0) continue
    ctx.strokeStyle = colors.bone
    ctx.beginPath()
    ctx.moveTo(ax * width, ay * height)
    ctx.lineTo(bx * width, by * height)
    ctx.stroke()
  }

  // Draw joints
  for (let i = 0; i < landmarks.length; i++) {
    const [x, y] = landmarks[i]
    if (x === 0 && y === 0) continue
    const isHand = HAND_INDICES.includes(i)
    ctx.fillStyle = isHand ? colors.hand : colors.joint
    ctx.beginPath()
    ctx.arc(x * width, y * height, isHand ? dotSize + 2 : dotSize, 0, Math.PI * 2)
    ctx.fill()
    // Glow ring on hands
    if (isHand) {
      ctx.strokeStyle = colors.hand
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(x * width, y * height, dotSize + 5, 0, Math.PI * 2)
      ctx.stroke()
    }
  }
}

/* ─── Overlay: crosshair + attention bars + LIVE pose landmarks ─── */
function drawOverlay(ctx, width, height, confidence, attention, landmarks, handLandmarks) {
  ctx.clearRect(0, 0, width, height)

  // Outer frame
  ctx.strokeStyle = 'rgba(34, 211, 238, 0.18)'
  ctx.lineWidth = 1.5
  ctx.strokeRect(6, 6, width - 12, height - 12)

  // Privacy: cover face
  const face = computeFaceRegion(landmarks, width, height)
  drawFaceMask(ctx, face)

  // Draw real pose landmarks
  drawLandmarks(ctx, landmarks, width, height, BODY_COLORS, 5)

  // Draw hand/finger landmarks
  drawAllHands(ctx, handLandmarks, width, height, HAND_COLORS, 4)

  // Attention bars at bottom
  const rgb = Math.max(0, Math.min(1, attention.rgb ?? 0.5))
  const pose = Math.max(0, Math.min(1, attention.pose ?? 0.5))
  ctx.fillStyle = 'rgba(15, 23, 42, 0.55)'
  ctx.fillRect(8, height - 22, width - 16, 14)
  ctx.fillStyle = 'rgba(34, 211, 238, 0.75)'
  ctx.fillRect(8, height - 22, (width - 16) * rgb, 6)
  ctx.fillStyle = 'rgba(168, 85, 247, 0.75)'
  ctx.fillRect(8, height - 14, (width - 16) * pose, 6)

  // Landmark count badge
  const n = landmarks?.length ?? 0
  const nh = handLandmarks?.reduce((s, h) => s + (h?.landmarks?.length ?? 0), 0) ?? 0
  if (n > 0 || nh > 0) {
    ctx.fillStyle = 'rgba(34, 211, 238, 0.55)'
    ctx.font = 'bold 10px monospace'
    ctx.fillText(`${n}+${nh} pts`, width - 75, 18)
  }
}

/* ─── RGB preview: enhanced color + attention heatmap ─── */
function drawRgbPreview(ctx, sourceCanvas, width, height, rgbAttention, landmarks) {
  ctx.clearRect(0, 0, width, height)
  ctx.filter = 'contrast(1.15) saturate(1.25)'
  ctx.drawImage(sourceCanvas, 0, 0, width, height)
  ctx.filter = 'none'

  const grad = ctx.createRadialGradient(
    width * 0.5, height * 0.5, 0,
    width * 0.5, height * 0.5, width * 0.48,
  )
  grad.addColorStop(0, `rgba(34, 211, 238, ${0.12 + rgbAttention * 0.25})`)
  grad.addColorStop(0.5, `rgba(34, 211, 238, ${0.04 + rgbAttention * 0.08})`)
  grad.addColorStop(1, 'rgba(34, 211, 238, 0)')
  ctx.globalCompositeOperation = 'screen'
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, width, height)
  ctx.globalCompositeOperation = 'source-over'

  ctx.fillStyle = 'rgba(0, 0, 0, 0.06)'
  for (let y = 0; y < height; y += 3) {
    ctx.fillRect(0, y, width, 1)
  }
  // Privacy: cover face
  const face = computeFaceRegion(landmarks, width, height)
  drawFaceMask(ctx, face)

  ctx.fillStyle = 'rgba(34, 211, 238, 0.65)'
  ctx.font = 'bold 10px monospace'
  ctx.fillText('RGB · ENHANCED', 6, height - 8)
}

/* ─── Pose preview: dark skeleton view with landmarks ─── */
function drawPosePreview(ctx, sourceCanvas, width, height, confidence, landmarks, handLandmarks) {
  ctx.clearRect(0, 0, width, height)

  // Dark purple-tinted silhouette
  ctx.filter = 'grayscale(1) brightness(0.25) contrast(1.2)'
  ctx.drawImage(sourceCanvas, 0, 0, width, height)
  ctx.filter = 'none'

  ctx.globalCompositeOperation = 'color'
  ctx.fillStyle = '#7c3aed'
  ctx.fillRect(0, 0, width, height)
  ctx.globalCompositeOperation = 'source-over'

  // Grid
  ctx.strokeStyle = 'rgba(168, 85, 247, 0.07)'
  ctx.lineWidth = 0.5
  for (let x = 20; x < width; x += 20) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke()
  }
  for (let y = 20; y < height; y += 20) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke()
  }

  // Privacy: cover face
  const face = computeFaceRegion(landmarks, width, height)
  drawFaceMask(ctx, face)

  // Draw real pose landmarks
  drawLandmarks(ctx, landmarks, width, height, POSE_COLORS, 4)

  // Draw hand/finger landmarks
  drawAllHands(ctx, handLandmarks, width, height, POSE_HAND_COLORS, 3)

  // Labels
  ctx.fillStyle = 'rgba(168, 85, 247, 0.7)'
  ctx.font = 'bold 10px monospace'
  ctx.fillText('POSE · SKELETON', 6, height - 8)
  if (confidence > 0) {
    ctx.fillText(`${(confidence * 100).toFixed(0)}%`, width - 34, height - 8)
  }
}

export default function useCamera() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const overlayRef = useRef(null)
  const rgbPreviewRef = useRef(null)
  const posePreviewRef = useRef(null)
  const captureIntervalRef = useRef(null)
  const drawRafRef = useRef(0)
  const frameSeq = useRef(0)
  const [cameraError, setCameraError] = useState('')

  const running = useStore((s) => s.running)
  const cameraActive = useStore((s) => s.cameraActive)
  const connected = useStore((s) => s.connected)
  const setCameraActive = useStore((s) => s.setCameraActive)
  const setRunning = useStore((s) => s.setRunning)
  const sendControl = useStore((s) => s.sendControl)
  const sendFrame = useStore((s) => s.sendFrame)
  const sendVideoStats = useStore((s) => s.sendVideoStats)
  const payload = useStore((s) => s.payload)

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera API unavailable in this browser')
      setCameraActive(false)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      })
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await video.play()
      setCameraError('')
      setCameraActive(true)
      frameSeq.current = 0

      const settings = stream.getVideoTracks()[0]?.getSettings?.() || {}
      sendVideoStats({
        camera_active: true,
        frame_hint: frameSeq.current,
        resolution: `${settings.width || 0}x${settings.height || 0}`,
      })
      if (connected) {
        sendControl('start')
        setRunning(true)
      }
    } catch (err) {
      const msg = err?.name === 'NotAllowedError'
        ? 'Camera permission denied. Allow webcam and retry.'
        : 'Unable to start camera stream.'
      setCameraError(msg)
      setCameraActive(false)
    }
  }, [connected, sendControl, sendVideoStats, setCameraActive, setRunning])

  useEffect(() => {
    if (!cameraActive || !connected) return

    const video = videoRef.current
    const stream = video?.srcObject
    const settings = stream?.getVideoTracks?.()?.[0]?.getSettings?.() || {}

    sendVideoStats({
      camera_active: true,
      frame_hint: frameSeq.current,
      resolution: `${settings.width || 320}x${settings.height || 240}`,
    })
  }, [cameraActive, connected, sendVideoStats])

  const stopCamera = useCallback(() => {
    const video = videoRef.current
    if (video?.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop())
      video.srcObject = null
    }
    if (connected) sendControl('stop')
    setRunning(false)
    setCameraActive(false)
    sendVideoStats({ camera_active: false, frame_hint: frameSeq.current, resolution: 'none' })

    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
    if (drawRafRef.current) {
      cancelAnimationFrame(drawRafRef.current)
      drawRafRef.current = 0
    }
  }, [connected, sendControl, sendVideoStats, setCameraActive, setRunning])

  useEffect(() => {
    if (!(running && cameraActive)) {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current)
        captureIntervalRef.current = null
      }
      return
    }

    captureIntervalRef.current = setInterval(() => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2) return

      const w = 640
      const h = 480
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.drawImage(video, 0, 0, w, h)
      const jpeg = canvas.toDataURL('image/jpeg', 0.72)
      const b64 = jpeg.split(',')[1]

      frameSeq.current += 1
      sendFrame(b64, frameSeq.current)
    }, 220)

    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current)
        captureIntervalRef.current = null
      }
    }
  }, [running, cameraActive, sendFrame, sendVideoStats])

  useEffect(() => {
    if (!cameraActive) return

    const loop = () => {
      const video = videoRef.current
      const source = canvasRef.current
      const overlay = overlayRef.current
      const rgbCanvas = rgbPreviewRef.current
      const poseCanvas = posePreviewRef.current

      if (video && source && video.readyState >= 2) {
        if (source.width !== 320 || source.height !== 240) {
          source.width = 320
          source.height = 240
        }
        const srcCtx = source.getContext('2d')
        if (srcCtx) srcCtx.drawImage(video, 0, 0, source.width, source.height)
      }

      const confidence = Math.max(0, Math.min(1, payload?.confidence ?? 0))
      const attention = payload?.attention ?? { rgb: 0.5, pose: 0.5 }
      const landmarks = payload?.pose_landmarks ?? []
      const handLandmarks = payload?.hand_landmarks ?? []

      if (overlay) {
        const ow = overlay.clientWidth || 640
        const oh = overlay.clientHeight || 360
        if (overlay.width !== ow || overlay.height !== oh) {
          overlay.width = ow
          overlay.height = oh
        }
        const octx = overlay.getContext('2d')
        if (octx) drawOverlay(octx, overlay.width, overlay.height, confidence, attention, landmarks, handLandmarks)
      }

      if (rgbCanvas && source) {
        const rw = rgbCanvas.clientWidth || 320
        const rh = rgbCanvas.clientHeight || 240
        if (rgbCanvas.width !== rw || rgbCanvas.height !== rh) {
          rgbCanvas.width = rw
          rgbCanvas.height = rh
        }
        const rctx = rgbCanvas.getContext('2d')
        if (rctx) drawRgbPreview(rctx, source, rgbCanvas.width, rgbCanvas.height, attention.rgb ?? 0.5, landmarks)
      }

      if (poseCanvas && source) {
        const pw = poseCanvas.clientWidth || 320
        const ph = poseCanvas.clientHeight || 240
        if (poseCanvas.width !== pw || poseCanvas.height !== ph) {
          poseCanvas.width = pw
          poseCanvas.height = ph
        }
        const pctx = poseCanvas.getContext('2d')
        if (pctx) drawPosePreview(pctx, source, poseCanvas.width, poseCanvas.height, confidence, landmarks, handLandmarks)
      }

      drawRafRef.current = requestAnimationFrame(loop)
    }

    drawRafRef.current = requestAnimationFrame(loop)
    return () => {
      if (drawRafRef.current) {
        cancelAnimationFrame(drawRafRef.current)
        drawRafRef.current = 0
      }
    }
  }, [cameraActive, payload])

  useEffect(() => () => stopCamera(), [stopCamera])

  return {
    videoRef,
    canvasRef,
    overlayRef,
    rgbPreviewRef,
    posePreviewRef,
    startCamera,
    stopCamera,
    cameraActive,
    cameraError,
  }
}
