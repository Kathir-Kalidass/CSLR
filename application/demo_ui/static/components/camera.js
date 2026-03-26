import { now } from "../core/utils.js";

function drawGrid(ctx, w, h) {
  ctx.save();
  ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 42) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSkeleton(ctx, points) {
  ctx.save();
  ctx.strokeStyle = "rgba(34, 211, 238, 0.86)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((pt, idx) => {
    if (idx === 0) {
      ctx.moveTo(pt.x, pt.y);
    } else {
      ctx.lineTo(pt.x, pt.y);
    }
  });
  ctx.stroke();

  points.forEach((pt, idx) => {
    const radius = idx % 3 === 0 ? 5 : 3;
    ctx.beginPath();
    ctx.fillStyle = idx % 2 === 0 ? "rgba(134, 239, 172, 0.94)" : "rgba(96, 165, 250, 0.95)";
    ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

function ensureCanvasSize(canvas, width, height) {
  if (!canvas) return;
  const w = Math.max(1, width);
  const h = Math.max(1, height);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

export function createCameraController({ state, refs, updateControlState, showPopup, sendVideoStats }) {
  function startFrameStreaming() {
    if (state.frameSendTimer) {
      return;
    }
    if (!state.frameCanvas) {
      state.frameCanvas = document.createElement("canvas");
      state.frameCanvas.width = 320;
      state.frameCanvas.height = 180;
    }
    const canvas = state.frameCanvas;
    const ctx = canvas.getContext("2d");

    state.frameSendTimer = setInterval(() => {
      if (!state.cameraActive || !refs.liveVideo.videoWidth || !refs.liveVideo.videoHeight) {
        return;
      }
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        return;
      }

      ctx.drawImage(refs.liveVideo, 0, 0, canvas.width, canvas.height);
      const jpegData = canvas.toDataURL("image/jpeg", 0.55).split(",")[1];
      state.frameSeq += 1;

      state.ws.send(
        JSON.stringify({
          type: "client_video_frame",
          frame_seq: state.frameSeq,
          image_jpeg_base64: jpegData,
          width: canvas.width,
          height: canvas.height,
        }),
      );
    }, 220);
  }

  function stopFrameStreaming() {
    if (state.frameSendTimer) {
      clearInterval(state.frameSendTimer);
      state.frameSendTimer = null;
    }
  }

  function startOverlayLoop() {
    const video = refs.liveVideo;
    const canvas = refs.overlayCanvas;
    const ctx = canvas.getContext("2d");
    const rgbCanvas = refs.rgbPreviewCanvas;
    const poseCanvas = refs.posePreviewCanvas;
    const rgbCtx = rgbCanvas ? rgbCanvas.getContext("2d") : null;
    const poseCtx = poseCanvas ? poseCanvas.getContext("2d") : null;
    const trails = [];

    const render = () => {
      if (!state.cameraActive || !video.videoWidth || !video.videoHeight) {
        state.overlayRaf = requestAnimationFrame(render);
        return;
      }

      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ensureCanvasSize(canvas, w, h);
      ensureCanvasSize(rgbCanvas, rgbCanvas?.clientWidth || 240, rgbCanvas?.clientHeight || 130);
      ensureCanvasSize(poseCanvas, poseCanvas?.clientWidth || 240, poseCanvas?.clientHeight || 130);

      ctx.clearRect(0, 0, w, h);
      drawGrid(ctx, w, h);

      const t = now() / 360;
      const cx = w * (0.5 + Math.sin(t * 1.7) * 0.18);
      const cy = h * (0.55 + Math.cos(t * 1.4) * 0.12);

      const points = [];
      for (let i = 0; i < 12; i += 1) {
        const angle = t + i * 0.5;
        const radius = 30 + i * 2.8;
        points.push({
          x: cx + Math.cos(angle) * radius,
          y: cy + Math.sin(angle * 1.1) * radius,
        });
      }

      trails.push(points.slice(0, 5));
      while (trails.length > 10) {
        trails.shift();
      }

      trails.forEach((trail, idx) => {
        const alpha = (idx + 1) / (trails.length * 1.4);
        ctx.strokeStyle = `rgba(56, 217, 255, ${alpha})`;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        trail.forEach((pt, i) => {
          if (i === 0) {
            ctx.moveTo(pt.x, pt.y);
          } else {
            ctx.lineTo(pt.x, pt.y);
          }
        });
        ctx.stroke();
      });

      drawSkeleton(ctx, points);

      if (rgbCtx && rgbCanvas) {
        rgbCtx.clearRect(0, 0, rgbCanvas.width, rgbCanvas.height);
        rgbCtx.drawImage(video, 0, 0, rgbCanvas.width, rgbCanvas.height);
      }

      if (poseCtx && poseCanvas) {
        poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
        drawGrid(poseCtx, poseCanvas.width, poseCanvas.height);
        const sx = poseCanvas.width / w;
        const sy = poseCanvas.height / h;
        const scaled = points.map((pt) => ({ x: pt.x * sx, y: pt.y * sy }));
        drawSkeleton(poseCtx, scaled);
      }

      state.frameCounter += 1;
      state.overlayRaf = requestAnimationFrame(render);
    };

    state.overlayRaf = requestAnimationFrame(render);
  }

  async function openCamera() {
    if (state.cameraActive) {
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showPopup("Camera Error", "<p>This browser does not support webcam access.</p>");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      state.stream = stream;
      refs.liveVideo.srcObject = stream;
      await refs.liveVideo.play();
      state.cameraActive = true;
      refs.cameraShell.classList.add("pulsing");
      updateControlState();
      startOverlayLoop();
      startFrameStreaming();
      sendVideoStats();
    } catch (error) {
      showPopup("Camera Permission", `<p>Webcam access failed: ${error?.message || "Unknown error"}</p>`);
    }
  }

  function closeCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
    }
    state.stream = null;
    state.cameraActive = false;
    stopFrameStreaming();
    refs.cameraShell.classList.remove("pulsing");
    cancelAnimationFrame(state.overlayRaf);
    state.overlayRaf = 0;

    const ctx = refs.overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, refs.overlayCanvas.width, refs.overlayCanvas.height);

    updateControlState();
    sendVideoStats();
  }

  return {
    openCamera,
    closeCamera,
    stopFrameStreaming,
  };
}
