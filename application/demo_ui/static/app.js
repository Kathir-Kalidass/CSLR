gsap.registerPlugin(TextPlugin);

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const latencyPill = document.getElementById("latencyPill");
const fpsPill = document.getElementById("fpsPill");
const accPill = document.getElementById("accPill");
const werPill = document.getElementById("werPill");
const bleuPill = document.getElementById("bleuPill");

const glossEl = document.getElementById("gloss");
const sentenceEl = document.getElementById("sentence");
const confBar = document.getElementById("confBar");
const confText = document.getElementById("confText");
const audioState = document.getElementById("audioState");
const parserConsole = document.getElementById("parserConsole");
const historyList = document.getElementById("historyList");

const flowDot = document.getElementById("flowDot");
const flowLane = document.getElementById("flowLane");

const cameraShell = document.getElementById("cameraShell");
const cameraBtn = document.getElementById("cameraBtn");
const cam = document.getElementById("cam");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const ttsBtn = document.getElementById("ttsBtn");

let stream = null;
let flowTween = null;
let wsRef = null;
let frameHintCounter = 0;
let statsTimer = null;

function sendControl(action) {
  if (!wsRef || wsRef.readyState !== WebSocket.OPEN) return;
  wsRef.send(JSON.stringify({ type: "control", action }));
}

startBtn.addEventListener("click", () => sendControl("start"));
stopBtn.addEventListener("click", () => sendControl("stop"));
clearBtn.addEventListener("click", () => sendControl("clear"));
ttsBtn.addEventListener("click", () => sendControl("toggle_tts"));

async function toggleCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
    cam.srcObject = null;
    cameraBtn.textContent = "Start Camera";
    cameraShell.classList.remove("pulsing");
    sendClientStats();
    if (statsTimer) {
      clearInterval(statsTimer);
      statsTimer = null;
    }
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cam.srcObject = stream;
    cameraBtn.textContent = "Stop Camera";
    cameraShell.classList.add("pulsing");
    if (!statsTimer) {
      statsTimer = setInterval(() => {
        frameHintCounter += 1;
        sendClientStats();
      }, 400);
    }
    sendClientStats();
  } catch (_) {
    cameraBtn.textContent = "Camera Blocked";
  }
}

cameraBtn.addEventListener("click", toggleCamera);

function animateCard(moduleKey) {
  const card = document.querySelector(`[data-module="${moduleKey}"]`);
  if (!card) return;
  card.classList.add("is-hot");
  gsap.fromTo(
    card,
    { boxShadow: "0 0 0px rgba(34,211,238,0)", y: 0 },
    {
      boxShadow: "0 0 26px rgba(34,211,238,0.24)",
      y: -2,
      duration: 0.32,
      yoyo: true,
      repeat: 1,
      ease: "power2.out",
      onComplete: () => card.classList.remove("is-hot"),
    }
  );
}

function animateTextSwap(el, text, duration = 0.65) {
  gsap.to(el, {
    duration,
    text,
    ease: "power2.out",
  });
}

function renderParseLines(container, lines) {
  if (!container) return;
  const safeLines = Array.isArray(lines) ? lines : [];
  container.innerHTML = safeLines.map((line) => `<div>- ${line}</div>`).join("");
  gsap.fromTo(container, { opacity: 0.35 }, { opacity: 1, duration: 0.35, ease: "power1.out" });
}

function updateModule(key, payload) {
  const card = document.querySelector(`[data-module="${key}"]`);
  if (!card || !payload) return;

  card.querySelector(".module-title").textContent = payload.title || key;
  card.querySelector(".in").textContent = payload.input || "--";
  card.querySelector(".proc").textContent = payload.process || "--";
  card.querySelector(".out").textContent = payload.output || "--";
  card.querySelector(".module-note").textContent = payload.note || "";

  renderParseLines(card.querySelector(".parser-lines"), payload.parse);
  animateCard(key);
}

function appendConsole(lines) {
  if (!Array.isArray(lines) || lines.length === 0) return;
  const html = lines.map((line) => `<div>${line}</div>`).join("");
  parserConsole.innerHTML = html;
  parserConsole.scrollTop = parserConsole.scrollHeight;
}

function setConnected(connected) {
  if (connected) {
    statusDot.classList.add("active");
    statusText.textContent = "Realtime stream active";
  } else {
    statusDot.classList.remove("active");
    statusText.textContent = "Disconnected - retrying";
  }
}

function setActiveStage(stageKey) {
  document.querySelectorAll(".flow-stage").forEach((el) => {
    el.classList.toggle("active", el.dataset.stage === stageKey);
  });
}

function animateFlowDot(stageKey) {
  const targetStage = document.querySelector(`.flow-stage[data-stage="${stageKey}"]`);
  if (!targetStage || !flowLane) return;

  const laneRect = flowLane.getBoundingClientRect();
  const stageRect = targetStage.getBoundingClientRect();
  const x = Math.max(0, stageRect.left - laneRect.left + stageRect.width / 2 - 8);

  if (flowTween) flowTween.kill();
  flowTween = gsap.to(flowDot, {
    x,
    duration: 0.48,
    ease: "power3.out",
  });
}

function renderHistory(history) {
  const rows = Array.isArray(history) ? history : [];
  if (rows.length === 0) {
    historyList.innerHTML = '<div class="history-item">No transcript yet.</div>';
    return;
  }
  historyList.innerHTML = rows
    .map((line, idx) => `<div class="history-item"><span class="idx">${idx + 1}</span>${line}</div>`)
    .join("");
}

function updateControls(controlState) {
  if (!controlState) return;
  const running = !!controlState.running;
  startBtn.disabled = running;
  stopBtn.disabled = !running;

  ttsBtn.textContent = controlState.tts_enabled ? "TTS: ON" : "TTS: OFF";
  ttsBtn.classList.toggle("ctrl-tts-off", !controlState.tts_enabled);
}

function updateMainPanels(msg) {
  animateTextSwap(glossEl, msg.partial_gloss || "--", 0.45);
  animateTextSwap(sentenceEl, msg.final_sentence || "Waiting for model output...", 0.5);

  const confPct = Math.round((msg.confidence || 0) * 100);
  gsap.to(confBar, {
    width: `${confPct}%`,
    duration: 0.6,
    ease: "power2.out",
  });
  confText.textContent = `${confPct}% confidence`;

  audioState.textContent = msg.audio_state || "idle";

  latencyPill.textContent = `Latency: ${msg.latency_ms} ms`;
  fpsPill.textContent = `FPS: ${msg.fps}`;

  if (msg.metrics) {
    werPill.textContent = `WER: ${(msg.metrics.wer_proxy * 100).toFixed(1)}%`;
    bleuPill.textContent = `BLEU: ${(msg.metrics.bleu_proxy * 100).toFixed(1)}`;
    accPill.textContent = `ACC: ${(msg.metrics.accuracy_proxy * 100).toFixed(1)}%`;
  }

  setActiveStage(msg.active_stage || "module1");
  animateFlowDot(msg.active_stage || "module1");
  appendConsole(msg.parser_console || []);
  renderHistory(msg.transcript_history || []);
  updateControls(msg.control_state || {});

  if (cameraShell.classList.contains("pulsing")) {
    gsap.fromTo(
      cameraShell,
      { boxShadow: "0 0 0 rgba(34,211,238,0)" },
      { boxShadow: "0 0 24px rgba(34,211,238,0.42)", duration: 0.35, yoyo: true, repeat: 1 }
    );
  }
}

function applyIdleAnimation() {
  gsap.to(".card", {
    y: "+=2",
    duration: 3.6,
    yoyo: true,
    repeat: -1,
    stagger: { each: 0.2, from: "random" },
    ease: "sine.inOut",
  });
}

function currentResolution() {
  if (!stream) return "unknown";
  const track = stream.getVideoTracks()[0];
  if (!track) return "unknown";
  const settings = track.getSettings();
  const w = settings.width || 0;
  const h = settings.height || 0;
  if (!w || !h) return "unknown";
  return `${w}x${h}`;
}

function sendClientStats() {
  if (!wsRef || wsRef.readyState !== WebSocket.OPEN) return;
  const payload = {
    type: "client_video_stats",
    camera_active: !!stream,
    frame_hint: frameHintCounter,
    resolution: currentResolution(),
  };
  wsRef.send(JSON.stringify(payload));
}

function connectSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/demo`);
  wsRef = ws;

  ws.onopen = () => {
    setConnected(true);
    gsap.from([".module"], {
      y: 14,
      opacity: 0,
      stagger: 0.05,
      duration: 0.45,
      ease: "power2.out",
    });
    applyIdleAnimation();
    sendClientStats();
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    updateMainPanels(msg);
    updateModule("module1", msg.module1);
    updateModule("module2", msg.module2);
    updateModule("module3", msg.module3);
    updateModule("module4", msg.module4);
    updateModule("module5", msg.module5);
    updateModule("module6", msg.module6);
    updateModule("module7", msg.module7);
  };

  ws.onclose = () => {
    setConnected(false);
    wsRef = null;
    setTimeout(connectSocket, 900);
  };

  ws.onerror = () => ws.close();
}

connectSocket();
