import { STAGE_EXPLANATIONS, STAGE_LABELS, STAGE_ORDER } from "../core/constants.js";
import { escapeHtml } from "../core/utils.js";
import { renderTimeline } from "./layout.js";

export function createPipelineUI({ state }) {
  function renderHighlightedText(el, text, tone) {
    if (!el) return;
    const clean = String(text || "").trim();
    if (!clean) {
      el.textContent = "--";
      return;
    }
    const words = clean.split(/\s+/).slice(0, 32);
    el.innerHTML = words
      .map((word, idx) => {
        const cls = idx === words.length - 1 ? "word-chip current" : "word-chip";
        return `<span class="${cls} ${tone}">${escapeHtml(word)}</span>`;
      })
      .join(" ");
  }

  function updateStatusChips() {
    const holder = document.getElementById("statusChips");
    if (!holder) {
      return;
    }

    const payload = state.payload;
    const model = payload?.model_info || state.latestSystemInfo?.model_info || {};
    const ttsEngine = payload?.tts_engine || state.latestSystemInfo?.tts_engine || "unavailable";
    const inferenceMode = payload?.inference_mode || state.latestSystemInfo?.inference_mode || "simulated";
    const runtimeStatus = payload?.runtime_status || state.latestSystemInfo?.runtime_status || "unknown";

    holder.innerHTML = `
      <span class="pill ${state.wsStatus === "connected" ? "pill-green" : "pill-amber"}">WS: ${state.wsStatus}</span>
      <span class="pill ${state.cameraActive ? "pill-cyan" : "pill-slate"}">Camera: ${state.cameraActive ? "active" : "off"}</span>
      <span class="pill ${state.running ? "pill-green" : "pill-slate"}">Pipeline: ${state.running ? "running" : "idle"}</span>
      <span class="pill ${inferenceMode === "real" ? "pill-green" : "pill-amber"}">Inference: ${escapeHtml(inferenceMode)}</span>
      <span class="pill ${state.ttsEnabled ? "pill-violet" : "pill-slate"}">TTS: ${state.ttsEnabled ? "on" : "off"} (${escapeHtml(ttsEngine)})</span>
      <span class="pill ${model.state === "loaded" ? "pill-green" : "pill-amber"}">best.pt: ${escapeHtml(model.state || "unknown")}</span>
      <span class="pill pill-slate">Runtime: ${escapeHtml(runtimeStatus)}</span>
    `;

    const modelBadge = document.getElementById("modelBadge");
    if (modelBadge) {
      const modelPath = (model.active_model || "best.pt").split("/").slice(-3).join("/");
      modelBadge.textContent = `model: ${modelPath}`;
    }
  }

  function updateControlState() {
    const camDot = document.getElementById("camDot");
    const camLabel = document.getElementById("camLabel");
    const btnTts = document.getElementById("btnTts");
    const btnGrand = document.getElementById("btnGrand");
    const btnAutoDemo = document.getElementById("btnAutoDemo");

    if (camDot) camDot.classList.toggle("active", state.cameraActive);
    if (camLabel) camLabel.textContent = state.cameraActive ? "Camera online" : "Camera idle";
    if (btnTts) {
      btnTts.textContent = state.ttsEnabled ? "TTS ON" : "TTS OFF";
      btnTts.classList.toggle("ctrl-tts-off", !state.ttsEnabled);
    }
    if (btnGrand) btnGrand.textContent = state.grandMode ? "Grand Mode ON" : "Grand Mode OFF";
    if (btnAutoDemo) {
      btnAutoDemo.textContent = state.autoDemo ? "Auto Demo ON" : "Auto Demo OFF";
      btnAutoDemo.classList.toggle("ctrl-auto-on", state.autoDemo);
    }

    document.body.classList.toggle("grand-mode", state.grandMode);
  }

  function updateHud(payload) {
    const fps = Number(payload.fps || 0);
    const latency = Number(payload.latency_ms || 0);
    const confidence = Number(payload.confidence || 0);
    document.getElementById("hudFps").textContent = `FPS: ${fps}`;
    document.getElementById("hudLatency").textContent = `Latency: ${latency} ms`;
    document.getElementById("hudConfidence").textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
  }

  function updateOutputs(payload) {
    renderHighlightedText(document.getElementById("glossText"), payload.partial_gloss || "--", "tone-gloss");
    renderHighlightedText(
      document.getElementById("sentenceText"),
      payload.final_sentence || "Waiting for live sign input...",
      "tone-sentence",
    );
    document.getElementById("audioState").textContent = payload.audio_state || "idle";
  }

  function updateNarration(activeStage, status) {
    const stageEl = document.getElementById("narrationStage");
    const textEl = document.getElementById("narrationText");
    if (!stageEl || !textEl) return;

    if (!activeStage || status !== "active") {
      stageEl.textContent = "Stage: idle";
      textEl.textContent = "Start pipeline to begin guided narration.";
      return;
    }

    const label = STAGE_LABELS[activeStage] || activeStage;
    stageEl.textContent = `Stage: ${label}`;
    textEl.textContent = STAGE_EXPLANATIONS[activeStage] || "Realtime pipeline stage is active.";
  }

  function updateFlow(payload) {
    const activeStage = payload.active_stage || "module1";
    const activeIndex = STAGE_ORDER.indexOf(activeStage);
    const stages = document.querySelectorAll(".flow-stage");

    stages.forEach((el, idx) => {
      const isActive = el.getAttribute("data-stage") === activeStage;
      el.classList.toggle("active", isActive);
      const fill = el.querySelector(".flow-progress-fill");
      if (fill) {
        const pct = idx < activeIndex ? 100 : idx === activeIndex ? 72 : 8;
        fill.style.width = `${pct}%`;
      }
    });
  }

  function updateModules(payload) {
    STAGE_ORDER.forEach((stage, idx) => {
      const module = payload[stage] || {};
      const card = document.querySelector(`.module[data-stage="${stage}"]`);
      if (!card) return;

      const isHot = payload.active_stage === stage;
      card.classList.toggle("is-hot", isHot);
      card.querySelector(".module-tag").textContent = isHot ? "active" : payload.status || "idle";
      card.querySelector(".module-input").textContent = module.input || "--";
      card.querySelector(".module-process").textContent = module.process || "--";
      card.querySelector(".module-output").textContent = module.output || "--";
      card.querySelector(".module-note").textContent = module.note || "No update.";

      const activeIndex = STAGE_ORDER.indexOf(payload.active_stage || "module1");
      const fill = card.querySelector(".mini-fill");
      const pct = idx < activeIndex ? 100 : idx === activeIndex ? 78 : 10;
      fill.style.width = `${pct}%`;
    });
  }

  function updateTimelineFromPayload(payload) {
    const windows = payload.timeline?.windows || ["frames[0:63]", "frames[32:95]", "frames[64:127]"];
    const active = payload.timeline?.active_window || windows[0];
    renderTimeline(windows);
    document.getElementById("timelineActive").textContent = active;

    const chips = document.querySelectorAll(".timeline-chip");
    chips.forEach((chip) => {
      chip.classList.toggle("active", chip.textContent.trim() === active.trim());
    });
  }

  function setGauge(name, percent, label) {
    const gauge = document.querySelector(`.gauge[data-gauge="${name}"]`);
    if (!gauge) return;
    const ring = gauge.querySelector(".gauge-ring");
    const clamped = Math.max(0, Math.min(100, percent));
    ring.style.setProperty("--p", `${clamped}%`);
    gauge.querySelector(".gauge-value").textContent = label;
  }

  function updateAnalytics(payload) {
    const metrics = payload.metrics || {};

    const accuracy = Number(metrics.accuracy_proxy ?? 0);
    const wer = Number(metrics.wer_proxy ?? 0);
    const bleu = Number(metrics.bleu_proxy ?? 0);
    const latency = Number(payload.latency_ms || 0);

    setGauge("accuracy", accuracy * 100, `${Math.round(accuracy * 100)}%`);
    setGauge("wer", Math.max(0, 100 - wer * 100), wer.toFixed(2));
    setGauge("bleu", bleu * 100, bleu.toFixed(2));
    const latencyScore = Math.max(0, 100 - Math.min(95, latency / 5));
    setGauge("latency", latencyScore, `${latency}ms`);

    const rgb = Number(payload.attention?.rgb ?? 0.5);
    const pose = Number(payload.attention?.pose ?? 0.5);
    document.getElementById("rgbBar").style.width = `${Math.round(rgb * 100)}%`;
    document.getElementById("poseBar").style.width = `${Math.round(pose * 100)}%`;
    document.getElementById("rgbVal").textContent = rgb.toFixed(2);
    document.getElementById("poseVal").textContent = pose.toFixed(2);
  }

  function updateHistory(history) {
    const list = document.getElementById("historyList");
    if (!history.length) {
      list.innerHTML = `<div class="history-empty">No translations yet.</div>`;
      return;
    }

    list.innerHTML = history
      .slice(0, 12)
      .map(
        (line, idx) => `
      <div class="history-item">
        <span class="idx">${idx + 1}</span>
        <span>${escapeHtml(line)}</span>
      </div>
    `,
      )
      .join("");
  }

  function updateParser(lines) {
    const consoleEl = document.getElementById("parserConsole");
    if (!lines.length) {
      consoleEl.textContent = "[system] waiting for stream";
      return;
    }

    const recent = lines.slice(-28).map((line) => `<div>${escapeHtml(line)}</div>`).join("");
    consoleEl.innerHTML = recent;
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  function applyPayload(payload) {
    state.payload = payload;
    state.running = !!payload?.control_state?.running;
    state.ttsEnabled = !!payload?.control_state?.tts_enabled;
    state.grandMode = !!payload?.control_state?.grand_mode;

    updateStatusChips();
    updateControlState();
    updateHud(payload);
    updateOutputs(payload);
    updateNarration(payload.active_stage, payload.status);
    updateFlow(payload);
    updateModules(payload);
    updateTimelineFromPayload(payload);
    updateAnalytics(payload);
    updateHistory(payload.transcript_history || []);
    updateParser(payload.parser_console || []);
  }

  return {
    updateStatusChips,
    updateControlState,
    updateNarration,
    applyPayload,
  };
}
