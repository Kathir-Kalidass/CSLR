import { escapeHtml } from "./utils.js";

export function bindMainEvents({
  state,
  popup,
  cameraController,
  sendControl,
  toggleTts,
  toggleGrandMode,
  toggleAutoDemo,
}) {
  document.getElementById("btnDemoGuide").addEventListener("click", () => {
    popup.showPopup(
      "Demo Flow",
      `
      <ol>
        <li>Open camera and allow webcam access.</li>
        <li>Start pipeline to stream the full recognition flow continuously.</li>
        <li>Watch frame capture, RGB/pose extraction, and temporal decoding update live.</li>
        <li>Observe gloss and corrected English sentence.</li>
        <li>Enable TTS for AI voice generation.</li>
        <li>Use Grand Mode for cinematic presentation.</li>
      </ol>
    `,
    );
  });

  document.getElementById("btnArchitecture").addEventListener("click", () => {
    const modelInfo = state.latestSystemInfo?.model_info || {};
    popup.showPopup(
      "Architecture + Runtime",
      `
      <p><b>Model:</b> ${escapeHtml(modelInfo.active_model || "unknown")}</p>
      <p><b>Model State:</b> ${escapeHtml(modelInfo.state || "unknown")}</p>
      <p><b>Model Size:</b> ${escapeHtml(String(modelInfo.size_mb || "n/a"))} MB</p>
      <p><b>Inference Mode:</b> ${escapeHtml(state.latestSystemInfo?.inference_mode || "simulated")}</p>
      <p><b>Runtime Status:</b> ${escapeHtml(state.latestSystemInfo?.runtime_status || "unknown")}</p>
      <p><b>TTS Engine:</b> ${escapeHtml(state.latestSystemInfo?.tts_engine || "unavailable")}</p>
      <p><b>Flow:</b> Capture -> Frame Window -> RGB/Pose Features -> Attention Decode -> Sentence -> Voice</p>
    `,
    );
  });

  document.getElementById("btnPresenter").addEventListener("click", () => {
    const sentence = state.payload?.final_sentence || "No sentence yet";
    const gloss = state.payload?.partial_gloss || "--";
    popup.showPopup(
      "Presenter Notes",
      `
      <p><b>Current Gloss:</b> ${escapeHtml(gloss)}</p>
      <p><b>Current Sentence:</b> ${escapeHtml(sentence)}</p>
      <p><b>Tip:</b> Explain dual-stream attention and sliding-window CTC alignment while this updates.</p>
    `,
    );
  });

  document.getElementById("btnOpenCamera").addEventListener("click", () => cameraController.openCamera());
  document.getElementById("btnCloseCamera").addEventListener("click", () => cameraController.closeCamera());
  document.getElementById("btnStart").addEventListener("click", () => sendControl("start"));
  document.getElementById("btnStop").addEventListener("click", () => sendControl("stop"));
  document.getElementById("btnClear").addEventListener("click", () => sendControl("clear"));
  document.getElementById("btnTts").addEventListener("click", toggleTts);
  document.getElementById("btnGrand").addEventListener("click", toggleGrandMode);
  document.getElementById("btnAutoDemo").addEventListener("click", toggleAutoDemo);

  document.getElementById("moduleGrid").addEventListener("click", (event) => {
    const moduleEl = event.target.closest(".module");
    if (!moduleEl) {
      return;
    }

    const stage = moduleEl.getAttribute("data-stage");
    const payload = state.payload;
    const moduleData = payload ? payload[stage] : null;
    if (!moduleData) {
      return;
    }

    const parseLines = Array.isArray(moduleData.parse)
      ? moduleData.parse.map((line) => `<li>${escapeHtml(line)}</li>`).join("")
      : "<li>No parser lines</li>";

    popup.showPopup(
      moduleData.title || stage,
      `
      <p><b>Input:</b> ${escapeHtml(moduleData.input || "--")}</p>
      <p><b>Process:</b> ${escapeHtml(moduleData.process || "--")}</p>
      <p><b>Output:</b> ${escapeHtml(moduleData.output || "--")}</p>
      <p><b>Note:</b> ${escapeHtml(moduleData.note || "--")}</p>
      <ul>${parseLines}</ul>
    `,
    );
  });
}
