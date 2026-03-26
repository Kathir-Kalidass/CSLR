import { STAGE_LABELS, STAGE_ORDER } from "../core/constants.js";

export function createShell(appRoot, appTitle, pageMode = "live") {
  const isLive = pageMode === "live";
  const isFlow = pageMode === "flow";
  const isInsights = pageMode === "insights";
  appRoot.innerHTML = `
    <section class="hero card p-4 md:p-6">
      <div class="hero-top">
        <div>
          <p class="hero-kicker">Continuous ISL Recognition and Translation</p>
          <h1 class="title-main">${appTitle}: AI Pipeline in Motion</h1>
          <p class="hero-sub">From your live sign in webcam to decoded text and AI voice, with real-time flow visualization.</p>
        </div>
        <div class="hero-actions">
          <a href="/live" class="page-link ${isLive ? "page-link-active" : ""}">Live</a>
          <a href="/flow" class="page-link ${isFlow ? "page-link-active" : ""}">Flow</a>
          <a href="/insights" class="page-link ${isInsights ? "page-link-active" : ""}">Insights</a>
          <button id="btnDemoGuide" class="ctrl-btn ctrl-info">Demo Guide</button>
          <button id="btnArchitecture" class="ctrl-btn ctrl-info">Architecture Notes</button>
        </div>
      </div>
      <div class="chip-row" id="statusChips"></div>
      <div class="chip-row"><span id="pageTag" class="pill pill-cyan">Live Recognition</span></div>
    </section>

    <section class="card controls p-4 mt-4">
      <div class="control-row" id="controlRow"></div>
    </section>

    <section class="card narration-card p-4 mt-4">
      <div class="sub-title-row">
        <h2 class="section-title">Live Narration</h2>
        <span id="narrationStage" class="pill-cyan">Stage: idle</span>
      </div>
      <p id="narrationText" class="narration-text">Start pipeline to begin guided narration.</p>
    </section>

    <section class="grid-panel mt-4">
      <div id="leftColumn" class="left-column">
        <article id="cameraCard" class="card camera-card p-3 md:p-4">
          <div class="section-head">
            <h2 class="section-title">Live Camera Stage</h2>
            <div class="live-pill"><span id="camDot" class="status-dot"></span><span id="camLabel">Camera idle</span></div>
          </div>
          <div class="camera-shell" id="cameraShell">
            <video id="liveVideo" autoplay playsinline muted></video>
            <canvas id="overlayCanvas"></canvas>
            <div class="scanline"></div>
            <div class="feed-hud">
              <span id="hudFps">FPS: 0</span>
              <span id="hudLatency">Latency: 0 ms</span>
              <span id="hudConfidence">Confidence: 0%</span>
            </div>
          </div>
          <div class="timeline-wrap mt-3">
            <div class="sub-title-row">
              <h3 class="sub-title">Temporal Sliding Window</h3>
              <span id="timelineActive" class="pill-cyan">frames[0:63]</span>
            </div>
            <div class="timeline-ribbon" id="timelineRibbon"></div>
          </div>
          <div class="stream-previews">
            <div class="stream-box">
              <h4 class="stream-title">RGB Stream</h4>
              <canvas id="rgbPreviewCanvas" class="stream-canvas"></canvas>
            </div>
            <div class="stream-box">
              <h4 class="stream-title">Pose Stream</h4>
              <canvas id="posePreviewCanvas" class="stream-canvas"></canvas>
            </div>
          </div>
        </article>

        <article id="outputCard" class="card output-card p-3 md:p-4 mt-4">
          <div class="outputs-grid">
            <div>
              <h3 class="sub-title">Live Sign Text</h3>
              <div id="glossText" class="panel-strong">--</div>
            </div>
            <div>
              <h3 class="sub-title">Live English Output</h3>
              <div id="sentenceText" class="panel-sentence">Open camera and start live recognition.</div>
            </div>
          </div>
          <div class="audio-block mt-3">
            <div class="sub-title-row">
              <h3 class="sub-title">AI Voice Output</h3>
              <span id="audioState" class="pill-amber">idle</span>
            </div>
            <div class="wave" id="waveBars"></div>
          </div>
        </article>
      </div>

      <div id="centerColumn" class="center-column">
        <article id="flowCard" class="card p-3 md:p-4">
          <div class="sub-title-row">
            <h2 class="section-title">Continuous Processing Flow</h2>
            <span id="modelBadge" class="pill-violet">model: checking</span>
          </div>
          <div class="flow-lane">
            <div class="flow-track" id="flowTrack"></div>
            <div class="flow-dot" id="flowDot"></div>
          </div>
          <div class="flow-stages" id="flowStages"></div>
        </article>

        <article id="flowStepsCard" class="card p-3 md:p-4 mt-4">
          <div class="sub-title-row">
            <h2 class="section-title">Flow Steps</h2>
            <span class="pill-green">capture -> voice</span>
          </div>
          <div class="module-grid" id="moduleGrid"></div>
        </article>
      </div>

      <div id="rightColumn" class="right-column">
        <article id="analyticsCard" class="card p-3 md:p-4">
          <h2 class="section-title">Live Analytics</h2>
          <div class="gauge-grid mt-3" id="gaugeGrid"></div>
          <div class="attention-card mt-3">
            <h3 class="sub-title">Attention Fusion</h3>
            <div class="attention-row">
              <span>RGB</span>
              <div class="bar"><div id="rgbBar" class="bar-fill rgb"></div></div>
              <span id="rgbVal">0.50</span>
            </div>
            <div class="attention-row">
              <span>Pose</span>
              <div class="bar"><div id="poseBar" class="bar-fill pose"></div></div>
              <span id="poseVal">0.50</span>
            </div>
          </div>
        </article>

        <article id="historyCard" class="card p-3 md:p-4 mt-4">
          <h2 class="section-title">Transcript History</h2>
          <div id="historyList" class="history-list"></div>
        </article>

        <article id="parserCard" class="card p-3 md:p-4 mt-4">
          <h2 class="section-title">Parser Console</h2>
          <div id="parserConsole" class="parser-console"></div>
        </article>
      </div>
    </section>

    <section id="bootOverlay" class="boot-overlay">
      <div class="boot-card">
        <h2>AI System Initialization</h2>
        <ul id="bootList"></ul>
      </div>
    </section>
  `;
}

export function renderControlRow() {
  const row = document.getElementById("controlRow");
  row.innerHTML = [
    `<button id="btnOpenCamera" class="ctrl-btn ctrl-cyan">Open Camera</button>`,
    `<button id="btnCloseCamera" class="ctrl-btn ctrl-slate">Close Camera</button>`,
    `<button id="btnStart" class="ctrl-btn ctrl-start">Start Pipeline</button>`,
    `<button id="btnStop" class="ctrl-btn ctrl-stop">Stop</button>`,
    `<button id="btnClear" class="ctrl-btn ctrl-clear">Clear History</button>`,
    `<button id="btnTts" class="ctrl-btn ctrl-tts">TTS ON</button>`,
    `<button id="btnGrand" class="ctrl-btn ctrl-grand">Grand Mode OFF</button>`,
    `<button id="btnAutoDemo" class="ctrl-btn ctrl-auto">Auto Demo OFF</button>`,
    `<button id="btnPresenter" class="ctrl-btn ctrl-presenter">Presenter Mode</button>`,
  ].join("");
}

export function renderFlowStages() {
  const root = document.getElementById("flowStages");
  root.innerHTML = STAGE_ORDER.map((stage) => {
    return `
      <div class="flow-stage" data-stage="${stage}">
        <span class="flow-name">${STAGE_LABELS[stage]}</span>
        <div class="flow-progress"><div class="flow-progress-fill"></div></div>
      </div>
    `;
  }).join("");
}

export function renderModuleCards() {
  const grid = document.getElementById("moduleGrid");
  grid.innerHTML = STAGE_ORDER.map((stage) => {
    return `
      <article class="module" data-stage="${stage}">
        <div class="module-head">
          <h3 class="module-title">${STAGE_LABELS[stage]}</h3>
          <span class="module-tag">idle</span>
        </div>
        <p class="module-field"><span>Input:</span> <span class="module-input">--</span></p>
        <p class="module-field"><span>Process:</span> <span class="module-process">--</span></p>
        <p class="module-field"><span>Output:</span> <span class="module-output">--</span></p>
        <div class="mini-progress"><div class="mini-fill"></div></div>
        <div class="module-note">No update yet.</div>
      </article>
    `;
  }).join("");
}

export function renderGauges() {
  const grid = document.getElementById("gaugeGrid");
  const gauges = [
    { key: "accuracy", label: "Accuracy" },
    { key: "wer", label: "WER" },
    { key: "bleu", label: "BLEU" },
    { key: "latency", label: "Latency" },
  ];

  grid.innerHTML = gauges
    .map(
      (g) => `
    <div class="gauge" data-gauge="${g.key}">
      <div class="gauge-ring"><div class="gauge-core"><span class="gauge-value">0</span></div></div>
      <div class="gauge-label">${g.label}</div>
    </div>
  `,
    )
    .join("");
}

export function renderWaveBars() {
  const wave = document.getElementById("waveBars");
  wave.innerHTML = new Array(24).fill(0).map((_, idx) => `<span class="wave-bar" data-wave-index="${idx}"></span>`).join("");
}

export function renderTimeline(windows) {
  const ribbon = document.getElementById("timelineRibbon");
  ribbon.innerHTML = windows
    .map((item, idx) => `<div class="timeline-chip ${idx === 0 ? "active" : ""}">${item}</div>`)
    .join("");
}

export function collectRefs() {
  return {
    popupBackdrop: document.getElementById("popupBackdrop"),
    popupPanel: document.getElementById("popupPanel"),
    liveVideo: document.getElementById("liveVideo"),
    overlayCanvas: document.getElementById("overlayCanvas"),
    rgbPreviewCanvas: document.getElementById("rgbPreviewCanvas"),
    posePreviewCanvas: document.getElementById("posePreviewCanvas"),
    cameraShell: document.getElementById("cameraShell"),
    flowTrack: document.getElementById("flowTrack"),
    flowDot: document.getElementById("flowDot"),
    bootOverlay: document.getElementById("bootOverlay"),
    bootList: document.getElementById("bootList"),
    particleCanvas: document.getElementById("particleCanvas"),
  };
}
