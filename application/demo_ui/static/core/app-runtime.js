import { DEFAULT_BOOT_SEQUENCE } from "./constants.js";
import { createAutoDemoController } from "./auto-demo.js";
import { bindMainEvents } from "./event-bindings.js";

import { animateWave, initFlowMotion, initParticleBackground, runBootSequence } from "../components/animations.js";

import { createSocketService } from "../services/socket.js";
import { fetchSystemInfo } from "../services/system.js";

export function createAppRuntime(context) {
  const {
    state,
    refs,
    popup,
    pipelineUI,
    ttsService,
    cameraController,
    sendControl,
    sendVideoStats,
    setWsStatus,
  } = context;

  async function handlePayload(payload) {
    if (!state.bootPlayed) {
      const seq = Array.isArray(payload.init_sequence) && payload.init_sequence.length
        ? payload.init_sequence
        : DEFAULT_BOOT_SEQUENCE;
      runBootSequence(state, refs, seq);
    }

    pipelineUI.applyPayload(payload);

    if (Array.isArray(payload.audio_wave)) {
      state.audioWave = payload.audio_wave;
      animateWave(payload.audio_wave, state);
    }

    await ttsService.maybeSpeak(payload);
  }

  async function loadSystemInfo() {
    const info = await fetchSystemInfo();
    if (info) {
      state.latestSystemInfo = info;
      pipelineUI.updateStatusChips();
    }
  }

  function toggleTts() {
    state.ttsEnabled = !state.ttsEnabled;
    sendControl("toggle_tts");
    pipelineUI.updateControlState();
  }

  function toggleGrandMode() {
    state.grandMode = !state.grandMode;
    sendControl("set_grand_mode", state.grandMode);
    pipelineUI.updateControlState();
    initFlowMotion(state, refs);
  }

  const { stopAutoDemo, toggleAutoDemo } = createAutoDemoController({
    state,
    pipelineUI,
    cameraController,
    sendControl,
    toggleGrandMode,
    toggleTts,
  });

  const socketService = createSocketService({
    state,
    onPayload: (payload) => {
      handlePayload(payload);
    },
    onOpen: () => {
      sendVideoStats();
      loadSystemInfo();
    },
    onStatus: setWsStatus,
  });

  function startPeriodicStatsPush() {
    if (state.statsPushHandle) {
      clearInterval(state.statsPushHandle);
    }
    state.statsPushHandle = setInterval(() => {
      sendVideoStats();
    }, 950);
  }

  function init() {
    bindMainEvents({
      state,
      popup,
      cameraController,
      sendControl,
      toggleTts,
      toggleGrandMode,
      toggleAutoDemo,
    });
    initFlowMotion(state, refs);
    window.addEventListener("resize", state.resizeFlowListener);
    initParticleBackground(state, refs.particleCanvas);
    socketService.connect();
    loadSystemInfo();
    startPeriodicStatsPush();
    pipelineUI.updateStatusChips();
    pipelineUI.updateControlState();
    pipelineUI.updateNarration("", "idle");
  }

  function dispose() {
    stopAutoDemo();
    cameraController.stopFrameStreaming();
    cameraController.closeCamera();

    if (state.statsPushHandle) {
      clearInterval(state.statsPushHandle);
      state.statsPushHandle = null;
    }
    if (state.reconnectHandle) {
      clearTimeout(state.reconnectHandle);
      state.reconnectHandle = null;
    }

    cancelAnimationFrame(state.particleRaf);
    if (state.ws) {
      state.ws.close();
    }

    window.removeEventListener("resize", state.resizeFlowListener);
  }

  state.resizeFlowListener = () => initFlowMotion(state, refs);

  return {
    init,
    dispose,
  };
}
