export function createControlActions({ state, refs, pipelineUI }) {
  function sendControl(action, value) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const packet = { type: "control", action };
    if (typeof value !== "undefined") {
      packet.value = value;
    }

    state.ws.send(JSON.stringify(packet));

    if (action === "start") {
      state.running = true;
    }
    if (action === "stop") {
      state.running = false;
    }
    pipelineUI.updateControlState();
  }

  function sendVideoStats() {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const resolution = refs.liveVideo && refs.liveVideo.videoWidth
      ? `${refs.liveVideo.videoWidth}x${refs.liveVideo.videoHeight}`
      : "unknown";

    state.ws.send(
      JSON.stringify({
        type: "client_video_stats",
        camera_active: state.cameraActive,
        frame_hint: state.frameCounter,
        resolution,
      }),
    );
  }

  function setWsStatus(status) {
    state.wsStatus = status;
    pipelineUI.updateStatusChips();
  }

  return {
    sendControl,
    sendVideoStats,
    setWsStatus,
  };
}
