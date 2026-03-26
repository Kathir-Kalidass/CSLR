export function createAutoDemoController({ state, pipelineUI, cameraController, sendControl, toggleGrandMode, toggleTts }) {
  function stopAutoDemo() {
    if (state.autoDemoHandle) {
      clearInterval(state.autoDemoHandle);
      state.autoDemoHandle = null;
    }
    state.autoDemo = false;
    state.autoStepIndex = 0;
    pipelineUI.updateControlState();
  }

  function startAutoDemo() {
    state.autoDemo = true;
    state.autoStepIndex = 0;
    pipelineUI.updateControlState();

    const steps = [
      async () => cameraController.openCamera(),
      async () => sendControl("start"),
      async () => toggleGrandMode(),
      async () => toggleTts(),
      async () => sendControl("clear"),
    ];

    state.autoDemoHandle = setInterval(async () => {
      if (!state.autoDemo) {
        return;
      }

      const step = steps[state.autoStepIndex % steps.length];
      await step();
      state.autoStepIndex += 1;

      if (state.autoStepIndex > 7) {
        stopAutoDemo();
      }
    }, 2400);
  }

  function toggleAutoDemo() {
    if (state.autoDemo) {
      stopAutoDemo();
    } else {
      startAutoDemo();
    }
  }

  return {
    startAutoDemo,
    stopAutoDemo,
    toggleAutoDemo,
  };
}
