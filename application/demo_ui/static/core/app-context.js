import { createInitialState } from "./state.js";
import { createControlActions } from "./control-actions.js";

import {
  collectRefs,
  createShell,
  renderControlRow,
  renderFlowStages,
  renderGauges,
  renderModuleCards,
  renderTimeline,
  renderWaveBars,
} from "../components/layout.js";
import { createPopupController } from "../components/popup.js";
import { createPipelineUI } from "../components/pipeline-ui.js";
import { createCameraController } from "../components/camera.js";
import { createTTSService } from "../services/tts.js";

function setPageModeLayout(pageMode) {
  const cameraCard = document.getElementById("cameraCard");
  const outputCard = document.getElementById("outputCard");
  const flowCard = document.getElementById("flowCard");
  const flowStepsCard = document.getElementById("flowStepsCard");
  const analyticsCard = document.getElementById("analyticsCard");
  const historyCard = document.getElementById("historyCard");
  const parserCard = document.getElementById("parserCard");
  const pageTag = document.getElementById("pageTag");

  const all = [cameraCard, outputCard, flowCard, flowStepsCard, analyticsCard, historyCard, parserCard].filter(Boolean);
  all.forEach((el) => el.classList.remove("hidden"));

  if (pageTag) {
    const labels = {
      live: "Live Recognition",
      flow: "Processing Flow",
      insights: "Insights",
    };
    pageTag.textContent = labels[pageMode] || "Live Recognition";
  }

  if (pageMode === "live") {
    flowStepsCard?.classList.add("hidden");
    analyticsCard?.classList.add("hidden");
    historyCard?.classList.add("hidden");
    parserCard?.classList.add("hidden");
  } else if (pageMode === "flow") {
    outputCard?.classList.add("hidden");
    analyticsCard?.classList.add("hidden");
    historyCard?.classList.add("hidden");
    parserCard?.classList.add("hidden");
  } else if (pageMode === "insights") {
    cameraCard?.classList.add("hidden");
    flowCard?.classList.add("hidden");
    flowStepsCard?.classList.add("hidden");
  }
}

export function createAppContext({ appRoot, appTitle, pageMode }) {
  const state = createInitialState();

  createShell(appRoot, appTitle, pageMode);
  renderControlRow();
  renderFlowStages();
  renderModuleCards();
  renderGauges();
  renderWaveBars();
  renderTimeline(["frames[0:63]", "frames[32:95]", "frames[64:127]"]);
  setPageModeLayout(pageMode);

  const refs = collectRefs();
  const popup = createPopupController({ popupBackdrop: refs.popupBackdrop, popupPanel: refs.popupPanel });
  const pipelineUI = createPipelineUI({ state });
  const ttsService = createTTSService({ state });
  const { sendControl, sendVideoStats, setWsStatus } = createControlActions({
    state,
    refs,
    pipelineUI,
  });

  const cameraController = createCameraController({
    state,
    refs,
    updateControlState: pipelineUI.updateControlState,
    showPopup: popup.showPopup,
    sendVideoStats,
  });

  return {
    state,
    refs,
    popup,
    pipelineUI,
    ttsService,
    cameraController,
    sendControl,
    sendVideoStats,
    setWsStatus,
  };
}
