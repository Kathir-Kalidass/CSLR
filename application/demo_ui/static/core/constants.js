export const APP_TITLE = "CSLR Grand Demo";

export const STAGE_ORDER = [
  "module1",
  "module2",
  "module3",
  "module4",
  "module5",
  "module6",
  "module7",
];

export const STAGE_LABELS = {
  module1: "Capture Frames",
  module2: "Extract RGB + Pose",
  module3: "Fuse + Decode",
  module4: "Clean Sign Tokens",
  module5: "Build Sentence",
  module6: "Generate Voice",
  module7: "Update Insights",
};

export const STAGE_EXPLANATIONS = {
  module1: "Live sign input is sampled and a short frame window is prepared for recognition.",
  module2: "Each frame is converted into RGB appearance features and pose motion features.",
  module3: "Both streams are fused with attention and decoded across the active temporal window.",
  module4: "Predicted sign tokens are cleaned by removing repeats and unstable fragments.",
  module5: "Tokens are converted into readable English output.",
  module6: "The final sentence is spoken as AI voice output when TTS is enabled.",
  module7: "Live performance signals are refreshed continuously while you sign.",
};

export const DEFAULT_BOOT_SEQUENCE = [
  "Initializing Vision Engine",
  "Loading EfficientNet + Pose Encoder",
  "Loading Attention Fusion + CTC Decoder",
  "Connecting Streaming Bus",
  "Model Ready",
];
