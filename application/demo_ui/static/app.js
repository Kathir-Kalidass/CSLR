const backendBase =
  window.BACKEND_BASE_URL ||
  "http://localhost:8000";

const inferenceVideoUrl = `${backendBase}/api/v1/inference/video`;
const inferenceFramesUrl = `${backendBase}/api/v1/inference/frames`;
const backendHealthUrl = `${backendBase}/health`;

const videoInput = document.getElementById("videoInput");
const uploadedVideo = document.getElementById("uploadedVideo");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const runUploadBtn = document.getElementById("runUploadBtn");
const speakUploadBtn = document.getElementById("speakUploadBtn");

const openCameraBtn = document.getElementById("openCameraBtn");
const captureCameraBtn = document.getElementById("captureCameraBtn");
const closeCameraBtn = document.getElementById("closeCameraBtn");
const speakCameraBtn = document.getElementById("speakCameraBtn");
const cameraVideo = document.getElementById("cameraVideo");
const captureCanvas = document.getElementById("captureCanvas");

const uploadGloss = document.getElementById("uploadGloss");
const uploadCaption = document.getElementById("uploadCaption");
const cameraGloss = document.getElementById("cameraGloss");
const cameraCaption = document.getElementById("cameraCaption");
const latencyBadge = document.getElementById("latencyBadge");
const confidenceBadge = document.getElementById("confidenceBadge");
const fpsBadge = document.getElementById("fpsBadge");
const backendStatus = document.getElementById("backendStatus");
const ttsStatus = document.getElementById("ttsStatus");
const globalMessage = document.getElementById("globalMessage");

let uploadFile = null;
let cameraStream = null;
let uploadSentence = "";
let cameraSentence = "";

function setMessage(text) {
  globalMessage.textContent = text;
}

function updateMetrics(result) {
  latencyBadge.textContent = `Latency: ${((result.processing_time || 0) * 1000).toFixed(0)} ms`;
  confidenceBadge.textContent = `Confidence: ${(((result.confidence || 0) * 100).toFixed(1))}%`;
  fpsBadge.textContent = `FPS: ${(result.fps || 0).toFixed(1)}`;
}

function speakText(text) {
  if (!text) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  utterance.rate = 1.02;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

async function checkBackend() {
  try {
    const response = await fetch(backendHealthUrl);
    if (!response.ok) throw new Error("Backend unavailable");
    backendStatus.textContent = "Backend: connected";
  } catch (error) {
    backendStatus.textContent = "Backend: unavailable";
    setMessage("Start the backend on port 8000 to run inference.");
  }
}

videoInput.addEventListener("change", () => {
  const [file] = videoInput.files || [];
  uploadFile = file || null;
  if (!uploadFile) {
    uploadedVideo.classList.add("hidden");
    uploadPlaceholder.classList.remove("hidden");
    runUploadBtn.disabled = true;
    return;
  }

  uploadedVideo.src = URL.createObjectURL(uploadFile);
  uploadedVideo.classList.remove("hidden");
  uploadPlaceholder.classList.add("hidden");
  runUploadBtn.disabled = false;
  setMessage(`Selected ${uploadFile.name}. Ready to generate caption.`);
});

runUploadBtn.addEventListener("click", async () => {
  if (!uploadFile) return;
  setMessage("Running video inference...");
  runUploadBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", uploadFile);
    const response = await fetch(inferenceVideoUrl, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    uploadGloss.textContent = (result.gloss || []).join(" ") || "No gloss detected.";
    uploadCaption.textContent = result.sentence || "No caption generated.";
    uploadSentence = result.sentence || "";
    speakUploadBtn.disabled = !uploadSentence;
    updateMetrics(result);
    setMessage("Video caption generated.");
    if (uploadSentence) speakText(uploadSentence);
  } catch (error) {
    setMessage(`Video inference failed: ${error.message}`);
  } finally {
    runUploadBtn.disabled = false;
  }
});

speakUploadBtn.addEventListener("click", () => speakText(uploadSentence));
speakCameraBtn.addEventListener("click", () => speakText(cameraSentence));

openCameraBtn.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    captureCameraBtn.disabled = false;
    closeCameraBtn.disabled = false;
    openCameraBtn.disabled = true;
    setMessage("Camera opened. Use Generate From Camera when ready.");
  } catch (error) {
    setMessage("Camera permission was denied or unavailable.");
  }
});

closeCameraBtn.addEventListener("click", () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }
  cameraStream = null;
  cameraVideo.srcObject = null;
  captureCameraBtn.disabled = true;
  closeCameraBtn.disabled = true;
  openCameraBtn.disabled = false;
  setMessage("Camera closed.");
});

async function captureFrames() {
  const width = cameraVideo.videoWidth || 640;
  const height = cameraVideo.videoHeight || 360;
  captureCanvas.width = width;
  captureCanvas.height = height;
  const ctx = captureCanvas.getContext("2d");
  const frames = [];

  for (let i = 0; i < 24; i += 1) {
    ctx.drawImage(cameraVideo, 0, 0, width, height);
    frames.push(captureCanvas.toDataURL("image/jpeg", 0.8));
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
  return frames;
}

captureCameraBtn.addEventListener("click", async () => {
  if (!cameraStream) return;
  captureCameraBtn.disabled = true;
  setMessage("Capturing camera frames and generating caption...");

  try {
    const frames = await captureFrames();
    const response = await fetch(inferenceFramesUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames, fps: 12 }),
    });
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    cameraGloss.textContent = (result.gloss || []).join(" ") || "No gloss detected.";
    cameraCaption.textContent = result.sentence || "No caption generated.";
    cameraSentence = result.sentence || "";
    speakCameraBtn.disabled = !cameraSentence;
    updateMetrics(result);
    setMessage("Camera caption generated.");
    if (cameraSentence) speakText(cameraSentence);
  } catch (error) {
    setMessage(`Camera inference failed: ${error.message}`);
  } finally {
    captureCameraBtn.disabled = false;
  }
});

if (!("speechSynthesis" in window)) {
  ttsStatus.textContent = "Voice: not supported";
} else {
  ttsStatus.textContent = "Voice: browser TTS";
}

checkBackend();
