const API_ENDPOINT = "/api/video/analyze/";
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

const elements = {
  fileInput: document.querySelector("#video-file"),
  chooseButton: document.querySelector("#choose-video"),
  chooseLabel: document.querySelector("#choose-label"),
  dropzone: document.querySelector("#dropzone"),
  dropzoneEmpty: document.querySelector("#dropzone-empty"),
  selectedFile: document.querySelector("#selected-file"),
  preview: document.querySelector("#video-preview"),
  fileName: document.querySelector("#file-name"),
  removeFile: document.querySelector("#remove-file"),
  uploadError: document.querySelector("#upload-error"),
  analyzeButton: document.querySelector("#analyze-button"),
  loadingState: document.querySelector("#loading-state"),
  resultEmpty: document.querySelector("#result-empty"),
  resultContent: document.querySelector("#result-content"),
  resultIcon: document.querySelector("#verdict-icon"),
  resultLabel: document.querySelector("#result-label"),
  resultScore: document.querySelector("#result-score"),
  resultPreview: document.querySelector("#result-preview"),
  resultMessage: document.querySelector("#result-message"),
  resultMessageText: document.querySelector("#result-message-text"),
};

let selectedVideo = null;
let previewUrl = null;
let analysisInProgress = false;

function isSupportedVideo(file) {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return ["mp4", "mov", "avi"].includes(extension);
}

function showUploadError(message) {
  elements.uploadError.textContent = message;
  elements.uploadError.hidden = false;
}

function clearUploadError() {
  elements.uploadError.textContent = "";
  elements.uploadError.hidden = true;
}

function releasePreviewUrl() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  elements.preview.removeAttribute("src");
  elements.resultPreview.removeAttribute("src");
  elements.preview.load();
  elements.resultPreview.load();
}

function resetSelectedVideo() {
  selectedVideo = null;
  releasePreviewUrl();
  elements.fileInput.value = "";
  elements.selectedFile.hidden = true;
  elements.dropzoneEmpty.hidden = false;
  elements.chooseLabel.textContent = "Sélectionner une vidéo";
  elements.analyzeButton.disabled = true;
  elements.resultContent.hidden = true;
  elements.resultEmpty.hidden = false;
  document.querySelector("#result-card").classList.remove("has-result");
  clearUploadError();
}

function setSelectedVideo(file) {
  if (analysisInProgress) return;
  clearUploadError();
  if (!file) return;

  if (!isSupportedVideo(file)) {
    resetSelectedVideo();
    showUploadError("Choisissez une vidéo MP4, MOV ou AVI.");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    resetSelectedVideo();
    showUploadError("La vidéo dépasse la limite de 500 Mo.");
    return;
  }

  releasePreviewUrl();
  selectedVideo = file;
  previewUrl = URL.createObjectURL(file);
  elements.preview.src = previewUrl;
  elements.fileName.textContent = file.name;
  elements.dropzoneEmpty.hidden = true;
  elements.selectedFile.hidden = false;
  elements.chooseLabel.textContent = "Changer la vidéo";
  elements.analyzeButton.disabled = false;
  elements.resultContent.hidden = true;
  elements.resultEmpty.hidden = false;
  document.querySelector("#result-card").classList.remove("has-result");
}

function getCookie(name) {
  const cookie = document.cookie.split(";").map((part) => part.trim()).find((part) => part.startsWith(`${name}=`));
  return cookie ? decodeURIComponent(cookie.slice(name.length + 1)) : "";
}

async function submitVideoToApi(file) {
  const formData = new FormData();
  formData.append("video", file, file.name);
  const headers = { Accept: "application/json" };
  const csrfToken = getCookie("csrftoken");
  if (csrfToken) headers["X-CSRFToken"] = csrfToken;

  const response = await fetch(API_ENDPOINT, {
    method: "POST",
    headers,
    body: formData,
    credentials: "same-origin",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Erreur du serveur (${response.status}).`);

  const fakeProbability = Number(payload.score);
  if (payload.score === null || payload.score === "" || !Number.isFinite(fakeProbability)) {
    throw new Error("La réponse du serveur ne contient pas de score valide.");
  }
  return {
    fakeProbability: Math.min(1, Math.max(0, fakeProbability)),
    elapsedSeconds: Number(payload.elapsed_seconds) || 0,
  };
}

function renderResult(result) {
  const isFake = result.fakeProbability >= 0.5;
  const verdictConfidence = isFake ? result.fakeProbability : 1 - result.fakeProbability;
  const percentage = `${(verdictConfidence * 100).toFixed(1)}%`;

  elements.resultIcon.classList.toggle("is-fake", isFake);
  elements.resultIcon.classList.toggle("is-real", !isFake);
  elements.resultLabel.classList.toggle("is-fake", isFake);
  elements.resultLabel.classList.toggle("is-real", !isFake);
  elements.resultLabel.textContent = isFake ? "Vidéo Suspecte" : "Vidéo Authentique";
  elements.resultScore.textContent = percentage;
  elements.resultMessage.classList.toggle("is-fake", isFake);
  elements.resultMessage.classList.toggle("is-real", !isFake);
  elements.resultMessage.querySelector(".message-icon").textContent = isFake ? "!" : "✓";
  elements.resultMessageText.textContent = isFake ? "Indices de génération détectés" : "Aucune anomalie détectée";
  elements.resultPreview.src = previewUrl;
  elements.resultPreview.load();
  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  document.querySelector("#result-card").classList.add("has-result");
}

async function handleAnalyze() {
  if (!selectedVideo || elements.analyzeButton.disabled) return;
  clearUploadError();
  elements.analyzeButton.disabled = true;
  elements.chooseButton.disabled = true;
  elements.removeFile.disabled = true;
  elements.loadingState.hidden = false;
  analysisInProgress = true;
  const startedAt = performance.now();

  try {
    const result = await submitVideoToApi(selectedVideo);
    if (!result.elapsedSeconds) result.elapsedSeconds = (performance.now() - startedAt) / 1000;
    renderResult(result);
    document.querySelector("#result-card").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    showUploadError(`L’analyse n’a pas abouti : ${error.message}`);
  } finally {
    analysisInProgress = false;
    elements.chooseButton.disabled = false;
    elements.removeFile.disabled = false;
    elements.loadingState.hidden = true;
    elements.analyzeButton.disabled = !selectedVideo;
  }
}

document.querySelector("#welcome-start").addEventListener("click", () => {
  document.querySelector("#analyse").scrollIntoView({ behavior: "smooth", block: "center" });
  elements.chooseButton.focus({ preventScroll: true });
});

elements.fileInput.addEventListener("change", (event) => setSelectedVideo(event.target.files?.[0]));
elements.chooseButton.addEventListener("click", () => elements.fileInput.click());
elements.dropzone.addEventListener("click", () => {
  if (!selectedVideo) elements.fileInput.click();
});
elements.dropzone.addEventListener("keydown", (event) => {
  if (!selectedVideo && ["Enter", " "].includes(event.key)) {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropzone.classList.add("is-dragging");
});
elements.dropzone.addEventListener("dragleave", (event) => {
  if (!elements.dropzone.contains(event.relatedTarget)) elements.dropzone.classList.remove("is-dragging");
});
elements.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropzone.classList.remove("is-dragging");
  setSelectedVideo(event.dataTransfer.files?.[0]);
});
elements.removeFile.addEventListener("click", resetSelectedVideo);
elements.analyzeButton.addEventListener("click", handleAnalyze);
document.querySelector("#new-analysis").addEventListener("click", () => {
  resetSelectedVideo();
  document.querySelector("#analyse").scrollIntoView({ behavior: "smooth", block: "center" });
  elements.chooseButton.focus({ preventScroll: true });
});
