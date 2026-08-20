const backendStatus = document.getElementById("backend-status");
const captureStatus = document.getElementById("capture-status");
const captureDetail = document.getElementById("capture-detail");
const clientIpValue = document.getElementById("client-ip-value");
const fpsValue = document.getElementById("fps-value");
const compressionValue = document.getElementById("compression-value");
const modelValue = document.getElementById("model-value");
const modelStatusValue = document.getElementById("model-status-value");
const modelDetail = document.getElementById("model-detail");
const knowledgeDatabaseValue = document.getElementById("knowledge-database-value");
const knowledgeDatabaseStatusValue = document.getElementById("knowledge-database-status-value");
const knowledgeDatabaseDetail = document.getElementById("knowledge-database-detail");
const detectedObjectValue = document.getElementById("detected-object-value");
const detectedObjectConfidenceValue = document.getElementById("detected-object-confidence-value");
const detectedObjectConfidenceBar = document.getElementById("detected-object-confidence-bar");
const portValue = document.getElementById("port-value");
const mdnsValue = document.getElementById("mdns-value");
const frameMeta = document.getElementById("frame-meta");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const chooseModelButton = document.getElementById("choose-model-button");
const chooseKnowledgeDatabaseButton = document.getElementById("choose-knowledge-database-button");
const shutdownButton = document.getElementById("shutdown-button");
const languageSelect = document.getElementById("language-select");
const debugMessageInput = document.getElementById("debug-message-input");
const debugSendButton = document.getElementById("debug-send-button");
const i18n = window.VI3DR_I18N;
const LOCALE_STORAGE_KEY = "vi3dr.locale";
let currentCaptureState = "idle";
let currentCaptureMessage = "";
let currentModelPath = "";
let currentModelStatus = "unknown";
let currentModelMessage = "";
let currentKnowledgeDatabasePath = "";
let currentKnowledgeDatabaseStatus = "unknown";
let currentKnowledgeDatabaseMessage = "";
let currentDetectionLabel = "";
let currentDetectionConfidence = 0;

function translate(key, params) {
  return i18n.t(key, params);
}

function setTextIfChanged(element, value) {
  if (element.textContent !== value) {
    element.textContent = value;
  }
}

function updateBackendStatus(status) {
  setTextIfChanged(backendStatus, translate(`status.${status || "unknown"}`));
}

function updateCaptureStatus(state, message) {
  const normalizedState = state || "idle";
  currentCaptureState = normalizedState;
  currentCaptureMessage = message || "";
  setTextIfChanged(captureStatus, translate(`status.${normalizedState}`));
  setTextIfChanged(captureDetail, translateCaptureMessage(normalizedState, message));
  updateDebugControls();

  if (normalizedState !== "connected" && !previewImage.src) {
    setTextIfChanged(
      previewPlaceholder,
      translateCaptureMessage(normalizedState, message),
    );
  }
}

function updateServer(host, port, mdnsIp) {
  setTextIfChanged(portValue, port || "-");
  setTextIfChanged(mdnsValue, mdnsIp || "-");
}

function updateCaptureMetrics(clientIp, fps, compression) {
  setTextIfChanged(clientIpValue, clientIp || "-");
  setTextIfChanged(fpsValue, fps || "-");
  setTextIfChanged(compressionValue, compression || "-");
}

function updateDetectionModel(modelPath, status, message) {
  currentModelPath = modelPath || "";
  currentModelStatus = status || "unknown";
  currentModelMessage = message || "";
  setTextIfChanged(modelValue, formatModelName(currentModelPath));
  setTextIfChanged(modelStatusValue, translate(`model_status.${currentModelStatus}`));
  setTextIfChanged(modelDetail, translateModelMessage(currentModelStatus, currentModelMessage));
  chooseModelButton.disabled = currentModelStatus === "loading";
  chooseModelButton.setAttribute("aria-busy", currentModelStatus === "loading" ? "true" : "false");
}

function updateKnowledgeDatabase(databasePath, status, message) {
  currentKnowledgeDatabasePath = databasePath || "";
  currentKnowledgeDatabaseStatus = status || "unknown";
  currentKnowledgeDatabaseMessage = message || "";
  setTextIfChanged(knowledgeDatabaseValue, formatFileName(currentKnowledgeDatabasePath));
  setTextIfChanged(
    knowledgeDatabaseStatusValue,
    translate(`model_status.${currentKnowledgeDatabaseStatus}`),
  );
  setTextIfChanged(
    knowledgeDatabaseDetail,
    translateKnowledgeDatabaseMessage(
      currentKnowledgeDatabaseStatus,
      currentKnowledgeDatabaseMessage,
    ),
  );
  chooseKnowledgeDatabaseButton.disabled = currentKnowledgeDatabaseStatus === "loading";
  chooseKnowledgeDatabaseButton.setAttribute(
    "aria-busy",
    currentKnowledgeDatabaseStatus === "loading" ? "true" : "false",
  );
}

function updateDetectionResult(label, confidence) {
  const parsedConfidence = Number(confidence);
  currentDetectionLabel = label || "";
  currentDetectionConfidence = Number.isFinite(parsedConfidence) ? parsedConfidence : 0;

  if (!currentDetectionLabel) {
    setTextIfChanged(detectedObjectValue, translate("messages.no_detection"));
    setTextIfChanged(detectedObjectConfidenceValue, "-");
    detectedObjectConfidenceBar.style.width = "0%";
    return;
  }

  const probability = Math.max(0, Math.min(currentDetectionConfidence, 1));
  const percent = Math.round(probability * 100);
  setTextIfChanged(detectedObjectValue, currentDetectionLabel);
  setTextIfChanged(detectedObjectConfidenceValue, `${percent}%`);
  detectedObjectConfidenceBar.style.width = `${percent}%`;
}

function updatePreviewFrame(frameDataUrl, width, height) {
  if (!frameDataUrl) {
    previewImage.style.display = "none";
    previewImage.removeAttribute("src");
    previewPlaceholder.style.display = "grid";
    setTextIfChanged(frameMeta, translate("messages.no_frame"));
    return;
  }

  previewImage.src = frameDataUrl;
  previewImage.style.display = "block";
  previewPlaceholder.style.display = "none";
  setTextIfChanged(frameMeta, translate("meta.frame_dimensions", { width, height }));
}

function translateCaptureMessage(state, message) {
  const normalizedMessage = message || "";
  const genericMessages = {
    connected: "Client connected",
    disconnected: "Session closed",
  };

  if (normalizedMessage && normalizedMessage !== genericMessages[state]) {
    return normalizedMessage;
  }

  const stateKey = `capture.${state}`;
  const translated = translate(stateKey);
  return translated === stateKey
    ? normalizedMessage || translate("messages.no_session_details")
    : translated;
}

function translateModelMessage(status, message) {
  const statusKey = `model_message.${status}`;
  const translated = translate(statusKey);
  return translated === statusKey
    ? message || translate("messages.model_waiting")
    : translated;
}

function translateKnowledgeDatabaseMessage(status, message) {
  const statusKey = `database_message.${status}`;
  const translated = translate(statusKey);
  return translated === statusKey
    ? message || translate("messages.database_waiting")
    : translated;
}

function formatModelName(modelPath) {
  return formatFileName(modelPath);
}

function formatFileName(modelPath) {
  if (!modelPath) {
    return "-";
  }

  return modelPath.split(/[\\/]/).pop() || modelPath;
}

function updateDebugControls() {
  const hasClient = currentCaptureState === "connected";
  const hasMessage = debugMessageInput.value.trim().length > 0;
  debugSendButton.disabled = !hasClient || !hasMessage;
}

function getInitialLocale() {
  const storedLocale = window.localStorage.getItem(LOCALE_STORAGE_KEY);
  if (storedLocale && i18n.supportedLocales.includes(storedLocale)) {
    return storedLocale;
  }

  return "en";
}

function setLocale(locale) {
  i18n.setLocale(locale);
  languageSelect.value = i18n.getLocale();

  if (!previewImage.src) {
    setTextIfChanged(frameMeta, translate("messages.no_frame"));
    setTextIfChanged(
      previewPlaceholder,
      translateCaptureMessage(currentCaptureState, currentCaptureMessage),
    );
  }
  updateDetectionModel(currentModelPath, currentModelStatus, currentModelMessage);
  updateKnowledgeDatabase(
    currentKnowledgeDatabasePath,
    currentKnowledgeDatabaseStatus,
    currentKnowledgeDatabaseMessage,
  );
  updateDetectionResult(currentDetectionLabel, currentDetectionConfidence);

  window.localStorage.setItem(LOCALE_STORAGE_KEY, i18n.getLocale());
}

function attachBridge() {
  setLocale(getInitialLocale());

  if (typeof qt === "undefined") {
    updateBackendStatus("error");
    updateCaptureStatus("error", translate("errors.qwebchannel_not_initialized"));
    return;
  }

  new QWebChannel(qt.webChannelTransport, (channel) => {
    const bridge = channel.objects.bridge;

    bridge.backendStatusChanged.connect(updateBackendStatus);
    bridge.serverDetailsChanged.connect(updateServer);
    bridge.captureSessionChanged.connect(updateCaptureStatus);
    bridge.captureMetricsChanged.connect(updateCaptureMetrics);
    bridge.previewFrameChanged.connect(updatePreviewFrame);
    bridge.detectionModelChanged.connect(updateDetectionModel);
    bridge.detectionResultChanged.connect(updateDetectionResult);
    bridge.knowledgeDatabaseChanged.connect(updateKnowledgeDatabase);
    bridge.backendErrorChanged.connect((message) => {
      updateBackendStatus("error");
      updateCaptureStatus("error", message || translate("errors.backend_error"));
    });

    chooseModelButton.addEventListener("click", () => {
      bridge.chooseDetectionModel();
    });

    chooseKnowledgeDatabaseButton.addEventListener("click", () => {
      bridge.chooseKnowledgeDatabase();
    });

    shutdownButton.addEventListener("click", () => {
      bridge.shutdownApplication();
    });

    debugMessageInput.addEventListener("input", updateDebugControls);

    debugSendButton.addEventListener("click", () => {
      const message = debugMessageInput.value.trim();
      if (!message) {
        return;
      }

      bridge.sendDebugInfoResponse(message);
    });

    languageSelect.addEventListener("change", (event) => {
      setLocale(event.target.value);
    });

    bridge.requestInitialState();
  });
}

window.addEventListener("DOMContentLoaded", attachBridge);
