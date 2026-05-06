const backendStatus = document.getElementById("backend-status");
const captureStatus = document.getElementById("capture-status");
const captureDetail = document.getElementById("capture-detail");
const portValue = document.getElementById("port-value");
const mdnsValue = document.getElementById("mdns-value");
const frameMeta = document.getElementById("frame-meta");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const shutdownButton = document.getElementById("shutdown-button");
const languageSelect = document.getElementById("language-select");
const i18n = window.VI3DR_I18N;
const LOCALE_STORAGE_KEY = "vi3dr.locale";
let currentCaptureState = "idle";
let currentCaptureMessage = "";

function translate(key, params) {
  return i18n.t(key, params);
}

function updateBackendStatus(status) {
  backendStatus.textContent = translate(`status.${status || "unknown"}`);
}

function updateCaptureStatus(state, message) {
  const normalizedState = state || "idle";
  currentCaptureState = normalizedState;
  currentCaptureMessage = message || "";
  captureStatus.textContent = translate(`status.${normalizedState}`);
  captureDetail.textContent = translateCaptureMessage(normalizedState, message);

  if (normalizedState !== "connected" && !previewImage.src) {
    previewPlaceholder.textContent = translateCaptureMessage(normalizedState, message);
  }
}

function updateServer(host, port, mdnsIp) {
  portValue.textContent = port || "-";
  mdnsValue.textContent = mdnsIp || "-";
}

function updatePreviewFrame(frameDataUrl, width, height) {
  if (!frameDataUrl) {
    previewImage.style.display = "none";
    previewImage.removeAttribute("src");
    previewPlaceholder.style.display = "grid";
    frameMeta.textContent = translate("messages.no_frame");
    return;
  }

  previewImage.src = frameDataUrl;
  previewImage.style.display = "block";
  previewPlaceholder.style.display = "none";
  frameMeta.textContent = translate("meta.frame_dimensions", { width, height });
}

function translateCaptureMessage(state, message) {
  const stateKey = `capture.${state}`;
  const translated = translate(stateKey);
  return translated === stateKey
    ? message || translate("messages.no_session_details")
    : translated;
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
    frameMeta.textContent = translate("messages.no_frame");
    previewPlaceholder.textContent = translateCaptureMessage(
      currentCaptureState,
      currentCaptureMessage,
    );
  }

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
    bridge.previewFrameChanged.connect(updatePreviewFrame);
    bridge.backendErrorChanged.connect((message) => {
      updateBackendStatus("error");
      updateCaptureStatus("error", message || translate("errors.backend_error"));
    });

    shutdownButton.addEventListener("click", () => {
      bridge.shutdownApplication();
    });

    languageSelect.addEventListener("change", (event) => {
      setLocale(event.target.value);
    });

    bridge.requestInitialState();
  });
}

window.addEventListener("DOMContentLoaded", attachBridge);
