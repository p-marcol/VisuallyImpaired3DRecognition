from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog


class DesktopBridge(QObject):
    backendStatusChanged = Signal(str)
    serverDetailsChanged = Signal(str, int, str)
    backendErrorChanged = Signal(str)
    captureSessionChanged = Signal(str, str)
    captureMetricsChanged = Signal(str, str, str)
    previewFrameChanged = Signal(str, int, int)
    detectionModelChanged = Signal(str, str, str)

    def __init__(self, backend_controller):
        super().__init__()
        self.backend_controller = backend_controller
        self.backend_controller.backendStatusChanged.connect(self.backendStatusChanged.emit)
        self.backend_controller.serverDetailsChanged.connect(self.serverDetailsChanged.emit)
        self.backend_controller.backendErrorChanged.connect(self.backendErrorChanged.emit)
        self.backend_controller.captureSessionChanged.connect(self.captureSessionChanged.emit)
        self.backend_controller.captureMetricsChanged.connect(self.captureMetricsChanged.emit)
        self.backend_controller.previewFrameChanged.connect(self.previewFrameChanged.emit)
        self.backend_controller.detectionModelChanged.connect(self.detectionModelChanged.emit)

    @Slot()
    def requestInitialState(self):
        state = self.backend_controller.get_state()
        self.backendStatusChanged.emit(state["status"])
        self.serverDetailsChanged.emit(state["host"], state["port"], state["mdns_ip"])
        self.captureSessionChanged.emit(state["capture_state"], state["capture_message"])
        self.captureMetricsChanged.emit(
            state["capture_client_ip"],
            state["capture_fps"],
            state["capture_compression"],
        )
        self.previewFrameChanged.emit(
            state["preview_frame"],
            state["frame_width"],
            state["frame_height"],
        )
        self.detectionModelChanged.emit(
            state["detection_model_path"],
            state["detection_status"],
            state["detection_message"],
        )

    @Slot()
    def chooseDetectionModel(self):
        model_path, _ = QFileDialog.getOpenFileName(
            None,
            "Choose YOLO model",
            "",
            "YOLO models (*.pt);;All files (*)",
        )
        if model_path:
            self.backend_controller.load_detection_model(model_path)

    @Slot()
    def shutdownApplication(self):
        self.backend_controller.stop(wait=True)
        app = QGuiApplication.instance()
        if app is not None:
            app.quit()
