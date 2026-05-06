from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QGuiApplication


class DesktopBridge(QObject):
    backendStatusChanged = Signal(str)
    serverDetailsChanged = Signal(str, int, str)
    backendErrorChanged = Signal(str)
    captureSessionChanged = Signal(str, str)
    previewFrameChanged = Signal(str, int, int)

    def __init__(self, backend_controller):
        super().__init__()
        self.backend_controller = backend_controller
        self.backend_controller.backendStatusChanged.connect(self.backendStatusChanged.emit)
        self.backend_controller.serverDetailsChanged.connect(self.serverDetailsChanged.emit)
        self.backend_controller.backendErrorChanged.connect(self.backendErrorChanged.emit)
        self.backend_controller.captureSessionChanged.connect(self.captureSessionChanged.emit)
        self.backend_controller.previewFrameChanged.connect(self.previewFrameChanged.emit)

    @Slot()
    def requestInitialState(self):
        state = self.backend_controller.get_state()
        self.backendStatusChanged.emit(state["status"])
        self.serverDetailsChanged.emit(state["host"], state["port"], state["mdns_ip"])
        self.captureSessionChanged.emit(state["capture_state"], state["capture_message"])
        self.previewFrameChanged.emit(
            state["preview_frame"],
            state["frame_width"],
            state["frame_height"],
        )

    @Slot()
    def shutdownApplication(self):
        self.backend_controller.stop()
        app = QGuiApplication.instance()
        if app is not None:
            app.quit()
