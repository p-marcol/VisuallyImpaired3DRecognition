import sys

from PySide6.QtWidgets import QApplication

from .backend_controller import BackendController
from .bridge import DesktopBridge
from .window import MainWindow


def run_desktop_app():
    qt_app = QApplication(sys.argv)
    backend_controller = BackendController()
    bridge = DesktopBridge(backend_controller)
    window = MainWindow(bridge)

    backend_controller.start()
    qt_app.aboutToQuit.connect(lambda: backend_controller.stop(wait=True))

    window.show()
    return qt_app.exec()
