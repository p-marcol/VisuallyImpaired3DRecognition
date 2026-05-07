from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow

WINDOW_TITLE = "VI3DR Control Room"
WINDOW_WIDTH = 1360
WINDOW_HEIGHT = 900


class MainWindow(QMainWindow):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.channel = QWebChannel(self)
        self.channel.registerObject("bridge", bridge)

        self.web_view = QWebEngineView(self)
        self.web_view.page().setWebChannel(self.channel)

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setCentralWidget(self.web_view)
        self._load_frontend()

    def _load_frontend(self):
        index_path = Path(__file__).resolve().parent / "assets" / "index.html"
        self.web_view.load(QUrl.fromLocalFile(str(index_path)))
