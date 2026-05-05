import cv2


class PreviewWindow:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self._is_open = False

    def open(self):
        if self._is_open:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._is_open = True

    def show(self, frame):
        if not self._is_open:
            self.open()
        cv2.imshow(self.window_name, frame)

    def should_close(self) -> bool:
        return cv2.waitKey(1) & 0xFF == ord("q")

    def close(self):
        if not self._is_open:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        finally:
            self._is_open = False
