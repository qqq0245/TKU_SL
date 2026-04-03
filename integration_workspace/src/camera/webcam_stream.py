from __future__ import annotations

import cv2


class WebcamStream:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index={camera_index}.")

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam.")
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()

    def __enter__(self) -> "WebcamStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
