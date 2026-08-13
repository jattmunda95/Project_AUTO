"""Webcam capture boundary."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CameraConfig:
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class Camera:
    """Own an OpenCV camera and release it reliably."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(self.config.device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open camera device {self.config.device}")
        self._capture = capture

    def read(self) -> NDArray:
        if self._capture is None:
            raise RuntimeError("Camera must be opened before reading")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Could not read a frame from the camera")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
