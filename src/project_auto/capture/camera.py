"""Camera capture, kept separate from inference and display.

Tries an external USB camera first and falls back to the built-in webcam when it is
not connected. Both are plain OpenCV device indices; there is no reliable cross-platform
way to identify a device as "USB" versus "built-in" by capability alone, so preference
is expressed purely as try-this-index-first, then-that-one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CameraConfig:
    usb_device: int = 1
    webcam_device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    @property
    def device_candidates(self) -> tuple[int, ...]:
        """Return device indices in try-first-to-try-last order, without duplicates."""
        if self.usb_device == self.webcam_device:
            return (self.usb_device,)
        return (self.usb_device, self.webcam_device)


class Camera:
    """Own one OpenCV video capture and release it reliably."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self.device: int | None = None

    def open(self) -> None:
        """Open the first working candidate device; the USB camera is tried first.

        A device can report open while producing no frames (e.g. a stale OS handle
        for a disconnected camera), so each candidate is confirmed with a real read
        before it is accepted, and released immediately if that read fails. Resolution
        and FPS must be set before that confirmation read, not after: changing them
        mid-stream on some backends (observed on Windows MSMF) corrupts the frame
        buffer and crashes the next read with a cv::Mat assertion.
        """
        if self._capture is not None:
            return

        for device in self.config.device_candidates:
            capture = cv2.VideoCapture(device)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                capture.set(cv2.CAP_PROP_FPS, self.config.fps)
                success, _ = capture.read()
                if success:
                    self._capture = capture
                    self.device = device
                    return
            capture.release()

        raise RuntimeError(
            "Could not open a camera on any configured device: "
            f"{self.config.device_candidates}"
        )

    def read(self) -> NDArray:
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Could not read a frame from the camera")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self.device = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
