"""Standalone camera device-selection tests using a mocked cv2.VideoCapture."""

from __future__ import annotations

from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from project_auto.capture.camera import Camera, CameraConfig


def make_capture(is_opened: bool, read_succeeds: bool) -> Mock:
    capture = Mock()
    capture.isOpened.return_value = is_opened
    frame = np.zeros((4, 4, 3), dtype=np.uint8) if read_succeeds else None
    capture.read.return_value = (read_succeeds, frame)
    return capture


def test_device_candidates_try_usb_before_webcam() -> None:
    config = CameraConfig(usb_device=1, webcam_device=0)

    assert config.device_candidates == (1, 0)


def test_device_candidates_deduplicate_when_equal() -> None:
    config = CameraConfig(usb_device=0, webcam_device=0)

    assert config.device_candidates == (0,)


def test_open_prefers_the_usb_camera_when_available() -> None:
    usb_capture = make_capture(is_opened=True, read_succeeds=True)
    with patch("project_auto.capture.camera.cv2.VideoCapture", return_value=usb_capture) as ctor:
        camera = Camera(CameraConfig(usb_device=1, webcam_device=0))
        camera.open()

    ctor.assert_called_once_with(1)
    assert camera.device == 1
    usb_capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)


def test_open_configures_resolution_before_the_confirmation_read() -> None:
    # Setting resolution/FPS after a read has already started can corrupt the frame
    # buffer on some backends (observed as a cv::Mat assertion on Windows MSMF).
    usb_capture = make_capture(is_opened=True, read_succeeds=True)
    call_order: list[str] = []
    usb_capture.set.side_effect = lambda *_: call_order.append("set")
    usb_capture.read.side_effect = lambda: (call_order.append("read"), (True, np.zeros((4, 4, 3))))[1]

    with patch("project_auto.capture.camera.cv2.VideoCapture", return_value=usb_capture):
        camera = Camera(CameraConfig(usb_device=1, webcam_device=0))
        camera.open()

    assert call_order == ["set", "set", "set", "read"]


def test_open_falls_back_to_webcam_when_usb_camera_is_absent() -> None:
    usb_capture = make_capture(is_opened=False, read_succeeds=False)
    webcam_capture = make_capture(is_opened=True, read_succeeds=True)
    captures = iter([usb_capture, webcam_capture])
    with patch("project_auto.capture.camera.cv2.VideoCapture", side_effect=lambda _: next(captures)):
        camera = Camera(CameraConfig(usb_device=1, webcam_device=0))
        camera.open()

    assert camera.device == 0
    usb_capture.release.assert_called_once()
    webcam_capture.release.assert_not_called()


def test_open_falls_back_when_usb_device_opens_but_produces_no_frames() -> None:
    usb_capture = make_capture(is_opened=True, read_succeeds=False)
    webcam_capture = make_capture(is_opened=True, read_succeeds=True)
    captures = iter([usb_capture, webcam_capture])
    with patch("project_auto.capture.camera.cv2.VideoCapture", side_effect=lambda _: next(captures)):
        camera = Camera(CameraConfig(usb_device=1, webcam_device=0))
        camera.open()

    assert camera.device == 0
    usb_capture.release.assert_called_once()


def test_open_raises_when_no_candidate_device_works() -> None:
    dead_capture = make_capture(is_opened=False, read_succeeds=False)
    with patch("project_auto.capture.camera.cv2.VideoCapture", return_value=dead_capture):
        camera = Camera(CameraConfig(usb_device=1, webcam_device=0))
        with pytest.raises(RuntimeError, match="Could not open a camera"):
            camera.open()

    assert camera.device is None


def test_open_is_idempotent_and_does_not_reopen() -> None:
    usb_capture = make_capture(is_opened=True, read_succeeds=True)
    with patch("project_auto.capture.camera.cv2.VideoCapture", return_value=usb_capture) as ctor:
        camera = Camera(CameraConfig(usb_device=1, webcam_device=0))
        camera.open()
        camera.open()

    ctor.assert_called_once()


def test_close_releases_and_resets_device() -> None:
    usb_capture = make_capture(is_opened=True, read_succeeds=True)
    with patch("project_auto.capture.camera.cv2.VideoCapture", return_value=usb_capture):
        camera = Camera(CameraConfig(usb_device=1, webcam_device=0))
        camera.open()
        camera.close()

    usb_capture.release.assert_called_once()
    assert camera.device is None
    with pytest.raises(RuntimeError, match="not open"):
        camera.read()
