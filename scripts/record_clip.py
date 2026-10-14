"""Record camera clips for offline detector comparison.

Press the toggle key (default ``r``) in the preview window to start a clip and press it
again to stop and save. ``q`` or Esc quits, finalizing any clip still in progress.
Clips land in ``data/recordings/`` as ``clip_YYYYmmdd_HHMMSS.mp4``.

The preview window must have focus for key presses to register. The REC overlay is drawn
on the preview only; saved frames are the raw camera frames.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from numpy.typing import NDArray

from project_auto.capture.camera import Camera, CameraConfig

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "recordings"
WINDOW = "Project AUTO clip recorder"
FPS_WINDOW_FRAMES = 30
RED = (0, 0, 255)


class FrameRateMeter:
    """Estimate the camera's real frame rate from recent frame arrival times."""

    def __init__(self, fallback: float) -> None:
        self._fallback = fallback
        self._times: deque[float] = deque(maxlen=FPS_WINDOW_FRAMES)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return self._fallback
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else self._fallback


class ClipWriter:
    """Write frames to one mp4 file, using the frame rate measured when it started."""

    def __init__(self, path: Path, frame: NDArray, fps: float) -> None:
        height, width = frame.shape[:2]
        self.path = path
        self.fps = fps
        self.frames = 0
        self._started = time.perf_counter()
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")

    def write(self, frame: NDArray) -> None:
        self._writer.write(frame)
        self.frames += 1

    def close(self) -> str:
        self._writer.release()
        seconds = time.perf_counter() - self._started
        return (
            f"Saved {self.path} ({self.frames} frames, {seconds:.1f}s, "
            f"written at {self.fps:.1f} fps)"
        )


def draw_status(frame: NDArray, recording: bool, seconds: float, key: str) -> NDArray:
    output = frame.copy()
    if recording:
        cv2.circle(output, (28, 28), 10, RED, -1)
        cv2.putText(output, f"REC {seconds:5.1f}s  [{key}] stop", (48, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    else:
        cv2.putText(output, f"[{key}] start recording   [q] quit", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output


def run(toggle_key: str) -> None:
    with (ROOT / "configs" / "camera.yaml").open(encoding="utf-8") as config_file:
        config = CameraConfig(**yaml.safe_load(config_file))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meter = FrameRateMeter(fallback=float(config.fps))
    clip: ClipWriter | None = None
    started = 0.0
    try:
        with Camera(config) as camera:
            print(f"Camera device {camera.device} open. "
                  f"Press '{toggle_key}' in the window to start/stop, 'q' to quit.")
            cv2.namedWindow(WINDOW)
            while True:
                frame = camera.read()
                meter.tick()
                if clip is not None:
                    clip.write(frame)

                seconds = time.perf_counter() - started if clip else 0.0
                cv2.imshow(WINDOW, draw_status(frame, clip is not None, seconds, toggle_key))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(toggle_key):
                    if clip is None:
                        name = f"clip_{datetime.now().astimezone():%Y%m%d_%H%M%S}.mp4"
                        clip = ClipWriter(OUTPUT_DIR / name, frame, meter.fps)
                        clip.write(frame)
                        started = time.perf_counter()
                        print(f"Recording {clip.path.name} at {clip.fps:.1f} fps...")
                    else:
                        print(clip.close())
                        clip = None
    finally:
        if clip is not None:
            print(clip.close())
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", default="r", help="single key that starts/stops a clip")
    args = parser.parse_args()
    if len(args.key) != 1 or args.key.lower() == "q":
        parser.error("--key must be a single character other than 'q'")
    run(args.key.lower())


if __name__ == "__main__":
    main()
