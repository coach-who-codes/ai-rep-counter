"""Thin wrapper around MediaPipe's Pose Landmarker (Tasks API).

Kept separate from the counting logic so the core can be unit-tested without
a camera, a GPU or even MediaPipe installed.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from .landmarks import Pose

MODEL_URLS = {
    variant: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
    )
    for variant in ("lite", "full", "heavy")
}
CACHE_DIR = Path(os.environ.get("REPCOUNTER_CACHE", Path.home() / ".cache" / "repcounter"))


def ensure_model(variant: str = "lite", path: Optional[str] = None) -> Path:
    """Return a local model path, downloading it on first use (~5-30 MB)."""
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        return p
    if variant not in MODEL_URLS:
        raise ValueError(f"variant must be one of {sorted(MODEL_URLS)}")
    target = CACHE_DIR / f"pose_landmarker_{variant}.task"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model ({variant}) -> {target}")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        tmp.replace(target)
    return target


class PoseDetector:
    """Detects one person per frame. Use as a context manager."""

    def __init__(self, model_path: Optional[str] = None, variant: str = "lite", min_confidence: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model(variant, model_path))),
            running_mode=vision.RunningMode.VIDEO,  # uses tracking between frames -> smoother
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[Pose]:
        """Run pose detection on a BGR frame (OpenCV format)."""
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return Pose.from_normalized(result.pose_landmarks[0], w, h)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
