"""Video -> pose -> reps pipeline, shared by the CLI and the Streamlit app."""

from __future__ import annotations

import time
from typing import Callable, Optional, Union

import cv2

from .counter import RepCounter
from .detector import PoseDetector
from .exercises import get_exercise
from .overlay import draw_hud
from .session import WorkoutSession

ProgressFn = Callable[[int, int], None]


def process_video(
    source: Union[int, str],
    exercise: str,
    output_path: Optional[str] = None,
    display: bool = False,
    client_id: str = "self",
    model_path: Optional[str] = None,
    model_variant: str = "lite",
    mirror: bool = False,
    progress: Optional[ProgressFn] = None,
) -> WorkoutSession:
    """Run the full pipeline on a webcam index or a video file path."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")

    is_live = isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    counter = RepCounter(get_exercise(exercise))
    session = WorkoutSession(counter, client_id=client_id)
    writer = None
    started = time.monotonic()
    frame_idx = 0

    with PoseDetector(model_path=model_path, variant=model_variant) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            # Webcam: wall-clock time. File: frame time (deterministic, independent of CPU speed).
            t = (time.monotonic() - started) if is_live else frame_idx / fps
            pose = detector.detect(frame, int(t * 1000))
            result = counter.update(pose, t)
            session.record(t, result.angle)
            draw_hud(frame, counter, result, pose)

            if output_path:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            if display:
                cv2.imshow("AI Rep Counter  (q to quit, r to reset)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    counter.reset()
            frame_idx += 1
            if progress:
                progress(frame_idx, total)

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    return session
