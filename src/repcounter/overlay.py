"""Draws the skeleton, joint angle, rep HUD and coaching cue onto a frame."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .counter import FrameResult, RepCounter
from .landmarks import SKELETON, Pose

GREEN = (80, 200, 120)
RED = (60, 60, 230)
YELLOW = (40, 200, 240)
WHITE = (245, 245, 245)
DARK = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(p) -> tuple:
    return int(round(p[0])), int(round(p[1]))


def draw_skeleton(frame: np.ndarray, pose: Pose, side: Optional[str], min_vis: float = 0.5) -> None:
    scale = max(1, frame.shape[1] // 640)
    for a, b in SKELETON:
        if pose.visibility(a) < min_vis or pose.visibility(b) < min_vis:
            continue
        on_side = side is not None and a.startswith(side) and b.startswith(side)
        cv2.line(
            frame, _pt(pose[a]), _pt(pose[b]), YELLOW if on_side else WHITE, 2 * scale + on_side * scale, cv2.LINE_AA
        )
    for name in pose.joints:
        if pose.visibility(name) >= min_vis:
            cv2.circle(frame, _pt(pose[name]), 3 * scale, DARK, -1, cv2.LINE_AA)
            cv2.circle(frame, _pt(pose[name]), 2 * scale, WHITE, -1, cv2.LINE_AA)


def _panel(frame, x1, y1, x2, y2, alpha=0.6):
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, DARK), alpha, 0)


def draw_hud(frame: np.ndarray, counter: RepCounter, result: FrameResult, pose: Optional[Pose]) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max(0.6, w / 960)  # scale text with resolution
    ex = counter.exercise

    if pose is not None:
        draw_skeleton(frame, pose, result.side)
        if result.angle is not None and result.side:
            jx, jy = _pt(pose[ex.primary_joint(result.side)])
            cv2.putText(
                frame,
                f"{result.angle:.0f}",
                (jx + int(12 * s), jy - int(8 * s)),
                FONT,
                0.8 * s,
                DARK,
                int(5 * s),
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{result.angle:.0f}",
                (jx + int(12 * s), jy - int(8 * s)),
                FONT,
                0.8 * s,
                YELLOW,
                int(2 * s),
                cv2.LINE_AA,
            )

    # --- rep counter panel (top-left)
    pw, ph = int(260 * s), int(150 * s)
    _panel(frame, 0, 0, min(pw, w), min(ph, h))
    cv2.putText(
        frame,
        ex.display_name.upper(),
        (int(14 * s), int(30 * s)),
        FONT,
        0.7 * s,
        WHITE,
        max(1, int(2 * s)),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, str(counter.reps), (int(14 * s), int(100 * s)), FONT, 2.3 * s, GREEN, max(2, int(5 * s)), cv2.LINE_AA
    )
    cv2.putText(frame, "REPS", (int(110 * s), int(75 * s)), FONT, 0.6 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(
        frame,
        f"good {counter.good_reps}  half {counter.partial_reps}",
        (int(110 * s), int(100 * s)),
        FONT,
        0.5 * s,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"phase: {result.phase.value}", (int(14 * s), int(135 * s)), FONT, 0.5 * s, YELLOW, 1, cv2.LINE_AA
    )

    # --- range-of-motion bar (right edge): 0% at the top position, 100% at full depth
    if result.angle is not None:
        span = max(1.0, ex.top_angle - ex.bottom_angle)
        depth = float(np.clip((ex.top_angle - result.angle) / span, 0, 1))
        bx1, bx2 = w - int(40 * s), w - int(18 * s)
        by1, by2 = int(40 * s), int(40 * s) + int(200 * s)
        _panel(frame, bx1, by1, bx2, by2, 0.5)
        fill_top = int(by2 - depth * (by2 - by1))
        cv2.rectangle(frame, (bx1, fill_top), (bx2, by2), GREEN if depth >= 1 else YELLOW, -1)
        cv2.putText(frame, "ROM", (bx1 - int(4 * s), by2 + int(22 * s)), FONT, 0.5 * s, WHITE, 1, cv2.LINE_AA)

    # --- coaching cue (bottom bar)
    if result.message:
        color = (
            GREEN
            if result.message.startswith("Good rep")
            else (RED if result.live_issues or "-" in result.message else WHITE)
        )
        bh = int(48 * s)
        _panel(frame, 0, h - bh, w, h, 0.65)
        cv2.putText(
            frame,
            result.message,
            (int(14 * s), h - int(16 * s)),
            FONT,
            0.75 * s,
            color,
            max(1, int(2 * s)),
            cv2.LINE_AA,
        )
    return frame
