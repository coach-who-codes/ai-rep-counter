"""Small, dependency-light geometry helpers used by every exercise.

All points are 2D pixel coordinates ``(x, y)`` where ``y`` grows *downwards*
(standard image convention).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

Point = Sequence[float]


def joint_angle(a: Point, b: Point, c: Point) -> float:
    """Interior angle ABC in degrees (0-180), i.e. the angle at joint ``b``.

    Example: for hip-knee-ankle this is the knee angle. Standing straight is
    ~180 deg, a deep squat is ~70-90 deg.

    Returns ``nan`` if either limb segment has zero length.
    """
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    norm = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if norm == 0:
        return float("nan")
    cos_angle = (bax * bcx + bay * bcy) / norm
    cos_angle = max(-1.0, min(1.0, cos_angle))  # guard against float drift
    return math.degrees(math.acos(cos_angle))


def angle_from_vertical(top: Point, bottom: Point) -> float:
    """Angle in degrees (0-180) between the segment bottom->top and straight up.

    0 deg = perfectly upright (e.g. torso during a squat), 90 deg = horizontal.
    """
    dx = top[0] - bottom[0]
    dy = bottom[1] - top[1]  # flip because image y points down
    if dx == 0 and dy == 0:
        return float("nan")
    return abs(math.degrees(math.atan2(dx, dy)))


def offset_from_line(p: Point, line_start: Point, line_end: Point) -> float:
    """Signed perpendicular distance (pixels) of ``p`` from a line.

    Positive means ``p`` sits *below* the line on screen (larger y), negative
    means above it. Used to tell sagging hips (below the shoulder-ankle line)
    from piked hips (above it) in a push-up.
    """
    lx, ly = line_end[0] - line_start[0], line_end[1] - line_start[1]
    length = math.hypot(lx, ly)
    if length == 0:
        return 0.0
    # 2D cross product gives signed area; divide by length for distance.
    cross = lx * (p[1] - line_start[1]) - ly * (p[0] - line_start[0])
    distance = cross / length
    # Make the sign independent of which way the person faces.
    return distance if lx >= 0 else -distance


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EMA:
    """Exponential moving average to smooth noisy per-frame joint angles.

    ``alpha`` close to 1 follows the signal tightly, close to 0 smooths heavily.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if math.isnan(x):
            return self.value if self.value is not None else x
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None
