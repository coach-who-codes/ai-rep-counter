"""Synthetic side-view skeletons with *known* joint angles.

Used for unit tests (we know the ground truth) and for the README demo.
Each generator builds a stick figure from limb lengths + target angles using
simple forward kinematics, so `primary_angle()` should give the angle back.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List, Tuple

from .landmarks import Joint, Pose

XY = Tuple[float, float]


def _rot(v: XY, deg: float) -> XY:
    r = math.radians(deg)
    return (v[0] * math.cos(r) - v[1] * math.sin(r), v[0] * math.sin(r) + v[1] * math.cos(r))


def _add(p: XY, v: XY, scale: float = 1.0) -> XY:
    return (p[0] + v[0] * scale, p[1] + v[1] * scale)


def _lerp(a: XY, b: XY, f: float) -> XY:
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def _to_pose(points: Dict[str, XY], facing: str, near_vis: float, far_vis: float, cx: float) -> Pose:
    """Both body sides share coordinates (side view); the far side is less visible."""
    joints = {}
    near, far = ("left", "right") if facing == "right" else ("right", "left")
    for name, (x, y) in points.items():
        if facing == "left":
            x = 2 * cx - x  # mirror so the person faces the other way
        if name == "nose":
            joints["nose"] = Joint(x, y, near_vis)
            continue
        joints[f"{near}_{name}"] = Joint(x, y, near_vis)
        joints[f"{far}_{name}"] = Joint(x, y, far_vis)
    return Pose(joints)


def squat_pose(
    knee_angle: float, torso_lean: float = 25.0, facing: str = "right", near_vis: float = 0.95, far_vis: float = 0.6
) -> Pose:
    """Side-view squat. ``torso_lean`` is the forward lean (deg) reached at the bottom."""
    ankle = (300.0, 400.0)
    shin_tilt = (180 - knee_angle) * 0.35  # knees travel forward as you sit down
    knee = _add(ankle, (math.sin(math.radians(shin_tilt)), -math.cos(math.radians(shin_tilt))), 100)
    knee_to_ankle = (-math.sin(math.radians(shin_tilt)), math.cos(math.radians(shin_tilt)))
    hip = _add(knee, _rot(knee_to_ankle, knee_angle), 100)
    depth = max(0.0, min(1.0, (180 - knee_angle) / 90))
    lean = 5 + (torso_lean - 5) * depth
    shoulder = _add(hip, (math.sin(math.radians(lean)), -math.cos(math.radians(lean))), 120)
    points = {
        "ankle": ankle,
        "heel": (ankle[0] - 15, ankle[1] + 5),
        "foot_index": (ankle[0] + 25, ankle[1] + 8),
        "knee": knee,
        "hip": hip,
        "shoulder": shoulder,
        "elbow": (shoulder[0] + 55, shoulder[1] + 10),
        "wrist": (shoulder[0] + 110, shoulder[1] + 12),
        "nose": _add(shoulder, (math.sin(math.radians(lean)) + 0.2, -math.cos(math.radians(lean))), 45),
    }
    return _to_pose(points, facing, near_vis, far_vis, cx=300)


def pushup_pose(
    elbow_angle: float, hip_offset: float = 0.0, facing: str = "right", near_vis: float = 0.95, far_vis: float = 0.6
) -> Pose:
    """Side-view push-up. ``hip_offset`` > 0 sags the hips (pixels), < 0 pikes them."""
    upper = fore = 70.0
    wrist = (400.0, 400.0)
    d = math.sqrt(upper**2 + fore**2 - 2 * upper * fore * math.cos(math.radians(elbow_angle)))
    shoulder = (wrist[0], wrist[1] - d)
    mid = _lerp(shoulder, wrist, 0.5)
    elbow = (mid[0] - math.sqrt(max(0.0, upper**2 - (d / 2) ** 2)), mid[1])
    ankle = (wrist[0] - 260, wrist[1] - 10)
    hip = _lerp(shoulder, ankle, 0.45)
    hip = (hip[0], hip[1] + hip_offset)
    knee = _lerp(shoulder, ankle, 0.72)
    knee = (knee[0], knee[1] + hip_offset * 0.5)
    points = {
        "wrist": wrist,
        "elbow": elbow,
        "shoulder": shoulder,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "heel": (ankle[0] - 8, ankle[1] - 5),
        "foot_index": (ankle[0] + 12, ankle[1] + 8),
        "nose": (shoulder[0] + 35, shoulder[1] - 5),
    }
    return _to_pose(points, facing, near_vis, far_vis, cx=300)


def curl_pose(
    elbow_angle: float,
    elbow_drift: float = 0.0,
    torso_lean: float = 0.0,
    facing: str = "right",
    near_vis: float = 0.95,
    far_vis: float = 0.6,
) -> Pose:
    """Side-view standing curl. ``elbow_drift`` swings the upper arm forward (deg)."""
    ankle, knee, hip = (300.0, 400.0), (300.0, 300.0), (300.0, 200.0)
    shoulder = _add(hip, (math.sin(math.radians(torso_lean)), -math.cos(math.radians(torso_lean))), 140)
    elbow = _add(shoulder, (math.sin(math.radians(elbow_drift)), math.cos(math.radians(elbow_drift))), 70)
    elbow_to_shoulder = (-math.sin(math.radians(elbow_drift)), -math.cos(math.radians(elbow_drift)))
    wrist = _add(elbow, _rot(elbow_to_shoulder, elbow_angle), 65)
    points = {
        "ankle": ankle,
        "heel": (ankle[0] - 15, ankle[1] + 5),
        "foot_index": (ankle[0] + 25, ankle[1] + 8),
        "knee": knee,
        "hip": hip,
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "nose": _add(shoulder, (0.25, -1.0), 45),
    }
    return _to_pose(points, facing, near_vis, far_vis, cx=300)


def rep_angles(
    top: float,
    bottom: float,
    reps: int,
    fps: int = 30,
    rep_seconds: float = 2.0,
    pause_seconds: float = 0.6,
    noise: float = 0.0,
    seed: int = 0,
) -> List[Tuple[float, float]]:
    """(time, angle) samples for ``reps`` smooth reps between ``top`` and ``bottom``."""
    rng = random.Random(seed)
    samples: List[Tuple[float, float]] = []
    t = 0.0
    dt = 1.0 / fps

    def emit(angle: float) -> None:
        nonlocal t
        samples.append((round(t, 4), angle + (rng.gauss(0, noise) if noise else 0.0)))
        t += dt

    for _ in range(int(pause_seconds * fps)):
        emit(top)
    for _ in range(reps):
        frames = int(rep_seconds * fps)
        for i in range(frames):
            phase = i / frames
            emit(bottom + (top - bottom) * (0.5 + 0.5 * math.cos(2 * math.pi * phase)))
        for _ in range(int(pause_seconds * fps)):
            emit(top)
    return samples


def iter_poses(generator, angles: List[Tuple[float, float]], **kwargs) -> Iterator[Tuple[float, Pose]]:
    for t, angle in angles:
        yield t, generator(angle, **kwargs)
