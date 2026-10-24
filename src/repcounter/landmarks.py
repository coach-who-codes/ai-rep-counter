"""Body landmark definitions (MediaPipe Pose's 33-point topology) and a Pose container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

# Indices from the MediaPipe Pose Landmarker model.
LANDMARK_INDEX: Dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Pairs of joints to draw as "bones" on the overlay.
SKELETON: Tuple[Tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    *(
        (f"{side}_{a}", f"{side}_{b}")
        for side in ("left", "right")
        for a, b in (
            ("shoulder", "elbow"),
            ("elbow", "wrist"),
            ("shoulder", "hip"),
            ("hip", "knee"),
            ("knee", "ankle"),
            ("ankle", "heel"),
            ("heel", "foot_index"),
        )
    ),
)


@dataclass
class Joint:
    x: float
    y: float
    visibility: float = 1.0

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Pose:
    """One person's pose in a single frame, in *pixel* coordinates."""

    joints: Dict[str, Joint] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Tuple[float, float]:
        return self.joints[name].xy

    def visibility(self, name: str) -> float:
        joint = self.joints.get(name)
        return joint.visibility if joint else 0.0

    def best_side(self, parts: Iterable[str]) -> str:
        """Pick 'left' or 'right' - whichever side the camera sees more clearly.

        Exercises are filmed side-on, so one side of the body is usually hidden.
        """
        parts = list(parts)
        left = sum(self.visibility(f"left_{p}") for p in parts)
        right = sum(self.visibility(f"right_{p}") for p in parts)
        return "left" if left >= right else "right"

    def side_visible(self, side: str, parts: Iterable[str], threshold: float = 0.5) -> bool:
        return all(self.visibility(f"{side}_{p}") >= threshold for p in parts)

    @classmethod
    def from_normalized(cls, landmarks, width: int, height: int) -> "Pose":
        """Build from MediaPipe's normalized landmarks (x, y in 0..1).

        Converting to pixels matters: normalized x and y are scaled by different
        amounts (width vs height), which would distort every angle.
        """
        joints = {}
        for name, idx in LANDMARK_INDEX.items():
            lm = landmarks[idx]
            vis = getattr(lm, "visibility", None)
            joints[name] = Joint(lm.x * width, lm.y * height, 1.0 if vis is None else float(vis))
        return cls(joints)
