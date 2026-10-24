"""Exercise rules: which joint angle drives the rep, the thresholds, and form checks.

Every exercise here follows the same pattern - start at a *high* joint angle
(standing / arms straight), travel to a *low* angle, and come back:

    squat      knee angle   (hip - knee - ankle)
    push-up    elbow angle  (shoulder - elbow - wrist)
    bicep curl elbow angle  (shoulder - elbow - wrist)

Thresholds are starting points from coaching practice (side-on camera view);
they are class attributes, so they're easy to tune for a different setup.
Adding a new exercise = subclass ``Exercise`` and register it in ``EXERCISES``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

from .geometry import angle_from_vertical, joint_angle, offset_from_line
from .landmarks import Pose


@dataclass(frozen=True)
class FormIssue:
    code: str
    message: str
    penalty: int = 25  # points removed from a rep's 100-point form score


class Exercise(ABC):
    name: str = ""
    display_name: str = ""
    parts: Tuple[str, ...] = ()  # body parts that must be visible (per side)

    top_angle: float = 160.0  # at/above: start & finish position
    partial_angle: float = 140.0  # below: a rep attempt has started
    bottom_angle: float = 100.0  # at/below: full range of motion reached
    min_rep_seconds: float = 0.4  # anything faster is treated as noise

    depth_issue: FormIssue = FormIssue("partial_range", "Use full range of motion", 100)

    @abstractmethod
    def primary_angle(self, pose: Pose, side: str) -> float:
        """The joint angle that tracks the rep."""

    @abstractmethod
    def primary_joint(self, side: str) -> str:
        """Joint where the primary angle is measured (for the overlay)."""

    def is_valid_position(self, pose: Pose, side: str) -> bool:
        """Is the person in a position where this exercise makes sense?"""
        return True

    def position_hint(self) -> str:
        return f"Get into {self.display_name.lower()} position"

    def check_form(self, pose: Pose, side: str, phase: str) -> List[FormIssue]:
        """Per-frame form checks while a rep is in progress."""
        return []


class Squat(Exercise):
    name = "squat"
    display_name = "Squat"
    parts = ("shoulder", "hip", "knee", "ankle")
    top_angle = 160.0
    partial_angle = 140.0
    bottom_angle = 100.0  # ~thighs parallel when filmed side-on

    depth_issue = FormIssue("too_shallow", "Go deeper - aim for thighs parallel", 100)
    CHEST_UP = FormIssue("chest_up", "Keep your chest up")
    MAX_TORSO_LEAN = 50.0

    def primary_angle(self, pose, side):
        return joint_angle(pose[f"{side}_hip"], pose[f"{side}_knee"], pose[f"{side}_ankle"])

    def primary_joint(self, side):
        return f"{side}_knee"

    def check_form(self, pose, side, phase):
        lean = angle_from_vertical(pose[f"{side}_shoulder"], pose[f"{side}_hip"])
        return [self.CHEST_UP] if lean > self.MAX_TORSO_LEAN else []


class PushUp(Exercise):
    name = "pushup"
    display_name = "Push-up"
    parts = ("shoulder", "elbow", "wrist", "hip", "ankle")
    top_angle = 150.0
    partial_angle = 130.0
    bottom_angle = 95.0

    depth_issue = FormIssue("too_shallow", "Go lower - chest close to the floor", 100)
    HIPS_SAG = FormIssue("hips_sagging", "Hips sagging - squeeze glutes, brace core")
    HIPS_PIKE = FormIssue("hips_piking", "Hips too high - make a straight line")
    MIN_BODY_LINE = 160.0  # shoulder-hip-ankle angle; 180 = perfect plank

    def primary_angle(self, pose, side):
        return joint_angle(pose[f"{side}_shoulder"], pose[f"{side}_elbow"], pose[f"{side}_wrist"])

    def primary_joint(self, side):
        return f"{side}_elbow"

    def is_valid_position(self, pose, side):
        # Body roughly horizontal -> stops arm movement while standing counting as reps.
        return angle_from_vertical(pose[f"{side}_shoulder"], pose[f"{side}_ankle"]) > 50

    def position_hint(self):
        return "Get into plank position, side-on to the camera"

    def check_form(self, pose, side, phase):
        shoulder, hip, ankle = pose[f"{side}_shoulder"], pose[f"{side}_hip"], pose[f"{side}_ankle"]
        if joint_angle(shoulder, hip, ankle) >= self.MIN_BODY_LINE:
            return []
        below_line = offset_from_line(hip, shoulder, ankle) > 0
        return [self.HIPS_SAG if below_line else self.HIPS_PIKE]


class BicepCurl(Exercise):
    name = "curl"
    display_name = "Bicep curl"
    parts = ("shoulder", "elbow", "wrist", "hip")
    top_angle = 150.0  # arm straight
    partial_angle = 120.0
    bottom_angle = 55.0  # fully curled
    min_rep_seconds = 0.5

    depth_issue = FormIssue("partial_range", "Squeeze all the way up", 100)
    ELBOW_DRIFT = FormIssue("elbow_drift", "Keep elbows pinned to your sides")
    BODY_SWING = FormIssue("body_swing", "Don't swing - keep your torso still")
    MAX_ELBOW_DRIFT = 30.0
    MAX_TORSO_LEAN = 15.0

    def primary_angle(self, pose, side):
        return joint_angle(pose[f"{side}_shoulder"], pose[f"{side}_elbow"], pose[f"{side}_wrist"])

    def primary_joint(self, side):
        return f"{side}_elbow"

    def is_valid_position(self, pose, side):
        return angle_from_vertical(pose[f"{side}_shoulder"], pose[f"{side}_hip"]) < 35

    def position_hint(self):
        return "Stand tall, side-on to the camera"

    def check_form(self, pose, side, phase):
        issues = []
        drift = joint_angle(pose[f"{side}_hip"], pose[f"{side}_shoulder"], pose[f"{side}_elbow"])
        if drift > self.MAX_ELBOW_DRIFT:
            issues.append(self.ELBOW_DRIFT)
        if angle_from_vertical(pose[f"{side}_shoulder"], pose[f"{side}_hip"]) > self.MAX_TORSO_LEAN:
            issues.append(self.BODY_SWING)
        return issues


EXERCISES: Dict[str, Type[Exercise]] = {cls.name: cls for cls in (Squat, PushUp, BicepCurl)}


def get_exercise(name: str) -> Exercise:
    try:
        return EXERCISES[name.lower()]()
    except KeyError:
        options = ", ".join(sorted(EXERCISES))
        raise ValueError(f"Unknown exercise '{name}'. Choose from: {options}") from None
