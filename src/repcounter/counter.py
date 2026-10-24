"""The rep-counting state machine.

    WAITING --angle >= top--> TOP --angle < partial--> DOWN --angle <= bottom--> BOTTOM
                               ^                         |                         |
                               |   back to top without   |                         |
                               +---- reaching bottom ----+  (partial rep, 0 pts)   |
                               |                                                   |
                               +---------------- angle >= top: rep counted --------+

Why a state machine instead of "count every time the angle dips"?
  * The gap between the thresholds acts as hysteresis, so jitter around a
    single threshold can't double-count.
  * We only count a rep once the person *returns* to the start position, the
    way a coach counts.
  * Half reps are detected and reported instead of silently counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .exercises import Exercise, FormIssue
from .geometry import EMA
from .landmarks import Pose


class Phase(str, Enum):
    WAITING = "waiting"  # haven't seen the start position yet
    TOP = "top"
    DOWN = "down"
    BOTTOM = "bottom"


@dataclass
class Rep:
    index: int  # 1-based attempt number
    start_time: float
    end_time: float
    min_angle: float
    completed: bool
    issues: List[str] = field(default_factory=list)
    score: int = 100

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_good(self) -> bool:
        return self.completed and not self.issues


@dataclass
class FrameResult:
    angle: Optional[float]
    phase: Phase
    side: Optional[str]
    reps: int
    message: str
    live_issues: List[str] = field(default_factory=list)
    finished_rep: Optional[Rep] = None


class RepCounter:
    def __init__(
        self,
        exercise: Exercise,
        smoothing: float = 0.4,
        min_visibility: float = 0.5,
        message_seconds: float = 1.5,
    ) -> None:
        self.exercise = exercise
        self.min_visibility = min_visibility
        self.message_seconds = message_seconds
        self._ema = EMA(smoothing)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.phase = Phase.WAITING
        self.attempts: List[Rep] = []
        self.side: Optional[str] = None
        self._ema.reset()
        self._rep_start = 0.0
        self._rep_min = math.inf
        self._rep_issues: dict = {}
        self._message = ""
        self._message_until = -math.inf

    @property
    def reps(self) -> int:
        return sum(r.completed for r in self.attempts)

    @property
    def good_reps(self) -> int:
        return sum(r.is_good for r in self.attempts)

    @property
    def partial_reps(self) -> int:
        return sum(not r.completed for r in self.attempts)

    # ----------------------------------------------------------------- update
    def update(self, pose: Optional[Pose], t: float) -> FrameResult:
        """Feed one frame (pose may be None if nobody was detected) at time ``t`` seconds."""
        ex = self.exercise
        if pose is None:
            return self._result(None, "No person detected - step into frame", t)

        # Only switch sides between reps, so the angle can't jump mid-rep.
        if self.side is None or self.phase in (Phase.WAITING, Phase.TOP):
            self.side = pose.best_side(ex.parts)
        side = self.side

        if not pose.side_visible(side, ex.parts, self.min_visibility):
            return self._result(None, "Turn side-on so your whole body is visible", t)
        if not ex.is_valid_position(pose, side):
            return self._result(None, ex.position_hint(), t)

        angle = self._ema.update(ex.primary_angle(pose, side))
        if math.isnan(angle):
            return self._result(None, "Tracking lost", t)

        finished: Optional[Rep] = None
        live: List[FormIssue] = []

        if self.phase == Phase.WAITING:
            if angle >= ex.top_angle:
                self.phase = Phase.TOP
        elif self.phase == Phase.TOP:
            if angle < ex.partial_angle:
                self._start_rep(t, angle)
                self.phase = Phase.DOWN
        elif self.phase in (Phase.DOWN, Phase.BOTTOM):
            self._rep_min = min(self._rep_min, angle)
            live = ex.check_form(pose, side, self.phase.value)
            for issue in live:
                self._rep_issues[issue.code] = issue
            if self.phase == Phase.DOWN and angle <= ex.bottom_angle:
                self.phase = Phase.BOTTOM
            elif angle >= ex.top_angle:
                finished = self._finish_rep(t, completed=self.phase == Phase.BOTTOM)
                self.phase = Phase.TOP

        if finished is not None:
            self._flash(self._rep_message(finished), t)

        default = self._default_message()
        live_msgs = [i.message for i in live]
        message = live_msgs[0] if live_msgs else default
        return self._result(angle, message, t, [i.code for i in live], finished)

    # ---------------------------------------------------------------- helpers
    def _start_rep(self, t: float, angle: float) -> None:
        self._rep_start = t
        self._rep_min = angle
        self._rep_issues = {}

    def _finish_rep(self, t: float, completed: bool) -> Optional[Rep]:
        if t - self._rep_start < self.exercise.min_rep_seconds:
            return None  # too fast to be a real rep -> tracking noise
        issues = dict(self._rep_issues)
        if not completed:
            issues[self.exercise.depth_issue.code] = self.exercise.depth_issue
        score = 0 if not completed else max(0, 100 - sum(i.penalty for i in issues.values()))
        rep = Rep(
            index=len(self.attempts) + 1,
            start_time=self._rep_start,
            end_time=t,
            min_angle=round(self._rep_min, 1),
            completed=completed,
            issues=sorted(issues),
            score=score,
        )
        self._last_issue_messages = [issues[c].message for c in sorted(issues)]
        self.attempts.append(rep)
        return rep

    def _rep_message(self, rep: Rep) -> str:
        if rep.is_good:
            return f"Good rep! ({rep.duration:.1f}s)"
        return self._last_issue_messages[0]

    def _flash(self, message: str, t: float) -> None:
        self._message = message
        self._message_until = t + self.message_seconds

    def _default_message(self) -> str:
        if self.phase == Phase.WAITING:
            return f"Start in the top position ({self.exercise.display_name})"
        return ""

    def _result(self, angle, message, t, live=None, finished=None) -> FrameResult:
        if t < self._message_until and not live:
            message = self._message  # keep "Good rep!" on screen for a moment
        return FrameResult(
            angle=angle,
            phase=self.phase,
            side=self.side,
            reps=self.reps,
            message=message,
            live_issues=live or [],
            finished_rep=finished,
        )
