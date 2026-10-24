"""AI Rep Counter - count reps and coach form from video using pose estimation."""

from .counter import Phase, Rep, RepCounter
from .exercises import EXERCISES, Exercise, get_exercise
from .landmarks import Pose
from .session import WorkoutSession

__all__ = ["EXERCISES", "Exercise", "Phase", "Pose", "Rep", "RepCounter", "WorkoutSession", "get_exercise"]
__version__ = "1.0.0"
