"""Session summary + logging (JSON per session, CSV history across sessions).

The CSV history is the bridge to the companion project
`client-dropout-predictor`: a client's average form score and workout
frequency become features for predicting who is about to drop out.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .counter import Rep, RepCounter

CSV_COLUMNS = [
    "date",
    "client_id",
    "exercise",
    "reps",
    "good_reps",
    "partial_reps",
    "form_score",
    "avg_tempo_s",
    "duration_s",
    "top_issue",
]


@dataclass
class SessionSummary:
    exercise: str
    client_id: str
    started_at: str
    duration_s: float
    reps: int
    good_reps: int
    partial_reps: int
    form_score: Optional[float]  # 0-100, averaged over every attempt
    avg_tempo_s: Optional[float]  # mean duration of completed reps
    issues: Dict[str, int] = field(default_factory=dict)
    rep_details: List[dict] = field(default_factory=list)

    @property
    def top_issue(self) -> str:
        return max(self.issues, key=self.issues.get) if self.issues else ""

    def pretty(self) -> str:
        lines = [
            f"Exercise      : {self.exercise}",
            f"Reps counted  : {self.reps}  (good form: {self.good_reps}, half reps: {self.partial_reps})",
            f"Form score    : {self.form_score if self.form_score is not None else '-'} / 100",
            f"Avg rep tempo : {self.avg_tempo_s if self.avg_tempo_s is not None else '-'} s",
        ]
        if self.issues:
            lines.append("Form issues   : " + ", ".join(f"{k} x{v}" for k, v in self.issues.items()))
        return "\n".join(lines)


class WorkoutSession:
    """Collects everything from one set/session so it can be summarised and saved."""

    def __init__(self, counter: RepCounter, client_id: str = "self") -> None:
        self.counter = counter
        self.client_id = client_id
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.angle_trace: List[Tuple[float, float]] = []
        self.last_time = 0.0

    def record(self, t: float, angle: Optional[float]) -> None:
        self.last_time = t
        if angle is not None:
            self.angle_trace.append((round(t, 3), round(angle, 2)))

    def summary(self) -> SessionSummary:
        attempts: List[Rep] = self.counter.attempts
        completed = [r for r in attempts if r.completed]
        issue_counts = Counter(code for r in attempts for code in r.issues)
        return SessionSummary(
            exercise=self.counter.exercise.name,
            client_id=self.client_id,
            started_at=self.started_at,
            duration_s=round(self.last_time, 2),
            reps=len(completed),
            good_reps=sum(r.is_good for r in attempts),
            partial_reps=len(attempts) - len(completed),
            form_score=round(sum(r.score for r in attempts) / len(attempts), 1) if attempts else None,
            avg_tempo_s=round(sum(r.duration for r in completed) / len(completed), 2) if completed else None,
            issues=dict(issue_counts.most_common()),
            rep_details=[{**asdict(r), "duration": round(r.duration, 2)} for r in attempts],
        )

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.summary())
        payload["angle_trace"] = self.angle_trace
        path.write_text(json.dumps(payload, indent=2))
        return path

    def append_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        row = {
            "date": s.started_at[:10],
            "client_id": s.client_id,
            "exercise": s.exercise,
            "reps": s.reps,
            "good_reps": s.good_reps,
            "partial_reps": s.partial_reps,
            "form_score": "" if s.form_score is None else s.form_score,
            "avg_tempo_s": "" if s.avg_tempo_s is None else s.avg_tempo_s,
            "duration_s": s.duration_s,
            "top_issue": s.top_issue,
        }
        new_file = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        return path
