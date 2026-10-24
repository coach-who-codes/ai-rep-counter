import csv
import json

from repcounter import RepCounter, WorkoutSession, get_exercise
from repcounter.session import CSV_COLUMNS
from repcounter.simulate import rep_angles, squat_pose


def make_session(bottoms):
    counter = RepCounter(get_exercise("squat"))
    session = WorkoutSession(counter, client_id="C042")
    t0 = 0.0
    for bottom in bottoms:
        for t, angle in rep_angles(175, bottom, 1):
            result = counter.update(squat_pose(angle), t0 + t)
            session.record(t0 + t, result.angle)
        t0 += 5
    return session


def test_summary_mixes_good_and_half_reps():
    s = make_session([85, 85, 125, 85]).summary()
    assert (s.reps, s.good_reps, s.partial_reps) == (3, 3, 1)
    assert s.form_score == 75.0  # (100 + 100 + 0 + 100) / 4
    assert s.issues == {"too_shallow": 1}
    assert s.top_issue == "too_shallow"
    assert s.avg_tempo_s is not None and s.avg_tempo_s > 0


def test_empty_session_has_no_score():
    s = make_session([]).summary()
    assert s.reps == 0 and s.form_score is None and s.avg_tempo_s is None and s.top_issue == ""
    assert "Form score" in s.pretty()


def test_save_json_and_append_csv(tmp_path):
    session = make_session([85, 85])
    data = json.loads(session.save_json(tmp_path / "s.json").read_text())
    assert data["reps"] == 2 and data["client_id"] == "C042" and len(data["angle_trace"]) > 10

    log = tmp_path / "logs" / "workout_log.csv"
    session.append_csv(log)
    session.append_csv(log)
    with log.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert rows[0]["client_id"] == "C042" and rows[0]["reps"] == "2"
