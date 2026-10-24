import pytest

from repcounter import Phase, RepCounter, get_exercise
from repcounter.simulate import curl_pose, iter_poses, pushup_pose, rep_angles, squat_pose


def run(exercise, generator, angles, **pose_kwargs):
    counter = RepCounter(get_exercise(exercise))
    for t, pose in iter_poses(generator, angles, **pose_kwargs):
        counter.update(pose, t)
    return counter


@pytest.mark.parametrize(
    "exercise, generator, top, bottom",
    [
        ("squat", squat_pose, 175, 85),
        ("pushup", pushup_pose, 168, 80),
        ("curl", curl_pose, 165, 40),
    ],
)
@pytest.mark.parametrize("reps", [1, 5, 12])
def test_counts_full_reps(exercise, generator, top, bottom, reps):
    counter = run(exercise, generator, rep_angles(top, bottom, reps, noise=2.0, seed=reps))
    assert counter.reps == reps
    assert counter.good_reps == reps
    assert counter.partial_reps == 0


def test_half_reps_are_flagged_not_counted():
    counter = run("squat", squat_pose, rep_angles(175, 125, 3))
    assert counter.reps == 0
    assert counter.partial_reps == 3
    assert all("too_shallow" in r.issues and r.score == 0 for r in counter.attempts)


def test_jitter_around_a_threshold_does_not_double_count():
    # hover around the bottom threshold with heavy noise, then stand up once
    angles = [(i / 30, a) for i, a in enumerate([175] * 20 + [100, 96, 103, 95, 104, 97, 102] * 6 + [175] * 20)]
    counter = run("squat", squat_pose, angles)
    assert counter.reps == 1


def test_nothing_counted_until_start_position_seen():
    # starts at the bottom of a squat (e.g. camera started late)
    angles = [(i / 30, 85.0) for i in range(30)]
    counter = run("squat", squat_pose, angles)
    assert counter.phase == Phase.WAITING
    assert counter.reps == 0


def test_too_fast_movement_is_ignored_as_noise():
    # a full "rep" in 0.2 s is a tracking glitch, not a squat
    angles = [(0.0, 175), (0.033, 175), (0.066, 90), (0.1, 80), (0.133, 85), (0.2, 175), (0.233, 175)]
    counter = RepCounter(get_exercise("squat"), smoothing=1.0)
    for t, a in angles:
        counter.update(squat_pose(a), t)
    assert counter.reps == 0


def test_works_facing_either_direction():
    left = run("squat", squat_pose, rep_angles(175, 85, 4), facing="left")
    right = run("squat", squat_pose, rep_angles(175, 85, 4), facing="right")
    assert left.reps == right.reps == 4


def test_missing_pose_frames_are_tolerated():
    counter = RepCounter(get_exercise("squat"))
    for i, (t, angle) in enumerate(rep_angles(175, 85, 3)):
        pose = None if i % 7 == 0 else squat_pose(angle)  # drop every 7th frame
        result = counter.update(pose, t)
        if pose is None:
            assert "No person" in result.message or result.message
    assert counter.reps == 3


def test_low_visibility_frames_are_skipped():
    counter = RepCounter(get_exercise("squat"))
    for t, angle in rep_angles(175, 85, 2):
        result = counter.update(squat_pose(angle, near_vis=0.2, far_vis=0.1), t)
    assert counter.reps == 0
    assert "visible" in result.message


def test_rep_records_timing_and_depth():
    counter = run("squat", squat_pose, rep_angles(175, 85, 2, rep_seconds=2.0))
    for rep in counter.attempts:
        assert 1.0 < rep.duration < 2.0  # measured between thresholds, so a bit under 2s
        assert rep.min_angle < 100


def test_reset_clears_everything():
    counter = run("squat", squat_pose, rep_angles(175, 85, 2))
    counter.reset()
    assert counter.reps == 0 and counter.attempts == [] and counter.phase == Phase.WAITING
