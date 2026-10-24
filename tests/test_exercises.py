import pytest

from repcounter import RepCounter, get_exercise
from repcounter.exercises import BicepCurl, PushUp, Squat
from repcounter.simulate import curl_pose, iter_poses, pushup_pose, rep_angles, squat_pose


@pytest.mark.parametrize("angle", [175, 140, 110, 90, 70])
def test_squat_primary_angle_matches_ground_truth(angle):
    assert Squat().primary_angle(squat_pose(angle), "left") == pytest.approx(angle, abs=0.01)


@pytest.mark.parametrize("angle", [170, 120, 80])
def test_pushup_and_curl_primary_angles(angle):
    assert PushUp().primary_angle(pushup_pose(angle), "left") == pytest.approx(angle, abs=0.01)
    assert BicepCurl().primary_angle(curl_pose(angle), "left") == pytest.approx(angle, abs=0.01)


def test_squat_flags_excessive_forward_lean():
    ex = Squat()
    assert ex.check_form(squat_pose(85, torso_lean=25), "left", "bottom") == []
    assert [i.code for i in ex.check_form(squat_pose(85, torso_lean=65), "left", "bottom")] == ["chest_up"]


@pytest.mark.parametrize("offset, expected", [(0, []), (40, ["hips_sagging"]), (-40, ["hips_piking"])])
@pytest.mark.parametrize("facing", ["left", "right"])
def test_pushup_body_line(offset, expected, facing):
    pose = pushup_pose(120, hip_offset=offset, facing=facing)
    side = pose.best_side(PushUp.parts)
    assert [i.code for i in PushUp().check_form(pose, side, "down")] == expected


def test_pushup_not_counted_while_standing():
    # curl-like arm movement while upright must not count as push-ups
    counter = RepCounter(get_exercise("pushup"))
    for t, pose in iter_poses(curl_pose, rep_angles(165, 40, 3)):
        result = counter.update(pose, t)
    assert counter.reps == 0
    assert "plank" in result.message


def test_curl_flags_elbow_drift_and_swing():
    ex = BicepCurl()
    assert ex.check_form(curl_pose(90), "left", "down") == []
    assert [i.code for i in ex.check_form(curl_pose(90, elbow_drift=45), "left", "down")] == ["elbow_drift"]
    codes = [i.code for i in ex.check_form(curl_pose(90, torso_lean=20), "left", "down")]
    assert "body_swing" in codes


def test_form_issue_lowers_rep_score():
    counter = RepCounter(get_exercise("squat"))
    for t, pose in iter_poses(squat_pose, rep_angles(175, 85, 3), torso_lean=65):
        counter.update(pose, t)
    assert counter.reps == 3 and counter.good_reps == 0
    assert all(r.score == 75 and r.issues == ["chest_up"] for r in counter.attempts)


def test_unknown_exercise_raises_helpful_error():
    with pytest.raises(ValueError, match="Choose from"):
        get_exercise("deadlift")
