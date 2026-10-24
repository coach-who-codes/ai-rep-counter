import math

import pytest

from repcounter.geometry import EMA, angle_from_vertical, joint_angle, offset_from_line


@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        ((1, 0), (0, 0), (0, 1), 90.0),
        ((1, 0), (0, 0), (-1, 0), 180.0),
        ((1, 0), (0, 0), (1, 1), 45.0),
        ((0, 1), (0, 0), (0, 1), 0.0),
    ],
)
def test_joint_angle(a, b, c, expected):
    assert joint_angle(a, b, c) == pytest.approx(expected, abs=1e-6)


def test_joint_angle_is_symmetric():
    assert joint_angle((3, 1), (0, 0), (-2, 5)) == pytest.approx(joint_angle((-2, 5), (0, 0), (3, 1)))


def test_joint_angle_degenerate_segment_is_nan():
    assert math.isnan(joint_angle((0, 0), (0, 0), (1, 1)))


def test_angle_from_vertical():
    assert angle_from_vertical((0, 0), (0, 10)) == pytest.approx(0)  # top directly above bottom
    assert angle_from_vertical((10, 10), (0, 10)) == pytest.approx(90)  # horizontal
    assert angle_from_vertical((10, 0), (0, 10)) == pytest.approx(45)
    assert angle_from_vertical((-10, 0), (0, 10)) == pytest.approx(45)  # direction-agnostic


def test_offset_from_line_sign_independent_of_facing():
    # hip 20px below a horizontal shoulder-ankle line (image y grows downwards)
    assert offset_from_line((100, 120), (0, 100), (200, 100)) == pytest.approx(20)
    assert offset_from_line((100, 120), (200, 100), (0, 100)) == pytest.approx(20)
    assert offset_from_line((100, 80), (0, 100), (200, 100)) == pytest.approx(-20)


def test_ema_smooths_and_ignores_nan():
    ema = EMA(alpha=0.5)
    assert ema.update(10) == 10
    assert ema.update(20) == 15
    assert ema.update(float("nan")) == 15
    ema.reset()
    assert ema.update(4) == 4


def test_ema_rejects_bad_alpha():
    with pytest.raises(ValueError):
        EMA(alpha=0)
