import pytest

from repcounter.cli import build_parser


def test_defaults():
    args = build_parser().parse_args([])
    assert args.exercise == "squat" and args.source == "0" and args.model_variant == "lite"


def test_rejects_unknown_exercise():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--exercise", "deadlift"])
