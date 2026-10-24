"""Render the README demo from a *simulated* skeleton (no camera needed).

Simulates a 5-rep squat set: 2 good reps, 1 half rep, 1 rep with too much
forward lean, then 1 good rep - and runs it through the real counter + overlay.

    python scripts/make_demo.py
Outputs: assets/demo_simulated.gif, assets/angle_trace.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repcounter import RepCounter, WorkoutSession, get_exercise  # noqa: E402
from repcounter.overlay import draw_hud  # noqa: E402
from repcounter.simulate import rep_angles, squat_pose  # noqa: E402

FPS = 30
PLAN = [  # (bottom knee angle, torso lean at bottom)
    (85, 25),
    (85, 25),
    (125, 25),  # half rep
    (85, 62),  # leaning too far forward
    (82, 25),
]


def draw_mannequin(canvas, pose) -> None:
    """Thick limbs + head so the simulated skeleton reads as a person."""
    body = (120, 112, 104)
    p = {k: (int(v.x), int(v.y)) for k, v in pose.joints.items()}
    limbs = [
        ("ankle", "knee", 26),
        ("knee", "hip", 32),
        ("hip", "shoulder", 40),
        ("shoulder", "elbow", 20),
        ("elbow", "wrist", 16),
        ("heel", "foot_index", 12),
    ]
    for a, b, width in limbs:
        cv2.line(canvas, p[f"left_{a}"], p[f"left_{b}"], body, width, cv2.LINE_AA)
    cv2.circle(canvas, p["nose"], 24, body, -1, cv2.LINE_AA)


def build_frames():
    t0, frames = 0.0, []
    for i, (bottom, lean) in enumerate(PLAN):
        pause = 0.8 if i == 0 else 0.5
        for t, angle in rep_angles(175, bottom, 1, fps=FPS, rep_seconds=2.0, pause_seconds=pause, noise=1.0, seed=i):
            frames.append((t0 + t, angle, lean))
        t0 = frames[-1][0] + 1 / FPS
    return frames


def main() -> None:
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    counter = RepCounter(get_exercise("squat"))
    session = WorkoutSession(counter, client_id="demo")
    gif_frames = []

    for i, (t, angle, lean) in enumerate(build_frames()):
        pose = squat_pose(angle, torso_lean=lean)
        # place the figure nicely on a 640x480 canvas
        for j in pose.joints.values():
            j.x += 40
            j.y += 30
        canvas = np.full((480, 640, 3), (48, 42, 38), np.uint8)
        cv2.line(canvas, (0, 440), (640, 440), (90, 90, 90), 2)
        draw_mannequin(canvas, pose)
        result = counter.update(pose, t)
        session.record(t, result.angle)
        draw_hud(canvas, counter, result, pose)
        cv2.putText(
            canvas, "simulated skeleton", (470, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA
        )
        if i % 2 == 0:  # 15 fps GIF keeps the file small
            gif_frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).quantize(colors=64))

    gif_frames[0].save(
        assets / "demo_simulated.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / (FPS / 2)),
        loop=0,
        optimize=True,
    )

    # angle trace with rep markers
    s = session.summary()
    ts, angles = zip(*session.angle_trace)
    ex = counter.exercise
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=120)
    ax.plot(ts, angles, color="#3b6fd1", lw=2, label="knee angle (smoothed)")
    ax.axhline(ex.top_angle, color="#888", ls="--", lw=1)
    ax.axhline(ex.partial_angle, color="#bbb", ls=":", lw=1)
    ax.axhline(ex.bottom_angle, color="#2e9e5b", ls="--", lw=1)
    ax.text(ts[-1], ex.top_angle + 2, "top", ha="right", fontsize=8, color="#666")
    ax.text(ts[-1], ex.partial_angle + 2, "rep started", ha="right", fontsize=8, color="#888")
    ax.text(ts[-1], ex.bottom_angle - 7, "full depth", ha="right", fontsize=8, color="#2e9e5b")
    for rep in s.rep_details:
        color = "#2e9e5b" if rep["completed"] and not rep["issues"] else "#d9822b" if rep["completed"] else "#d64545"
        label = "good" if color == "#2e9e5b" else ("half rep" if not rep["completed"] else rep["issues"][0])
        ax.axvspan(rep["start_time"], rep["end_time"], color=color, alpha=0.12)
        ax.text(
            (rep["start_time"] + rep["end_time"]) / 2,
            182,
            f"#{rep['index']} {label}",
            ha="center",
            fontsize=8,
            color=color,
        )
    ax.set_ylim(60, 192)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("degrees")
    ax.set_title(
        f"Squat set: {s.reps} reps counted, {s.good_reps} good, {s.partial_reps} half - form score {s.form_score}",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(assets / "angle_trace.png")
    print(s.pretty())
    print("wrote", assets / "demo_simulated.gif", "and", assets / "angle_trace.png")


if __name__ == "__main__":
    main()
