"""Turn an annotated output video into a small GIF for the README.

python scripts/video_to_gif.py annotated.mp4 assets/demo.gif --start 2 --seconds 8 --width 480
"""

from __future__ import annotations

import argparse

import cv2
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("gif")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--seconds", type=float, default=8.0, help="clip length")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / args.fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    frames, i = [], 0
    while len(frames) < args.seconds * args.fps:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (args.width, int(h * args.width / w)))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).quantize(colors=128))
        i += 1
    if not frames:
        raise SystemExit("No frames read - check the path and --start")
    frames[0].save(
        args.gif, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True
    )
    print(f"wrote {args.gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
