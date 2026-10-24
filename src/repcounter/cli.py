"""Command line entry point.

Examples
--------
Webcam, live squat counting:
    python -m repcounter --exercise squat

Analyse a recorded video and save an annotated copy:
    python -m repcounter --exercise pushup --source my_set.mp4 --output annotated.mp4 --no-display
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .exercises import EXERCISES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="repcounter", description="AI rep counter & form checker")
    p.add_argument("-e", "--exercise", choices=sorted(EXERCISES), default="squat")
    p.add_argument("-s", "--source", default="0", help="webcam index (default 0) or path to a video file")
    p.add_argument("-o", "--output", help="save annotated video to this .mp4 path")
    p.add_argument("--client", default="self", help="client id stored in the workout log")
    p.add_argument("--log-dir", default="logs", help="where session JSON + workout_log.csv are written")
    p.add_argument("--no-display", action="store_true", help="don't open a preview window")
    p.add_argument("--mirror", action="store_true", help="flip horizontally (selfie view)")
    p.add_argument("--model", help="path to a pose_landmarker .task file (downloaded automatically if omitted)")
    p.add_argument(
        "--model-variant", choices=["lite", "full", "heavy"], default="lite", help="bigger = more accurate but slower"
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .pipeline import process_video  # heavy imports only when actually running

    source = int(args.source) if args.source.isdigit() else args.source
    session = process_video(
        source,
        args.exercise,
        output_path=args.output,
        display=not args.no_display,
        client_id=args.client,
        model_path=args.model,
        model_variant=args.model_variant,
        mirror=args.mirror,
    )

    summary = session.summary()
    log_dir = Path(args.log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = session.save_json(log_dir / f"{stamp}_{args.client}_{args.exercise}.json")
    csv_path = session.append_csv(log_dir / "workout_log.csv")

    print("\n" + summary.pretty())
    print(f"\nSaved session -> {json_path}\nWorkout history -> {csv_path}")
    if args.output:
        print(f"Annotated video -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
