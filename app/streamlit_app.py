"""Upload a workout video -> get reps, form score, coaching cues and an annotated video.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repcounter.exercises import EXERCISES, get_exercise  # noqa: E402
from repcounter.pipeline import process_video  # noqa: E402


def to_browser_mp4(src: Path) -> Path:
    """OpenCV writes mp4v, which browsers won't play. Re-encode to H.264 if ffmpeg is available."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
    )
    return dst


st.set_page_config(page_title="AI Rep Counter", page_icon=":weight_lifter:", layout="wide")
st.title("AI Rep Counter & Form Checker")
st.caption("Pose estimation (MediaPipe) + joint-angle state machine. Film yourself side-on, full body in frame.")

with st.sidebar:
    exercise = st.selectbox("Exercise", sorted(EXERCISES), format_func=lambda n: get_exercise(n).display_name)
    client_id = st.text_input("Client ID", value="self")
    variant = st.radio("Model", ["lite", "full"], horizontal=True, help="full = more accurate, slower")
    st.markdown("**Tips**\n- Phone at hip height, 2-3 m away\n- Side-on view\n- Good lighting, plain background")

uploaded = st.file_uploader("Upload a video of your set", type=["mp4", "mov", "avi", "m4v"])

if uploaded and st.button("Analyse set", type="primary"):
    workdir = Path(tempfile.mkdtemp())
    src = workdir / uploaded.name
    src.write_bytes(uploaded.getbuffer())
    out = workdir / "annotated.mp4"

    bar = st.progress(0.0, text="Detecting pose & counting reps...")

    def on_progress(done: int, total: int) -> None:
        if total:
            bar.progress(min(done / total, 1.0), text=f"Frame {done}/{total}")

    session = process_video(
        str(src), exercise, output_path=str(out), client_id=client_id, model_variant=variant, progress=on_progress
    )
    bar.empty()
    s = session.summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reps counted", s.reps)
    c2.metric("Good-form reps", s.good_reps)
    c3.metric("Half reps", s.partial_reps)
    c4.metric("Form score", f"{s.form_score:.0f}/100" if s.form_score is not None else "-")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Annotated video")
        playable = to_browser_mp4(out)
        st.video(playable.read_bytes())
        st.download_button("Download annotated video", playable.read_bytes(), file_name="annotated.mp4")
    with right:
        st.subheader("Joint angle over time")
        if session.angle_trace:
            trace = pd.DataFrame(session.angle_trace, columns=["time_s", "angle"]).set_index("time_s")
            st.line_chart(trace)
        st.subheader("Rep by rep")
        if s.rep_details:
            reps = pd.DataFrame(s.rep_details)[["index", "completed", "duration", "min_angle", "score", "issues"]]
            reps["issues"] = reps["issues"].apply(lambda x: ", ".join(x) or "-")
            st.dataframe(reps, hide_index=True, width="stretch")
        else:
            st.info("No reps detected - check the camera angle and that your whole body is visible.")

    st.download_button(
        "Download session JSON",
        data=Path(session.save_json(workdir / "session.json")).read_text(),
        file_name="session.json",
        mime="application/json",
    )
