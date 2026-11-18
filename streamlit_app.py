import os
import tempfile
import shutil
import streamlit as st
import pandas as pd
from clipper import (
    COLS, ASPECT_PRESETS,
    download_video, extract_audio, transcribe_audio, analyze_clips,
    render_clip, probe_duration, broll_keywords_for_clip,
)
from groq import Groq

st.set_page_config(page_title="AutoShorts", layout="wide")
st.title("🎬 AutoShorts")
st.caption("Turn YouTube videos into Shorts. Free · No GPU · Runs on Streamlit Cloud.")

# Initialize session state
for key in ["workdir", "video", "words", "clips_df", "rendered_files"]:
    if key not in st.session_state:
        st.session_state[key] = None

def _client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY not set. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    return Groq(api_key=key)

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("Spoken language", ["auto", "en", "hi", "es", "fr", "de", "pt", "id", "ja", "ko"])
    n_clips = st.slider("Number of clips", 1, 10, 4)
    min_len = st.slider("Min clip length (s)", 10, 45, 20)
    max_len = st.slider("Max clip length (s)", 30, 90, 60)
    st.divider()
    st.header("🎨 Creative Controls")
    aspect = st.radio("Aspect ratio", list(ASPECT_PRESETS.keys()), index=0)
    captions = st.checkbox("Burn-in captions", value=True)
    emojis_cb = st.checkbox("Auto emojis", value=True)
    hook_cb = st.checkbox("Hook title overlay", value=True)
    zoom_cb = st.checkbox("Punch-in zoom", value=False, help="Slower on CPU")
    broll_cb = st.checkbox("B-roll overlays", value=False, help="Needs PEXELS_API_KEY")
    brand_color = st.text_input("Caption highlight colour", "&H00FFFF&")

# --- Main: Input ---
url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=…")
upload = st.file_uploader("…or upload a video file", type=["mp4", "mov", "webm", "mkv"])
cookies = st.file_uploader("cookies.txt (optional, fixes YouTube bot checks)", type=["txt"])

# --- Analyze Button ---
if st.button("1️⃣ Analyze Video", type="primary", use_container_width=True):
    if not url and not upload:
        st.error("Paste a URL or upload a file.")
        st.stop()
    if max_len <= min_len:
        st.error("Max length must be greater than min length.")
        st.stop()

    # Clean previous workdir
    if st.session_state.workdir and os.path.isdir(st.session_state.workdir):
        shutil.rmtree(st.session_state.workdir, ignore_errors=True)
    workdir = tempfile.mkdtemp(prefix="autoshorts_")

    with st.status("Analyzing…", expanded=True) as status:
        try:
            client = _client()
            st.write("Getting video…")
            if upload:
                video = os.path.join(workdir, "source.mp4")
                with open(video, "wb") as f:
                    f.write(upload.getbuffer())
            else:
                ck = None
                if cookies:
                    ck = os.path.join(workdir, "cookies.txt")
                    with open(ck, "wb") as f:
                        f.write(cookies.getbuffer())
                video = download_video(url.strip(), workdir, ck)

            duration, W, H = probe_duration(video)
            st.write(f"Video: {duration/60:.1f} min, {W}×{H}")

            st.write("Extracting audio…")
            audio = extract_audio(video, workdir)

            st.write("Transcribing with Whisper (Groq)…")
            words, segments = transcribe_audio(audio, client, language)
            if not segments:
                st.error("No speech detected.")
                st.stop()

            st.write("Finding viral moments with Llama…")
            clips = analyze_clips(segments, client, n_clips, min_len, max_len, duration)

            rows = [[True, c["start"], c["end"], c["title"], c["score"], c["reason"]] for c in clips]
            df = pd.DataFrame(rows, columns=COLS)

            st.session_state.workdir = workdir
            st.session_state.video = video
            st.session_state.words = words
            st.session_state.clips_df = df
            st.session_state.rendered_files = None

            status.update(label="Analysis complete!", state="complete")
        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.exception(e)

# --- Clip Table + Render Button ---
if st.session_state.clips_df is not None:
    st.subheader("Detected Clips")
    st.info("Edit the table below (uncheck clips you don't want, adjust start/end), then click Render.")
    edited_df = st.data_editor(
        st.session_state.clips_df,
        column_config={
            "use": st.column_config.CheckboxColumn("Use", default=True),
            "start": st.column_config.NumberColumn("Start (s)", format="%.2f"),
            "end": st.column_config.NumberColumn("End (s)", format="%.2f"),
            "score": st.column_config.NumberColumn("Score", min_value=1, max_value=10),
        },
        hide_index=True,
        use_container_width=True,
        key="clip_editor",
    )

    if st.button("2️⃣ Render Selected Clips", type="primary", use_container_width=True):
        selected = edited_df[edited_df["use"] == True]
        if selected.empty:
            st.warning("No clips selected.")
            st.stop()

        workdir = st.session_state.workdir
        video_name = os.path.basename(st.session_state.video)
        words = st.session_state.words
        client = _client() if broll_cb else None
        outputs = []

        progress_bar = st.progress(0, text="Starting render…")
        for i, (_, row) in enumerate(selected.iterrows(), start=1):
            clip = {"start": float(row["start"]), "end": float(row["end"]), "title": str(row["title"])}
            if clip["end"] <= clip["start"]:
                continue
            if broll_cb and client:
                clip["broll_keywords"] = broll_keywords_for_clip(clip["title"], client)

            progress_bar.progress((i - 1) / len(selected), text=f"Rendering clip {i}/{len(selected)}…")
            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in clip["title"])[:40].strip() or f"clip{i}"
            out_name = f"{i:02d}_{safe_title.replace(' ', '_')}.mp4"
            try:
                path = render_clip(
                    workdir, video_name, clip, words, out_name,
                    aspect=aspect, captions=captions, hook=hook_cb,
                    emojis=emojis_cb, broll=broll_cb, zoom=zoom_cb,
                    brand_color=brand_color,
                )
                outputs.append(path)
            except Exception as e:
                st.error(f"Clip {i} failed: {e}")

        progress_bar.progress(1.0, text="Done!")
        st.session_state.rendered_files = outputs

# --- Download Section ---
if st.session_state.rendered_files:
    st.subheader("Download Clips")
    for path in st.session_state.rendered_files:
        with open(path, "rb") as f:
            st.download_button(
                label=f"⬇️ {os.path.basename(path)}",
                data=f,
                file_name=os.path.basename(path),
                mime="video/mp4",
                use_container_width=True,
            )