import os
import streamlit as st
import numpy as np
from vision_module import VideoDetector
from audio_module import AudioDetector

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Deepfake Detector")
st.caption("Frame-level video analysis + audio analysis · Works on MP4, WAV, MP3, JPG, PNG")

uploaded_file = st.file_uploader(
    "Upload a file to analyse",
    type=["mp4", "wav", "mp3", "jpg", "jpeg", "png"]
)

def verdict_ui(score: float):
    st.divider()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if score > 0.60:
            st.error(f"### 🚨 DEEPFAKE DETECTED\nConfidence: **{score:.0%}**")
        elif score > 0.40:
            st.warning(f"### ⚠️ INCONCLUSIVE\nConfidence: **{score:.0%}** — manual review recommended")
        else:
            st.success(f"### ✅ LIKELY AUTHENTIC\nConfidence of being fake: **{score:.0%}**")

def score_bar(label: str, value: float, help_text: str = ""):
    col1, col2, col3 = st.columns([2, 4, 1])
    col1.caption(label)
    color = "🔴" if value > 0.6 else "🟡" if value > 0.4 else "🟢"
    col2.progress(float(value))
    col3.caption(f"{color} {value:.2f}")

if uploaded_file is not None:
    ext      = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_path = f"temp_upload{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size/1024:.1f} KB")

    if st.button("🔍 Run Deepfake Analysis", type="primary", use_container_width=True):

        final_scores = []

        # ── VIDEO (MP4) ────────────────────────────────────────────────────
        if ext == ".mp4":
            left, right = st.columns(2)

            # Frame analysis
            with left:
                st.markdown("#### 🎞️ Frame-Level Analysis")
                with st.spinner("Extracting and analysing frames…"):
                    result = VideoDetector().analyze_video_file(tmp_path, n_frames=30)

                if "error" in result and result.get("frames_analysed", 0) == 0:
                    st.error(f"Frame analysis failed: {result['error']}")
                else:
                    vs = result["score"]
                    final_scores.append(vs)
                    st.metric("Video Score", f"{vs:.2f}",
                              delta=f"{result['frames_analysed']} frames analysed",
                              delta_color="off")
                    score_bar("Avg frame suspicion", result["mean_score"])
                    score_bar("Peak frame suspicion", result["max_score"])
                    score_bar("Temporal inconsistency", result["temporal"])

                    # Frame score sparkline
                    if result["frame_scores"]:
                        st.caption("Frame-by-frame suspicion score:")
                        fs = result["frame_scores"]
                        # show as bar chart
                        import pandas as pd
                        chart_data = pd.DataFrame({"Suspicion": fs})
                        st.bar_chart(chart_data, height=120)

            # Audio analysis
            with right:
                st.markdown("#### 🔊 Audio Track Analysis")
                with st.spinner("Analysing audio track…"):
                    audio_result = AudioDetector().predict_audio_file(tmp_path)

                if "error" in audio_result:
                    st.warning(f"Audio: {audio_result['error']}")
                else:
                    av = audio_result["score"]
                    final_scores.append(av)
                    st.metric("Audio Score", f"{av:.2f}")
                    score_bar("MFCC flatness",      audio_result.get("mfcc_flatness", 0))
                    score_bar("Pitch smoothness",   audio_result.get("pitch_smooth",  0))
                    score_bar("Spectral flux",      audio_result.get("spec_flux",     0))
                    score_bar("Silence pattern",    audio_result.get("silence",       0))
                    score_bar("Harmonic-to-noise",  audio_result.get("hnr",           0))

        # ── IMAGE ──────────────────────────────────────────────────────────
        elif ext in (".jpg", ".jpeg", ".png"):
            with st.spinner("Analysing image…"):
                vs = VideoDetector().analyze_image_file(tmp_path)
            final_scores.append(vs)
            st.metric("Image Score", f"{vs:.2f}")
            st.progress(vs)

        # ── AUDIO ONLY ─────────────────────────────────────────────────────
        elif ext in (".wav", ".mp3"):
            with st.spinner("Analysing audio…"):
                audio_result = AudioDetector().predict_audio_file(tmp_path)
            if "error" in audio_result:
                st.error(audio_result["error"])
            else:
                av = audio_result["score"]
                final_scores.append(av)
                st.metric("Audio Score", f"{av:.2f}")
                score_bar("MFCC flatness",     audio_result.get("mfcc_flatness", 0))
                score_bar("Pitch smoothness",  audio_result.get("pitch_smooth",  0))
                score_bar("Spectral flux",     audio_result.get("spec_flux",     0))
                score_bar("Silence pattern",   audio_result.get("silence",       0))
                score_bar("Harmonic-to-noise", audio_result.get("hnr",           0))

        # ── Final verdict ──────────────────────────────────────────────────
        if final_scores:
            final = float(np.max(final_scores))   # take worst-case score
            verdict_ui(final)
        else:
            st.warning("No analysis completed.")
