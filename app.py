import os
import numpy as np
import streamlit as st
from vision_module import VideoDetector
from audio_module import AudioDetector

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Multimodal Deepfake Detector")
st.caption("Frame-level video + audio analysis · MP4, WAV, MP3, JPG, PNG")

uploaded_file = st.file_uploader(
    "Upload a file to analyse",
    type=["mp4", "wav", "mp3", "jpg", "jpeg", "png"]
)


def verdict_ui(score: float):
    st.divider()
    _, col, _ = st.columns([1, 2, 1])
    with col:
        pct = f"{score:.0%}"
        if score > 0.60:
            st.error(f"## 🚨 DEEPFAKE DETECTED\nSuspicion score: **{pct}**")
        elif score > 0.40:
            st.warning(f"## ⚠️ INCONCLUSIVE\nSuspicion score: **{pct}** — manual review recommended")
        else:
            st.success(f"## ✅ LIKELY AUTHENTIC\nSuspicion score: **{pct}**")


def score_bar(label: str, value: float):
    c1, c2, c3 = st.columns([2, 4, 1])
    c1.caption(label)
    c2.progress(float(np.clip(value, 0.0, 1.0)))
    icon = "🔴" if value > 0.6 else "🟡" if value > 0.4 else "🟢"
    c3.caption(f"{icon} {value:.2f}")


if uploaded_file is not None:
    ext      = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_path = f"temp_upload{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

    if st.button("🔍 Run Deepfake Analysis", type="primary", use_container_width=True):
        final_scores = []

        # ── MP4 Video ────────────────────────────────────────────────────
        if ext == ".mp4":
            col_v, col_a = st.columns(2)

            # Frame-level analysis
            with col_v:
                st.markdown("#### 🎞️ Frame-Level Analysis")
                with st.spinner("Decoding and analysing frames…"):
                    vr = VideoDetector().analyze_video_file(tmp_path, n_frames=30)

                if vr.get("frames_analysed", 0) == 0:
                    st.error(f"Frame analysis failed: {vr.get('error', 'unknown')}")
                else:
                    vs = vr["score"]
                    final_scores.append(vs)
                    st.metric("Video Score", f"{vs:.2f}",
                              delta=f"{vr['frames_analysed']} / {vr['total_frames']} frames",
                              delta_color="off")
                    score_bar("Avg frame suspicion",   vr["mean_score"])
                    score_bar("Peak frame suspicion",  vr["max_score"])
                    score_bar("Temporal inconsistency",vr["temporal"])
                    if vr["frame_scores"]:
                        import pandas as pd
                        st.caption("Frame-by-frame suspicion:")
                        st.bar_chart(pd.DataFrame({"score": vr["frame_scores"]}), height=130)

            # Audio-track analysis
            with col_a:
                st.markdown("#### 🔊 Audio Track Analysis")
                with st.spinner("Extracting and analysing audio…"):
                    ar = AudioDetector().predict_audio_file(tmp_path)

                if "error" in ar:
                    st.warning(f"Audio analysis: {ar['error']}")
                    # Still append 0 so verdict doesn't ignore the channel
                    final_scores.append(0.0)
                else:
                    av = ar["score"]
                    final_scores.append(av)
                    st.metric("Audio Score", f"{av:.2f}")
                    score_bar("MFCC flatness",    ar.get("mfcc_flatness", 0))
                    score_bar("Pitch smoothness", ar.get("pitch_smooth",  0))
                    score_bar("Spectral flux",    ar.get("spec_flux",     0))
                    score_bar("Silence pattern",  ar.get("silence",       0))
                    score_bar("Harmonic ratio",   ar.get("harmonic_ratio",0))
                    with st.expander("📐 Raw diagnostics"):
                        st.json({
                            "mfcc_std":  ar.get("_mfcc_std"),
                            "log_flux":  ar.get("_log_flux"),
                            "silence_r": ar.get("_silence_r"),
                        })

        # ── Image ────────────────────────────────────────────────────────
        elif ext in (".jpg", ".jpeg", ".png"):
            with st.spinner("Analysing image…"):
                vs = VideoDetector().analyze_image_file(tmp_path)
            final_scores.append(vs)
            st.metric("Image Suspicion Score", f"{vs:.2f}")
            st.progress(float(vs))

        # ── Audio only ───────────────────────────────────────────────────
        elif ext in (".wav", ".mp3"):
            with st.spinner("Analysing audio…"):
                ar = AudioDetector().predict_audio_file(tmp_path)
            if "error" in ar:
                st.error(f"Audio analysis failed: {ar['error']}")
                final_scores.append(0.0)
            else:
                av = ar["score"]
                final_scores.append(av)
                st.metric("Audio Suspicion Score", f"{av:.2f}")
                score_bar("MFCC flatness",    ar.get("mfcc_flatness", 0))
                score_bar("Pitch smoothness", ar.get("pitch_smooth",  0))
                score_bar("Spectral flux",    ar.get("spec_flux",     0))
                score_bar("Silence pattern",  ar.get("silence",       0))
                score_bar("Harmonic ratio",   ar.get("harmonic_ratio",0))
                with st.expander("📐 Raw diagnostics"):
                    st.json({
                        "mfcc_std":  ar.get("_mfcc_std"),
                        "log_flux":  ar.get("_log_flux"),
                        "silence_r": ar.get("_silence_r"),
                    })

        # ── Verdict ──────────────────────────────────────────────────────
        if final_scores:
            # Use max score — if either modality looks synthetic, flag it
            verdict_ui(float(np.max(final_scores)))
        else:
            st.warning("No analysis completed.")
