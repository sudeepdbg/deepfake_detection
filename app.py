import os
import numpy as np
import streamlit as st
from vision_module import VideoDetector
from audio_module import AudioDetector

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Multimodal Deepfake Detector")
st.caption("Frame-level video + audio analysis · MP4, WAV, MP3, JPG, PNG")

# ── Threshold: 0.50 = DEEPFAKE (calibrated on real deepfake data)
DEEPFAKE_THRESHOLD  = 0.50
UNCERTAIN_THRESHOLD = 0.35

uploaded_file = st.file_uploader(
    "Upload a file to analyse",
    type=["mp4", "wav", "mp3", "jpg", "jpeg", "png"]
)


def verdict_ui(score: float, video_score: float, audio_score: float,
               audio_failed: bool):
    st.divider()
    _, col, _ = st.columns([1, 2, 1])
    with col:
        pct = f"{score:.0%}"
        if score >= DEEPFAKE_THRESHOLD:
            st.error(f"## 🚨 DEEPFAKE DETECTED\nSuspicion score: **{pct}**")
        elif score >= UNCERTAIN_THRESHOLD:
            st.warning(f"## ⚠️ INCONCLUSIVE\nSuspicion score: **{pct}** — manual review recommended")
        else:
            st.success(f"## ✅ LIKELY AUTHENTIC\nSuspicion score: **{pct}**")

        if audio_failed:
            st.caption("⚠️ Audio analysis failed — verdict based on video only")

        with st.expander("ℹ️ Score breakdown"):
            st.caption(
                f"**Video score:** {video_score:.2f} · "
                f"**Audio score:** {'N/A' if audio_failed else f'{audio_score:.2f}'}  \n"
                f"Final score = max(video, audio).  "
                f"Thresholds: ≥ {DEEPFAKE_THRESHOLD:.0%} = Deepfake · "
                f"≥ {UNCERTAIN_THRESHOLD:.0%} = Inconclusive · "
                f"< {UNCERTAIN_THRESHOLD:.0%} = Authentic"
            )


def score_bar(label: str, value: float):
    c1, c2, c3 = st.columns([2, 4, 1])
    c1.caption(label)
    c2.progress(float(np.clip(value, 0.0, 1.0)))
    icon = "🔴" if value > 0.5 else "🟡" if value > 0.35 else "🟢"
    c3.caption(f"{icon} {value:.2f}")


if uploaded_file is not None:
    ext      = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_path = f"temp_upload{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

    if st.button("🔍 Run Deepfake Analysis", type="primary", use_container_width=True):

        video_score = 0.0
        audio_score = 0.0
        audio_failed = False

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
                    video_score = vr["score"]
                    st.metric("Video Score", f"{video_score:.2f}",
                              delta=f"{vr['frames_analysed']} / {vr['total_frames']} frames",
                              delta_color="off")
                    score_bar("Noise residual (GAN fingerprint)", vr["mean_score"])
                    score_bar("Peak frame suspicion",             vr["max_score"])
                    score_bar("Temporal inconsistency",           vr["temporal"])
                    if vr["frame_scores"]:
                        import pandas as pd
                        st.caption("Frame-by-frame suspicion score:")
                        st.bar_chart(
                            pd.DataFrame({"suspicion": vr["frame_scores"]}),
                            height=130
                        )

            # Audio-track analysis
            with col_a:
                st.markdown("#### 🔊 Audio Track Analysis")
                with st.spinner("Extracting and analysing audio via ffmpeg…"):
                    ar = AudioDetector().predict_audio_file(tmp_path)

                if "error" in ar:
                    st.warning(f"Audio: {ar['error']}")
                    audio_failed = True
                else:
                    audio_score = ar["score"]
                    dur = ar.get("_duration_s", "?")
                    st.metric("Audio Score", f"{audio_score:.2f}",
                              delta=f"{dur}s analysed", delta_color="off")
                    score_bar("Spectral flux",     ar.get("spec_flux",   0))
                    score_bar("Energy variation",  ar.get("energy_var",  0))
                    score_bar("ZCR consistency",   ar.get("zcr_consist", 0))
                    score_bar("Spectral rolloff",  ar.get("rolloff",     0))
                    score_bar("Silence pattern",   ar.get("silence",     0))
                    with st.expander("📐 Raw diagnostics"):
                        st.json({k: v for k, v in ar.items() if k.startswith("_")})

        # ── Image ────────────────────────────────────────────────────────
        elif ext in (".jpg", ".jpeg", ".png"):
            with st.spinner("Analysing image…"):
                video_score = VideoDetector().analyze_image_file(tmp_path)
            st.metric("Image Suspicion Score", f"{video_score:.2f}")
            st.progress(float(video_score))
            audio_failed = True

        # ── Audio only ───────────────────────────────────────────────────
        elif ext in (".wav", ".mp3"):
            with st.spinner("Analysing audio…"):
                ar = AudioDetector().predict_audio_file(tmp_path)
            if "error" in ar:
                st.error(f"Audio analysis failed: {ar['error']}")
                audio_failed = True
            else:
                audio_score = ar["score"]
                st.metric("Audio Suspicion Score", f"{audio_score:.2f}")
                score_bar("Spectral flux",    ar.get("spec_flux",   0))
                score_bar("Energy variation", ar.get("energy_var",  0))
                score_bar("ZCR consistency",  ar.get("zcr_consist", 0))
                score_bar("Spectral rolloff", ar.get("rolloff",     0))
                score_bar("Silence pattern",  ar.get("silence",     0))

        # ── Verdict ──────────────────────────────────────────────────────
        final_score = max(video_score, audio_score)
        verdict_ui(final_score, video_score, audio_score, audio_failed)
