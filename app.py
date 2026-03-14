import os
import streamlit as st
from vision_module import VideoDetector
from audio_module import AudioDetector

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️")
st.title("🛡️ Multimodal Deepfake Detector")
st.caption("Supports MP4 video (audio track), WAV, MP3, and image files (JPG/PNG).")

uploaded_file = st.file_uploader(
    "Upload a file to analyse",
    type=["mp4", "wav", "mp3", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    ext      = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_path = f"temp_file{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

    if st.button("🔍 Run Deepfake Analysis", type="primary"):
        v_score = 0.0
        a_score = 0.0
        v_ran   = False
        a_ran   = False

        # ── Image analysis (JPG / PNG) ────────────────────────────────────
        if ext in (".jpg", ".jpeg", ".png"):
            with st.spinner("Analysing image…"):
                try:
                    v_score = VideoDetector().analyze_image_file(tmp_path)
                    v_ran   = True
                    col1, col2 = st.columns([1, 2])
                    col1.metric("🖼️ Image Score", f"{v_score:.2f}")
                    col2.progress(v_score)
                except Exception as e:
                    st.warning(f"Image analysis error: {e}")

        # ── MP4 video — audio track analysis ─────────────────────────────
        elif ext == ".mp4":
            st.info(
                "🎞️ **Video detected.** Extracting and analysing the audio track "
                "(frame-level video analysis requires a local environment with OpenCV)."
            )
            with st.spinner("Extracting audio from MP4 and analysing…"):
                try:
                    a_score = AudioDetector().predict_audio_file(tmp_path)
                    a_ran   = True
                    col1, col2 = st.columns([1, 2])
                    col1.metric("🔊 Audio Track Score", f"{a_score:.2f}")
                    col2.progress(a_score)
                except Exception as e:
                    st.warning(f"Audio extraction failed: {e}")

        # ── Audio files (WAV / MP3) ───────────────────────────────────────
        elif ext in (".wav", ".mp3"):
            with st.spinner("Analysing audio…"):
                try:
                    a_score = AudioDetector().predict_audio_file(tmp_path)
                    a_ran   = True
                    col1, col2 = st.columns([1, 2])
                    col1.metric("🔊 Audio Score", f"{a_score:.2f}")
                    col2.progress(a_score)
                except Exception as e:
                    st.warning(f"Audio analysis error: {e}")

        # ── Verdict ───────────────────────────────────────────────────────
        if v_ran or a_ran:
            st.divider()
            score = max(v_score, a_score)

            if score > 0.6:
                st.error("🚨 **Conclusion: Synthetic / Manipulated Content Detected**")
            elif score > 0.4:
                st.warning("⚠️ **Conclusion: Inconclusive — manual review recommended**")
            else:
                st.success("✅ **Conclusion: Likely Authentic**")

            with st.expander("📊 Score breakdown"):
                c1, c2 = st.columns(2)
                c1.metric("Image / Video Score", f"{v_score:.2f}",
                          help="Based on pixel anomaly heuristics (image only)")
                c2.metric("Audio Score", f"{a_score:.2f}",
                          help="Based on MFCC, spectral & dynamic-range features")
                st.caption(
                    "**Score guide:** 0.0 = likely authentic · 1.0 = likely synthetic  \n"
                    "These are heuristic scores — not a trained deepfake model. "
                    "Replace `_score_frame()` and `predict_audio_file()` with your model's inference."
                )
        else:
            st.warning("No analysis was completed for this file type.")
