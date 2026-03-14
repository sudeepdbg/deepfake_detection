import os
import streamlit as st
from vision_module import VideoDetector
from audio_module import AudioDetector

st.title("🛡️ Multimodal Deepfake Detector")
st.caption("Upload a video, audio, or image file for AI-based authenticity analysis.")

uploaded_file = st.file_uploader(
    "Upload a file to analyse",
    type=["mp4", "wav", "mp3", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_path = f"temp_file{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

    if st.button("🔍 Run Deepfake Analysis"):
        v_score = 0.0
        a_score = 0.0
        v_ran   = False
        a_ran   = False

        # ── Video (MP4) ───────────────────────────────────────────────────
        if ext == ".mp4":
            st.warning(
                "⚠️ **MP4 video frame analysis requires a server-side OpenCV "
                "installation** (not available on Streamlit Cloud). "
                "Upload a `.jpg` / `.png` still frame instead, or run the "
                "app locally with `opencv-python` installed."
            )

        # ── Image (still frame) ───────────────────────────────────────────
        if ext in (".jpg", ".jpeg", ".png"):
            with st.spinner("Analysing image…"):
                try:
                    v_score = VideoDetector().analyze_image_file(tmp_path)
                    v_ran   = True
                    st.write(f"### 🖼️ Image Analysis Score: `{v_score:.2f}`")
                except Exception as e:
                    st.warning(f"Image analysis error: {e}")

        # ── Audio (.wav / .mp3) ───────────────────────────────────────────
        if ext in (".wav", ".mp3"):
            with st.spinner("Analysing audio…"):
                try:
                    a_score = AudioDetector().predict_audio_file(tmp_path)
                    a_ran   = True
                    st.write(f"### 🔊 Audio Score: `{a_score:.2f}`")
                except Exception as e:
                    st.warning(f"Audio analysis error: {e}")
        elif ext == ".mp4":
            st.info("ℹ️ Audio-track extraction from MP4 is not yet implemented.")

        # ── Verdict ───────────────────────────────────────────────────────
        if v_ran or a_ran:
            st.divider()
            max_score = max(v_score, a_score)
            if max_score > 0.6:
                st.error("🚨 **Synthetic / Manipulated Content Detected**")
            elif max_score > 0.4:
                st.warning("⚠️ **Inconclusive — manual review recommended**")
            else:
                st.success("✅ **Likely Authentic**")

            with st.expander("📊 Score details"):
                c1, c2 = st.columns(2)
                c1.metric("Image / Video Score", f"{v_score:.2f}")
                c2.metric("Audio Score",          f"{a_score:.2f}")
                st.caption(
                    "Scores are placeholder heuristics. "
                    "Replace `VideoDetector._score_frame` and "
                    "`AudioDetector.predict_audio_file` with your trained model."
                )
