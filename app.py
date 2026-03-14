import os
import streamlit as st
from vision_module import VideoDetector
from audio_module import AudioDetector

st.title("🛡️ Multimodal Deepfake Detector")
st.caption("Upload a video or audio file for AI-based authenticity analysis.")

uploaded_file = st.file_uploader(
    "Upload a file to analyse",
    type=["mp4", "wav", "mp3"]
)

if uploaded_file is not None:
    # Save to a temp file with the correct extension so librosa / cv2 can read it
    ext = os.path.splitext(uploaded_file.name)[1].lower()   # e.g. ".mp4"
    tmp_path = f"temp_file{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"File saved: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("🔍 Run Deepfake Analysis"):
        v_score = 0.0
        a_score = 0.0

        # ── Video analysis ──────────────────────────────────────────────
        if ext == ".mp4":
            with st.spinner("Analysing video frames…"):
                try:
                    v_detector = VideoDetector()
                    v_score    = v_detector.analyze_video_file(tmp_path)
                    st.write(f"### 🎥 Video Confidence Score: `{v_score:.2f}`")
                except Exception as e:
                    st.warning(f"Video analysis skipped: {e}")

        # ── Audio analysis ──────────────────────────────────────────────
        # FIX 5: Only run AudioDetector on audio files (.wav / .mp3).
        # Previously it always ran, including on .mp4 where audio extraction
        # is not implemented — causing misleading / incorrect scores.
        if ext in (".wav", ".mp3"):
            with st.spinner("Analysing audio signal…"):
                try:
                    a_detector = AudioDetector()
                    a_score    = a_detector.predict_audio_file(tmp_path)
                    st.write(f"### 🔊 Audio Confidence Score: `{a_score:.2f}`")
                except Exception as e:
                    st.warning(f"Audio analysis skipped: {e}")
        elif ext == ".mp4":
            st.info("ℹ️ Audio track analysis is not yet supported for MP4 files.")

        # ── Final verdict ───────────────────────────────────────────────
        st.divider()
        if v_score == 0.0 and a_score == 0.0:
            st.warning("⚠️ No analysis could be completed. Check that the model file exists.")
        elif v_score > 0.6 or a_score > 0.6:
            st.error("🚨 Conclusion: **Synthetic / Manipulated Content Detected**")
        elif v_score > 0.4 or a_score > 0.4:
            st.warning("⚠️ Conclusion: **Inconclusive — manual review recommended**")
        else:
            st.success("✅ Conclusion: **Likely Authentic**")

        # ── Score breakdown ─────────────────────────────────────────────
        with st.expander("📊 Score details"):
            col1, col2 = st.columns(2)
            col1.metric("Video Score",  f"{v_score:.2f}", help="0 = likely real · 1 = likely synthetic")
            col2.metric("Audio Score",  f"{a_score:.2f}", help="0 = likely real · 1 = likely synthetic")
            st.caption(
                "Scores are heuristic placeholders. "
                "Replace `VideoDetector.analyze_video_file` and "
                "`AudioDetector.predict_audio_file` with your trained model's inference."
            )
