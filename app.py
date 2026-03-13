import streamlit as st
import cv2
import tempfile
from vision_module import VideoDetector
from audio_module import AudioDetector

st.title("Deepfake Detection System (File Upload)")

uploaded_file = st.file_uploader("Upload video/audio file", type=["mp4", "wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("Analyze"):
        # 1. Vision Analysis
        v_detector = VideoDetector()
        # Logic: Open the temp file, grab a sample frame, get score
        v_score = v_detector.analyze_video_file(tfile.name)
        
        # 2. Audio Analysis
        a_detector = AudioDetector()
        a_score = a_detector.analyze_audio_file(tfile.name)
        
        # 3. Present Results
        st.write(f"Video Score: {v_score:.2f}")
        st.write(f"Audio Score: {a_score:.2f}")
        
        if (v_score + a_score) / 2 > 0.5:
            st.error("Conclusion: Machine Generated / Deepfake")
        else:
            st.success("Conclusion: Likely Authentic")
