import streamlit as st
import cv2
import librosa
from vision_module import VideoDetector
from audio_module import AudioDetector

st.title("🛡️ Multimodal Deepfake Detector")

# File uploader accepts both Video and Audio
uploaded_file = st.file_uploader("Upload a file to analyze", type=["mp4", "wav", "mp3"])

if uploaded_file is not None:
    # Save to a temporary file path
    with open("temp_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Run Deepfake Analysis"):
        v_score = 0.0
        a_score = 0.0
        
        # Determine if it's video or audio based on extension
        if uploaded_file.name.endswith(".mp4"):
            v_detector = VideoDetector()
            v_score = v_detector.analyze_video_file("temp_file")
            st.write(f"### Video Confidence: {v_score:.2f}")
        
        else:
            a_detector = AudioDetector()
            a_score = a_detector.predict_audio_file("temp_file")
            st.write(f"### Audio Confidence: {a_score:.2f}")
            
        # Final Conclusion
if v_score > 0.6 or a_score > 0.6:
    st.error("Conclusion: Synthetic Content Detected (Triggered by high confidence in video or audio)")
elif v_score > 0.4 or a_score > 0.4:
    st.warning("Conclusion: Inconclusive - Further analysis required")
else:
    st.success("Conclusion: Likely Authentic")
