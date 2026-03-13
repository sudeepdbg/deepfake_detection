import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from vision_module import VideoDetector
from audio_module import AudioDetector

# Initialize detectors
video_detector = VideoDetector()
audio_detector = AudioDetector()

st.set_page_config(page_title="Real-Time Multimodal Deepfake Detector", layout="wide")
st.title("🛡️ Multimodal Deepfake Detection System")
st.markdown("Analyzing both visual and auditory cues in real-time.")

# ICE Servers for STUN/TURN (Helps bypass firewalls)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class DetectionProcessor:
    def __init__(self):
        self.v_score = 0.0
        self.a_score = 0.0

    def video_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Vision Analysis
        self.v_score = video_detector.get_face_score(img)
        
        # 2. Logic for labels based on Multimodal Fusion
        status = "Authentic"
        color = (0, 255, 0) # Green

        if self.v_score > 0.5 and self.a_score > 0.5:
            status = "FULL MANIPULATION"
            color = (0, 0, 255)
        elif self.v_score > 0.5:
            status = "VIDEO FAKE"
            color = (0, 165, 255) # Orange
        elif self.a_score > 0.5:
            status = "AUDIO FAKE"
            color = (255, 0, 255) # Magenta

        # Overlay results on the frame
        cv2.putText(img, f"STATUS: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(img, f"V-Score: {self.v_score:.2f} | A-Score: {self.a_score:.2f}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def audio_callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert audio frame to numpy array for RawNet2
        sound = frame.to_ndarray()
        
        # 3. Audio Analysis (process in chunks)
        # We flatten because RawNet2 usually expects a 1D waveform
        self.a_score = audio_detector.predict_audio(sound.flatten())
        
        return frame

# Initialize the processor
processor = DetectionProcessor()

# WebRTC Streamer UI
webrtc_streamer(
    key="deepfake-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=processor.video_callback,
    audio_frame_callback=processor.audio_callback,
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True,
)

st.sidebar.header("System Metrics")
st.sidebar.write("Latency: < 1.0s")
st.sidebar.write("Models: MesoNet (Video), RawNet2 (Audio)")
