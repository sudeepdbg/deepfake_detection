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

st.set_page_config(page_title="Multimodal Deepfake Detector", layout="wide")
st.title("🛡️ Multimodal Deepfake Detection System")

# Standard Google STUN servers to help establish connection
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class DetectionProcessor:
    def __init__(self):
        self.v_score = 0.0
        self.a_score = 0.0

    def video_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to BGR for OpenCV processing
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Vision Analysis
        self.v_score = video_detector.get_face_score(img)
        
        # 2. Multimodal Fusion Logic
        status = "Authentic"
        color = (0, 255, 0) # Green

        if self.v_score > 0.5 and self.a_score > 0.5:
            status = "FULL DEEPFAKE"
            color = (0, 0, 255) # Red
        elif self.v_score > 0.5:
            status = "VIDEO MANIPULATION"
            color = (0, 165, 255) # Orange
        elif self.a_score > 0.5:
            status = "AUDIO MANIPULATION"
            color = (255, 0, 255) # Magenta

        # Visual UI Overlay
        cv2.putText(img, f"STATUS: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(img, f"V-Score: {self.v_score:.2f} | A-Score: {self.a_score:.2f}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def audio_callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert audio frame to numpy array
        sound = frame.to_ndarray()
        
        # 3. Audio Analysis
        # Assuming your audio_module has a method to process these arrays
        self.a_score = audio_detector.predict_audio(sound.flatten())
        
        return frame

# Initialize the stateful processor
processor = DetectionProcessor()

# WebRTC UI Component
webrtc_streamer(
    key="deepfake-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=processor.video_callback,
    audio_frame_callback=processor.audio_callback,
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True,
)
