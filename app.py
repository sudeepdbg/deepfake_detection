import cv2
import threading
from vision_module import VideoDetector
from audio_module import AudioDetector

def run_detection():
    video = VideoDetector()
    audio = AudioDetector()
    cap = cv2.VideoCapture(0) # Starts your MacBook webcam

    print("System Starting... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Get scores
        v_prob = video.get_face_score(frame)
        
        # Determine Label
        status = "Authentic"
        color = (0, 255, 0) # Green
        
        if v_prob > 0.5:
            status = "Video Deepfake Detected!"
            color = (0, 0, 255) # Red

        # UI Overlay
        cv2.putText(frame, f"Status: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Deepfake Detection POC', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()
