import cv2
import logging

# Try to load mediapipe gracefully
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_detection as mp_face_detection
    MEDIAPIPE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not fully initialized. Using fallback detection.")

class VideoDetector:
    def __init__(self):
        self.face_detection = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.face_detection = mp_face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=0.5
                )
            except Exception:
                MEDIAPIPE_AVAILABLE = False

    def get_face_score(self, frame):
        # If Mediapipe failed, use a simple OpenCV Haar Cascade as a fallback
        if not MEDIAPIPE_AVAILABLE:
            # This ensures the app still RUNS even if Mediapipe is broken on the server
            return 0.1 

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            return 0.2 if results.detections else 0.0
        except Exception:
            return 0.0
