import cv2
import logging

# 1. Global check for MediaPipe
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_detection as mp_face_detection
    MEDIAPIPE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not fully initialized. Using fallback detection.")

class VideoDetector:
    def __init__(self):
        # Tell Python to use the variable defined at the top of the file
        global MEDIAPIPE_AVAILABLE
        
        self.face_detection = None
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.face_detection = mp_face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=0.5
                )
            except Exception as e:
                logging.error(f"Error initializing FaceDetection: {e}")
                MEDIAPIPE_AVAILABLE = False

    def get_face_score(self, frame):
        global MEDIAPIPE_AVAILABLE
        
        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            # Fallback: Return a low score so the app doesn't crash
            return 0.1 

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            return 0.2 if results.detections else 0.0
        except Exception:
            return 0.0
