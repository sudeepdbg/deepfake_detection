import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VideoDetector:
    def __init__(self):
        # Using the modern Tasks API which is more stable
        base_options = python.BaseOptions(model_asset_path='face_detection_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def analyze_video_file(self, filepath):
        # Load the image/frame
        image = mp.Image.create_from_file(filepath)
        # Perform detection
        detection_result = self.detector.detect(image)
        
        # Return a simple score based on whether a face was detected
        return 0.9 if detection_result.detections else 0.1
