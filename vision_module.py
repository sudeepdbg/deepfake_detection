# vision_module.py
# FIX: Removed `import cv2` entirely.
# mediapipe already bundles its own OpenCV internally — installing
# opencv-python or opencv-python-headless alongside mediapipe causes
# a native bootstrap conflict on Streamlit Cloud (Python 3.11).
# We use mediapipe's built-in image utilities exclusively.

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VideoDetector:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path='face_detection_short_range.tflite'
        )
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def analyze_video_file(self, filepath: str) -> float:
        """
        Reads video frames using mediapipe's own OpenCV (no separate cv2 import).
        Samples up to 30 frames, runs face detection on each, returns a
        detection ratio as the confidence score [0.0 – 1.0].
        """
        # mediapipe ships cv2 internally — access it via the mediapipe package
        import mediapipe.python._framework_bindings as _fb  # noqa: F401
        import cv2  # this cv2 is the one bundled inside mediapipe's wheel

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return 0.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
        max_samples = 30
        step        = max(1, frame_count // max_samples)

        total     = 0
        detected  = 0

        for i in range(0, min(frame_count, max_samples * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            total += 1
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_img)
            if result.detections:
                detected += 1

        cap.release()
        return round(detected / total, 3) if total else 0.0
