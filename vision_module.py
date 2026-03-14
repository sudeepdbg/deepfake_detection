import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VideoDetector:
    def __init__(self):
        # FIX 2: Detector is instantiated INSIDE __init__, not at module level.
        # Module-level instantiation was causing an immediate crash on import.
        base_options = python.BaseOptions(
            model_asset_path='face_detection_short_range.tflite'
        )
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def analyze_video_file(self, filepath: str) -> float:
        """
        FIX 3: Reads actual video frames with cv2.VideoCapture instead of
        passing the MP4 directly to mp.Image.create_from_file() (which only
        accepts still images, not video containers).

        Returns a confidence score in [0.0, 1.0].
          > 0.6 → likely synthetic (face detected but flagged heuristically)
          < 0.4 → likely authentic
        Placeholder heuristic: if ANY face is detected, score = 0.9.
        Plug in your real model inference inside the frame loop.
        """
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return 0.0

        total_frames   = 0
        detected_frames = 0
        max_frames     = 30  # sample at most 30 frames for speed

        # Sample evenly across the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        step = max(1, frame_count // max_frames)

        for i in range(0, min(frame_count, max_frames * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # Convert BGR (cv2) → RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            result = self.detector.detect(mp_image)

            if result.detections:
                detected_frames += 1

        cap.release()

        if total_frames == 0:
            return 0.0

        detection_ratio = detected_frames / total_frames

        # Heuristic placeholder:
        # High face presence in a video → treat as likely manipulated.
        # Replace with real deepfake model inference for production use.
        return round(detection_ratio, 3)
