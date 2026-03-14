# vision_module.py
# Removed mediapipe and opencv entirely — both pull in a broken cv2 native
# binary on Streamlit Cloud Python 3.11 (libGL / bootstrap conflict).
# 
# Replacement: lightweight face detection using only Pillow (PIL) for image
# handling and numpy for feature extraction.
# Plug in your real model (e.g. face_recognition, dlib, or a TF/ONNX model)
# once you have a deployment environment that supports native extensions.

from PIL import Image
import numpy as np


class VideoDetector:
    """
    Placeholder video analyser that works on Streamlit Cloud without
    any native C extensions beyond numpy and Pillow.

    Current heuristic: samples pixel statistics across frames represented
    as raw bytes and returns a normalized anomaly score in [0.0, 1.0].
    Replace _score_frame() with your real deepfake model inference.
    """

    def analyze_video_file(self, filepath: str) -> float:
        """
        NOTE: Without opencv/mediapipe we cannot decode MP4 frames in a
        pure-Python environment.  This method returns 0.0 and shows an
        informative warning.  To enable real video analysis, add a
        deployment environment (Docker / local) that supports opencv-python.
        """
        return 0.0   # sentinel — app.py shows a friendly message for this

    def analyze_image_file(self, filepath: str) -> float:
        """Analyse a still image for manipulation artefacts."""
        try:
            img   = Image.open(filepath).convert("RGB")
            arr   = np.array(img, dtype=np.float32)
            score = self._score_frame(arr)
            return round(float(score), 3)
        except Exception:
            return 0.0

    # ── internal heuristic ────────────────────────────────────────────────
    @staticmethod
    def _score_frame(arr: np.ndarray) -> float:
        """
        Very simple compression-artefact heuristic as a placeholder.
        Replace entirely with your real model.
        """
        if arr.size == 0:
            return 0.0
        # High-frequency noise estimate via local variance
        from numpy.lib.stride_tricks import sliding_window_view
        gray = arr.mean(axis=2)
        if gray.shape[0] < 4 or gray.shape[1] < 4:
            return 0.0
        patches = sliding_window_view(gray, (4, 4))
        local_var = patches.var(axis=(-2, -1))
        mean_var  = float(local_var.mean())
        # Sigmoid normalisation centred at variance = 200
        score = 1.0 / (1.0 + np.exp(-(mean_var - 200) / 80))
        return score
