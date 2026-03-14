"""
vision_module.py
Frame-level deepfake detection using imageio + imageio-ffmpeg.
Works on Streamlit Cloud — no OpenCV, no mediapipe, no libGL needed.

Detection heuristics applied per frame:
  1. JPEG blocking artefacts  — GAN/diffusion outputs often show grid noise
  2. Colour channel correlation — natural faces have strong R/G/B correlation;
     deepfakes generated per-channel often deviate
  3. High-frequency residual (noise analysis) — deepfakes leave a distinct
     high-freq fingerprint after subtracting a smoothed version
  4. Edge coherence — real faces have smooth edge transitions; GAN faces
     often have incoherent edges around hair/background boundaries
  5. Temporal inconsistency — between consecutive frames, deepfakes show
     unnatural pixel-level jitter in face regions
"""

import numpy as np
from PIL import Image, ImageFilter


# ── Per-frame feature extractors ─────────────────────────────────────────────

def _blocking_artefact_score(arr: np.ndarray) -> float:
    """
    Measures 8×8 DCT-block discontinuities — a GAN/compression artefact.
    High score → suspicious.
    """
    gray = arr.mean(axis=2).astype(np.float32)
    h, w = gray.shape
    # Horizontal block boundaries at every 8 pixels
    h_diff = np.abs(gray[:, 7:w - 1:8] - gray[:, 8:w:8]).mean()
    # Vertical block boundaries
    v_diff = np.abs(gray[7:h - 1:8, :] - gray[8:h:8, :]).mean()
    # Interior differences (should be similar in natural images)
    h_int  = np.abs(np.diff(gray, axis=1)).mean()
    v_int  = np.abs(np.diff(gray, axis=0)).mean()
    interior = (h_int + v_int) / 2 + 1e-6
    ratio = (h_diff + v_diff) / 2 / interior
    # deepfakes: ratio tends to be lower (smooth blocks) or higher (sharp edges)
    return float(np.clip(abs(ratio - 1.0) / 2.0, 0, 1))


def _channel_correlation_score(arr: np.ndarray) -> float:
    """
    Natural skin has strong R-G and R-B correlation.
    Deepfakes generated independently per-channel break this.
    Low correlation → high suspicion.
    """
    r = arr[:, :, 0].flatten().astype(np.float32)
    g = arr[:, :, 1].flatten().astype(np.float32)
    b = arr[:, :, 2].flatten().astype(np.float32)
    rg = float(np.corrcoef(r, g)[0, 1])
    rb = float(np.corrcoef(r, b)[0, 1])
    avg_corr = (rg + rb) / 2
    # Natural images: avg_corr ~ 0.85–0.99; deepfakes: lower
    return float(np.clip(1.0 - avg_corr, 0, 1))


def _noise_residual_score(arr: np.ndarray) -> float:
    """
    Extracts high-frequency residual by subtracting a Gaussian-blurred version.
    Deepfakes leave a characteristic noise pattern at specific frequencies.
    """
    pil  = Image.fromarray(arr)
    blur = pil.filter(ImageFilter.GaussianBlur(radius=2))
    residual = np.abs(np.array(pil, dtype=np.float32) -
                      np.array(blur, dtype=np.float32))
    mean_res  = residual.mean()
    # Normalize: deepfake residuals tend to cluster in 8–25 range
    score = np.clip((mean_res - 4.0) / 20.0, 0, 1)
    return float(score)


def _edge_coherence_score(arr: np.ndarray) -> float:
    """
    Measures smoothness of edges using Laplacian variance.
    GAN outputs often have incoherent / aliased edges.
    """
    from PIL import ImageFilter
    gray = Image.fromarray(arr).convert("L")
    lap  = gray.filter(ImageFilter.Kernel(
        size=3, kernel=[-1,-1,-1,-1,8,-1,-1,-1,-1], scale=1, offset=128
    ))
    lap_arr = np.array(lap, dtype=np.float32)
    var = float(lap_arr.var())
    # Very high or very low Laplacian variance is suspicious
    # Natural images: ~200–2000
    if var < 50:
        return 0.8   # suspiciously smooth (over-processed deepfake)
    if var > 5000:
        return 0.7   # suspiciously noisy
    return float(np.clip(1.0 - (var - 50) / 4950, 0, 1))


def score_frame(frame_arr: np.ndarray) -> float:
    """
    Combines all per-frame heuristics into a single [0, 1] score.
    Returns higher values for more suspicious frames.
    """
    arr = np.array(frame_arr, dtype=np.uint8)
    if arr.ndim == 2:                      # grayscale → RGB
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 4:               # RGBA → RGB
        arr = arr[:, :, :3]

    s1 = _blocking_artefact_score(arr)
    s2 = _channel_correlation_score(arr)
    s3 = _noise_residual_score(arr)
    s4 = _edge_coherence_score(arr)

    combined = (s1 * 0.20 + s2 * 0.35 + s3 * 0.25 + s4 * 0.20)
    return float(np.clip(combined, 0.0, 1.0))


# ── Temporal inconsistency (between frames) ──────────────────────────────────

def _temporal_inconsistency(frames: list) -> float:
    """
    Measures unnatural jitter between consecutive frames.
    Deepfakes often show frame-to-frame inconsistency in texture.
    Returns a [0, 1] suspicion score.
    """
    if len(frames) < 2:
        return 0.0
    diffs = []
    for a, b in zip(frames[:-1], frames[1:]):
        fa = np.array(a, dtype=np.float32)
        fb = np.array(b, dtype=np.float32)
        if fa.shape != fb.shape:
            continue
        diff = np.abs(fa - fb).mean()
        diffs.append(diff)
    if not diffs:
        return 0.0
    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs)
    # High variance in inter-frame diff = unnatural jitter
    jitter_score = np.clip(std_diff / (mean_diff + 1e-6), 0, 1)
    return float(jitter_score)


# ── Main VideoDetector class ──────────────────────────────────────────────────

class VideoDetector:

    def analyze_video_file(self, filepath: str, n_frames: int = 30) -> dict:
        """
        Extracts frames via imageio-ffmpeg (no OpenCV needed),
        runs all heuristics, returns a detailed result dict.
        """
        try:
            import imageio.v3 as iio
        except ImportError:
            import imageio as iio

        frame_scores = []
        sampled_frames = []

        try:
            # Read video metadata first to compute sampling step
            props   = iio.improps(filepath, plugin="pyav")
            total_f = props.n_images if hasattr(props, "n_images") else 0
        except Exception:
            total_f = 0

        try:
            reader = iio.imiter(filepath, plugin="pyav")
            all_frames = []
            for frame in reader:
                all_frames.append(frame)
                if len(all_frames) >= 300:   # cap memory usage
                    break
        except Exception as e:
            return {"error": str(e), "score": 0.0, "frame_scores": [],
                    "verdict": "Analysis failed"}

        if not all_frames:
            return {"error": "No frames decoded", "score": 0.0,
                    "frame_scores": [], "verdict": "Analysis failed"}

        # Sample evenly
        step = max(1, len(all_frames) // n_frames)
        sampled = all_frames[::step][:n_frames]

        for frame in sampled:
            s = score_frame(frame)
            frame_scores.append(s)
            sampled_frames.append(frame)

        if not frame_scores:
            return {"error": "No frames scored", "score": 0.0,
                    "frame_scores": [], "verdict": "Analysis failed"}

        # Temporal inconsistency across sampled frames
        temporal = _temporal_inconsistency(sampled_frames)

        mean_score = float(np.mean(frame_scores))
        max_score  = float(np.max(frame_scores))
        # Weighted: average frame suspicion + boost from temporal jitter
        final_score = np.clip(
            mean_score * 0.55 + max_score * 0.25 + temporal * 0.20,
            0.0, 1.0
        )

        return {
            "score":        round(float(final_score), 3),
            "mean_score":   round(mean_score, 3),
            "max_score":    round(max_score, 3),
            "temporal":     round(temporal, 3),
            "frame_scores": [round(s, 3) for s in frame_scores],
            "frames_analysed": len(frame_scores),
            "total_frames":    len(all_frames),
        }

    def analyze_image_file(self, filepath: str) -> float:
        try:
            img = Image.open(filepath).convert("RGB")
            return score_frame(np.array(img))
        except Exception:
            return 0.0
