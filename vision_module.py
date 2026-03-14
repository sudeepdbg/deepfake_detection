"""
vision_module.py  — deepfake frame analysis, no OpenCV / mediapipe / libGL needed.
Uses imageio + av (PyAV) for frame decoding on Streamlit Cloud.

Per-frame heuristics:
  1. Channel correlation   — deepfakes break natural R/G/B skin correlation
  2. Noise residual        — GAN fingerprint after Gaussian blur subtraction
  3. Edge coherence        — incoherent edges at face/hair boundaries
  4. Blocking artefacts    — 8×8 DCT grid discontinuities
  5. Temporal inconsistency — unnatural frame-to-frame jitter
"""

import numpy as np
from PIL import Image, ImageFilter


# ── Per-frame heuristics ──────────────────────────────────────────────────────

def _channel_correlation(arr: np.ndarray) -> float:
    """
    Natural skin has R/G/B strongly correlated.
    Deepfakes generated per-channel break this correlation.
    BUG FIX: np.nan_to_num handles uniform/constant colour frames.
    """
    r = arr[:, :, 0].flatten().astype(np.float32)
    g = arr[:, :, 1].flatten().astype(np.float32)
    b = arr[:, :, 2].flatten().astype(np.float32)
    rg = float(np.nan_to_num(np.corrcoef(r, g)[0, 1], nan=1.0))
    rb = float(np.nan_to_num(np.corrcoef(r, b)[0, 1], nan=1.0))
    avg = (rg + rb) / 2.0
    return float(np.clip(1.0 - avg, 0.0, 1.0))


def _noise_residual(arr: np.ndarray) -> float:
    """GAN outputs leave a characteristic high-frequency noise fingerprint."""
    pil      = Image.fromarray(arr)
    blurred  = pil.filter(ImageFilter.GaussianBlur(radius=2))
    residual = np.abs(
        np.array(pil, dtype=np.float32) - np.array(blurred, dtype=np.float32)
    )
    mean_res = float(residual.mean())
    return float(np.clip((mean_res - 4.0) / 20.0, 0.0, 1.0))


def _edge_coherence(arr: np.ndarray) -> float:
    """
    GAN faces have incoherent/aliased edges. Laplacian variance measures this.
    BUG FIX: PIL ImageFilter.Kernel size must be a tuple (w, h), not an int.
    """
    gray = Image.fromarray(arr).convert("L")
    # FIX: size=(3,3) — was incorrectly size=3 causing TypeError
    lap = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1,
                -1,  8, -1,
                -1, -1, -1],
        scale=1,
        offset=128
    ))
    var = float(np.array(lap, dtype=np.float32).var())
    if var < 50:    return 0.8   # over-smoothed — heavy deepfake processing
    if var > 5000:  return 0.7   # over-noisy
    return float(np.clip(1.0 - (var - 50.0) / 4950.0, 0.0, 1.0))


def _blocking_artefact(arr: np.ndarray) -> float:
    """8×8 block boundary discontinuities — GAN compression signature."""
    gray = arr.mean(axis=2).astype(np.float32)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    h_bound  = np.abs(gray[:, 7:w-1:8] - gray[:, 8:w:8]).mean()
    v_bound  = np.abs(gray[7:h-1:8, :] - gray[8:h:8, :]).mean()
    h_int    = np.abs(np.diff(gray, axis=1)).mean()
    v_int    = np.abs(np.diff(gray, axis=0)).mean()
    interior = (h_int + v_int) / 2.0 + 1e-6
    ratio    = (h_bound + v_bound) / 2.0 / interior
    return float(np.clip(abs(ratio - 1.0) / 2.0, 0.0, 1.0))


def score_frame(frame) -> float:
    """Combine all per-frame heuristics → single [0,1] suspicion score."""
    arr = np.array(frame, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    s1 = _channel_correlation(arr)
    s2 = _noise_residual(arr)
    s3 = _edge_coherence(arr)
    s4 = _blocking_artefact(arr)

    return float(np.clip(s1 * 0.35 + s2 * 0.25 + s3 * 0.20 + s4 * 0.20, 0.0, 1.0))


def _temporal_inconsistency(frames: list) -> float:
    """Unnatural frame-to-frame pixel jitter is a deepfake signature."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    for a, b in zip(frames[:-1], frames[1:]):
        fa = np.array(a, dtype=np.float32)
        fb = np.array(b, dtype=np.float32)
        if fa.shape == fb.shape:
            diffs.append(float(np.abs(fa - fb).mean()))
    if not diffs:
        return 0.0
    mean_d = float(np.mean(diffs))
    std_d  = float(np.std(diffs))
    return float(np.clip(std_d / (mean_d + 1e-6), 0.0, 1.0))


# ── Frame decoder (3-tier fallback) ──────────────────────────────────────────

def _decode_frames(filepath: str, max_frames: int = 300) -> tuple:
    """
    Returns (frames_list, error_string_or_None).
    Tries: pyav plugin → FFMPEG plugin → imageio v2 legacy API.
    """
    # Attempt 1: PyAV plugin (requires `av` package)
    try:
        import imageio.v3 as iio
        frames = []
        for frame in iio.imiter(filepath, plugin="pyav"):
            frames.append(np.array(frame))
            if len(frames) >= max_frames:
                break
        if frames:
            return frames, None
    except Exception as e1:
        pass

    # Attempt 2: imageio-ffmpeg FFMPEG plugin
    try:
        import imageio.v3 as iio
        frames = []
        for frame in iio.imiter(filepath, plugin="FFMPEG"):
            frames.append(np.array(frame))
            if len(frames) >= max_frames:
                break
        if frames:
            return frames, None
    except Exception as e2:
        pass

    # Attempt 3: imageio v2 legacy API
    try:
        import imageio
        vid    = imageio.get_reader(filepath, "ffmpeg")
        frames = []
        for frame in vid:
            frames.append(np.array(frame))
            if len(frames) >= max_frames:
                break
        vid.close()
        if frames:
            return frames, None
    except Exception as e3:
        return [], f"All decoders failed — pyav, FFMPEG, ffmpeg-v2: {e3}"

    return [], "No frames decoded"


# ── Main class ────────────────────────────────────────────────────────────────

class VideoDetector:

    def analyze_video_file(self, filepath: str, n_frames: int = 30) -> dict:
        all_frames, err = _decode_frames(filepath)

        if err or not all_frames:
            return {
                "error": err or "No frames decoded",
                "score": 0.0, "frame_scores": [], "frames_analysed": 0
            }

        # Sample evenly across the video
        step     = max(1, len(all_frames) // n_frames)
        sampled  = all_frames[::step][:n_frames]

        frame_scores = [score_frame(f) for f in sampled]
        temporal     = _temporal_inconsistency(sampled)

        mean_s = float(np.mean(frame_scores))
        max_s  = float(np.max(frame_scores))
        final  = float(np.clip(mean_s * 0.55 + max_s * 0.25 + temporal * 0.20,
                               0.0, 1.0))

        return {
            "score":           round(final, 3),
            "mean_score":      round(mean_s, 3),
            "max_score":       round(max_s, 3),
            "temporal":        round(temporal, 3),
            "frame_scores":    [round(s, 3) for s in frame_scores],
            "frames_analysed": len(frame_scores),
            "total_frames":    len(all_frames),
        }

    def analyze_image_file(self, filepath: str) -> float:
        try:
            img = Image.open(filepath).convert("RGB")
            return score_frame(np.array(img))
        except Exception:
            return 0.0
