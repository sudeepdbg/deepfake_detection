"""
vision_module.py — frame-level deepfake detection.
No OpenCV / mediapipe / libGL. Uses imageio-ffmpeg (FFMPEG plugin).

Heuristics calibrated against real deepfake video data:
  1. Noise residual   (weight 0.50) — strongest signal: GAN noise residual ~16-17
  2. Edge coherence   (weight 0.30) — Laplacian var ~4600 hits 'over-noisy' branch
  3. Blocking artefact(weight 0.10) — weak signal but adds value
  4. Channel corr     (weight 0.10) — weak for face-swap but included
  5. Temporal jitter  (separate)    — frame-to-frame inconsistency
"""

import numpy as np
from PIL import Image, ImageFilter


def _noise_residual(arr: np.ndarray) -> float:
    """
    GAN/diffusion outputs leave a characteristic high-frequency noise fingerprint.
    Calibrated: deepfake residual ~16-17 → score ~0.63
    Natural video residual typically 3-8 → score ~0.0-0.2
    """
    pil     = Image.fromarray(arr)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=2))
    residual = np.abs(
        np.array(pil, dtype=np.float32) - np.array(blurred, dtype=np.float32)
    )
    mean_res = float(residual.mean())
    # Threshold: > 12 is suspicious, > 20 is strongly synthetic
    return float(np.clip((mean_res - 6.0) / 16.0, 0.0, 1.0))


def _edge_coherence(arr: np.ndarray) -> float:
    """
    Laplacian variance measures edge sharpness/coherence.
    Calibrated: deepfake Laplacian var ~4600-4800 → score 0.7 (over-noisy branch)
    Natural face: ~200-2000
    """
    gray = Image.fromarray(arr).convert("L")
    lap = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1,
                -1,  8, -1,
                -1, -1, -1],
        scale=1, offset=128
    ))
    var = float(np.array(lap, dtype=np.float32).var())
    if var < 80:    return 0.75   # over-smooth — heavy GAN post-processing
    if var > 3500:  return 0.70   # over-noisy — GAN texture artefacts
    # Natural range 80-3500: low score
    return float(np.clip(1.0 - (var - 80.0) / 3420.0, 0.0, 0.35))


def _blocking_artefact(arr: np.ndarray) -> float:
    """8×8 DCT-block boundary discontinuities — GAN compression signature."""
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


def _channel_correlation(arr: np.ndarray) -> float:
    """
    Natural skin: strong R/G/B correlation.
    Face-swap deepfakes maintain this (weak discriminator for this type).
    Still useful for image manipulation detection.
    """
    r = arr[:, :, 0].flatten().astype(np.float32)
    g = arr[:, :, 1].flatten().astype(np.float32)
    b = arr[:, :, 2].flatten().astype(np.float32)
    rg = float(np.nan_to_num(np.corrcoef(r, g)[0, 1], nan=1.0))
    rb = float(np.nan_to_num(np.corrcoef(r, b)[0, 1], nan=1.0))
    avg = (rg + rb) / 2.0
    return float(np.clip(1.0 - avg, 0.0, 1.0))


def score_frame(frame) -> float:
    """
    Combine heuristics with weights calibrated on real deepfake data.
    Noise residual carries 50% weight — strongest signal.
    """
    arr = np.array(frame, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    s1 = _noise_residual(arr)      # 0.50 — calibrated, strongest signal
    s2 = _edge_coherence(arr)      # 0.30 — good signal for GAN textures
    s3 = _blocking_artefact(arr)   # 0.10 — weak but adds value
    s4 = _channel_correlation(arr) # 0.10 — weak for face-swap

    return float(np.clip(s1 * 0.50 + s2 * 0.30 + s3 * 0.10 + s4 * 0.10,
                         0.0, 1.0))


def _temporal_inconsistency(frames: list) -> float:
    """Unnatural frame-to-frame pixel jitter — deepfake signature."""
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
    return float(np.clip(np.std(diffs) / (np.mean(diffs) + 1e-6), 0.0, 1.0))


def _decode_frames(filepath: str, max_frames: int = 300):
    """Decode video frames using imageio FFMPEG plugin (imageio-ffmpeg)."""
    # Primary: imageio v3 FFMPEG plugin
    try:
        import imageio.v3 as iio
        frames = []
        for frame in iio.imiter(filepath, plugin="FFMPEG"):
            frames.append(np.array(frame))
            if len(frames) >= max_frames:
                break
        if frames:
            return frames, None
    except Exception as e1:
        pass

    # Fallback: imageio v2 legacy API
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
    except Exception as e2:
        return [], f"Frame decoding failed: {e2}"

    return [], "No frames decoded"


class VideoDetector:

    def analyze_video_file(self, filepath: str, n_frames: int = 30) -> dict:
        all_frames, err = _decode_frames(filepath)
        if err or not all_frames:
            return {"error": err or "No frames decoded", "score": 0.0,
                    "frame_scores": [], "frames_analysed": 0}

        step     = max(1, len(all_frames) // n_frames)
        sampled  = all_frames[::step][:n_frames]

        frame_scores = [score_frame(f) for f in sampled]
        temporal     = _temporal_inconsistency(sampled)

        mean_s = float(np.mean(frame_scores))
        max_s  = float(np.max(frame_scores))
        # Weighted: mean dominates, max catches peak, temporal adds consistency signal
        final  = float(np.clip(mean_s * 0.50 + max_s * 0.30 + temporal * 0.20,
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
