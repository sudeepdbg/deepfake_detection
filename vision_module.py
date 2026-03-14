"""
vision_module.py — unified deepfake & AI-image detection.
Works on video frames AND still images using adaptive scoring.

KEY INSIGHT: The optimal signals differ by media type:
  - Video frames (raw pixels): noise residual + edge coherence are strong
  - JPEG images (compressed):  noise is destroyed; use ELA + skin texture instead

ADAPTIVE STRATEGY: measure noise residual to auto-detect media type,
then weight signals accordingly. Blend smoothly in the middle range.

Signals used:
  1. Noise residual    — GAN fingerprint after blur subtraction (strong for video)
  2. Edge coherence    — Laplacian variance, GAN texture artefacts (strong for both)
  3. ELA uniformity    — AI images have uniform error levels across regions (JPEG only)
  4. Skin smoothness   — AI faces lack micro-texture; pores/grain absent (JPEG only)
  5. Block smoothness  — 8x8 block variance distribution (useful for both)
  6. Face/BG separation— AI portrait bokeh is mathematically perfect (JPEG only)
"""

import io
import numpy as np
from PIL import Image, ImageFilter
from numpy.lib.stride_tricks import sliding_window_view


# ══════════════════════════════════════════════════════════════
# INDIVIDUAL SIGNAL FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _noise_residual(pil_img: Image.Image, arr: np.ndarray) -> float:
    """
    GAN fingerprint: subtract Gaussian blur, measure remaining noise.
    Strong for raw video frames (score ~0.65 for deepfake video).
    Destroyed by JPEG compression (score ~0.0 for JPEG AI images).
    """
    blurred  = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
    residual = np.abs(arr.astype(np.float32) -
                      np.array(blurred, dtype=np.float32))
    mean_res = float(residual.mean())
    # Calibrated: deepfake video frame ~16-17 → 0.65
    return float(np.clip((mean_res - 6.0) / 16.0, 0.0, 1.0))


def _edge_coherence(pil_img: Image.Image) -> float:
    """
    Laplacian variance detects GAN texture artefacts.
    Works for both media types:
    - Deepfake video: var ~4600 → score 0.70
    - AI JPEG image: var ~190 → score 0.35 (moderate)
    """
    gray = pil_img.convert("L")
    lap  = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1,
                -1,  8, -1,
                -1, -1, -1],
        scale=1, offset=128
    ))
    var = float(np.array(lap, dtype=np.float32).var())
    if var < 80:    return 0.75   # over-smooth → heavy GAN post-processing
    if var > 3500:  return 0.70   # over-noisy  → GAN texture artefacts
    return float(np.clip(1.0 - (var - 80.0) / 3420.0, 0.0, 0.35))


def _ela_uniformity(pil_img: Image.Image, arr: np.ndarray) -> float:
    """
    Error Level Analysis: re-compress at Q75, measure regional variation (CV).
    AI images: ELA unnaturally uniform across regions (CV < 0.3) → score 0.72.
    Real photos: ELA varies strongly by region (CV > 0.5) → score ~0.
    JPEG-specific: ineffective for raw video frames.
    """
    h, w = arr.shape[:2]
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=75)
    buf.seek(0)
    comp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    ela  = np.abs(arr.astype(np.float32) - comp)

    cell_means = []
    for cy in range(4):
        for cx in range(4):
            y1, y2 = cy * h // 4, (cy + 1) * h // 4
            x1, x2 = cx * w // 4, (cx + 1) * w // 4
            cell_means.append(float(ela[y1:y2, x1:x2].mean()))

    ela_cv = float(np.std(cell_means) / (np.mean(cell_means) + 1e-6))
    # Calibrated: AI portrait CV=0.248 → score=0.72; natural photo CV>0.5 → score=0
    return float(np.clip((0.50 - ela_cv) / 0.35, 0.0, 1.0))


def _skin_smoothness(gray: np.ndarray) -> float:
    """
    AI-generated portraits lack micro-texture (pores, fine hairs, skin grain).
    Measures fraction of 5×5 face-region patches with local_std < 3.5.
    Calibrated: AI portrait smooth_frac=0.54-0.62 → score=0.97-1.0.
    Real skin: smooth_frac < 0.25 → score ~0.
    """
    h, w = gray.shape
    face = gray[h // 6: 3 * h // 4, w // 5: 4 * w // 5]
    if face.size < 100:
        return 0.0
    wins        = sliding_window_view(face, (5, 5))
    local_std   = wins.std(axis=(-1, -2))
    smooth_frac = float(np.mean(local_std < 3.5))
    return float(np.clip((smooth_frac - 0.25) / 0.30, 0.0, 1.0))


def _block_smoothness(gray: np.ndarray) -> float:
    """
    AI images have a high proportion of suspiciously smooth 8×8 blocks.
    Calibrated: AI portrait=0.512 → score=0.75; video frame=0.28 → score=0.08.
    """
    h, w = gray.shape
    block_vars = []
    for y in range(0, h - 8, 8):
        for x in range(0, w - 8, 8):
            block_vars.append(float(gray[y:y + 8, x:x + 8].var()))
    if not block_vars:
        return 0.0
    very_smooth = float(np.mean(np.array(block_vars) < 5.0))
    return float(np.clip((very_smooth - 0.20) / 0.35, 0.0, 1.0))


def _face_bg_separation(pil_img: Image.Image, gray: np.ndarray) -> float:
    """
    AI portrait generators produce mathematically perfect bokeh.
    Face is extremely sharp; background is uniformly smooth.
    Real cameras: natural transition with lens aberrations.
    Calibrated: AI portrait ratio=1.285 → score=0.57; video frame ≈0.
    """
    h, w = gray.shape
    edges    = np.array(pil_img.convert("L").filter(ImageFilter.FIND_EDGES),
                        dtype=np.float32)
    face_e   = float(edges[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean())
    bg_top   = edges[:h // 5, :].flatten()
    bg_bot   = edges[4 * h // 5:, :].flatten()
    bg_e     = float(np.concatenate([bg_top, bg_bot]).mean())
    ratio    = face_e / (bg_e + 1e-6)
    return float(np.clip((ratio - 1.0) / 0.5, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════
# ADAPTIVE UNIFIED SCORER
# ══════════════════════════════════════════════════════════════

def score_frame_or_image(pil_img: Image.Image) -> dict:
    """
    Unified scoring for both video frames and still images.

    AUTO-DETECTS media type using noise residual magnitude:
      noise > 5  → raw video frame → use video weights (noise + edge dominant)
      noise < 3  → JPEG image      → use image weights (ELA + skin dominant)
      3–5        → blend both sets smoothly

    Returns score [0,1] and individual feature scores.
    """
    arr  = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    gray = np.array(pil_img.convert("L"),   dtype=np.float32)

    # ── Compute all 6 signals ────────────────────────────────
    s_noise = _noise_residual(pil_img, arr)
    s_edge  = _edge_coherence(pil_img)
    s_ela   = _ela_uniformity(pil_img, arr)
    s_skin  = _skin_smoothness(gray)
    s_block = _block_smoothness(gray)
    s_sep   = _face_bg_separation(pil_img, gray)

    # ── Detect media type via raw noise level ─────────────────
    blurred  = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
    raw_noise = float(np.abs(arr.astype(np.float32) -
                             np.array(blurred, dtype=np.float32)).mean())

    # Blend factor: 0.0 = pure JPEG image, 1.0 = pure raw video frame
    video_weight = float(np.clip((raw_noise - 3.0) / 4.0, 0.0, 1.0))
    image_weight = 1.0 - video_weight

    # ── Weight sets calibrated on real data ──────────────────
    # VIDEO weights — calibrated: deepfake video frame → 0.58
    score_video = (s_noise * 0.50 +
                   s_edge  * 0.30 +
                   s_block * 0.20)

    # IMAGE weights — calibrated: AI portrait → 0.75
    score_image = (s_ela   * 0.35 +
                   s_skin  * 0.30 +
                   s_edge  * 0.15 +
                   s_sep   * 0.10 +
                   s_block * 0.10)

    # ── Blend ──────────────────────────────────────────────────
    final = float(np.clip(
        video_weight * score_video + image_weight * score_image,
        0.0, 1.0
    ))

    media_type = (
        "raw video frame"  if raw_noise > 5 else
        "JPEG image"       if raw_noise < 3 else
        "mixed"
    )

    return {
        "score":         round(final, 3),
        "noise_residual":round(s_noise, 3),
        "edge_coherence":round(s_edge,  3),
        "ela_uniformity":round(s_ela,   3),
        "skin_smooth":   round(s_skin,  3),
        "block_smooth":  round(s_block, 3),
        "bg_separation": round(s_sep,   3),
        "_raw_noise":    round(raw_noise, 2),
        "_media_type":   media_type,
        "_video_w":      round(video_weight, 2),
        "_image_w":      round(image_weight, 2),
    }


# ══════════════════════════════════════════════════════════════
# TEMPORAL INCONSISTENCY (video only)
# ══════════════════════════════════════════════════════════════

def _temporal_inconsistency(frames: list) -> float:
    """Unnatural frame-to-frame pixel jitter — deepfake video signature."""
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


# ══════════════════════════════════════════════════════════════
# FRAME DECODER
# ══════════════════════════════════════════════════════════════

def _decode_frames(filepath: str, max_frames: int = 300):
    """Decode video using imageio FFMPEG plugin (imageio-ffmpeg)."""
    try:
        import imageio.v3 as iio
        frames = []
        for frame in iio.imiter(filepath, plugin="FFMPEG"):
            frames.append(np.array(frame))
            if len(frames) >= max_frames:
                break
        if frames:
            return frames, None
    except Exception:
        pass
    try:
        import imageio
        vid    = imageio.get_reader(filepath, "ffmpeg")
        frames = [np.array(f) for f in vid][:max_frames]
        vid.close()
        if frames:
            return frames, None
    except Exception as e:
        return [], f"Frame decoding failed: {e}"
    return [], "No frames decoded"


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

class VideoDetector:

    def analyze_video_file(self, filepath: str, n_frames: int = 30) -> dict:
        """
        Analyse video — applies unified scorer to each frame then combines.
        """
        all_frames, err = _decode_frames(filepath)
        if err or not all_frames:
            return {"error": err or "No frames decoded", "score": 0.0,
                    "frame_scores": [], "frames_analysed": 0}

        step     = max(1, len(all_frames) // n_frames)
        sampled  = all_frames[::step][:n_frames]

        frame_results = []
        for f in sampled:
            arr = np.array(f, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            elif arr.ndim != 3 or arr.shape[2] != 3:
                continue
            pil = Image.fromarray(arr)
            frame_results.append(score_frame_or_image(pil))

        if not frame_results:
            return {"error": "No valid frames scored", "score": 0.0,
                    "frame_scores": [], "frames_analysed": 0}

        temporal     = _temporal_inconsistency(sampled)
        frame_scores = [r["score"] for r in frame_results]
        mean_s       = float(np.mean(frame_scores))
        max_s        = float(np.max(frame_scores))

        # Boost from temporal inconsistency
        final = float(np.clip(
            mean_s * 0.50 + max_s * 0.30 + temporal * 0.20,
            0.0, 1.0
        ))

        # Aggregate per-signal means for display
        sig_keys = ["noise_residual", "edge_coherence", "ela_uniformity",
                    "skin_smooth", "block_smooth", "bg_separation"]
        sig_means = {k: round(float(np.mean([r[k] for r in frame_results])), 3)
                     for k in sig_keys}

        return {
            "score":           round(final, 3),
            "mean_score":      round(mean_s, 3),
            "max_score":       round(max_s, 3),
            "temporal":        round(temporal, 3),
            "frame_scores":    [round(s, 3) for s in frame_scores],
            "frames_analysed": len(frame_results),
            "total_frames":    len(all_frames),
            **sig_means,
        }

    def analyze_image_file(self, filepath: str) -> dict:
        """
        Analyse still image — uses the same unified scorer as video frames.
        """
        try:
            img = Image.open(filepath).convert("RGB")
            return score_frame_or_image(img)
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
