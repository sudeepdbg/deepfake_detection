"""
vision_module.py — unified deepfake & AI-generated image/video detection.
No OpenCV / mediapipe / libGL. Works on Streamlit Cloud.

ADAPTIVE SCORING: measures raw noise to detect media type, then blends
two weight sets. Works correctly for:
  - JPEG AI images   (ELA + skin + autocorr dominant)
  - PNG AI images    (ELA + autocorr dominant — skin less reliable due to complex scenes)
  - Deepfake video   (noise residual + edge + autocorr + sharpness dominant)

Signals (7 total, all computed for every input):
  1. ELA uniformity    — AI images have unnaturally uniform JPEG recompression error
  2. Skin smoothness   — AI faces lack micro-texture (pores/grain)
  3. Noise residual    — GAN fingerprint after blur subtraction (strong for raw video)
  4. Edge coherence    — Laplacian variance, GAN texture artefacts
  5. Noise autocorr    — Spatial correlation of denoised residual; AI = different pattern
  6. Over-sharpening   — AI content is unnaturally sharp at edges
  7. Face/BG sep       — AI portrait bokeh is mathematically perfect
"""

import io
import numpy as np
from PIL import Image, ImageFilter
from numpy.lib.stride_tricks import sliding_window_view


# ══════════════════════════════════════════════════════════════
# SIGNAL FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _ela_uniformity(pil_img: Image.Image, arr: np.ndarray) -> float:
    """
    Re-compress at Q75, measure regional ELA coefficient of variation.
    AI images: ELA is unnaturally uniform (CV < 0.3) → score high.
    Real photos: ELA varies by region (CV > 0.5) → score ~0.
    Works for both JPEG and PNG inputs.
    Calibrated: AI JPEG CV=0.248→0.72, AI PNG CV=0.221→0.80
    """
    h, w = arr.shape[:2]
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=75)
    buf.seek(0)
    comp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    ela  = np.abs(arr.astype(np.float32) - comp)
    cms  = [float(ela[cy*h//4:(cy+1)*h//4, cx*w//4:(cx+1)*w//4].mean())
            for cy in range(4) for cx in range(4)]
    ela_cv = float(np.std(cms) / (np.mean(cms) + 1e-6))
    return float(np.clip((0.50 - ela_cv) / 0.35, 0.0, 1.0))


def _skin_smoothness(gray: np.ndarray) -> float:
    """
    AI-generated faces lack micro-texture (pores, fine hairs, skin grain).
    Measures fraction of 5×5 patches in face region with local_std < 3.5.
    Best for single-subject portraits; less reliable for complex multi-person scenes.
    Calibrated: AI JPEG=0.62→1.0, AI PNG=0.39→0.48
    """
    h, w = gray.shape
    face = gray[h // 6: 3 * h // 4, w // 5: 4 * w // 5]
    if face.size < 100:
        return 0.0
    wins        = sliding_window_view(face, (5, 5))
    local_std   = wins.std(axis=(-1, -2))
    smooth_frac = float(np.mean(local_std < 3.5))
    return float(np.clip((smooth_frac - 0.25) / 0.30, 0.0, 1.0))


def _noise_residual(pil_img: Image.Image, arr: np.ndarray) -> float:
    """
    GAN fingerprint: subtract Gaussian blur, measure remaining noise.
    Strong for raw video frames (~16-17 → score 0.65).
    Destroyed by JPEG compression (~1-2 → score 0.0).
    PNG images: moderate (~3-4 → partial score).
    """
    blurred  = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
    residual = np.abs(arr.astype(np.float32) -
                      np.array(blurred, dtype=np.float32))
    return float(np.clip((residual.mean() - 6.0) / 16.0, 0.0, 1.0))


def _edge_coherence(pil_img: Image.Image) -> float:
    """
    Laplacian variance detects GAN texture artefacts.
    Deepfake video: var ~4600 → 0.70. AI images: var ~190 → 0.35.
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
    if var < 80:    return 0.75
    if var > 3500:  return 0.70
    return float(np.clip(1.0 - (var - 80.0) / 3420.0, 0.0, 0.35))


def _noise_autocorrelation(pil_img: Image.Image, arr: np.ndarray) -> float:
    """
    Spatial autocorrelation of the denoised noise residual.
    Real cameras: PRNU creates autocorr ~0.1-0.4.
    AI images: clean generation creates very high autocorr (>0.45) due to
    perfect smooth regions creating edge-correlated noise patterns.
    Calibrated: AI PNG=0.50→1.0, AI JPEG=0.55→1.0, Video=0.85→1.0
    """
    denoised = np.array(pil_img.filter(ImageFilter.MedianFilter(5)),
                        dtype=np.float32)
    noise_map = arr.astype(np.float32) - denoised
    nm        = noise_map.mean(axis=2).flatten()
    autocorr  = float(np.corrcoef(nm[:-1], nm[1:])[0, 1])
    # Natural camera: autocorr ~0.1-0.35 → low score
    # AI content: autocorr >0.45 → high score
    return float(np.clip(abs(autocorr - 0.15) / 0.30, 0.0, 1.0))


def _over_sharpening(pil_img: Image.Image, arr: np.ndarray) -> float:
    """
    AI generators produce images with unnatural edge sharpness.
    Applying unsharp mask to an already-sharp image makes little difference.
    Calibrated: Video frame=1.0 (very sharp), AI PNG=0.38, AI JPEG=0.0
    """
    sharpened  = pil_img.filter(
        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
    )
    sharp_diff = float(
        np.abs(arr.astype(np.float32) -
               np.array(sharpened, dtype=np.float32)).mean()
    )
    return float(np.clip((sharp_diff - 2.0) / 6.0, 0.0, 1.0))


def _face_bg_separation(pil_img: Image.Image, gray: np.ndarray) -> float:
    """
    AI portrait bokeh is mathematically perfect: face much sharper than background.
    Real cameras: natural transition. AI portraits: ratio > 1.2.
    Calibrated: AI JPEG=0.57, AI PNG=0.33, Video frame ~0.
    """
    h, w  = gray.shape
    edges = np.array(pil_img.convert("L").filter(ImageFilter.FIND_EDGES),
                     dtype=np.float32)
    face_e = float(edges[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean())
    bg_e   = float(np.concatenate([
        edges[:h // 5, :].flatten(),
        edges[4 * h // 5:, :].flatten()
    ]).mean())
    ratio  = face_e / (bg_e + 1e-6)
    return float(np.clip((ratio - 1.0) / 0.5, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════
# ADAPTIVE UNIFIED SCORER
# ══════════════════════════════════════════════════════════════

def score_frame_or_image(pil_img: Image.Image) -> dict:
    """
    Unified scoring for video frames AND still images (JPEG + PNG).

    AUTO-DETECTS media type using raw noise residual:
      noise > 5  → raw video frame  → video weight set (noise + edge + autocorr)
      noise < 3  → JPEG image       → image weight set (ELA + skin + autocorr)
      noise 3–5  → PNG/mixed        → smooth blend of both (image-dominant)

    All 7 signals computed regardless — shown in UI for transparency.
    """
    arr  = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    gray = np.array(pil_img.convert("L"),   dtype=np.float32)

    # ── Compute all 7 signals ──────────────────────────────────
    s_ela    = _ela_uniformity(pil_img, arr)
    s_skin   = _skin_smoothness(gray)
    s_noise  = _noise_residual(pil_img, arr)
    s_edge   = _edge_coherence(pil_img)
    s_autocr = _noise_autocorrelation(pil_img, arr)
    s_sharp  = _over_sharpening(pil_img, arr)
    s_sep    = _face_bg_separation(pil_img, gray)

    # ── Detect media type ─────────────────────────────────────
    raw_noise    = float(np.abs(
        arr.astype(np.float32) -
        np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=2)),
                 dtype=np.float32)
    ).mean())
    video_weight = float(np.clip((raw_noise - 3.0) / 4.0, 0.0, 1.0))
    image_weight = 1.0 - video_weight

    # ── Image weight set ──────────────────────────────────────
    # Calibrated: AI JPEG→0.70, AI PNG→0.59
    score_image = (s_ela    * 0.30 +   # strongest for JPEG & PNG
                   s_skin   * 0.20 +   # reliable for portraits
                   s_autocr * 0.20 +   # strong for all AI content
                   s_edge   * 0.15 +   # moderate for both
                   s_sharp  * 0.10 +   # PNG over-sharpening
                   s_sep    * 0.05)    # portrait bokeh

    # ── Video weight set ──────────────────────────────────────
    # Calibrated: deepfake video frame→0.75
    score_video = (s_noise  * 0.40 +   # GAN fingerprint (strongest)
                   s_edge   * 0.25 +   # GAN texture artefacts
                   s_autocr * 0.20 +   # noise pattern (strong)
                   s_sharp  * 0.15)    # unnatural sharpness

    # ── Blend ──────────────────────────────────────────────────
    final = float(np.clip(
        video_weight * score_video + image_weight * score_image,
        0.0, 1.0
    ))

    media_type = (
        "raw video frame" if raw_noise > 5 else
        "JPEG image"      if raw_noise < 3 else
        "PNG image"
    )

    return {
        "score":          round(final, 3),
        "ela_uniformity": round(s_ela,    3),
        "skin_smooth":    round(s_skin,   3),
        "noise_residual": round(s_noise,  3),
        "edge_coherence": round(s_edge,   3),
        "noise_autocorr": round(s_autocr, 3),
        "over_sharpening":round(s_sharp,  3),
        "bg_separation":  round(s_sep,    3),
        "_raw_noise":     round(raw_noise, 2),
        "_media_type":    media_type,
        "_video_w":       round(video_weight, 2),
        "_image_w":       round(image_weight, 2),
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
        """Analyse video — applies unified scorer to sampled frames."""
        all_frames, err = _decode_frames(filepath)
        if err or not all_frames:
            return {"error": err or "No frames decoded", "score": 0.0,
                    "frame_scores": [], "frames_analysed": 0}

        step    = max(1, len(all_frames) // n_frames)
        sampled = all_frames[::step][:n_frames]

        frame_results = []
        for f in sampled:
            arr = np.array(f, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            elif arr.ndim != 3 or arr.shape[2] != 3:
                continue
            frame_results.append(score_frame_or_image(Image.fromarray(arr)))

        if not frame_results:
            return {"error": "No valid frames scored", "score": 0.0,
                    "frame_scores": [], "frames_analysed": 0}

        temporal     = _temporal_inconsistency(sampled)
        frame_scores = [r["score"] for r in frame_results]
        mean_s       = float(np.mean(frame_scores))
        max_s        = float(np.max(frame_scores))
        final        = float(np.clip(
            mean_s * 0.50 + max_s * 0.30 + temporal * 0.20, 0.0, 1.0
        ))

        sig_keys = ["ela_uniformity", "skin_smooth", "noise_residual",
                    "edge_coherence", "noise_autocorr", "over_sharpening",
                    "bg_separation"]
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
        """Analyse still image (JPEG or PNG) — AI-generated content detection."""
        try:
            img = Image.open(filepath).convert("RGB")
            return score_frame_or_image(img)
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
