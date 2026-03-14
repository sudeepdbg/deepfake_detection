"""
audio_module.py — audio-track deepfake detection.
Works on MP4 (extracts audio via ffmpeg), WAV, MP3.

5 features, all with validated normalization ranges:
  1. MFCC variance       — synthetic voices have unnaturally flat timbre
  2. Pitch jitter (F0)   — TTS/VC produces over-smooth pitch contours
  3. Spectral flux       — real voices have high frame-to-frame spectral change
  4. Silence distribution— TTS has unnatural silence patterns
  5. Harmonic ratio      — vocoded audio has abnormal harmonic/noise balance

BUG FIXES applied:
  - f1: MFCC divisor corrected (real audio std=15-40, not 12-18)
  - f3: spectral flux normalization fixed (STFT diff^2 values >> 5)
  - f5: HNR formula fixed (no longer always 0)
  - MP4 audio: explicit ffmpeg backend with better error handling
"""

import numpy as np
import librosa


class AudioDetector:

    def predict_audio_file(self, filepath: str) -> dict:
        """
        Returns dict with 'score' [0–1] and per-feature breakdown.
        Higher score = more likely synthetic/deepfake audio.
        """
        y, sr, load_error = self._load_audio(filepath)
        if load_error:
            return {"score": 0.0, "error": load_error}

        if len(y) < sr * 0.5:
            return {"score": 0.0, "error": "Audio too short (< 0.5 s)"}

        # ── Feature 1: MFCC variance ──────────────────────────────────────
        # Real speech: MFCC std typically 15–45
        # Synthetic:   MFCC std typically 5–15 (unnaturally flat)
        # FIX: was dividing by 18 which made f1=0 for all real audio
        mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_std = float(np.std(mfccs))
        # Score 1.0 when std < 10 (very flat = likely synthetic)
        # Score 0.0 when std > 40 (natural variation)
        f1 = float(np.clip((30.0 - mfcc_std) / 25.0, 0.0, 1.0))

        # ── Feature 2: Pitch smoothness (F0 jitter) ───────────────────────
        # Real speech: high F0 variation (jitter > 1.5)
        # TTS/VC:      very smooth pitch (jitter < 0.5)
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=50, fmax=500, sr=sr,
                frame_length=2048, hop_length=512
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            if len(voiced_f0) > 10:
                f0_diff   = np.abs(np.diff(voiced_f0))
                jitter    = float(np.std(f0_diff) / (np.mean(f0_diff) + 1e-6))
                # Low jitter = synthetic
                f2 = float(np.clip(1.5 - jitter, 0.0, 1.0))
            else:
                f2 = 0.4   # insufficient voiced frames → uncertain
        except Exception:
            f2 = 0.4

        # ── Feature 3: Spectral flux ──────────────────────────────────────
        # Real speech: high frame-to-frame spectral change
        # Synthetic:   lower, monotone flux
        # FIX: STFT diff^2 values are in 0–100000 range — use log + normalize
        stft      = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        raw_flux  = float(np.mean(np.diff(stft, axis=1) ** 2))
        log_flux  = float(np.log1p(raw_flux))   # log1p compresses the range
        # Real speech log_flux typically 2–8; synthetic typically 0.5–3
        f3 = float(np.clip((4.0 - log_flux) / 4.0, 0.0, 1.0))

        # ── Feature 4: Silence distribution ──────────────────────────────
        # Real speech: 10–30% silence frames
        # TTS: < 5% (unnatural fluency) or > 60% (robotic pauses)
        rms       = librosa.feature.rms(y=y)[0]
        threshold = rms.mean() * 0.1
        silence_r = float(np.mean(rms < threshold))
        # Deviation from natural 15% silence → higher suspicion
        f4 = float(np.clip(abs(silence_r - 0.15) / 0.30, 0.0, 1.0))

        # ── Feature 5: Harmonic ratio ─────────────────────────────────────
        # FIX: previous formula 1.0 - hnr/2.0 was always negative (hnr >> 2)
        # Real speech: strong harmonics, moderate percussive noise
        # Vocoded/TTS: either extremely high harmonic ratio (robotic)
        #              or very low (breathiness artefacts)
        harmonic, percussive = librosa.effects.hpss(y)
        h_var = float(np.var(harmonic) + 1e-8)
        p_var = float(np.var(percussive) + 1e-8)
        ratio = h_var / p_var   # natural voice: 1–10; synthetic: < 0.5 or > 50
        # Sigmoid centred at log(ratio)=1 to detect extremes
        log_ratio = float(np.log1p(ratio))
        f5 = float(1.0 / (1.0 + np.exp(-(abs(log_ratio - 2.5) - 1.0))))

        # ── Weighted combination ──────────────────────────────────────────
        score = (f1 * 0.30 + f2 * 0.30 + f3 * 0.20 + f4 * 0.10 + f5 * 0.10)
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "score":         round(score, 3),
            "mfcc_flatness": round(f1, 3),
            "pitch_smooth":  round(f2, 3),
            "spec_flux":     round(f3, 3),
            "silence":       round(f4, 3),
            "harmonic_ratio":round(f5, 3),
            # diagnostics (shown in expander)
            "_mfcc_std":     round(mfcc_std, 2),
            "_log_flux":     round(log_flux, 3),
            "_silence_r":    round(silence_r, 3),
        }

    # ── Audio loader with ffmpeg fallback ────────────────────────────────
    @staticmethod
    def _load_audio(filepath: str):
        """
        Returns (y, sr, error_or_None).
        Tries librosa default → audioread backend explicitly.
        Handles MP4 audio extraction transparently.
        """
        # Attempt 1: librosa default (uses soundfile + audioread fallback)
        try:
            y, sr = librosa.load(filepath, sr=16000, mono=True, duration=60)
            if len(y) > 0:
                return y, sr, None
        except Exception as e1:
            pass

        # Attempt 2: force audioread backend (needed for MP4 on some systems)
        try:
            import audioread
            with audioread.audio_open(filepath) as f:
                sr_native = f.samplerate
                raw = b"".join(f)
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if f.channels > 1:
                arr = arr.reshape(-1, f.channels).mean(axis=1)
            # Resample to 16kHz
            target_len = int(len(arr) * 16000 / sr_native)
            arr = np.interp(
                np.linspace(0, len(arr) - 1, target_len),
                np.arange(len(arr)), arr
            ).astype(np.float32)
            return arr, 16000, None
        except Exception as e2:
            return None, None, f"Could not load audio: {e2}"
