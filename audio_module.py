"""
audio_module.py
Audio-track deepfake detection using librosa.
Works on MP4, WAV, MP3 via ffmpeg backend.

Features:
  1. MFCC flatness         — synthetic voices have unnaturally uniform timbre
  2. Pitch consistency     — TTS produces overly smooth F0 contours
  3. Spectral flux         — real voices have high frame-to-frame spectral change
  4. Silence ratio         — TTS often has unnatural silence distribution
  5. Harmonic-to-noise     — vocoded/synthesised audio has lower HNR
"""

import numpy as np
import librosa


class AudioDetector:

    def predict_audio_file(self, filepath: str) -> dict:
        """
        Returns a dict with score [0–1] and feature breakdown.
        Higher score = more likely synthetic/deepfake.
        """
        try:
            y, sr = librosa.load(filepath, sr=16000, mono=True, duration=60)
        except Exception as e:
            return {"score": 0.0, "error": str(e)}

        if len(y) < sr * 0.5:
            return {"score": 0.0, "error": "Audio too short"}

        # ── Feature 1: MFCC flatness ────────────────────────────────────
        mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_std  = float(np.std(mfccs))
        # Synthetic: std < 10 (suspiciously flat); natural: 12–30
        f1 = float(np.clip(1.0 - mfcc_std / 18.0, 0, 1))

        # ── Feature 2: Pitch consistency (F0 smoothness) ────────────────
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=50, fmax=400, sr=sr,
                frame_length=2048, hop_length=512
            )
            f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
            if len(f0_voiced) > 10:
                f0_diff = np.abs(np.diff(f0_voiced))
                f0_jitter = float(np.std(f0_diff) / (np.mean(f0_diff) + 1e-6))
                # TTS: very smooth → low jitter → high suspicion
                f2 = float(np.clip(1.0 - f0_jitter / 3.0, 0, 1))
            else:
                f2 = 0.3   # no voiced frames — uncertain
        except Exception:
            f2 = 0.3

        # ── Feature 3: Spectral flux ─────────────────────────────────────
        stft  = np.abs(librosa.stft(y))
        flux  = np.mean(np.diff(stft, axis=1) ** 2)
        # High flux = natural; low flux = synthetic monotony
        f3 = float(np.clip(1.0 - flux / 5.0, 0, 1))

        # ── Feature 4: Silence ratio ─────────────────────────────────────
        rms        = librosa.feature.rms(y=y)[0]
        silence_r  = float(np.mean(rms < rms.mean() * 0.1))
        # TTS: silence ratio often < 0.05 (unnaturally little silence)
        # or  > 0.5 (unnatural gaps)
        f4 = float(np.clip(abs(silence_r - 0.15) / 0.35, 0, 1))

        # ── Feature 5: Harmonic-to-noise ratio ──────────────────────────
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = float(np.var(harmonic) / (np.var(percussive) + 1e-8))
        # Natural voice: hnr > 1; synthesised: often < 0.8
        f5 = float(np.clip(1.0 - hnr / 2.0, 0, 1))

        # ── Weighted combination ─────────────────────────────────────────
        score = (f1 * 0.25 + f2 * 0.30 + f3 * 0.20 + f4 * 0.10 + f5 * 0.15)
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "score":         round(score, 3),
            "mfcc_flatness": round(f1, 3),
            "pitch_smooth":  round(f2, 3),
            "spec_flux":     round(f3, 3),
            "silence":       round(f4, 3),
            "hnr":           round(f5, 3),
        }
