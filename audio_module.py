# audio_module.py
import numpy as np
import librosa


class AudioDetector:
    """
    Analyses audio for synthetic / manipulated content.
    Works on .wav, .mp3, AND .mp4 (audio track extracted via ffmpeg,
    which is pre-installed on Streamlit Cloud).
    """

    def predict_audio_file(self, filepath: str) -> float:
        """
        Returns a confidence score in [0.0, 1.0].
        0.0 = likely authentic   |   1.0 = likely synthetic

        Feature pipeline:
          1. MFCCs  — captures timbral texture
          2. Spectral centroid — brightness / naturalness indicator
          3. Zero-crossing rate — noisiness / synthesis artefacts
          4. RMS energy variance — unnatural flatness in synthetic audio
        All features are normalized and combined into one score via sigmoid.
        """
        try:
            # librosa uses audioread → ffmpeg fallback for MP4/video containers
            y, sr = librosa.load(filepath, sr=16000, mono=True, duration=60)
        except Exception as e:
            return 0.0

        if len(y) < sr * 0.5:   # less than 0.5 s — not enough signal
            return 0.0

        # ── Feature 1: MFCCs ──────────────────────────────────────────────
        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean  = np.abs(np.mean(mfccs, axis=1)).mean()   # overall energy
        mfcc_std   = np.std(mfccs)                           # variation

        # ── Feature 2: Spectral centroid ──────────────────────────────────
        centroid     = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_std = float(np.std(centroid))   # low std → suspiciously uniform

        # ── Feature 3: Zero-crossing rate ────────────────────────────────
        zcr     = librosa.feature.zero_crossing_rate(y)
        zcr_std = float(np.std(zcr))

        # ── Feature 4: RMS energy variance ───────────────────────────────
        rms        = librosa.feature.rms(y=y)
        rms_var    = float(np.var(rms))

        # ── Combine into single score ─────────────────────────────────────
        # Heuristic weights — replace with trained model output in production.
        # Synthetic audio tends to have:
        #   • lower MFCC std (flatter spectrum)
        #   • lower centroid std (less tonal variation)
        #   • lower RMS variance (less dynamic range)
        flatness_score = (
            max(0, 1 - mfcc_std / 15.0)       * 0.40 +
            max(0, 1 - centroid_std / 800.0)   * 0.30 +
            max(0, 1 - rms_var / 0.01)         * 0.20 +
            max(0, 1 - zcr_std / 0.05)         * 0.10
        )

        # Sigmoid to keep in (0, 1) and avoid hard clipping
        score = 1.0 / (1.0 + np.exp(-6 * (flatness_score - 0.5)))
        return round(float(np.clip(score, 0.0, 1.0)), 3)
