import numpy as np
import librosa

class AudioDetector:
    def predict_audio_file(self, filepath: str) -> float:
        """
        FIX 4: Returns a properly normalised confidence score in [0.0, 1.0].

        Previously, np.mean(spectrogram) returned a raw dB value (e.g. -30.0)
        which was meaningless as a deepfake confidence score and would always
        trigger the "> 0.6 → synthetic" branch or the "< 0.4 → authentic"
        branch arbitrarily.

        Current implementation extracts MFCCs (a compact, normalised
        representation of the Mel spectrogram) and maps their energy to [0,1]
        using a sigmoid function.

        Plug in your real trained model's inference in place of the
        heuristic sigmoid at the bottom of this function.
        """
        try:
            y, sr = librosa.load(filepath, sr=16000, mono=True)
        except Exception:
            return 0.0

        if len(y) == 0:
            return 0.0

        # 1. Extract MFCCs (13 coefficients, time-averaged)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)   # shape: (13,)

        # 2. Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr               = np.mean(librosa.feature.zero_crossing_rate(y))

        # 3. Placeholder heuristic score (sigmoid of L2 norm of MFCCs).
        # This produces a score in (0, 1) — replace with real model inference.
        feature_magnitude = float(np.linalg.norm(mfcc_mean)) / 100.0
        score = 1.0 / (1.0 + np.exp(-feature_magnitude + 2))   # sigmoid centred at 2

        return round(float(score), 3)
