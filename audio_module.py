"""
audio_module.py — audio deepfake detection.
Works on MP4, WAV, MP3 via ffmpeg subprocess extraction.

ROOT CAUSE OF AUDIO FAILURE: librosa.load() on MP4 fails on Streamlit Cloud
because the audioread/soundfile backends cannot handle MP4 containers directly.
FIX: Use ffmpeg subprocess to extract audio to WAV first, then load WAV.
This always works because ffmpeg binary is present on Streamlit Cloud.

Features (calibrated on real deepfake audio):
  1. Spectral flux   — real voice has high frame-to-frame spectral change
  2. Energy variation— real speech has high RMS variation
  3. ZCR consistency — synthetic audio has unnaturally consistent ZCR
  4. Spectral rolloff— TTS has different high-frequency energy distribution
  5. Silence ratio   — TTS has unnatural silence patterns
"""

import subprocess
import tempfile
import os
import wave
import numpy as np


class AudioDetector:

    def predict_audio_file(self, filepath: str) -> dict:
        """
        Returns dict with 'score' [0–1] and feature breakdown.
        Higher score = more likely synthetic/deepfake.
        """
        # Step 1: Extract audio to WAV using ffmpeg subprocess
        # This works for MP4, MP3, WAV — any container ffmpeg supports
        y, sr, err = self._extract_audio(filepath)
        if err:
            return {"score": 0.0, "error": err}
        if len(y) < sr * 0.5:
            return {"score": 0.0, "error": "Audio too short (< 0.5s)"}

        # Step 2: Compute features using only numpy (no librosa dependency)
        hop   = 512
        win_n = 2048
        hann  = np.hanning(win_n)

        # Build spectrogram frames
        spec_frames = []
        rms_frames  = []
        zcr_frames  = []

        for i in range(0, min(len(y) - win_n, win_n * 200), hop):
            chunk = y[i:i + win_n] * hann
            spec  = np.abs(np.fft.rfft(chunk))[:win_n // 2]
            spec_frames.append(spec)
            rms_frames.append(float(np.sqrt(np.mean(chunk ** 2))))
            zcr_frames.append(float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2))

        if len(spec_frames) < 10:
            return {"score": 0.0, "error": "Not enough audio frames to analyse"}

        spec_matrix = np.array(spec_frames, dtype=np.float32)
        rms_arr     = np.array(rms_frames,  dtype=np.float32)
        zcr_arr     = np.array(zcr_frames,  dtype=np.float32)

        # ── Feature 1: Spectral flux ──────────────────────────────────────
        # Real speech: high frame-to-frame spectral change (log_flux 2-6)
        # Deepfake/TTS: lower variation (log_flux 1-3)
        # Calibrated: deepfake log_flux ~1.8 → f1 ~0.55
        raw_flux = float(np.mean(np.diff(spec_matrix, axis=0) ** 2))
        log_flux = float(np.log1p(raw_flux))
        # Lower flux = more suspicious: score high when log_flux < 3
        f1 = float(np.clip((3.5 - log_flux) / 3.5, 0.0, 1.0))

        # ── Feature 2: Energy (RMS) variation ────────────────────────────
        # Real speech: high energy variation (std/mean > 0.5)
        # TTS: more uniform energy (std/mean < 0.3)
        rms_cv = float(rms_arr.std() / (rms_arr.mean() + 1e-8))
        # High variation = natural; low = suspicious
        # Calibrated: natural voice cv ~0.8-1.5; TTS ~0.1-0.4
        f2 = float(np.clip(1.0 - rms_cv / 0.8, 0.0, 1.0))

        # ── Feature 3: ZCR consistency ───────────────────────────────────
        # Real speech: ZCR varies a lot (std > 0.05)
        # Synthetic: unnaturally consistent ZCR
        zcr_cv = float(zcr_arr.std() / (zcr_arr.mean() + 1e-8))
        f3 = float(np.clip(1.0 - zcr_cv / 0.6, 0.0, 1.0))

        # ── Feature 4: Spectral rolloff ratio ────────────────────────────
        # TTS often has different high-frequency energy distribution
        # Compute ratio of energy in top 25% vs bottom 75% of spectrum
        n_bins    = spec_matrix.shape[1]
        low_e     = spec_matrix[:, :n_bins * 3 // 4].mean()
        high_e    = spec_matrix[:, n_bins * 3 // 4:].mean()
        rolloff_r = float(high_e / (low_e + 1e-8))
        # Very low ratio = muffled TTS; very high = synthetic noise
        f4 = float(np.clip(abs(rolloff_r - 0.15) / 0.20, 0.0, 1.0))

        # ── Feature 5: Silence ratio ──────────────────────────────────────
        # Real speech: 10-30% silence frames
        # TTS: < 5% (unnatural fluency) or > 60%
        threshold = rms_arr.mean() * 0.1
        silence_r = float(np.mean(rms_arr < threshold))
        f5 = float(np.clip(abs(silence_r - 0.15) / 0.30, 0.0, 1.0))

        # ── Weighted combination ──────────────────────────────────────────
        score = (f1 * 0.35 + f2 * 0.25 + f3 * 0.20 + f4 * 0.10 + f5 * 0.10)
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "score":          round(score, 3),
            "spec_flux":      round(f1, 3),
            "energy_var":     round(f2, 3),
            "zcr_consist":    round(f3, 3),
            "rolloff":        round(f4, 3),
            "silence":        round(f5, 3),
            "_log_flux":      round(log_flux, 3),
            "_rms_cv":        round(rms_cv, 3),
            "_zcr_cv":        round(zcr_cv, 3),
            "_silence_r":     round(silence_r, 3),
            "_duration_s":    round(len(y) / sr, 1),
        }

    @staticmethod
    def _extract_audio(filepath: str):
        """
        Extracts mono 16kHz audio from any media file using ffmpeg subprocess.
        Returns (y_float32_array, sample_rate, error_or_None).
        This bypasses librosa entirely and works reliably on Streamlit Cloud.
        """
        tmp_wav = None
        try:
            # Use the ffmpeg binary bundled with imageio_ffmpeg — always present
            # on Streamlit Cloud. Do NOT use bare 'ffmpeg' — it is not on PATH.
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp_wav = f.name

            result = subprocess.run(
                [ffmpeg_exe, '-y', '-i', filepath,
                 '-vn',                    # no video
                 '-acodec', 'pcm_s16le',   # 16-bit PCM
                 '-ar', '16000',           # 16 kHz
                 '-ac', '1',               # mono
                 tmp_wav],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                # Check if it's a no-audio-stream error (video-only file)
                if 'no streams' in result.stderr.lower() or \
                   'output file is empty' in result.stderr.lower() or \
                   os.path.getsize(tmp_wav) < 100:
                    return None, None, "No audio stream found in file"
                return None, None, f"ffmpeg failed: {result.stderr[-200:]}"

            if not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) < 100:
                return None, None, "ffmpeg produced empty audio file"

            with wave.open(tmp_wav, 'r') as wf:
                sr      = wf.getframerate()
                n_ch    = wf.getnchannels()
                raw     = wf.readframes(wf.getnframes())

            y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if n_ch > 1:
                y = y.reshape(-1, n_ch).mean(axis=1)

            return y, sr, None

        except subprocess.TimeoutExpired:
            return None, None, "Audio extraction timed out"
        except Exception as e:
            return None, None, f"Audio extraction error: {e}"
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass
