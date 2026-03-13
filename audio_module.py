import sounddevice as sd
import numpy as np

class AudioDetector:
    def __init__(self, sr=16000):
        self.sr = sr

    def capture_and_predict(self):
        # Captures 2 seconds of audio
        duration = 2 
        recording = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1)
        sd.wait()
        # Logic: Pass recording to RawNet2 model here
        return 0.1 # Returns fake probability
