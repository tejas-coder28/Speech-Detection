import os
import tempfile
import unittest
import wave

import numpy as np

from model import extract_embedding


class ModelTest(unittest.TestCase):
    def test_extract_embedding_returns_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.wav")
            sample_rate = 16000
            duration = 0.2
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())

            embedding = extract_embedding(path)
            self.assertEqual(embedding.ndim, 1)
            self.assertGreater(embedding.size, 0)


if __name__ == "__main__":
    unittest.main()
