"""
denoise.py

Lightweight noise-reduction preprocessing for the speaker recognition
pipeline. Runs BEFORE audio reaches the ECAPA embedding model, so it helps
both enrollment and verification without touching any matching/threshold
logic.

What it does:
1. Loads a WAV file.
2. Estimates the noise profile from the quietest section of the clip
   (assumes there's at least a brief gap of near-silence -- true for most
   short recordings) and subtracts it from the whole signal.
3. Trims leading/trailing near-silence so the embedding model focuses on
   the actual speech rather than dead air.
4. Overwrites the file in-place with the cleaned audio so nothing else in
   the pipeline needs to change.

What it does NOT do (see caveats discussed earlier):
- It does not separate two overlapping speakers. If two people are talking
  at once, noisereduce will NOT isolate one voice -- that needs a separate
  speaker-separation/diarization model, not this.
- It won't fix a fundamentally bad recording (clipping, extremely low
  volume, mic right next to a fan).
"""

import numpy as np
import soundfile as sf
import noisereduce as nr

from vad import strip_non_speech


def denoise_wav(audio_path: str, noise_sample_seconds: float = 0.5) -> str:
    """
    Clean up a WAV file in-place: strip non-speech frames (VAD), reduce
    steady background noise, and trim silence from the start/end. Returns
    the same path for convenient chaining.
    """
    # Run VAD first so the noise profile below is estimated from a clip
    # that's already mostly speech, not raw unfiltered audio.
    strip_non_speech(audio_path)

    try:
        data, sample_rate = sf.read(audio_path)
    except Exception as e:
        print(f"[denoise.py] WARNING: could not read {audio_path} for denoising, skipping. Reason: {e}")
        return audio_path

    if data.size == 0:
        return audio_path

    # noisereduce expects float; soundfile already gives float for most WAVs,
    # but normalize just in case it's int-encoded.
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # Use the first N seconds as the "noise profile" sample. Most short
    # recordings have a brief quiet moment right at the start before the
    # person begins speaking.
    noise_sample_len = int(noise_sample_seconds * sample_rate)
    noise_clip = data[:noise_sample_len] if data.size > noise_sample_len else data

    try:
        reduced = nr.reduce_noise(y=data, sr=sample_rate, y_noise=noise_clip, stationary=True)
    except Exception as e:
        print(f"[denoise.py] WARNING: noise reduction failed for {audio_path}, using original audio. Reason: {e}")
        reduced = data

    trimmed = _trim_silence(reduced, sample_rate)

    try:
        sf.write(audio_path, trimmed, sample_rate)
    except Exception as e:
        print(f"[denoise.py] WARNING: could not write denoised audio back to {audio_path}. Reason: {e}")

    return audio_path


def _trim_silence(signal: np.ndarray, sample_rate: int, threshold_ratio: float = 0.02) -> np.ndarray:
    """
    Trim leading/trailing near-silence based on amplitude threshold relative
    to the loudest point in the clip. Leaves the middle of the recording
    untouched -- this only cuts dead air at the edges.
    """
    if signal.size == 0:
        return signal

    peak = np.max(np.abs(signal))
    if peak <= 0:
        return signal

    threshold = peak * threshold_ratio
    above_threshold = np.where(np.abs(signal) > threshold)[0]

    if above_threshold.size == 0:
        return signal  # entire clip is near-silent; leave as-is rather than returning empty

    start = above_threshold[0]
    end = above_threshold[-1] + 1
    return signal[start:end]