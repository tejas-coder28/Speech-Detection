"""
vad.py

Voice Activity Detection (VAD) using a pure-NumPy energy-based approach.

No external C-extensions required (webrtcvad removed).

Algorithm
---------
1. Convert audio to mono float32.
2. Split into short overlapping frames (30 ms, 10 ms hop).
3. Compute the RMS energy of each frame.
4. Keep frames whose RMS exceeds a threshold relative to the peak frame.
5. Write the kept frames back to the same file (in-place).

This is simpler than WebRTC VAD but sufficient for clean studio/microphone
recordings, avoids the platform-specific webrtcvad wheel, and works in every
Python environment without any extra installation.

Tunable constants
-----------------
_FRAME_MS      Frame duration in milliseconds (default 30).
_HOP_MS        Hop (step) between frames in milliseconds (default 10).
_ENERGY_RATIO  A frame is kept when its RMS ≥ peak_rms × _ENERGY_RATIO.
               Lower values keep more frames (less aggressive).
               Higher values are stricter (only loud frames kept).
"""

import numpy as np
import soundfile as sf

_FRAME_MS    = 30     # ms — frame length for energy analysis
_HOP_MS      = 10     # ms — hop between consecutive frames
_ENERGY_RATIO = 0.05  # fraction of peak RMS below which a frame is silence
_TARGET_SR   = 16000  # preferred sample rate for VAD analysis

# Minimum number of samples the output must contain.
# ECAPA-TDNN requires ≥ 1 s of input (≈ 100 mel frames).
_MIN_SPEECH_SAMPLES = _TARGET_SR  # 1 second @ 16 kHz


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mono(data: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if data.ndim > 1:
        return np.mean(data, axis=1)
    return data


def _resample_to_16k(data: np.ndarray, orig_sr: int) -> np.ndarray:
    """Linear-interpolation resample to _TARGET_SR.  Fast, no librosa needed."""
    if orig_sr == _TARGET_SR:
        return data
    target_len = int(len(data) * _TARGET_SR / orig_sr)
    if target_len <= 0:
        return data
    orig_idx   = np.linspace(0, len(data) - 1, num=len(data))
    target_idx = np.linspace(0, len(data) - 1, num=target_len)
    return np.interp(target_idx, orig_idx, data)


def _energy_vad(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Return a copy of `signal` with silence frames removed.

    Parameters
    ----------
    signal      : 1-D float32 array, normalised to [-1, 1].
    sample_rate : sample rate of `signal`.

    Returns
    -------
    1-D float32 array containing only the voiced frames, or the original
    `signal` if the result would be too short (< _MIN_SPEECH_SAMPLES).
    """
    frame_len = int(sample_rate * _FRAME_MS / 1000)
    hop_len   = int(sample_rate * _HOP_MS  / 1000)

    if len(signal) < frame_len:
        return signal  # too short to analyse; keep as-is

    # Build a boolean mask — True for samples that belong to a voiced frame.
    mask = np.zeros(len(signal), dtype=bool)

    rms_values = []
    starts = list(range(0, len(signal) - frame_len + 1, hop_len))

    for start in starts:
        frame = signal[start : start + frame_len]
        rms_values.append(float(np.sqrt(np.mean(frame ** 2))))

    if not rms_values:
        return signal

    rms_arr  = np.array(rms_values)
    peak_rms = rms_arr.max()

    if peak_rms == 0.0:
        return signal  # complete silence; leave unchanged

    threshold = peak_rms * _ENERGY_RATIO

    for i, (start, rms) in enumerate(zip(starts, rms_arr)):
        if rms >= threshold:
            end = min(start + frame_len, len(signal))
            mask[start:end] = True

    voiced = signal[mask]

    if len(voiced) == 0:
        return signal  # nothing passed threshold; keep original

    # Safety guard: never return less than 1 second (ECAPA minimum).
    if len(voiced) < _MIN_SPEECH_SAMPLES:
        return signal

    return voiced



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strip_non_speech(audio_path: str) -> str:
    """
    Overwrite a WAV file in-place, keeping only energy-active (speech) frames.
    Returns the same path for convenient chaining.

    Falls back to leaving the file untouched on any error, so the pipeline
    is never blocked by a preprocessing failure.
    """
    try:
        data, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception as exc:
        print(f"[vad.py] WARNING: could not read {audio_path}, skipping VAD. "
              f"Reason: {exc}")
        return audio_path

    if data.size == 0:
        return audio_path

    mono     = _to_mono(data)
    resampled = _resample_to_16k(mono, sample_rate)

    voiced = _energy_vad(resampled, _TARGET_SR)

    # Only write back if VAD actually changed the audio.
    if len(voiced) == len(resampled):
        return audio_path  # nothing was stripped

    try:
        sf.write(audio_path, voiced, _TARGET_SR)
    except Exception as exc:
        print(f"[vad.py] WARNING: could not write VAD result to {audio_path}. "
              f"Reason: {exc}")

    return audio_path