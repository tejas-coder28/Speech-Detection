import os
import wave
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


# ---------------------------------------------------------------------------
# Silence-detection & Consistency thresholds
# ---------------------------------------------------------------------------
_RMS_THRESHOLD  = float(os.environ.get("SILENCE_RMS_THRESHOLD",  "0.005"))
_PEAK_THRESHOLD = float(os.environ.get("SILENCE_PEAK_THRESHOLD", "0.02"))

# Empirical threshold for within-clip window consistency.
# Minimum cosine similarity below this raises MultipleSpeakersError.
# Start at 0.5 as a placeholder; needs empirical tuning: test with clean
# single-speaker clips vs. clips with a deliberate second voice cutting in,
# log the actual minimum-similarity values for each, and set the threshold
# where the two groups separate.
_CONSISTENCY_THRESHOLD = float(os.environ.get("CONSISTENCY_THRESHOLD", "0.4"))


class SilenceDetectedError(RuntimeError):
    """
    Raised when a recording contains no meaningful audio signal.
    Callers should catch this and notify the user rather than crashing.
    """


class MultipleSpeakersError(RuntimeError):
    """
    Raised when a recording is detected to contain multiple voices.
    """


class RecordingCancelledError(RuntimeError):
    """
    Raised when audio recording is cancelled by user request.
    """


def _is_silent(recording: np.ndarray) -> bool:
    """
    Return True if float32 numpy audio array is below RMS and peak thresholds.
    """
    if recording.size == 0:
        return True
    rms  = float(np.sqrt(np.mean(np.square(recording))))
    peak = float(np.max(np.abs(recording)))
    return rms < _RMS_THRESHOLD and peak < _PEAK_THRESHOLD


def _check_silence(recording: np.ndarray, filename: str) -> None:
    """
    Inspect a float32 numpy audio array and raise SilenceDetectedError
    if the signal is below both the RMS and peak thresholds.
    """
    if recording.size == 0:
        raise SilenceDetectedError(
            "Recording is empty (0 samples captured). "
            "Check that your microphone is connected and not muted."
        )

    rms  = float(np.sqrt(np.mean(np.square(recording))))
    peak = float(np.max(np.abs(recording)))

    if rms < _RMS_THRESHOLD and peak < _PEAK_THRESHOLD:
        raise SilenceDetectedError(
            f"No voice detected in '{os.path.basename(filename)}'. "
            f"(RMS={rms:.4f}, peak={peak:.4f}). "
            "Please speak clearly into the microphone and try again."
        )


def check_multiple_speakers(audio_path: str) -> bool:
    """
    Analyze a recorded audio clip for within-clip embedding consistency
    to reject recordings containing 2+ voices.

    Splits the audio into 1.5s windows with 0.5s hop. Skips silent windows.
    For each remaining window, extracts an embedding using model.extract_embedding.
    If fewer than 3 valid (non-silent) windows remain, skips the check.
    Computes L2-normalized centroid and minimum cosine similarity between each
    window embedding and the centroid. Raises MultipleSpeakersError if the
    minimum similarity is below _CONSISTENCY_THRESHOLD.

    Returns:
        True if check passes.

    Raises:
        MultipleSpeakersError: if 2+ voices are detected.
    """
    if not os.path.exists(audio_path):
        return True

    try:
        with wave.open(audio_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            raw_bytes = wav_file.readframes(n_frames)
    except Exception as exc:
        print(f"[audio.py] WARNING: Could not read WAV for consistency check: {exc}")
        return True

    if n_frames == 0 or sample_rate <= 0:
        return True

    if sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    win_samples = int(1.5 * sample_rate)
    hop_samples = int(0.5 * sample_rate)
    total_samples = len(audio)

    if total_samples < win_samples:
        return True

    from model import extract_embedding

    valid_embeddings = []
    temp_dir = os.path.dirname(audio_path) or "recordings"

    window_idx = 0
    start = 0
    while start + win_samples <= total_samples:
        win_audio = audio[start:start + win_samples]

        if not _is_silent(win_audio):
            temp_win_path = os.path.join(temp_dir, f"_temp_win_{window_idx}_{os.path.basename(audio_path)}")
            try:
                win_peak = np.max(np.abs(win_audio))
                norm_win = win_audio / win_peak if win_peak > 0 else win_audio
                win_int16 = np.int16(norm_win * 32767)

                with wave.open(temp_win_path, "wb") as w_out:
                    w_out.setnchannels(1)
                    w_out.setsampwidth(2)
                    w_out.setframerate(sample_rate)
                    w_out.writeframes(win_int16.tobytes())

                emb = extract_embedding(temp_win_path)
                valid_embeddings.append(emb)
            except Exception as exc:
                print(f"[audio.py] WARNING: Window embedding extraction failed: {exc}")
            finally:
                if os.path.exists(temp_win_path):
                    try: os.remove(temp_win_path)
                    except Exception: pass

        window_idx += 1
        start += hop_samples

    if len(valid_embeddings) < 3:
        return True

    stacked = np.stack(valid_embeddings, axis=0)
    centroid = np.mean(stacked, axis=0)
    c_norm = np.linalg.norm(centroid)
    if c_norm > 0:
        centroid = centroid / c_norm

    min_sim = 1.0
    for emb in valid_embeddings:
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb_norm = emb / norm
            sim = float(np.dot(emb_norm, centroid))
        else:
            sim = 0.0
        if sim < min_sim:
            min_sim = sim

    print(f"[audio.py] Consistency check for '{os.path.basename(audio_path)}': "
          f"valid_windows={len(valid_embeddings)}, min_sim={min_sim:.4f}, threshold={_CONSISTENCY_THRESHOLD}")

    if min_sim < _CONSISTENCY_THRESHOLD:
        raise MultipleSpeakersError(
            f"Multiple voices detected in '{os.path.basename(audio_path)}' "
            f"(minimum window similarity={min_sim:.4f} < {_CONSISTENCY_THRESHOLD}). "
            "Please ensure only one person is speaking, then try again."
        )

    return True



def record_audio(filename: str, duration: int = 5,
                 sample_rate: int = 16000, channels: int = 1,
                 cancel_event: Optional[threading.Event] = None) -> str:
    """
    Record audio from the default microphone for `duration` seconds,
    save as 16-bit PCM WAV, and return the file path.

    Raises:
        SilenceDetectedError: if the captured audio contains no voice.
        RecordingCancelledError: if cancelled via cancel_event.
        sounddevice.PortAudioError: if the microphone is unavailable.
    """
    os.makedirs("recordings", exist_ok=True)

    # Resolve output path: keep absolute paths as-is, prefix bare names.
    if os.path.isabs(filename) or os.path.dirname(filename):
        output_path = filename
    else:
        output_path = os.path.join("recordings", filename)

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # --- Capture audio in chunks so cancel_event can interrupt immediately ---
    chunks = []
    chunk_samples = int(0.1 * sample_rate)  # 100ms chunks
    total_samples = int(duration * sample_rate)
    samples_read = 0

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32") as stream:
            while samples_read < total_samples:
                if cancel_event and cancel_event.is_set():
                    if os.path.exists(output_path):
                        try: os.remove(output_path)
                        except Exception: pass
                    raise RecordingCancelledError("Recording cancelled by user.")
                to_read = min(chunk_samples, total_samples - samples_read)
                data, _ = stream.read(to_read)
                chunks.append(data)
                samples_read += len(data)

        if chunks:
            recording = np.concatenate(chunks, axis=0)
            recording = np.squeeze(recording)
        else:
            recording = np.array([], dtype="float32")
    except Exception as exc:
        if isinstance(exc, (SilenceDetectedError, RecordingCancelledError)):
            raise
        # Fallback to standard sd.rec if InputStream fails on some hardware
        if cancel_event and cancel_event.is_set():
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except Exception: pass
            raise RecordingCancelledError("Recording cancelled by user.")
        frames = int(duration * sample_rate)
        recording = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32")
        # Poll sd.wait in loop for cancel
        start_time = threading.Event()
        while sd.get_stream().active if hasattr(sd, "get_stream") else False:
            if cancel_event and cancel_event.is_set():
                sd.stop()
                if os.path.exists(output_path):
                    try: os.remove(output_path)
                    except Exception: pass
                raise RecordingCancelledError("Recording cancelled by user.")
            start_time.wait(0.05)
        sd.wait()
        recording = np.squeeze(recording)

    # --- Silence guard (runs BEFORE writing to disk) ---
    try:
        _check_silence(recording, filename)
    except SilenceDetectedError:
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except Exception: pass
        raise

    # --- Normalise and write WAV ---
    peak = np.max(np.abs(recording)) if recording.size else 0.0
    if peak > 0:
        recording = recording / peak

    audio_int16 = np.int16(recording * 32767)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return output_path