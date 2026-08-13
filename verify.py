import numpy as np
import os

from audio import (
    record_audio,
    SilenceDetectedError,
    RecordingCancelledError,
    MultipleSpeakersError,
    check_multiple_speakers
)
from model import extract_embedding
from database import load_all_speakers

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Minimum rescaled cosine similarity to accept a match.
# Rescaled cosine = (raw_cosine + 1) / 2  → [0, 1].
# 0.72 is the RESCALED threshold; equivalent raw cosine = 2×0.72−1 = 0.44.
THRESHOLD = float(os.environ.get("VERIFY_THRESHOLD", "0.72"))

# Minimum margin that the best score must exceed the second-best score.
# Prevents ambiguous identifications when two enrolled speakers score close.
# Only enforced when two or more speakers are enrolled.
MARGIN = float(os.environ.get("VERIFY_MARGIN", "0.05"))

# Duration of the verification clip (seconds).
_CLIP_DURATION = 5


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1-D float32 vectors.
    Returns a value in [-1, 1].  Safe against zero-norm vectors.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _rescale(raw_cosine: float) -> float:
    """
    Map raw cosine similarity from [-1, 1] → [0, 1] so that:
      - 0.0 = completely opposite embeddings
      - 0.5 = orthogonal (random/unrelated speaker)
      - 1.0 = perfect match
    """
    return (raw_cosine + 1.0) / 2.0


def _centroid_score(test_emb: np.ndarray, stored: np.ndarray) -> float:
    """
    Compute the rescaled cosine similarity between test_emb and the
    centroid of the stored profile.

    If stored is a (N, D) matrix (N > 1 segments):
      - Each row is L2-normalised individually.
      - The normalised rows are averaged.
      - The average is L2-normalised once more so the centroid lies on
        the unit hypersphere (consistent with cosine similarity geometry).
    If stored is already a 1-D (D,) vector it is used as-is — fully
    backward-compatible with existing enrolled single-vector profiles.

    Returns a rescaled score in [0, 1].
    """
    stored = np.atleast_2d(stored)          # always (N, D)

    if stored.shape[0] == 1:
        # Single stored vector — use directly as centroid.
        centroid = stored[0]
    else:
        # Normalize each row onto the unit hypersphere.
        norms = np.linalg.norm(stored, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)   # avoid division by zero
        normed_rows = stored / norms
        # Average the normalized rows, then re-normalize the result.
        avg = normed_rows.mean(axis=0)
        avg_norm = np.linalg.norm(avg)
        centroid = avg / avg_norm if avg_norm > 0 else avg

    return _rescale(_cosine_sim(test_emb, centroid))


def identify_voice(user_email: str,
                   threshold: float = THRESHOLD,
                   cancel_event=None) -> tuple[str, float, str | None, tuple[str, float] | None, tuple[str, float] | None, float | None]:
    """
    Record a short clip and compare it against every enrolled speaker
    belonging to user_email.  Speakers from other accounts are never
    loaded — they are completely invisible to this call.

    Args:
        user_email: The logged-in account email (session['user_email']).

    Returns:
        (speaker_name, score, detail, top_match, runner_up, margin)
          speaker_name: matched name or a sentinel string:
            "No Users Registered" / "Unknown Speaker" / "No Voice Detected" /
            "Multiple Voices Detected" / "Cancelled"
          score:      best rescaled cosine score in [0, 1].
          detail:     None on a genuine match; otherwise one of:
            "below_threshold"  – score never reached THRESHOLD.
            "ambiguous_match"  – score passed THRESHOLD but margin to the
                                 second-best enrolled speaker was too small.
          top_match:  tuple of (best_speaker_name, best_score) or None if no speakers enrolled.
          runner_up:  tuple of (runner_up_name, runner_up_score) when 2+ speakers enrolled; None otherwise.
          margin:     float (top_score - runner_up_score) when 2+ speakers enrolled; None otherwise.

    Raises:
        Nothing – all exceptions are caught and returned as sentinel names.
    """
    speakers = load_all_speakers(user_email)
    if not speakers:
        return "No Users Registered", 0.0, None, None, None, None

    def _clean_temp_files():
        for path in ["test_temp.wav", os.path.join("recordings", "test_temp.wav")]:
            if os.path.exists(path):
                try: os.remove(path)
                except Exception: pass

    # --- Record test clip ---
    try:
        test_audio = record_audio("test_temp.wav", duration=_CLIP_DURATION, cancel_event=cancel_event)
        check_multiple_speakers(test_audio)
    except RecordingCancelledError as exc:
        print(f"[verify.py] Verification cancelled: {exc}")
        _clean_temp_files()
        return "Cancelled", 0.0, None, None, None, None
    except SilenceDetectedError as exc:
        print(f"[verify.py] Silence detected: {exc}")
        _clean_temp_files()
        return "No Voice Detected", 0.0, None, None, None, None
    except MultipleSpeakersError as exc:
        print(f"[verify.py] Multiple speakers detected: {exc}")
        _clean_temp_files()
        return "Multiple Voices Detected", 0.0, None, None, None, None
    except Exception as exc:
        print(f"[verify.py] Recording failed: {exc}")
        _clean_temp_files()
        return "No Voice Detected", 0.0, None, None, None, None


    # --- Extract embedding ---
    try:
        test_emb = extract_embedding(test_audio)
    except Exception as exc:
        print(f"[verify.py] Embedding extraction failed: {exc}")
        return "Unknown Speaker", 0.0, "below_threshold", None, None, None

    # L2-normalise the test embedding once (cheap, avoids repeated work).
    test_norm = np.linalg.norm(test_emb)
    if test_norm > 0:
        test_emb = test_emb / test_norm

    # --- Compare against every enrolled speaker ---
    scores = []
    for name, stored_emb in speakers.items():
        stored_emb = np.array(stored_emb, dtype=np.float32)

        # Guard against embedding dimension mismatches from old profiles.
        stored_flat = stored_emb.flatten()
        if stored_flat.shape[0] != test_emb.flatten().shape[0]:
            # Try multi-row interpretation: (N, D) where last dim matches.
            if stored_emb.ndim == 2 and stored_emb.shape[1] == test_emb.shape[0]:
                pass  # handled by _centroid_score below
            else:
                print(
                    f"[verify.py] Skipping '{name}': embedding shape mismatch "
                    f"({stored_emb.shape} vs {test_emb.shape}). Re-register this user."
                )
                continue

        score = _centroid_score(test_emb, stored_emb)
        scores.append((name, float(score)))

    if not scores:
        return "Unknown Speaker", 0.0, "below_threshold", None, None, None

    scores.sort(key=lambda x: x[1], reverse=True)

    top_match = scores[0]
    best_name, best_score = top_match

    runner_up = scores[1] if len(scores) >= 2 else None
    margin = (best_score - runner_up[1]) if runner_up is not None else None

    # --- Apply threshold + margin check ---
    if best_score >= threshold:
        # Margin check: only enforce when a second speaker exists.
        if runner_up is not None and margin < MARGIN:
            print(
                f"[verify.py] Ambiguous match rejected: best={best_score:.4f} ({best_name}) "
                f"runner_up={runner_up[1]:.4f} ({runner_up[0]}) margin={margin:.4f}"
            )
            return "Unknown Speaker", best_score, "ambiguous_match", top_match, runner_up, margin
        return best_name, best_score, None, top_match, runner_up, margin
    else:
        return "Unknown Speaker", best_score, "below_threshold", top_match, runner_up, margin


# ---------------------------------------------------------------------------
# File-path adapter for Gradio / external callers
# ---------------------------------------------------------------------------
#
# identify_voice() records from the microphone internally, which makes it
# unsuitable for Gradio (audio is captured in the browser and passed as a
# file path).  verify_from_file() mirrors the *exact* same pipeline —
# load speakers → extract embedding → L2-normalise → compare → threshold —
# but skips the recording step, accepting an on-disk WAV instead.
#
# Keeping this in verify.py (rather than in the Gradio app) means the
# scoring logic lives in exactly one place.
# ---------------------------------------------------------------------------

def verify_from_file(user_email: str,
                     audio_path: str,
                     threshold: float = THRESHOLD) -> tuple[str, float, str | None, tuple[str, float] | None, tuple[str, float] | None, float | None]:
    """
    Run speaker verification from an existing audio file, matching only
    against speakers enrolled under user_email.

    Args:
        user_email: The logged-in account email (session['user_email']).
        audio_path: Path to a WAV file on disk (e.g. Gradio temp file).
        threshold:  Minimum rescaled cosine similarity to accept.

    Returns:
        (speaker_name, score, detail, top_match, runner_up, margin) — same contract as
        identify_voice().
    """
    speakers = load_all_speakers(user_email)
    if not speakers:
        return "No Users Registered", 0.0, None, None, None, None

    # --- Extract embedding from the provided file ---
    try:
        test_emb = extract_embedding(audio_path)
    except Exception as exc:
        print(f"[verify.py] verify_from_file – embedding failed: {exc}")
        return "Unknown Speaker", 0.0, "below_threshold", None, None, None

    # L2-normalise (same as identify_voice)
    test_norm = np.linalg.norm(test_emb)
    if test_norm > 0:
        test_emb = test_emb / test_norm

    # --- Compare against every enrolled speaker ---
    scores = []
    for name, stored_emb in speakers.items():
        stored_emb = np.array(stored_emb, dtype=np.float32)

        stored_flat = stored_emb.flatten()
        if stored_flat.shape[0] != test_emb.flatten().shape[0]:
            if stored_emb.ndim == 2 and stored_emb.shape[1] == test_emb.shape[0]:
                pass  # multi-row handled by _centroid_score
            else:
                print(
                    f"[verify.py] Skipping '{name}': shape mismatch "
                    f"({stored_emb.shape} vs {test_emb.shape})."
                )
                continue

        score = _centroid_score(test_emb, stored_emb)
        scores.append((name, float(score)))

    if not scores:
        return "Unknown Speaker", 0.0, "below_threshold", None, None, None

    scores.sort(key=lambda x: x[1], reverse=True)

    top_match = scores[0]
    best_name, best_score = top_match

    runner_up = scores[1] if len(scores) >= 2 else None
    margin = (best_score - runner_up[1]) if runner_up is not None else None

    # --- Apply threshold + margin check ---
    if best_score >= threshold:
        if runner_up is not None and margin < MARGIN:
            print(
                f"[verify.py] verify_from_file: ambiguous match rejected: "
                f"best={best_score:.4f} ({best_name}) runner_up={runner_up[1]:.4f} ({runner_up[0]}) margin={margin:.4f}"
            )
            return "Unknown Speaker", best_score, "ambiguous_match", top_match, runner_up, margin
        return best_name, best_score, None, top_match, runner_up, margin
    else:
        return "Unknown Speaker", best_score, "below_threshold", top_match, runner_up, margin


# ---------------------------------------------------------------------------
# Uploaded-file format conversion  (used by app.py upload route)
# ---------------------------------------------------------------------------

# Upload guardrails
_MIN_UPLOAD_SECONDS = 1.0
_MAX_UPLOAD_SECONDS = 30.0
_UPLOAD_SAMPLE_RATE = 16000   # target SR, matching record_audio()


class AudioTooShortError(ValueError):
    """Raised when an uploaded audio file is shorter than the minimum."""


def convert_upload_to_wav(src_path: str, dst_path: str) -> str:
    """
    Read any format supported by soundfile / libsndfile (WAV, FLAC, OGG)
    or fallback through torchaudio (MP3, M4A), resample to 16 kHz mono
    16-bit PCM WAV — matching what record_audio() produces.

    Duration guardrails:
      - < 1 s  -> raises AudioTooShortError
      - > 30 s -> silently trimmed to first 30 s

    Returns dst_path for chaining.
    """
    import wave
    import soundfile as sf

    # --- Load audio data ---
    data = None
    sr = None
    try:
        data, sr = sf.read(src_path, dtype="float32", always_2d=True)
    except Exception:
        pass

    if data is None:
        try:
            import av
            container = av.open(src_path)
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("No audio stream found in file.")
            sr = stream.codec_context.sample_rate or stream.rate or 16000
            resampler = av.AudioResampler(format="flt", layout="mono")
            frames = []
            for frame in container.decode(stream):
                resample_frames = resampler.resample(frame)
                if resample_frames:
                    for rf in resample_frames:
                        frames.append(rf.to_ndarray().flatten())
            if not frames:
                raise ValueError("Audio stream contains no frames.")
            data = np.concatenate(frames)
        except Exception:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(src_path)
                data = waveform.numpy().T
                sr = int(sr)
            except Exception as exc:
                raise ValueError(
                    f"Could not decode audio file: {exc}"
                ) from exc

    # --- Mono mixdown ---
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data.squeeze()

    # --- Resample to target SR ---
    if sr != _UPLOAD_SAMPLE_RATE:
        try:
            import torchaudio
            import torch
            tensor = torch.from_numpy(data).unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=_UPLOAD_SAMPLE_RATE
            )
            tensor = resampler(tensor)
            data = tensor.squeeze(0).numpy()
        except Exception as exc:
            print(f"[verify.py] torchaudio resample failed ({exc}), using scipy")
            from scipy.signal import resample as scipy_resample
            target_len = int(len(data) * _UPLOAD_SAMPLE_RATE / sr)
            data = scipy_resample(data, target_len).astype(np.float32)
        sr = _UPLOAD_SAMPLE_RATE

    # --- Duration guardrails ---
    duration_s = len(data) / sr
    if duration_s < _MIN_UPLOAD_SECONDS:
        raise AudioTooShortError(
            f"Audio is too short ({duration_s:.1f}s). "
            f"Please upload at least {_MIN_UPLOAD_SECONDS:.0f} second of speech."
        )
    max_samples = int(_MAX_UPLOAD_SECONDS * sr)
    if len(data) > max_samples:
        print(
            f"[verify.py] Trimming upload from {duration_s:.1f}s "
            f"to {_MAX_UPLOAD_SECONDS:.0f}s"
        )
        data = data[:max_samples]

    # --- Normalise and write 16-bit PCM WAV ---
    peak = np.max(np.abs(data)) if data.size else 0.0
    if peak > 0:
        data = data / peak
    audio_int16 = np.int16(data * 32767)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with wave.open(dst_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

    return dst_path
