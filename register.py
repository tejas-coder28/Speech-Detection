"""
register.py

Enrolls a new speaker by recording THREE separate 5-second clips,
extracting an ECAPA embedding from each, and storing their L2-normalised
average as the voiceprint.

Averaging multiple embeddings dramatically improves enrollment quality
over a single long recording because:
  - It smooths out within-session variability (pitch drift, breaths).
  - Each segment is independently denoised and VAD-trimmed.
  - Bad/silent clips raise SilenceDetectedError so the UI can prompt
    the user to re-speak rather than enrolling a corrupt voiceprint.
"""

import os
import numpy as np

from audio import (
    record_audio,
    SilenceDetectedError,
    RecordingCancelledError,
    MultipleSpeakersError,
    check_multiple_speakers
)
from model import extract_embedding
from database import register_speaker, safe_user_id


# Number of clips to capture and average for enrollment.
_ENROLL_SEGMENTS = 3
_SEGMENT_DURATION = 5  # seconds per segment


def enroll_new_user(user_email: str, name: str,
                    progress_callback=None, cancel_event=None) -> bool:
    """
    Record _ENROLL_SEGMENTS clips, extract an embedding from each,
    and save the averaged, L2-normalised embedding to THIS USER's
    private database.

    All recording files are stored under
    recordings/{safe_user_id(user_email)}/ so different accounts
    enrolling a speaker with the same name never collide.

    Args:
        user_email:        Logged-in account email (session['user_email']).
        name:              Speaker name (key in this user's database).
        progress_callback: Optional callable(step, total, message) for
                           UI progress updates between segments.
        cancel_event:      Optional threading.Event to check for early cancellation.

    Returns:
        True on success.

    Raises:
        SilenceDetectedError:    If a recording segment is silent.
        MultipleSpeakersError:   If multiple voices are detected in a segment.
        RecordingCancelledError: If cancelled by user.
        RuntimeError:            If all embeddings fail extraction.
    """
    uid = safe_user_id(user_email)
    user_recordings_dir = os.path.join("recordings", uid)
    os.makedirs(user_recordings_dir, exist_ok=True)

    embeddings = []
    recorded_files = []

    seg = 0
    while seg < _ENROLL_SEGMENTS:
        if cancel_event and cancel_event.is_set():
            cancel_event.clear()

        if progress_callback:
            progress_callback(
                seg, _ENROLL_SEGMENTS,
                f"Recording segment {seg + 1} of {_ENROLL_SEGMENTS} "
                f"({_SEGMENT_DURATION}s)... Speak clearly."
            )

        # Build the full path: record_audio handles any path with a directory
        # component as-is (no "recordings/" prefix is added a second time).
        seg_file_path = os.path.join(user_recordings_dir, f"{name}_enroll_seg{seg}.wav")

        try:
            audio_file = record_audio(
                seg_file_path,
                duration=_SEGMENT_DURATION,
                cancel_event=cancel_event,
            )
            # Perform multiple-speaker consistency check
            check_multiple_speakers(audio_file)
        except RecordingCancelledError as exc:
            if os.path.exists(seg_file_path):
                try: os.remove(seg_file_path)
                except Exception: pass
            if cancel_event:
                cancel_event.clear()
            if progress_callback:
                progress_callback(
                    seg, _ENROLL_SEGMENTS,
                    f"Segment {seg + 1} cancelled. Restarting segment {seg + 1}..."
                )
            continue
        except (SilenceDetectedError, MultipleSpeakersError) as exc:
            # Clean up all files recorded so far in this attempt
            for fpath in recorded_files:
                if os.path.exists(fpath):
                    try: os.remove(fpath)
                    except Exception: pass
            if os.path.exists(seg_file_path):
                try: os.remove(seg_file_path)
                except Exception: pass
            raise exc
        except Exception as exc:
            for fpath in recorded_files:
                if os.path.exists(fpath):
                    try: os.remove(fpath)
                    except Exception: pass
            if os.path.exists(seg_file_path):
                try: os.remove(seg_file_path)
                except Exception: pass
            raise exc

        try:
            emb = extract_embedding(audio_file)
            recorded_files.append(audio_file)
        except Exception as exc:
            print(f"[register.py] WARNING: embedding failed for segment "
                  f"{seg + 1}: {exc}. Skipping segment.")
            if os.path.exists(audio_file):
                try: os.remove(audio_file)
                except Exception: pass
            continue

        norm = np.linalg.norm(emb)
        if norm > 0:
            embeddings.append(emb / norm)

        seg += 1

    if not embeddings:
        for fpath in recorded_files:
            if os.path.exists(fpath):
                try: os.remove(fpath)
                except Exception: pass
        raise RuntimeError(
            "Could not extract any valid embeddings. "
            "Check your microphone and try again."
        )

    # Average the normalised segment embeddings and re-normalise.
    avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    avg_norm = np.linalg.norm(avg_embedding)
    if avg_norm > 0:
        avg_embedding = avg_embedding / avg_norm

    register_speaker(user_email, name, avg_embedding)

    if progress_callback:
        progress_callback(
            _ENROLL_SEGMENTS, _ENROLL_SEGMENTS,
            f"Voiceprint for '{name}' saved successfully."
        )

    return True


def enroll_user_from_files(user_email: str, name: str, saved_files: list[dict],
                           progress_callback=None, cancel_event=None) -> tuple[bool, list[dict], str]:
    """
    Process uploaded audio file(s) to enroll a speaker under user_email.
    - If 1 single audio file is uploaded: requires minimum 15.0 seconds of audio,
      slices it into 3 x 5s segments (seg0, seg1, seg2), and averages their embeddings.
    - If multiple audio files are uploaded: processes each file as an individual segment.
    """
    import wave
    import shutil
    from verify import convert_upload_to_wav, AudioTooShortError
    from audio import _check_silence

    uid = safe_user_id(user_email)
    user_recordings_dir = os.path.join("recordings", uid)
    os.makedirs(user_recordings_dir, exist_ok=True)

    embeddings = []
    file_results = []
    recorded_files = []

    # --- CASE 1: Single file upload (Auto-slice 15s file into 3 x 5s segments) ---
    if len(saved_files) == 1:
        f_info = saved_files[0]
        orig_filename = f_info["filename"]
        src_path = f_info["path"]
        temp_wav = src_path + ".converted.wav"

        try:
            if progress_callback:
                progress_callback(0, 3, f"Converting '{orig_filename}'...")

            convert_upload_to_wav(src_path, temp_wav)

            # Read full WAV audio data
            with wave.open(temp_wav, "rb") as wf:
                sr = wf.getframerate()
                nframes = wf.getnframes()
                sampwidth = wf.getsampwidth()
                raw_bytes = wf.readframes(nframes)

            duration = nframes / sr if sr > 0 else 0
            if duration < 15.0:
                file_results.append({
                    "filename": orig_filename,
                    "status": "rejected",
                    "reason": f"Audio clip too short ({duration:.1f}s). Minimum 15 seconds required for single-file enrollment."
                })
                return False, file_results, f"Single-file enrollment requires at least 15 seconds of audio (provided: {duration:.1f}s)."

            # Decode samples to float32
            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

            # Slice into 3 segments of 5.0 seconds (5 * sr samples)
            seg_samples_len = int(5.0 * sr)
            for seg_idx in range(3):
                if cancel_event and cancel_event.is_set():
                    for fpath in recorded_files:
                        if os.path.exists(fpath):
                            try: os.remove(fpath)
                            except Exception: pass
                    return False, file_results, "Enrollment cancelled."

                if progress_callback:
                    progress_callback(
                        seg_idx, 3,
                        f"Processing 5s segment {seg_idx + 1} of 3..."
                    )

                start_idx = seg_idx * seg_samples_len
                end_idx = start_idx + seg_samples_len
                seg_samples = samples[start_idx:end_idx]

                # Check silence on 5s segment
                _check_silence(seg_samples, f"{orig_filename} [segment {seg_idx + 1}]")

                # Save 5s WAV segment: {name}_enroll_seg{seg_idx}.wav
                target_filename = f"{name}_enroll_seg{seg_idx}.wav"
                target_path = os.path.join(user_recordings_dir, target_filename)

                seg_int16 = np.int16(np.clip(seg_samples, -1.0, 1.0) * 32767)
                with wave.open(target_path, "wb") as swf:
                    swf.setnchannels(1)
                    swf.setsampwidth(2)
                    swf.setframerate(sr)
                    swf.writeframes(seg_int16.tobytes())

                # Check multiple speakers
                check_multiple_speakers(target_path)

                # Extract embedding
                emb = extract_embedding(target_path)
                norm = np.linalg.norm(emb)
                if norm == 0:
                    raise RuntimeError(f"Zero embedding for segment {seg_idx + 1}.")

                embeddings.append(emb / norm)
                recorded_files.append(target_path)

            file_results.append({
                "filename": orig_filename,
                "status": "accepted",
                "reason": None
            })

        except (AudioTooShortError, SilenceDetectedError, MultipleSpeakersError, Exception) as exc:
            for fpath in recorded_files:
                if os.path.exists(fpath):
                    try: os.remove(fpath)
                    except Exception: pass
            file_results.append({
                "filename": orig_filename,
                "status": "rejected",
                "reason": str(exc)
            })
            return False, file_results, f"Enrollment failed: {exc}"
        finally:
            for p in [src_path, temp_wav]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass

    # --- CASE 2: Multiple uploaded files (Process each file as 1 segment) ---
    else:
        total_files = len(saved_files)
        for idx, f_info in enumerate(saved_files):
            if cancel_event and cancel_event.is_set():
                for rem in saved_files[idx:]:
                    if os.path.exists(rem["path"]):
                        try: os.remove(rem["path"])
                        except Exception: pass
                return False, file_results, "Enrollment cancelled."

            orig_filename = f_info["filename"]
            src_path = f_info["path"]
            temp_wav = src_path + ".converted.wav"

            if progress_callback:
                progress_callback(
                    idx, total_files,
                    f"Validating '{orig_filename}' ({idx + 1} of {total_files})…"
                )

            try:
                convert_upload_to_wav(src_path, temp_wav)

                with wave.open(temp_wav, "rb") as wf:
                    raw_bytes = wf.readframes(wf.getnframes())
                    if raw_bytes:
                        sampwidth = wf.getsampwidth()
                        if sampwidth == 2:
                            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sampwidth == 4:
                            samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    else:
                        samples = np.array([], dtype=np.float32)

                _check_silence(samples, orig_filename)
                check_multiple_speakers(temp_wav)

                emb = extract_embedding(temp_wav)
                norm = np.linalg.norm(emb)
                if norm == 0:
                    raise RuntimeError("Extracted zero embedding vector.")

                emb_norm = emb / norm

                seg_idx = len(recorded_files)
                target_filename = f"{name}_enroll_seg{seg_idx}.wav"
                target_path = os.path.join(user_recordings_dir, target_filename)

                shutil.move(temp_wav, target_path)

                embeddings.append(emb_norm)
                recorded_files.append(target_path)
                file_results.append({
                    "filename": orig_filename,
                    "status": "accepted",
                    "reason": None
                })

            except (AudioTooShortError, SilenceDetectedError, MultipleSpeakersError, Exception) as exc:
                file_results.append({
                    "filename": orig_filename,
                    "status": "rejected",
                    "reason": str(exc)
                })
            finally:
                for p in [src_path, temp_wav]:
                    if os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass

    if not embeddings:
        return False, file_results, "All uploaded files failed validation. No voiceprint profile created."

    # Average normalized segment embeddings and re-normalize
    avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    avg_norm = np.linalg.norm(avg_embedding)
    if avg_norm > 0:
        avg_embedding = avg_embedding / avg_norm

    register_speaker(user_email, name, avg_embedding)

    accepted_count = len(embeddings)
    msg = f"'{name}' enrolled successfully with {accepted_count} audio segments (5s each)."

    return True, file_results, msg