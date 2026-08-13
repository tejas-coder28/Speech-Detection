import os
import sys
import types
import wave
from typing import Optional

import numpy as np

# WORKAROUND 1: SpeechBrain's lazy-import system sometimes touches an optional
# k2/ASR-decoding module (irrelevant to speaker recognition) during internal
# warning checks, and crashes if k2 isn't installed. Pre-registering a harmless
# dummy module short-circuits that lazy import so it never fails.
for _fake_mod in ("speechbrain.integrations", "speechbrain.integrations.k2_fsa"):
    if _fake_mod not in sys.modules:
        sys.modules[_fake_mod] = types.ModuleType(_fake_mod)

# WORKAROUND 2: SpeechBrain registers several optional integrations
# (huggingface, wordemb, encodec, k2_fsa, ...) as LazyModule placeholders in
# sys.modules. They're only supposed to actually import their real dependency
# the first time something touches a real attribute on them.
#
# The problem: deep inside torch, when it registers custom ops, it calls
# inspect.getframeinfo() for debugging info. That function walks through
# EVERY module in sys.modules and does hasattr(module, '__file__') on each
# one -- including these lazy placeholders. Just checking hasattr() triggers
# a real import attempt, which can cascade into transformers/torch.distributed
# and crash -- even though this has nothing to do with speaker recognition.
#
# Fix: make dunder-attribute probes (like __file__) fail quietly with a
# normal AttributeError instead of forcing an eager import. This is exactly
# what inspect.getmodule() already expects and handles gracefully for
# ordinary modules that lack that attribute. Real, intentional use of these
# lazy modules (e.g. actually calling into wordemb) still works normally,
# since only dunder access is intercepted.
import speechbrain.utils.importutils as _sb_importutils

_original_lazy_getattr = _sb_importutils.LazyModule.__getattr__


def _safe_lazy_getattr(self, attr):
    if attr.startswith("__") and attr.endswith("__"):
        raise AttributeError(attr)
    return _original_lazy_getattr(self, attr)


_sb_importutils.LazyModule.__getattr__ = _safe_lazy_getattr

import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

verification_model = None

try:
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    verification_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        local_strategy=LocalStrategy.COPY,
    )
    print("[model.py] ECAPA model loaded OK - using real embeddings.")
except Exception as e:
    verification_model = None
    print(f"[model.py] WARNING: ECAPA failed to load, using WEAK fallback features. Reason: {e}")


def _load_audio_vector(audio_path):
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    if audio.size == 0:
        return np.zeros(64, dtype=np.float32)

    if audio.size > 1:
        audio = audio / max(abs(audio).max(), 1.0)

    energy = np.mean(np.square(audio))
    if energy <= 0:
        return np.zeros(64, dtype=np.float32)

    features = []
    for start in range(0, len(audio), 512):
        chunk = audio[start:start + 512]
        if chunk.size == 0:
            continue
        features.append(float(np.mean(np.abs(chunk))))
        features.append(float(np.std(chunk)))

    if not features:
        features = [0.0, 0.0]

    vector = np.array(features, dtype=np.float32)
    if vector.size < 64:
        vector = np.pad(vector, (0, 64 - vector.size), mode="constant")
    return vector[:64]


def extract_embedding(audio_path):
    # Clean up background noise and trim silence before extracting the
    # embedding. This runs for both enrollment and verification audio,
    # since both go through this same function.
    from denoise import denoise_wav
    audio_path = denoise_wav(audio_path)

    if verification_model is not None:
        try:
            signal = verification_model.load_audio(audio_path)
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)  # (samples,) -> (1, samples)

            # ----------------------------------------------------------------
            # ECAPA-TDNN requires at least 1 second of audio (16 000 samples
            # @ 16 kHz).  Its CNN stack applies (2, 2) padding at every layer,
            # so feeding fewer than ~100 mel frames causes:
            #   "Padding size should be less than the corresponding input
            #    dimension … input [1, 80, 1]"
            # Pad with zeros on the right to guarantee the minimum length.
            # Zero-padding does not distort speaker identity because ECAPA
            # uses temporal attention to focus on voiced frames.
            # ----------------------------------------------------------------
            MIN_SAMPLES = 16_000  # 1 second @ 16 kHz
            if signal.shape[-1] < MIN_SAMPLES:
                pad_amount = MIN_SAMPLES - signal.shape[-1]
                signal = torch.nn.functional.pad(signal, (0, pad_amount))
                print(
                    f"[model.py] INFO: signal too short "
                    f"({signal.shape[-1] - pad_amount} samples); "
                    f"zero-padded to {MIN_SAMPLES} samples for ECAPA."
                )

            with torch.no_grad():
                embedding = verification_model.encode_batch(signal)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"[model.py] WARNING: real embedding failed for {audio_path}, using WEAK fallback. Reason: {e}")

    return _load_audio_vector(audio_path)