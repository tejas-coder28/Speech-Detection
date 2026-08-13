"""
tts.py

Offline Text-to-Speech (TTS) helper using pyttsx3.
Saves greeting audio files for client-side web audio playback on remote deployments.
"""

import os
import re
import sys

_TTS_DIR = os.path.join(os.path.dirname(__file__), "static", "tts")
os.makedirs(_TTS_DIR, exist_ok=True)


def speak_greeting(name: str) -> str | None:
    """
    Saves spoken audio greeting to static/tts/ for remote/web audio playback.
    Returns relative URL path (e.g. '/static/tts/greeting_name.wav') or None if failed.
    """
    if not name or not name.strip():
        return None

    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip())
    filename = f"greeting_{clean_name}.wav"
    filepath = os.path.join(_TTS_DIR, filename)
    greeting_text = f"Welcome, {name}."

    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(greeting_text, filepath)
            engine.runAndWait()

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return f"/static/tts/{filename}"
    except Exception as exc:
        print(f"[tts.py] TTS Notice: Could not generate audio file: {exc}", file=sys.stderr)

    return None
