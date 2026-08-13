"""
app.py

Flask web server for the Voiceprint Identity System.

Routes
------
GET  /                              – dashboard (login_required)
GET  /register                      – create account form
POST /register                      – save new account → redirect to /login
GET  /login                         – login form
POST /login                         – validate credentials, set session
GET  /logout                        – clear session, redirect to /login
POST /api/enroll                    – start enrollment  (login_required)
GET  /api/enroll/status/<id>        – poll enrollment   (login_required)
POST /api/verify                    – start verification (login_required)
POST /api/verify/upload             – verify from uploaded audio file
GET  /api/verify/status/<id>        – poll result        (login_required)
GET  /api/speakers                  – list speakers      (login_required)
POST /api/speaker/remove            – delete speaker     (login_required)
POST /api/reset                     – wipe database      (login_required)

User accounts
-------------
Stored in users.json as { email: hashed_password }.
Workflow: Register → redirect to /login → Log in → dashboard.
Register never auto-authenticates the new account.
⚠ register.py is the VOICE ENROLLMENT module — unrelated to this route.

Per-account data isolation
--------------------------
Every call that touches speaker embeddings or recordings is scoped to
session['user_email'] via database.safe_user_id().  Two accounts that
enrol a speaker with the same name (e.g. "Mom") are completely isolated:
separate pickle files, separate recordings sub-folders, zero cross-account
visibility.
"""

import csv
import json
import os
import functools
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime

from flask import (
    Flask, jsonify, redirect, render_template,
    request, session, url_for, send_from_directory,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from database import (
    load_all_speakers,
    register_speaker,
    rename_speaker,
    init_db,
    safe_user_id,
    _db_path,
    _recordings_dir,
)
from audio import SilenceDetectedError, RecordingCancelledError, MultipleSpeakersError
from register import enroll_new_user, enroll_user_from_files, _ENROLL_SEGMENTS, _SEGMENT_DURATION
from verify import identify_voice, verify_from_file, convert_upload_to_wav, AudioTooShortError
from tts import speak_greeting          # offline pyttsx3 TTS helper


# ---------------------------------------------------------------------------
# Flask app + session secret
# ---------------------------------------------------------------------------
app = Flask(__name__)

# REQUIRED for Flask sessions to work. Read from SECRET_KEY env var or generate random key.
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24).hex()

# Deployment configuration flag
REMOTE_DEPLOYMENT = os.environ.get("REMOTE_DEPLOYMENT", "false").lower() in ("true", "1", "yes")

# Max upload size: 15 MB (enforced by Flask / Werkzeug before the view runs).
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# ---------------------------------------------------------------------------
# Startup: purge orphan enroll segment files
# ---------------------------------------------------------------------------
def _cleanup_orphan_enroll_files() -> None:
    """
    On startup, scan every user's recordings sub-folder and delete segment
    files for any speaker who does NOT have all three segments (seg0, seg1,
    seg2).  A complete enrollment always produces exactly these three files.
    Partial sets are leftover from failed or interrupted enrollments and
    should be removed so the recordings folder stays clean.
    """
    recordings_root = "recordings"
    if not os.path.isdir(recordings_root):
        return
    for uid_dir in os.listdir(recordings_root):
        uid_path = os.path.join(recordings_root, uid_dir)
        if not os.path.isdir(uid_path):
            continue
        # Group enroll-seg WAVs by speaker name
        from collections import defaultdict as _dd
        seg_map: dict[str, list[str]] = _dd(list)
        try:
            for fname in os.listdir(uid_path):
                if "_enroll_seg" not in fname or not fname.endswith(".wav"):
                    continue
                speaker_name = fname.split("_enroll_seg")[0]
                seg_map[speaker_name].append(os.path.join(uid_path, fname))
        except Exception as exc:
            print(f"[startup] Error scanning '{uid_path}': {exc}")
            continue
        # Delete any speaker whose segment set is not exactly {seg0, seg1, seg2}
        for speaker_name, fpaths in seg_map.items():
            seg_nums = set()
            for fp in fpaths:
                try:
                    num = int(os.path.basename(fp)
                              .split("_enroll_seg")[1]
                              .replace(".wav", ""))
                    seg_nums.add(num)
                except Exception:
                    pass
            if seg_nums != {0, 1, 2}:
                for fp in fpaths:
                    try:
                        os.remove(fp)
                        print(f"[startup] Removed incomplete enroll file: {fp}")
                    except Exception as exc:
                        print(f"[startup] Could not remove '{fp}': {exc}")


_cleanup_orphan_enroll_files()

# Allowed audio extensions for the /api/verify/upload and /api/enroll/upload routes.
_ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}

# ---------------------------------------------------------------------------
# File-based user account store  (users.json)
# ---------------------------------------------------------------------------
# Format: { "email@example.com": "<werkzeug_hash>" }
# Created automatically on first registration if it doesn't exist.
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
_users_lock = threading.Lock()   # guard concurrent reads/writes

_test_log_lock = threading.Lock()  # Lock for per-user test log CSV files


def _load_users() -> dict[str, str]:
    """Load users.json → dict.  Returns {} if the file doesn't exist yet."""
    with _users_lock:
        if not os.path.exists(USERS_FILE):
            return {}
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def _save_users(users: dict[str, str]) -> None:
    """Automically write the users dict to users.json."""
    with _users_lock:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)


# ---------------------------------------------------------------------------
# login_required decorator
# ---------------------------------------------------------------------------
# Applied to every dashboard route and API endpoint so that unauthenticated
# callers (e.g. someone hitting /api/verify directly) get a 401 instead of
# silently bypassing the login screen.
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            # JSON API calls → 401 JSON; browser page calls → redirect
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required."}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# In-memory job store  { job_id: dict }
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    cancel_evt = threading.Event()
    with _jobs_lock:
        _jobs[job_id] = {
            "state":        "running",   # running | done | error | cancelled
            "step":         0,
            "total":        _ENROLL_SEGMENTS,
            "message":      "",
            "speaker":      None,
            "score":        None,
            "detail":       None,        # "below_threshold" | "ambiguous_match" | None
            "top_match":    None,        # {"name": str, "score": float} | None
            "runner_up":    None,        # {"name": str, "score": float} | None
            "margin":       None,        # float | None
            "cancel_event": cancel_evt,
        }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        res = dict(job)
        res.pop("cancel_event", None)
        return res


def _get_job_raw(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _enroll_worker(job_id: str, user_email: str, name: str) -> None:
    _update_job(job_id, state="running", message="Starting enrollment…")
    raw_job = _get_job_raw(job_id)
    cancel_evt = raw_job.get("cancel_event") if raw_job else None

    try:
        def on_progress(step: int, total: int, msg: str) -> None:
            _update_job(job_id, step=step, total=total, message=msg,
                        seg_start_ts=time.time())

        enroll_new_user(user_email, name,
                        progress_callback=on_progress, cancel_event=cancel_evt)

        job = _get_job(job_id)
        if job and job.get("state") == "cancelled":
            return

        _update_job(
            job_id,
            state="done",
            step=_ENROLL_SEGMENTS,
            total=_ENROLL_SEGMENTS,
            message=f"'{name}' enrolled successfully.",
        )

    except SilenceDetectedError as exc:
        _update_job(job_id, state="error", error_type="silence",
                    message=str(exc))
    except MultipleSpeakersError as exc:
        _update_job(job_id, state="error", error_type="multiple_voices",
                    message=str(exc))
    except RecordingCancelledError:
        _update_job(job_id, state="cancelled", message="Enrollment cancelled.")
    except Exception as exc:
        job = _get_job(job_id)
        if job and job.get("state") == "cancelled":
            return
        _update_job(job_id, state="error", error_type="general",
                    message=str(exc))
    finally:
        # If enrollment did NOT complete (speaker absent from DB), clean up
        # any partial segment files that were written during this attempt.
        try:
            db = load_all_speakers(user_email)
            if name not in db:
                uid = safe_user_id(user_email)
                rec_dir = os.path.join("recordings", uid)
                if os.path.isdir(rec_dir):
                    for fname in os.listdir(rec_dir):
                        if fname.startswith(f"{name}_enroll_seg") and fname.endswith(".wav"):
                            fpath = os.path.join(rec_dir, fname)
                            try:
                                os.remove(fpath)
                                print(f"[enroll] Cleaned up partial file: {fpath}")
                            except Exception as rm_exc:
                                print(f"[enroll] Could not remove '{fpath}': {rm_exc}")
        except Exception:
            pass  # never let cleanup crash the worker


def _log_test_attempt(user_email: str, predicted_name: str, score: float,
                      detail: str | None = None, runner_up: tuple | None = None,
                      margin: float | None = None, expected_name: str = "") -> None:
    """Log every verification attempt to test_logs/{safe_user_id}.csv."""
    try:
        if not user_email:
            return
        uid = safe_user_id(user_email)
        os.makedirs("test_logs", exist_ok=True)
        log_file = os.path.join("test_logs", f"{uid}.csv")
        file_exists = os.path.exists(log_file)

        runner_up_name = runner_up[0] if runner_up and len(runner_up) > 0 else ""
        runner_up_score = f"{runner_up[1]:.4f}" if runner_up and len(runner_up) > 1 else ""
        margin_str = f"{margin:.4f}" if margin is not None else ""
        detail_str = detail or ""

        correct_str = ""
        if expected_name:
            correct_str = "True" if predicted_name == expected_name else "False"

        with _test_log_lock:
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "timestamp", "user_email", "expected_name", "predicted_name",
                    "confidence", "detail", "runner_up_name", "runner_up_score",
                    "margin", "correct"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "user_email": user_email,
                    "expected_name": expected_name or "",
                    "predicted_name": predicted_name or "",
                    "confidence": f"{score:.4f}" if isinstance(score, (float, int)) else "0.0000",
                    "detail": detail_str,
                    "runner_up_name": runner_up_name,
                    "runner_up_score": runner_up_score,
                    "margin": margin_str,
                    "correct": correct_str,
                })
    except Exception as exc:
        print(f"[app.py] Error logging to test_logs: {exc}")


def _log_verification_run(user_email: str, speaker: str, score: float, top_match=None, detail: str = ""):
    """Log a verification attempt (matched or rejected) to test_log.csv for dashboard stats."""
    try:
        uid = safe_user_id(user_email)
        user_dir = os.path.join("recordings", uid)
        os.makedirs(user_dir, exist_ok=True)
        log_path = os.path.join(user_dir, "test_log.csv")
        file_exists = os.path.exists(log_path)

        _DENIED_SENTINELS = {"Unknown Speaker", "No Voice Detected", "No Users Registered", "Cancelled", "Multiple Voices Detected"}
        is_matched = bool(speaker and speaker not in _DENIED_SENTINELS)

        spk_name = speaker if is_matched else (top_match[0] if top_match and isinstance(top_match, (list, tuple)) else "Unknown")
        correct_str = "true" if is_matched else "false"

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["timestamp", "speaker", "score", "status", "correct", "detail"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "speaker": spk_name,
                "score": f"{score:.4f}" if isinstance(score, (float, int)) else "0.0000",
                "status": "matched" if is_matched else "rejected",
                "correct": correct_str,
                "detail": detail or ""
            })
    except Exception as exc:
        print(f"[app.py] Error logging verification run: {exc}")


def _verify_worker(job_id: str, user_email: str, expected_name: str = "") -> None:
    _update_job(job_id, state="running",
                message=f"Listening ({_SEGMENT_DURATION}s)…",
                rec_start_ts=time.time())
    raw_job = _get_job_raw(job_id)
    cancel_evt = raw_job.get("cancel_event") if raw_job else None

    try:
        speaker, score, detail, top_match, runner_up, margin = identify_voice(user_email, cancel_event=cancel_evt)

        if speaker == "Cancelled":
            _update_job(job_id, state="cancelled", message="Verification cancelled.")
            return

        if speaker == "Multiple Voices Detected":
            _update_job(job_id, state="error", error_type="multiple_voices",
                        message="Multiple Voices Detected: Please ensure only one person is speaking, then try again.")
            _log_test_attempt(user_email, speaker, score, detail, runner_up, margin, expected_name)
            return

        # Server-side log top_match, runner_up, margin
        if runner_up is not None:
            print(
                f"[app.py] Verification for {user_email}: "
                f"best={score:.4f} ({top_match[0] if top_match else 'None'}) "
                f"runner_up={runner_up[1]:.4f} ({runner_up[0]}) margin={margin:.4f} — detail='{detail}'"
            )

        _DENIED_SENTINELS = {"Unknown Speaker", "No Voice Detected",
                             "No Users Registered", "Cancelled", "Multiple Voices Detected"}
        tts_url = None
        if speaker and speaker not in _DENIED_SENTINELS:
            tts_url = speak_greeting(speaker)   # Generate greeting audio file URL

        top_match_dict = {"name": top_match[0], "score": round(top_match[1] * 100, 1)} if top_match else None
        runner_up_dict = {"name": runner_up[0], "score": round(runner_up[1] * 100, 1)} if runner_up else None
        margin_val = round(margin * 100, 1) if margin is not None else None

        _update_job(job_id, state="done", speaker=speaker, score=score,
                    detail=detail, top_match=top_match_dict, runner_up=runner_up_dict,
                    margin=margin_val, tts_url=tts_url, message="Analysis complete.")

        _log_verification_run(user_email, speaker, score, top_match, detail)
        _log_test_attempt(user_email, speaker, score, detail, runner_up, margin, expected_name)

    except Exception as exc:
        _update_job(job_id, state="error", error_type="general",
                    message=str(exc))


def _verify_upload_worker(job_id: str, user_email: str, saved_path: str, expected_name: str = "") -> None:
    """
    Background worker for uploaded-file verification.
    Mirrors _verify_worker but calls verify_from_file() instead of
    identify_voice().  Produces the identical job dict shape so the
    frontend polling + result modal need zero changes.
    """
    _update_job(job_id, state="running",
                message="Processing uploaded audio…")
    try:
        # Convert to 16 kHz mono WAV (in-place temp file)
        wav_path = saved_path + ".converted.wav"
        try:
            convert_upload_to_wav(saved_path, wav_path)
        except AudioTooShortError as exc:
            _update_job(job_id, state="error", error_type="general",
                        message=str(exc))
            return
        except ValueError as exc:
            _update_job(job_id, state="error", error_type="general",
                        message=str(exc))
            return

        _update_job(job_id, message="Analysing voiceprint…")

        speaker, score, detail, top_match, runner_up, margin = verify_from_file(
            user_email, wav_path
        )

        if speaker == "Multiple Voices Detected":
            _update_job(job_id, state="error", error_type="multiple_voices",
                        message="Multiple Voices Detected: Please ensure only "
                                "one person is speaking in the uploaded clip.")
            _log_test_attempt(user_email, speaker, score, detail, runner_up, margin, expected_name)
            return

        # Server-side log top_match, runner_up, margin
        if runner_up is not None:
            print(
                f"[app.py] Upload verification for {user_email}: "
                f"best={score:.4f} ({top_match[0] if top_match else 'None'}) "
                f"runner_up={runner_up[1]:.4f} ({runner_up[0]}) margin={margin:.4f} — "
                f"detail='{detail}'"
            )

        _DENIED_SENTINELS = {"Unknown Speaker", "No Voice Detected",
                             "No Users Registered", "Multiple Voices Detected"}
        tts_url = None
        if speaker and speaker not in _DENIED_SENTINELS:
            tts_url = speak_greeting(speaker)

        top_match_dict = {"name": top_match[0], "score": round(top_match[1] * 100, 1)} if top_match else None
        runner_up_dict = {"name": runner_up[0], "score": round(runner_up[1] * 100, 1)} if runner_up else None
        margin_val = round(margin * 100, 1) if margin is not None else None

        _update_job(job_id, state="done", speaker=speaker, score=score,
                    detail=detail, top_match=top_match_dict, runner_up=runner_up_dict,
                    margin=margin_val, tts_url=tts_url, message="Analysis complete.")

        _log_verification_run(user_email, speaker, score, top_match, detail)
        _log_test_attempt(user_email, speaker, score, detail, runner_up, margin, expected_name)

    except Exception as exc:
        _update_job(job_id, state="error", error_type="general",
                    message=str(exc))
    finally:
        # Cleanup: remove both the original upload and the converted WAV
        for p in [saved_path, saved_path + ".converted.wav"]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as exc:
                    print(f"[app.py] Cleanup error for '{p}': {exc}")




# ---------------------------------------------------------------------------
# Auth routes  (no login_required — these ARE the auth flow)
# ---------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    """
    Create a new user account.

    On success: save hashed password to users.json and redirect to
    /login with a flash message — NEVER auto-authenticates.

    NOTE: This route is completely unrelated to register.py, which is
    the voice-enrollment module.  Do not confuse them.
    """
    if session.get("authenticated"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        email    = (request.form.get("email")    or "").strip().lower()
        password = (request.form.get("password") or "")
        confirm  = (request.form.get("confirm")  or "")

        if not email or not password:
            error = "Email and password are required."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            users = _load_users()
            if email in users:
                error = "Email already registered. Please log in."
            else:
                users[email] = generate_password_hash(password)
                _save_users(users)
                # Redirect to login — do NOT set session here.
                return redirect(url_for("login", registered="1"))

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Email + password form.  On success: set session and redirect to /."""
    if session.get("authenticated"):
        return redirect(url_for("index"))

    # Show a success notice when redirected here after registration.
    registered = request.args.get("registered") == "1"

    error = None
    if request.method == "POST":
        email    = (request.form.get("email")    or "").strip().lower()
        password = (request.form.get("password") or "")

        users  = _load_users()
        hashed = users.get(email)
        if hashed and check_password_hash(hashed, password):
            session.clear()
            session["authenticated"] = True
            session["user_email"]    = email
            return redirect(url_for("index"))
        else:
            error = "Invalid email or password."

    return render_template("login.html", error=error, registered=registered)


@app.route("/logout")
def logout():
    """Clear session and return to login screen."""
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Dashboard route
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    current_theme = session.get("ui_theme", "blue")
    if current_theme not in {"blue", "red", "green", "yellow"}:
        current_theme = "blue"
    is_testing_mode = (request.args.get("test") == "1")
    return render_template("index.html",
                           enroll_segments=_ENROLL_SEGMENTS,
                           segment_duration=_SEGMENT_DURATION,
                           user_email=session.get("user_email", ""),
                           current_theme=current_theme,
                           is_testing_mode=is_testing_mode,
                           remote_deployment=REMOTE_DEPLOYMENT)

@app.route("/api/tts/<path:filename>")
def serve_tts(filename):
    """Serve generated TTS audio files."""
    tts_dir = os.path.join(os.path.dirname(__file__), "static", "tts")
    return send_from_directory(tts_dir, filename)


# ---------------------------------------------------------------------------
# API — Enroll
# ---------------------------------------------------------------------------

@app.route("/api/enroll", methods=["POST"])
@login_required
def api_enroll():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required."}), 400

    user_email = session["user_email"]
    job_id = _new_job()
    t = threading.Thread(target=_enroll_worker,
                         args=(job_id, user_email, name), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/enroll/status/<job_id>")
@login_required
def api_enroll_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/api/enroll/cancel/<job_id>", methods=["POST"])
@login_required
def api_enroll_cancel(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        cancel_evt = job.get("cancel_event")
        if cancel_evt:
            cancel_evt.set()
        job["state"] = "cancelled"
    return jsonify({"ok": True, "message": "Enrollment cancellation requested."})


def _enroll_upload_worker(job_id: str, user_email: str, name: str,
                          saved_files: list[dict]) -> None:
    """
    Background worker for /api/enroll/upload.
    Calls enroll_user_from_files() which handles per-file validation,
    conversion, silence/multi-speaker checks, and embedding averaging.
    Reports progress via the same step/total fields as the live enroll worker.
    """
    raw_job = _get_job_raw(job_id)
    cancel_evt = raw_job.get("cancel_event") if raw_job else None

    total_files = len(saved_files)
    _update_job(job_id, state="running",
                message=f"Processing {total_files} file(s)…",
                step=0, total=total_files)

    def progress_callback(step: int, total: int, message: str) -> None:
        _update_job(job_id, state="running",
                    message=message, step=step, total=total)

    try:
        success, file_results, msg = enroll_user_from_files(
            user_email=user_email,
            name=name,
            saved_files=saved_files,
            progress_callback=progress_callback,
            cancel_event=cancel_evt,
        )

        accepted = [r for r in file_results if r["status"] == "accepted"]
        rejected = [r for r in file_results if r["status"] == "rejected"]

        print(
            f"[app.py] Enroll upload for '{name}' ({user_email}): "
            f"{len(accepted)} accepted, {len(rejected)} rejected."
        )
        for r in rejected:
            print(f"  ✗ {r['filename']}: {r['reason']}")

        if not success:
            _update_job(job_id, state="error", error_type="general",
                        message=msg, file_results=file_results,
                        step=total_files, total=total_files)
        else:
            _update_job(job_id, state="done",
                        message=msg, file_results=file_results,
                        step=len(accepted), total=total_files)

    except Exception as exc:
        job = _get_job(job_id)
        if job and job.get("state") == "cancelled":
            return
        _update_job(job_id, state="error", error_type="general",
                    message=str(exc))


@app.route("/api/enroll/upload", methods=["POST"])
@login_required
def api_enroll_upload():
    """
    Accept one or more uploaded audio files for speaker enrollment.
    Validates extension + per-file 15 MB size limit, saves them to temp
    paths, then kicks off _enroll_upload_worker in the same async-job
    pattern as /api/enroll.
    """
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Speaker name is required."}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No audio files provided."}), 400

    os.makedirs("recordings", exist_ok=True)
    saved_files = []
    rejected_immediately = []

    for f in files:
        if not f.filename:
            continue

        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in _ALLOWED_AUDIO_EXTS:
            rejected_immediately.append({
                "filename": f.filename,
                "status": "rejected",
                "reason": f"Unsupported file type '{ext}'"
            })
            continue

        # Per-file 15 MB size guard
        f.seek(0, 2)  # seek to end
        size = f.tell()
        f.seek(0)
        if size > 15 * 1024 * 1024:
            rejected_immediately.append({
                "filename": f.filename,
                "status": "rejected",
                "reason": "File exceeds 15 MB limit"
            })
            continue

        safe_name = secure_filename(f.filename) or "upload"
        temp_path = os.path.join(
            "recordings",
            f"_enroll_upload_{uuid.uuid4().hex[:8]}_{safe_name}"
        )
        try:
            f.save(temp_path)
        except Exception as exc:
            rejected_immediately.append({
                "filename": f.filename,
                "status": "rejected",
                "reason": f"Failed to save: {exc}"
            })
            continue

        saved_files.append({"path": temp_path, "filename": f.filename})

    if not saved_files:
        return jsonify({
            "error": "No valid audio files could be accepted. "
                     "Check extensions (.wav .mp3 .m4a .flac .ogg) and file sizes (max 15 MB each).",
            "file_results": rejected_immediately,
        }), 400

    user_email = session["user_email"]
    job_id = _new_job()
    t = threading.Thread(
        target=_enroll_upload_worker,
        args=(job_id, user_email, name, saved_files),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# API — Verify
# ---------------------------------------------------------------------------

@app.route("/api/verify", methods=["POST"])
@login_required
def api_verify():
    data = request.get_json(silent=True) or {}
    expected_name = (data.get("expected_name") or "").strip()
    user_email = session["user_email"]
    job_id = _new_job()
    t = threading.Thread(target=_verify_worker,
                         args=(job_id, user_email, expected_name), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/verify/status/<job_id>")
@login_required
def api_verify_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/api/verify/cancel/<job_id>", methods=["POST"])
@login_required
def api_verify_cancel(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        cancel_evt = job.get("cancel_event")
        if cancel_evt:
            cancel_evt.set()
        job["state"] = "cancelled"
    return jsonify({"ok": True, "message": "Verification cancelled."})


@app.route("/api/verify/upload", methods=["POST"])
@login_required
def api_verify_upload():
    """
    Accept an uploaded audio file for speaker verification.
    Validates extension + size, saves to a temp path, then kicks off
    _verify_upload_worker in the same async-job pattern as /api/verify.
    """
    from werkzeug.utils import secure_filename

    if "file" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No audio file selected."}), 400

    # Extension allowlist
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in _ALLOWED_AUDIO_EXTS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. "
                     f"Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXTS))}"
        }), 400

    expected_name = (request.form.get("expected_name") or "").strip()

    # Save to a unique temp path inside recordings/
    os.makedirs("recordings", exist_ok=True)
    safe_name = secure_filename(f.filename) or "upload"
    temp_path = os.path.join(
        "recordings",
        f"_upload_{uuid.uuid4().hex[:8]}_{safe_name}"
    )
    try:
        f.save(temp_path)
    except Exception as exc:
        return jsonify({"error": f"Failed to save upload: {exc}"}), 500

    user_email = session["user_email"]
    job_id = _new_job()
    t = threading.Thread(
        target=_verify_upload_worker,
        args=(job_id, user_email, temp_path, expected_name),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id})



# ---------------------------------------------------------------------------
# API — Speakers (list / remove / rename)
# ---------------------------------------------------------------------------

@app.route("/api/speakers")
@login_required
def api_speakers():
    """Return a JSON list of enrolled speaker names for the current account."""
    user_email = session["user_email"]
    try:
        db = load_all_speakers(user_email)
        return jsonify({"speakers": sorted(db.keys())})
    except Exception:
        return jsonify({"speakers": []})


@app.route("/api/speaker/remove", methods=["POST"])
@login_required
def api_speaker_remove():
    """Delete a single speaker from THIS user's database and clean up their recordings."""
    import pickle
    user_email = session["user_email"]
    uid        = safe_user_id(user_email)

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required."}), 400

    try:
        db = load_all_speakers(user_email)
    except Exception:
        return jsonify({"error": "Could not read database."}), 500

    if name not in db:
        return jsonify({"error": f"Speaker '{name}' not found."}), 404

    del db[name]
    db_file = _db_path(user_email)
    with open(db_file, "wb") as f:
        pickle.dump(db, f)

    # Clean up audio recordings in THIS user's sub-folder only
    user_rec_dir = os.path.join("recordings", uid)
    if os.path.exists(user_rec_dir):
        prefix = f"{name}_enroll_seg"
        try:
            for fname in os.listdir(user_rec_dir):
                if fname.startswith(prefix):
                    fpath = os.path.join(user_rec_dir, fname)
                    try:
                        if os.path.isfile(fpath) or os.path.islink(fpath):
                            os.remove(fpath)
                    except Exception as exc:
                        print(f"[app.py] Error deleting recording '{fpath}': {exc}")
        except Exception as exc:
            print(f"[app.py] Error scanning recordings directory: {exc}")

    return jsonify({"ok": True, "message": f"Speaker '{name}' removed."})


@app.route("/api/speaker/rename", methods=["POST"])
@login_required
def api_speaker_rename():
    """Rename an enrolled speaker in THIS user's database and rename their recording files."""
    import pickle
    user_email = session["user_email"]
    uid        = safe_user_id(user_email)

    data     = request.get_json(silent=True) or {}
    old_name = (data.get("old_name") or "").strip()
    new_name = (data.get("new_name") or "").strip()

    if not old_name or not new_name:
        return jsonify({"error": "Name cannot be empty."}), 400

    if old_name == new_name:
        return jsonify({"ok": True, "message": "Name unchanged."})

    try:
        db = load_all_speakers(user_email)
    except Exception:
        return jsonify({"error": "Could not read database."}), 500

    if old_name not in db:
        return jsonify({"error": f"Speaker '{old_name}' not found."}), 404

    if new_name in db:
        return jsonify({"error": f"Speaker '{new_name}' already exists."}), 400

    # Rename key in database
    db[new_name] = db.pop(old_name)
    db_file = _db_path(user_email)
    with open(db_file, "wb") as f:
        pickle.dump(db, f)

    # Rename matching recording files in THIS user's sub-folder only
    user_rec_dir = os.path.join("recordings", uid)
    if os.path.exists(user_rec_dir):
        old_prefix = f"{old_name}_enroll_seg"
        new_prefix = f"{new_name}_enroll_seg"
        try:
            for fname in os.listdir(user_rec_dir):
                if fname.startswith(old_prefix):
                    old_path = os.path.join(user_rec_dir, fname)
                    new_fname = new_prefix + fname[len(old_prefix):]
                    new_path = os.path.join(user_rec_dir, new_fname)
                    try:
                        os.rename(old_path, new_path)
                    except Exception as exc:
                        print(f"[app.py] Error renaming file '{old_path}' to '{new_path}': {exc}")
        except Exception as exc:
            print(f"[app.py] Error scanning recordings directory during rename: {exc}")

    return jsonify({"ok": True, "message": f"Speaker '{old_name}' renamed to '{new_name}'."})



# ---------------------------------------------------------------------------
# API — Reset  (THIS user's data only)
# ---------------------------------------------------------------------------

@app.route("/api/reset", methods=["POST"])
@login_required
def api_reset():
    """
    Wipe ALL speaker data belonging to the currently logged-in account.
    Other accounts are completely unaffected.
    """
    user_email = session["user_email"]
    uid        = safe_user_id(user_email)

    # Delete this user's pickle file
    db_file = _db_path(user_email)
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception as exc:
            print(f"[app.py] Error deleting database for {uid}: {exc}")

    # Delete this user's recordings sub-folder contents
    user_rec_dir = os.path.join("recordings", uid)
    if os.path.exists(user_rec_dir):
        try:
            for fname in os.listdir(user_rec_dir):
                fpath = os.path.join(user_rec_dir, fname)
                try:
                    if os.path.isfile(fpath) or os.path.islink(fpath):
                        os.remove(fpath)
                except Exception as exc:
                    print(f"[app.py] Error deleting file '{fpath}': {exc}")
        except Exception as exc:
            print(f"[app.py] Error scanning recordings directory: {exc}")

    # Also clean up any leftover temp file outside recordings/ if present
    for temp_file in ["test_temp.wav"]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as exc:
                print(f"[app.py] Error deleting temp file '{temp_file}': {exc}")

    return jsonify({"ok": True, "message": "Your speaker profiles and recordings have been cleared."})


# ---------------------------------------------------------------------------
# Theme toggle
# ---------------------------------------------------------------------------

@app.route("/api/theme", methods=["POST"])
@login_required
def api_theme():
    data  = request.get_json(silent=True) or {}
    theme = data.get("theme", "blue")
    valid_themes = {"blue", "red", "green", "yellow"}
    if theme not in valid_themes:
        theme = "blue"
    session["ui_theme"] = theme
    return jsonify({"ok": True, "theme": theme})


# ---------------------------------------------------------------------------
# Stats dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard/stats")
@login_required
def dashboard_stats():
    user_email = session["user_email"]
    uid        = safe_user_id(user_email)
    current_theme = session.get("ui_theme", "blue")
    if current_theme not in {"blue", "red", "green", "yellow"}:
        current_theme = "blue"

    # Load enrolled speakers
    try:
        speakers = load_all_speakers(user_email)
        speaker_names = list(speakers.keys())
    except Exception:
        speaker_names = []

    # Read test_log.csv — scoped to this user (old per-user-recordings log)
    log_path = os.path.join("recordings", uid, "test_log.csv")
    accuracy_by_day   = {}   # {date_str: {correct: int, total: int}}
    attempts_by_speaker = defaultdict(int)   # {speaker_name: count}
    labeled_rows = 0

    # Pre-populate attempts for enrolled speakers so they appear on chart
    for name in speaker_names:
        attempts_by_speaker[name] = 0


    # Success / failure counters — seeded from the old recordings log
    _REJECTION_SENTINELS = {
        "Unknown Speaker", "No Voice Detected",
        "Multiple Voices Detected", "No Users Registered", "",
    }
    verify_success = 0
    verify_failed  = 0

    if os.path.exists(log_path):
        try:
            with open(log_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    correct_field = row.get("correct", "").strip().lower()
                    status_field  = row.get("status", "").strip().lower()

                    if correct_field in ("true", "1") or status_field == "matched":
                        is_correct = True
                    elif correct_field in ("false", "0") or status_field == "rejected":
                        is_correct = False
                    else:
                        continue  # skip malformed row

                    labeled_rows += 1

                    # Feed old-log rows into success/failure totals
                    if is_correct:
                        verify_success += 1
                    else:
                        verify_failed += 1

                    ts = row.get("timestamp", "") or row.get("date", "")
                    try:
                        day = datetime.fromisoformat(ts).strftime("%Y-%m-%d")
                    except Exception:
                        day = ts[:10] if len(ts) >= 10 else "unknown"

                    if day not in accuracy_by_day:
                        accuracy_by_day[day] = {"correct": 0, "total": 0}
                    accuracy_by_day[day]["total"] += 1
                    if is_correct:
                        accuracy_by_day[day]["correct"] += 1

                    spk = row.get("speaker", "").strip()
                    if spk and spk != "Unknown":
                        attempts_by_speaker[spk] += 1
        except Exception as exc:
            print(f"[stats] Error reading log: {exc}")

    # Also fold in newer test_logs/{uid}.csv rows (predicted_name format)
    new_log_path = os.path.join("test_logs", f"{uid}.csv")
    if os.path.exists(new_log_path):
        try:
            with open(new_log_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predicted = (row.get("predicted_name") or "").strip()
                    labeled_rows += 1  # count every new-log row toward Verification Runs
                    if predicted and predicted not in _REJECTION_SENTINELS:
                        verify_success += 1
                    else:
                        verify_failed += 1
        except Exception as exc:
            print(f"[stats] Error reading test_logs: {exc}")


    # Build sorted chart data
    sorted_days = sorted(accuracy_by_day.keys())
    accuracy_labels = sorted_days
    accuracy_data   = [
        round(accuracy_by_day[d]["correct"] / accuracy_by_day[d]["total"] * 100, 1)
        if accuracy_by_day[d]["total"] > 0 else 0
        for d in sorted_days
    ]

    # Cumulative (running) accuracy line
    cum_correct = 0
    cum_total   = 0
    cumulative_accuracy_data = []
    for d in sorted_days:
        cum_correct += accuracy_by_day[d]["correct"]
        cum_total   += accuracy_by_day[d]["total"]
        cumulative_accuracy_data.append(
            round(cum_correct / cum_total * 100, 1) if cum_total > 0 else 0
        )

    # Overall accuracy across all runs
    total_runs = verify_success + verify_failed
    overall_accuracy = round(verify_success / total_runs * 100, 1) if total_runs > 0 else None

    return render_template("stats.html",
                           user_email=user_email,
                           current_theme=current_theme,
                           speaker_names=speaker_names,
                           speaker_count=len(speaker_names),
                           labeled_rows=labeled_rows,
                           verify_success=verify_success,
                           verify_failed=verify_failed,
                           overall_accuracy=overall_accuracy,
                           accuracy_labels=json.dumps(accuracy_labels),
                           accuracy_data=json.dumps(accuracy_data),
                           cumulative_accuracy_data=json.dumps(cumulative_accuracy_data),
                           attempts_by_speaker=json.dumps(dict(attempts_by_speaker)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
