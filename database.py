"""
database.py

Per-user speaker embedding storage.

Each account's voiceprints are isolated in their own pickle file:
    database/speaker_dict_{safe_user_id}.pkl

where safe_user_id is the account email with every non-alphanumeric
character replaced by an underscore (so tejasmuralidhar2@gmail.com
becomes tejasmuralidhar2_gmail_com).

ALL public functions require a user_id argument (the raw email string
from session['user_email']).  There is no global / default path — this
forces every call-site to be explicit about which account it operates on.
"""

import os
import pickle
import re

import numpy as np


# ---------------------------------------------------------------------------
# Path helper  (single source of truth for the sanitisation rule)
# ---------------------------------------------------------------------------

def safe_user_id(user_email: str) -> str:
    """
    Convert an email address to a filesystem-safe identifier.

    All characters that are not ASCII letters or digits are replaced
    with underscores.  This is the ONLY place the rule lives — every
    other function calls this helper.

    Example:
        "tejasmuralidhar2@gmail.com" → "tejasmuralidhar2_gmail_com"
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", user_email)


def _db_path(user_email: str) -> str:
    """Return the absolute path of this user's pickle file."""
    return os.path.join("database", f"speaker_dict_{safe_user_id(user_email)}.pkl")


def _recordings_dir(user_email: str) -> str:
    """Return the absolute path of this user's recordings sub-folder."""
    return os.path.join("recordings", safe_user_id(user_email))


# ---------------------------------------------------------------------------
# Core CRUD
# ---------------------------------------------------------------------------

def init_db(user_email: str) -> None:
    """Create the database directory and the user's pickle file if absent."""
    os.makedirs("database", exist_ok=True)
    path = _db_path(user_email)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump({}, f)


def register_speaker(user_email: str, name: str, embedding) -> None:
    """
    Save (or overwrite) a speaker's averaged embedding in this user's
    private pickle file.
    """
    init_db(user_email)
    path = _db_path(user_email)
    with open(path, "rb") as f:
        db = pickle.load(f)
    db[name] = np.array(embedding).flatten()
    with open(path, "wb") as f:
        pickle.dump(db, f)


def load_all_speakers(user_email: str) -> dict:
    """
    Load and return this user's speaker dict  { name: embedding }.
    Returns {} if no speakers have been enrolled yet.
    """
    init_db(user_email)
    path = _db_path(user_email)
    with open(path, "rb") as f:
        return pickle.load(f)


def rename_speaker(user_email: str, old_name: str, new_name: str) -> None:
    """Rename a speaker key inside this user's pickle file."""
    init_db(user_email)
    path = _db_path(user_email)
    with open(path, "rb") as f:
        db = pickle.load(f)
    if old_name in db:
        db[new_name] = db.pop(old_name)
        with open(path, "wb") as f:
            pickle.dump(db, f)