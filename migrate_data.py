"""
migrate_data.py

One-time migration of the pre-isolation shared data to the per-user layout.

Run ONCE after deploying the per-account isolation changes:

    python migrate_data.py [--owner tejasmuralidhar2@gmail.com]

What it does
------------
1. Reads the old flat database/speaker_dict.pkl.
2. Writes every entry into database/speaker_dict_{safe_id}.pkl
   (the new per-user file for the owner account).
3. Moves every WAV file from the old flat recordings/ folder into
   recordings/{safe_id}/ (creating the sub-folder if needed).
4. Renames the old flat speaker_dict.pkl to speaker_dict.pkl.migrated
   so the app never reads from both locations at once.
5. Prints a full summary of what was migrated.

If the old flat speaker_dict.pkl does not exist the script exits early
(migration already done or nothing to migrate).
"""

import argparse
import os
import pickle
import re
import shutil
import sys


# ---------------------------------------------------------------------------
# Helpers (inline copies so this script is self-contained)
# ---------------------------------------------------------------------------

def _safe_user_id(email):
    return re.sub(r"[^a-zA-Z0-9]", "_", email)


OLD_PKL = os.path.join("database", "speaker_dict.pkl")
OLD_REC_DIR = "recordings"


def migrate(owner_email):
    uid = _safe_user_id(owner_email)
    new_pkl = os.path.join("database", "speaker_dict_{}.pkl".format(uid))
    new_rec_dir = os.path.join("recordings", uid)

    # -----------------------------------------------------------------------
    # Guard: nothing to migrate
    # -----------------------------------------------------------------------
    if not os.path.exists(OLD_PKL):
        print("[migrate] Old flat database '{}' not found -- "
              "nothing to migrate (already done or clean install).".format(OLD_PKL))
        return

    print("[migrate] Migrating shared data -> owner account: {}".format(owner_email))
    print("[migrate] Safe user ID : {}".format(uid))
    print("[migrate] New pkl path : {}".format(new_pkl))
    print("[migrate] New rec dir  : {}".format(new_rec_dir))
    print()

    # -----------------------------------------------------------------------
    # 1. Read old shared pickle
    # -----------------------------------------------------------------------
    with open(OLD_PKL, "rb") as f:
        old_db = pickle.load(f)

    print("[migrate] Speakers found in old database ({}):".format(len(old_db)))
    for name in sorted(old_db.keys()):
        print("           * {}".format(name))
    print()

    # -----------------------------------------------------------------------
    # 2. Merge into new per-user pickle (keep existing entries if any)
    # -----------------------------------------------------------------------
    os.makedirs("database", exist_ok=True)
    if os.path.exists(new_pkl):
        with open(new_pkl, "rb") as f:
            new_db = pickle.load(f)
        print("[migrate] Existing entries in new database: {}".format(sorted(new_db.keys())))
    else:
        new_db = {}

    new_db.update(old_db)

    with open(new_pkl, "wb") as f:
        pickle.dump(new_db, f)
    print("[migrate] OK: Wrote {} speakers to {}".format(len(new_db), new_pkl))
    print()

    # -----------------------------------------------------------------------
    # 3. Move WAV files from flat recordings/ -> recordings/{uid}/
    # -----------------------------------------------------------------------
    os.makedirs(new_rec_dir, exist_ok=True)

    moved_files = []
    skipped_files = []

    if os.path.isdir(OLD_REC_DIR):
        for fname in os.listdir(OLD_REC_DIR):
            src = os.path.join(OLD_REC_DIR, fname)
            # Skip sub-directories (including the newly-created uid sub-folder)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(new_rec_dir, fname)
            try:
                shutil.move(src, dst)
                moved_files.append(fname)
            except Exception as exc:
                skipped_files.append((fname, str(exc)))

    print("[migrate] Recording files moved ({}):".format(len(moved_files)))
    for fname in sorted(moved_files):
        print("           * {}".format(fname))

    if skipped_files:
        print("\n[migrate] WARNING -- files that could NOT be moved ({}):".format(len(skipped_files)))
        for fname, reason in skipped_files:
            print("           FAIL {}: {}".format(fname, reason))
    print()

    # -----------------------------------------------------------------------
    # 4. Rename old flat pickle so it's never read again
    # -----------------------------------------------------------------------
    archived = OLD_PKL + ".migrated"
    try:
        os.rename(OLD_PKL, archived)
        print("[migrate] OK: Archived old pickle -> {}".format(archived))
    except Exception as exc:
        print("[migrate] WARNING: could not rename old pickle: {}".format(exc))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("[migrate] Migration complete.")
    print("  Speakers migrated : {}".format(len(old_db)))
    print("  Recordings moved  : {}".format(len(moved_files)))
    print("  New database path : {}".format(new_pkl))
    print("  New recordings dir: {}".format(new_rec_dir))
    if skipped_files:
        print("  WARNING Files skipped: {} -- check warnings above".format(len(skipped_files)))
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate pre-isolation shared speaker data to a single owner account."
    )
    parser.add_argument(
        "--owner",
        default="tejasmuralidhar2@gmail.com",
        help="Email of the account that owns the existing data "
             "(default: tejasmuralidhar2@gmail.com)",
    )
    args = parser.parse_args()

    # Change to the project root so relative paths resolve correctly.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    migrate(args.owner)
