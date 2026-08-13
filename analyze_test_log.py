"""
analyze_test_log.py

Per-user accuracy analysis script for Voiceprint Identity System.
Reads test_logs/{safe_user_id}.csv for a specific user email and displays:
  - Overall accuracy (% correct for ground-truth labeled attempts)
  - Per-speaker accuracy (grouped by expected_name)
  - False-accept count (predicted a wrong enrolled speaker)
  - Average confidence scores on correct vs. incorrect attempts
"""

import sys
import os
import csv
from collections import defaultdict
from database import safe_user_id

REJECTION_SENTINELS = {
    "Unknown Speaker",
    "No Voice Detected",
    "No Users Registered",
    "Multiple Voices Detected",
    "Cancelled",
    "",
}


def analyze_log(user_email: str = None):
    if not user_email:
        if len(sys.argv) > 1:
            user_email = sys.argv[1].strip()
            if user_email == "--email" and len(sys.argv) > 2:
                user_email = sys.argv[2].strip()
        else:
            try:
                user_email = input("Enter user email for accuracy analysis: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled.")
                return

    if not user_email:
        print("User email is required to analyze test logs.")
        return

    uid = safe_user_id(user_email)
    log_path = os.path.join("test_logs", f"{uid}.csv")

    if not os.path.exists(log_path):
        print(f"No logged attempts yet for '{user_email}' — run some verifications first")
        return

    total_attempts = 0
    labeled_attempts = 0
    correct_count = 0

    speaker_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    false_accept_count = 0

    correct_confidences = []
    incorrect_confidences = []

    try:
        with open(log_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_attempts += 1

                expected_name = (row.get("expected_name") or "").strip()
                predicted_name = (row.get("predicted_name") or "").strip()

                try:
                    confidence = float(row.get("confidence", 0.0) or 0.0)
                except ValueError:
                    confidence = 0.0

                c_str = (row.get("correct") or "").strip().lower()

                is_correct = None
                if c_str in ("true", "1"):
                    is_correct = True
                elif c_str in ("false", "0"):
                    is_correct = False

                if is_correct is not None:
                    labeled_attempts += 1
                    if is_correct:
                        correct_count += 1
                        correct_confidences.append(confidence)
                    else:
                        incorrect_confidences.append(confidence)
                        if predicted_name not in REJECTION_SENTINELS:
                            false_accept_count += 1

                    if expected_name:
                        speaker_stats[expected_name]["total"] += 1
                        if is_correct:
                            speaker_stats[expected_name]["correct"] += 1

    except Exception as exc:
        print(f"Error reading {log_path}: {exc}")
        return

    print("=" * 55)
    print("        PER-USER VOICE BIOMETRIC ACCURACY ANALYSIS")
    print("=" * 55)
    print(f"User Account:                       {user_email}")
    print(f"Log File:                           {log_path}")
    print(f"Total Logged Verification Attempts: {total_attempts}")
    print(f"Labeled (Testing Mode) Attempts:    {labeled_attempts}")
    print(f"Unlabeled Attempts:                 {total_attempts - labeled_attempts}")
    print("-" * 55)

    if labeled_attempts == 0:
        print("\nNo labeled test attempts found in log.")
        print("Run verifications with Testing Mode (?test=1) to generate accuracy statistics.")
        print("=" * 55)
        return

    overall_acc = (correct_count / labeled_attempts) * 100.0
    print(f"\n--- OVERALL ACCURACY ---")
    print(f"Accuracy: {overall_acc:.2f}% ({correct_count} / {labeled_attempts} correct)")

    print(f"\n--- PER-SPEAKER ACCURACY ---")
    if speaker_stats:
        for spk, stats in sorted(speaker_stats.items()):
            tot = stats["total"]
            corr = stats["correct"]
            acc = (corr / tot * 100.0) if tot > 0 else 0.0
            print(f"  • {spk}: {acc:.2f}% ({corr} / {tot} correct)")
    else:
        print("  (No expected_name values recorded)")

    print(f"\n--- SECURITY & PERFORMANCE METRICS ---")
    print(f"False-Accept Count: {false_accept_count} (predicted wrong enrolled speaker)")

    avg_correct_conf = (
        sum(correct_confidences) / len(correct_confidences)
        if correct_confidences
        else None
    )
    avg_incorrect_conf = (
        sum(incorrect_confidences) / len(incorrect_confidences)
        if incorrect_confidences
        else None
    )

    avg_corr_str = f"{avg_correct_conf:.4f}" if avg_correct_conf is not None else "N/A"
    avg_incorr_str = f"{avg_incorrect_conf:.4f}" if avg_incorrect_conf is not None else "N/A"

    print(f"Average Confidence (Correct Attempts):   {avg_corr_str}")
    print(f"Average Confidence (Incorrect Attempts): {avg_incorr_str}")
    print("=" * 55)


if __name__ == "__main__":
    analyze_log()
