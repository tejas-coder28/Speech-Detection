import os
import pickle
import unittest
from app import app, DB_PATH

class TestDeletionEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        # Enable session authentication for tests
        with self.client.session_transaction() as sess:
            sess["authenticated"] = True
            sess["user_email"] = "test@example.com"

        # Ensure database directory and dummy speaker dict
        os.makedirs("database", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)

        self.dummy_db = {
            "SpeakerA": [0.1] * 192,
            "SpeakerB": [0.2] * 192
        }
        with open(DB_PATH, "wb") as f:
            pickle.dump(self.dummy_db, f)

        # Create dummy recording files
        self.files = [
            os.path.join("recordings", "SpeakerA_enroll_seg0.wav"),
            os.path.join("recordings", "SpeakerA_enroll_seg1.wav"),
            os.path.join("recordings", "SpeakerB_enroll_seg0.wav"),
            os.path.join("recordings", "test_temp.wav")
        ]
        for path in self.files:
            with open(path, "w") as f:
                f.write("dummy audio content")

    def test_single_speaker_remove(self):
        res = self.client.post("/api/speaker/remove", json={"name": "SpeakerA"})
        self.assertEqual(res.status_code, 200)

        # Verify SpeakerA removed from database
        with open(DB_PATH, "rb") as f:
            db = pickle.load(f)
        self.assertNotIn("SpeakerA", db)
        self.assertIn("SpeakerB", db)

        # Verify SpeakerA recordings deleted, but SpeakerB and test_temp remain
        self.assertFalse(os.path.exists(os.path.join("recordings", "SpeakerA_enroll_seg0.wav")))
        self.assertFalse(os.path.exists(os.path.join("recordings", "SpeakerA_enroll_seg1.wav")))
        self.assertTrue(os.path.exists(os.path.join("recordings", "SpeakerB_enroll_seg0.wav")))
        self.assertTrue(os.path.exists(os.path.join("recordings", "test_temp.wav")))

    def test_reset_everything(self):
        res = self.client.post("/api/reset")
        self.assertEqual(res.status_code, 200)

        # Verify DB file deleted
        self.assertFalse(os.path.exists(DB_PATH))

        # Verify recordings folder exists but is empty
        self.assertTrue(os.path.exists("recordings"))
        self.assertEqual(len(os.listdir("recordings")), 0)

if __name__ == "__main__":
    unittest.main()
