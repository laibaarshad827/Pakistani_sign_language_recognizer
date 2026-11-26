
import tempfile
import unittest
from pathlib import Path

from database import Database
from auth import Auth
from profiles import ProfileManager


class PlatformTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "test.db"

        self.database = Database(db_path)
        self.auth = Auth(self.database)
        self.profiles = ProfileManager(self.database)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_registration_and_login(self):
        user_id = self.auth.register(
            "laiba_test",
            "SecurePass123",
            "Laiba"
        )

        self.assertGreater(user_id, 0)

        user = self.auth.login("laiba_test", "SecurePass123")

        self.assertIsNotNone(user)
        self.assertEqual(user["user_id"], user_id)

    def test_wrong_password(self):
        self.auth.register("test_user", "SecurePass123")

        result = self.auth.login("test_user", "WrongPassword")

        self.assertIsNone(result)

    def test_duplicate_username(self):
        self.auth.register("test_user", "SecurePass123")

        with self.assertRaises(ValueError):
            self.auth.register("test_user", "AnotherPass123")

    def test_individual_progress(self):
        user1 = self.auth.register("user_one", "SecurePass123")
        user2 = self.auth.register("user_two", "SecurePass123")

        self.profiles.record_attempt(user1, "Alif", True)
        self.profiles.record_attempt(user2, "Bay", False)

        progress1 = self.profiles.get_progress(user1)
        progress2 = self.profiles.get_progress(user2)

        self.assertEqual(len(progress1), 1)
        self.assertEqual(len(progress2), 1)

        self.assertEqual(progress1[0]["sign"], "Alif")
        self.assertEqual(progress2[0]["sign"], "Bay")

    def test_profile_and_level(self):
        user_id = self.auth.register(
            "profile_user",
            "SecurePass123",
            "Laiba"
        )

        profile = self.profiles.get_profile(user_id)

        self.assertEqual(profile["display_name"], "Laiba")
        self.assertEqual(profile["current_level"], "Beginner")

        self.profiles.update_level(user_id, "Intermediate")

        profile = self.profiles.get_profile(user_id)

        self.assertEqual(profile["current_level"], "Intermediate")


if __name__ == "__main__":
    unittest.main()
