
import tempfile
import unittest

from pathlib import Path

from database import Database
from auth import Auth
from hierarchy import LearningHierarchy
from recommendations import PersonalizedLearningAgent


class LearningAgentTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        db_path = Path(self.temp_dir.name) / "test.db"

        self.database = Database(db_path)
        self.auth = Auth(self.database)

        self.classes = [
            "alif", "bay", "pay",
            "tay", "say", "chay",
            "hay", "khay", "dal"
        ]

        self.user_id = self.auth.register(
            "learner_one",
            "SecurePass123"
        )

        self.agent = PersonalizedLearningAgent(
            self.database,
            self.user_id,
            self.classes,
            sign_order=self.classes
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hierarchy_creation(self):
        hierarchy = LearningHierarchy(
            self.classes,
            sign_order=self.classes
        )

        levels = hierarchy.get_all_levels()

        self.assertEqual(len(levels["Beginner"]), 3)
        self.assertEqual(len(levels["Intermediate"]), 3)
        self.assertEqual(len(levels["Advanced"]), 3)

    def test_beginner_unlocked_initially(self):
        progress = self.agent.get_level_progress()

        self.assertTrue(progress[0]["unlocked"])
        self.assertFalse(progress[1]["unlocked"])
        self.assertFalse(progress[2]["unlocked"])

    def test_intermediate_unlocks_after_mastery(self):
        beginner_signs = self.agent.hierarchy.get_signs("Beginner")

        for sign in beginner_signs:
            self.agent.record_result(sign, True)
            self.agent.record_result(sign, True)

        progress = self.agent.get_level_progress()

        self.assertTrue(progress[0]["unlocked"])
        self.assertTrue(progress[1]["unlocked"])
        self.assertFalse(progress[2]["unlocked"])

    def test_recommendation_comes_from_unlocked_level(self):
        recommendation = self.agent.get_recommendation()

        self.assertIn(
            recommendation["sign"],
            self.agent.hierarchy.get_signs("Beginner")
        )

        self.assertEqual(recommendation["level"], "Beginner")

    def test_progress_persists_after_reloading_agent(self):
        self.agent.record_result("alif", True)

        new_agent = PersonalizedLearningAgent(
            self.database,
            self.user_id,
            self.classes,
            sign_order=self.classes
        )

        self.assertEqual(
            new_agent.learner.attempts("alif"),
            1
        )

        self.assertAlmostEqual(
            new_agent.learner.mastery("alif"),
            2 / 3
        )

    def test_users_have_separate_progress(self):
        second_user = self.auth.register(
            "learner_two",
            "SecurePass123"
        )

        self.agent.record_result("alif", True)

        second_agent = PersonalizedLearningAgent(
            self.database,
            second_user,
            self.classes,
            sign_order=self.classes
        )

        self.assertEqual(
            self.agent.learner.attempts("alif"),
            1
        )

        self.assertEqual(
            second_agent.learner.attempts("alif"),
            0
        )

        self.assertAlmostEqual(
            second_agent.learner.mastery("alif"),
            0.5
        )

    def test_dashboard_metrics(self):
        self.agent.record_result("alif", True)

        dashboard = self.agent.get_dashboard()

        self.assertEqual(dashboard["total_signs"], 9)
        self.assertEqual(dashboard["attempted_signs"], 1)
        self.assertEqual(dashboard["mastered_signs"], 0)
        self.assertIsNotNone(dashboard["recommendation"]["sign"])


if __name__ == "__main__":
    unittest.main()
