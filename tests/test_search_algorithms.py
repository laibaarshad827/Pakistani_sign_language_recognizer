
import unittest
import tempfile

from pathlib import Path

from database import Database
from auth import Auth
from recommendations import PersonalizedLearningAgent

from search_algorithms import (
    bfs_path,
    dfs_path,
    a_star_path,
    LearningPathPlanner
)

from csp import PracticeSequenceGenerator


class SearchAlgorithmTests(unittest.TestCase):

    def setUp(self):
        self.graph = {
            "A": {"B": 1, "C": 4},
            "B": {"A": 1, "C": 2, "D": 5},
            "C": {"A": 4, "B": 2, "D": 1},
            "D": {"B": 5, "C": 1}
        }

        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "test.db"

        self.database = Database(db_path)
        self.auth = Auth(self.database)

        self.user_id = self.auth.register(
            "search_user",
            "SecurePass123"
        )

        self.classes = [
            "alif", "bay", "pay",
            "tay", "say", "chay",
            "hay", "khay", "dal"
        ]

        self.agent = PersonalizedLearningAgent(
            self.database,
            self.user_id,
            self.classes,
            sign_order=self.classes
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bfs_finds_shortest_path_by_edges(self):
        path = bfs_path(self.graph, "A", "D")

        self.assertEqual(path, ["A", "B", "D"])

    def test_dfs_finds_a_valid_path(self):
        path = dfs_path(self.graph, "A", "D")

        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "D")

        for index in range(len(path) - 1):
            self.assertIn(path[index + 1], self.graph[path[index]])

    def test_astar_finds_lowest_cost_path(self):
        path = a_star_path(self.graph, "A", "D")

        self.assertEqual(path, ["A", "B", "C", "D"])

    def test_search_returns_none_for_unreachable_goal(self):
        graph = {
            "A": {"B": 1},
            "B": {"A": 1},
            "C": {}
        }

        self.assertIsNone(bfs_path(graph, "A", "C"))
        self.assertIsNone(dfs_path(graph, "A", "C"))
        self.assertIsNone(a_star_path(graph, "A", "C"))

    def test_astar_rejects_negative_edge_cost(self):
        graph = {
            "A": {"B": -1},
            "B": {}
        }

        with self.assertRaises(ValueError):
            a_star_path(graph, "A", "B")

    def test_planner_only_includes_unlocked_signs(self):
        planner = LearningPathPlanner(self.agent)

        beginner_signs = self.agent.hierarchy.get_signs("Beginner")
        intermediate_signs = self.agent.hierarchy.get_signs("Intermediate")

        self.assertEqual(set(planner.graph), set(beginner_signs))

        for sign in intermediate_signs:
            self.assertNotIn(sign, planner.graph)

    def test_planner_rejects_locked_target(self):
        planner = LearningPathPlanner(self.agent)

        beginner_sign = self.agent.hierarchy.get_signs("Beginner")[0]
        locked_sign = self.agent.hierarchy.get_signs("Intermediate")[0]

        with self.assertRaises(ValueError):
            planner.find_bfs_route(beginner_sign, locked_sign)

    def test_planner_finds_route_inside_unlocked_level(self):
        planner = LearningPathPlanner(self.agent)

        signs = self.agent.hierarchy.get_signs("Beginner")

        path = planner.find_bfs_route(signs[0], signs[2])

        self.assertEqual(path[0], signs[0])
        self.assertEqual(path[-1], signs[2])

    def test_csp_generates_unique_practice_signs(self):
        generator = PracticeSequenceGenerator(self.agent)

        sequence = generator.generate(length=3)

        signs = [item["sign"] for item in sequence]

        self.assertEqual(len(sequence), 3)
        self.assertEqual(len(set(signs)), 3)

    def test_csp_excludes_recent_signs(self):
        generator = PracticeSequenceGenerator(self.agent)

        recent = self.agent.hierarchy.get_signs("Beginner")[0]

        sequence = generator.generate(
            length=2,
            recent_signs=[recent]
        )

        signs = [item["sign"] for item in sequence]

        self.assertNotIn(recent, signs)

    def test_csp_preserves_level_order(self):
        beginner_signs = self.agent.hierarchy.get_signs("Beginner")

        for sign in beginner_signs:
            self.agent.record_result(sign, True)
            self.agent.record_result(sign, True)

        generator = PracticeSequenceGenerator(self.agent)
        sequence = generator.generate(length=5)

        level_positions = {
            "Beginner": 0,
            "Intermediate": 1,
            "Advanced": 2
        }

        indices = [
            level_positions[item["level"]]
            for item in sequence
        ]

        self.assertEqual(indices, sorted(indices))

    def test_csp_rejects_impossible_sequence_length(self):
        generator = PracticeSequenceGenerator(self.agent)

        with self.assertRaises(ValueError):
            generator.generate(length=100)


if __name__ == "__main__":
    unittest.main()
