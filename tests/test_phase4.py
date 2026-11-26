
import tempfile
import unittest

from pathlib import Path

import numpy as np

from database import Database
from auth import Auth
from clustering import SignClusterAnalyzer
from q_learning import QLearningAgent
from genetic_algorithm import GeneticSVMOptimizer


class Phase4Tests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.database = Database(
            Path(self.temp_dir.name) / "test.db"
        )

        self.auth = Auth(self.database)

        self.user_id = self.auth.register(
            "phase4_user",
            "SecurePass123"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_kmeans_clustering(self):
        rng = np.random.default_rng(42)

        X = np.vstack([
            rng.normal(0, 0.1, (10, 4)),
            rng.normal(5, 0.1, (10, 4)),
            rng.normal(10, 0.1, (10, 4)),
            rng.normal(15, 0.1, (10, 4))
        ])

        y = np.array(
            ["alif"] * 10
            + ["bay"] * 10
            + ["pay"] * 10
            + ["tay"] * 10
        )

        analyzer = SignClusterAnalyzer(n_clusters=2)

        result = analyzer.analyze(X, y, top_n=3)

        self.assertEqual(len(analyzer.sign_names), 4)
        self.assertEqual(len(result["clusters"]), 2)
        self.assertEqual(len(result["similar_signs"]), 3)

    def test_similar_sign_distances(self):
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1]
        ])

        y = np.array([
            "alif", "alif",
            "bay", "bay"
        ])

        analyzer = SignClusterAnalyzer(n_clusters=2)
        analyzer.fit(X, y)

        pairs = analyzer.get_similar_signs()

        self.assertEqual(len(pairs), 1)
        self.assertGreaterEqual(pairs[0]["distance"], 0)

    def test_q_learning_state_creation(self):
        state = QLearningAgent.make_state(
            "Beginner",
            0.3,
            0.8
        )

        self.assertEqual(
            state,
            "Beginner|low|good"
        )

    def test_q_learning_update(self):
        agent = QLearningAgent(
            self.database,
            self.user_id,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0
        )

        new_value = agent.update(
            "Beginner|low|poor",
            "alif",
            reward=1,
            next_state="Beginner|medium|good",
            next_actions=["bay", "pay"]
        )

        self.assertAlmostEqual(new_value, 0.1)

        self.assertAlmostEqual(
            agent.get_q_value(
                "Beginner|low|poor",
                "alif"
            ),
            0.1
        )

    def test_q_table_persistence(self):
        agent = QLearningAgent(
            self.database,
            self.user_id
        )

        agent.update(
            "Beginner|low|poor",
            "alif",
            reward=1,
            next_state="Beginner|medium|good",
            done=True
        )

        new_agent = QLearningAgent(
            self.database,
            self.user_id
        )

        self.assertAlmostEqual(
            new_agent.get_q_value(
                "Beginner|low|poor",
                "alif"
            ),
            0.1
        )

    def test_q_tables_are_user_specific(self):
        second_user = self.auth.register(
            "phase4_user2",
            "SecurePass123"
        )

        agent1 = QLearningAgent(
            self.database,
            self.user_id
        )

        agent2 = QLearningAgent(
            self.database,
            second_user
        )

        agent1.update(
            "Beginner|low|poor",
            "alif",
            1,
            "Beginner|medium|good",
            done=True
        )

        self.assertEqual(
            agent1.get_q_value(
                "Beginner|low|poor",
                "alif"
            ),
            0.1
        )

        self.assertEqual(
            agent2.get_q_value(
                "Beginner|low|poor",
                "alif"
            ),
            0.0
        )

    def test_q_learning_reward(self):
        reward = QLearningAgent.calculate_reward(
            True,
            0.5,
            0.7
        )

        self.assertAlmostEqual(reward, 1.4)

    def test_genetic_algorithm(self):
        rng = np.random.default_rng(42)

        X = rng.normal(size=(36, 4))

        y = np.array(
            ["alif"] * 12
            + ["bay"] * 12
            + ["pay"] * 12
        )

        X[:12, 0] += 3
        X[12:24, 1] += 3
        X[24:, 2] += 3

        optimizer = GeneticSVMOptimizer(
            population_size=4,
            generations=1,
            mutation_rate=0.1,
            cv_folds=2,
            random_state=42
        )

        optimizer.fit(X, y)

        result = optimizer.get_best_parameters()

        self.assertGreater(result["number_of_features"], 0)
        self.assertLessEqual(
            result["number_of_features"],
            X.shape[1]
        )

        self.assertIn(result["C"], optimizer.C_VALUES)
        self.assertIn(result["gamma"], optimizer.GAMMA_VALUES)

        self.assertEqual(len(result["history"]), 1)

    def test_genetic_algorithm_model_creation(self):
        rng = np.random.default_rng(42)

        X = rng.normal(size=(36, 4))

        y = np.array(
            ["alif"] * 12
            + ["bay"] * 12
            + ["pay"] * 12
        )

        X[:12, 0] += 3
        X[12:24, 1] += 3
        X[24:, 2] += 3

        optimizer = GeneticSVMOptimizer(
            population_size=4,
            generations=1,
            cv_folds=2
        )

        optimizer.fit(X, y)

        model = optimizer.build_model()

        selected = optimizer.get_best_parameters()[
            "selected_features"
        ]

        model.fit(X[:, selected], y)

        predictions = model.predict(X[:, selected])

        self.assertEqual(len(predictions), len(y))


if __name__ == "__main__":
    unittest.main()
