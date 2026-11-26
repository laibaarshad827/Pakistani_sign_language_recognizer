
"""User-specific Q-Learning for personalized PSL practice."""

import random

from database import Database


class QLearningAgent:

    def __init__(
        self,
        database,
        user_id,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.2,
        seed=42
    ):
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1.")

        if not 0 <= discount_factor <= 1:
            raise ValueError("Discount factor must be between 0 and 1.")

        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1.")

        self.database = database
        self.user_id = user_id

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.rng = random.Random(seed)

    @staticmethod
    def make_state(level, mastery, recent_success_rate):
        """Convert learner measurements into a discrete state."""

        if level not in ["Beginner", "Intermediate", "Advanced"]:
            raise ValueError("Invalid learning level.")

        if not 0 <= mastery <= 1:
            raise ValueError("Mastery must be between 0 and 1.")

        if not 0 <= recent_success_rate <= 1:
            raise ValueError("Success rate must be between 0 and 1.")

        if mastery < 0.4:
            mastery_bucket = "low"
        elif mastery < 0.7:
            mastery_bucket = "medium"
        else:
            mastery_bucket = "high"

        if recent_success_rate < 0.4:
            performance_bucket = "poor"
        elif recent_success_rate < 0.7:
            performance_bucket = "average"
        else:
            performance_bucket = "good"

        return (
            f"{level}|{mastery_bucket}|{performance_bucket}"
        )

    def get_q_value(self, state, action):
        row = self.database.fetch_one(
            """
            SELECT q_value
            FROM q_values
            WHERE user_id = ? AND state = ? AND action = ?
            """,
            (self.user_id, state, action)
        )

        return float(row["q_value"]) if row else 0.0

    def choose_action(self, state, available_actions):
        """Epsilon-greedy action selection."""

        actions = list(dict.fromkeys(available_actions))

        if not actions:
            raise ValueError("At least one action must be available.")

        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        values = {
            action: self.get_q_value(state, action)
            for action in actions
        }

        best_value = max(values.values())

        best_actions = [
            action for action, value in values.items()
            if value == best_value
        ]

        return self.rng.choice(best_actions)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_actions=None,
        done=False
    ):
        """Apply the Q-Learning Bellman update and persist it."""

        if not isinstance(reward, (int, float)):
            raise ValueError("Reward must be numeric.")

        if not isinstance(done, bool):
            raise ValueError("done must be boolean.")

        current_q = self.get_q_value(state, action)

        if done:
            future_value = 0.0
        else:
            actions = list(next_actions or [])

            future_value = max(
                (self.get_q_value(next_state, a) for a in actions),
                default=0.0
            )

        target = (
            reward
            + self.discount_factor * future_value
        )

        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        self.database.execute(
            """
            INSERT INTO q_values
            (user_id, state, action, q_value)
            VALUES (?, ?, ?, ?)

            ON CONFLICT(user_id, state, action)
            DO UPDATE SET q_value = excluded.q_value
            """,
            (self.user_id, state, action, new_q)
        )

        return new_q

    @staticmethod
    
    def calculate_reward(success, mastery_before, mastery_after):
        """Reward practice outcomes and improvement in mastery."""

        if not isinstance(success, bool):
            raise ValueError("Success must be boolean.")

        if not 0 <= mastery_before <= 1:
            raise ValueError("Invalid previous mastery.")

        if not 0 <= mastery_after <= 1:
            raise ValueError("Invalid updated mastery.")

        outcome_reward = 1.0 if success else -1.0

        improvement_reward = (
            mastery_after - mastery_before
        ) * 2.0

        return outcome_reward + improvement_reward

    def get_q_table(self):
        """Return the current user's persisted Q-table."""

        rows = self.database.fetch_all(
            """
            SELECT state, action, q_value
            FROM q_values
            WHERE user_id = ?
            ORDER BY state, action
            """,
            (self.user_id,)
        )

        return [
            {
                "state": row["state"],
                "action": row["action"],
                "q_value": float(row["q_value"])
            }
            for row in rows
        ]
