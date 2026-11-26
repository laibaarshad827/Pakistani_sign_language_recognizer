
"""Personalized learning agent for PSL Trainer."""

from learner import LearnerModel
from hierarchy import LearningHierarchy, LEVEL_ORDER
from profiles import ProfileManager
from q_learning import QLearningAgent


class PersonalizedLearningAgent:

    MASTERY_TARGET = 0.80
    RECENT_ATTEMPTS = 10

    def __init__(
        self,
        database,
        user_id,
        classes,
        sign_order=None,
        seed=42
    ):
        self.database = database
        self.user_id = user_id
        self.classes = list(classes)

        self.learner = LearnerModel(self.classes, seed=seed)

        self.hierarchy = LearningHierarchy(
            self.classes,
            sign_order=sign_order
        )

        self.profiles = ProfileManager(database)

        self.q_agent = QLearningAgent(
            database,
            user_id,
            seed=seed
        )

        self.load_progress()

    def load_progress(self):
        """Load this user's Bayesian mastery values from SQLite."""

        rows = self.database.fetch_all(
            """
            SELECT sign, alpha, beta
            FROM sign_progress
            WHERE user_id = ?
            """,
            (self.user_id,)
        )

        for row in rows:
            sign = row["sign"]

            if sign in self.learner.alpha:
                self.learner.alpha[sign] = float(row["alpha"])
                self.learner.beta[sign] = float(row["beta"])

    def _recent_success_rate(self):
        """Calculate success rate from the user's last ten attempts."""

        rows = self.database.fetch_all(
            """
            SELECT success
            FROM learning_attempts
            WHERE user_id = ?
            ORDER BY attempt_id DESC
            LIMIT ?
            """,
            (self.user_id, self.RECENT_ATTEMPTS)
        )

        if not rows:
            return 0.5

        return sum(row["success"] for row in rows) / len(rows)

    def _get_active_stage(self):
        """Find the earliest unlocked level with unfinished signs."""

        for level in LEVEL_ORDER:
            signs = self.hierarchy.get_signs(level)

            unlocked = self.hierarchy.is_unlocked(
                level,
                self.learner.mastery,
                self.learner.attempts
            )

            if not unlocked:
                continue

            available = [
                sign for sign in signs
                if self.learner.mastery(sign) < self.MASTERY_TARGET
            ]

            if available:
                return level, signs, available

        return LEVEL_ORDER[-1], self.hierarchy.get_signs(
            LEVEL_ORDER[-1]
        ), []

    def _make_current_state(self):
        """Build a Q-learning state from current learner performance."""

        level, signs, available = self._get_active_stage()

        mastery = (
            sum(self.learner.mastery(sign) for sign in signs) / len(signs)
            if signs else 1.0
        )

        success_rate = self._recent_success_rate()

        state = self.q_agent.make_state(
            level,
            mastery,
            success_rate
        )

        return state, level, signs, available

    def record_result(
        self,
        sign,
        success,
        predicted_sign=None,
        confidence=None,
        feedback=None
    ):
        """Save practice, update mastery, and learn from the outcome."""

        if sign not in self.classes:
            raise ValueError(f"Unknown sign: {sign}")

        if not isinstance(success, bool):
            raise ValueError("Success must be boolean.")

        # Capture the learner's state before this attempt.
        state, _, _, _ = self._make_current_state()
        mastery_before = self.learner.mastery(sign)

        # Save the attempt exactly once.
        self.profiles.record_attempt(
            user_id=self.user_id,
            sign=sign,
            success=success,
            predicted_sign=predicted_sign,
            confidence=confidence,
            feedback=feedback
        )

        # Update the Bayesian learner model.
        self.learner.update(sign, success)

        mastery_after = self.learner.mastery(sign)

        # Determine the learner's new state and available actions.
        next_state, _, _, next_actions = self._make_current_state()

        reward = self.q_agent.calculate_reward(
            success,
            mastery_before,
            mastery_after
        )

        # Update the Q-value for the sign just practised.
        self.q_agent.update(
            state=state,
            action=sign,
            reward=reward,
            next_state=next_state,
            next_actions=next_actions,
            done=not next_actions
        )

    def get_level_progress(self):
        """Return progress and unlocking information for every level."""

        results = []

        for level in LEVEL_ORDER:
            signs = self.hierarchy.get_signs(level)

            mastered = sum(
                self.learner.mastery(sign) >= self.MASTERY_TARGET
                for sign in signs
            )

            attempted = sum(
                self.learner.attempts(sign) > 0
                for sign in signs
            )

            average_mastery = (
                sum(self.learner.mastery(sign) for sign in signs)
                / len(signs)
                if signs else 0.0
            )

            unlocked = self.hierarchy.is_unlocked(
                level,
                self.learner.mastery,
                self.learner.attempts
            )

            results.append({
                "level": level,
                "total_signs": len(signs),
                "attempted": attempted,
                "mastered": mastered,
                "average_mastery": average_mastery,
                "unlocked": unlocked
            })

        return results

    def get_learning_path(self):
        """Return every sign with its current learning status."""

        path = []
        level_progress = {
            item["level"]: item
            for item in self.get_level_progress()
        }

        for level in LEVEL_ORDER:
            unlocked = level_progress[level]["unlocked"]

            signs = []

            for sign in self.hierarchy.get_signs(level):
                mastery = self.learner.mastery(sign)
                attempts = self.learner.attempts(sign)

                if attempts == 0:
                    status = "Not started"
                elif mastery >= self.MASTERY_TARGET:
                    status = "Mastered"
                elif mastery < 0.65:
                    status = "Needs revision"
                else:
                    status = "In progress"

                signs.append({
                    "sign": sign,
                    "mastery": mastery,
                    "attempts": attempts,
                    "status": status
                })

            path.append({
                "level": level,
                "unlocked": unlocked,
                "signs": signs
            })

        return path

    def get_recommendation(self):
        """Use Q-learning to select the next sign in the active level."""

        state, level, _, available = self._make_current_state()

        if not available:
            return {
                "sign": None,
                "level": None,
                "mastery": None,
                "attempts": 0,
                "reason": (
                    "Congratulations! All signs have reached "
                    "the mastery target."
                )
            }

        recommended_sign = self.q_agent.choose_action(
            state,
            available
        )

        mastery = self.learner.mastery(recommended_sign)
        attempts = self.learner.attempts(recommended_sign)

        if attempts == 0:
            reason = "This sign is new. Start learning it."
        elif mastery < 0.65:
            reason = "Your mastery is low. This sign needs revision."
        else:
            reason = "Keep practising until you reach 80% mastery."

        return {
            "sign": recommended_sign,
            "level": level,
            "mastery": mastery,
            "attempts": attempts,
            "reason": reason,
            "strategy": "Q-learning adaptive recommendation"
        }

    def get_dashboard(self):
        """Return the information needed by the learner dashboard."""

        total_signs = len(self.classes)

        attempted = sum(
            self.learner.attempts(sign) > 0
            for sign in self.classes
        )

        mastered = sum(
            self.learner.mastery(sign) >= self.MASTERY_TARGET
            for sign in self.classes
        )

        overall_mastery = (
            sum(self.learner.mastery(sign) for sign in self.classes)
            / total_signs
            if total_signs else 0.0
        )

        return {
            "total_signs": total_signs,
            "attempted_signs": attempted,
            "mastered_signs": mastered,
            "overall_mastery": overall_mastery,
            "recent_success_rate": self._recent_success_rate(),
            "levels": self.get_level_progress(),
            "recommendation": self.get_recommendation()
        }
