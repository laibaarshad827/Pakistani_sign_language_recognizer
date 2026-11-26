
from database import Database


class ProfileManager:
    def __init__(self, database):
        self.database = database

    def get_profile(self, user_id):
        return self.database.fetch_one(
            """
            SELECT u.user_id, u.username,
                   p.display_name, p.current_level,
                   p.created_at
            FROM users u
            JOIN profiles p ON u.user_id = p.user_id
            WHERE u.user_id = ?
            """,
            (user_id,)
        )

    def update_level(self, user_id, level):
        valid_levels = ["Beginner", "Intermediate", "Advanced"]

        if level not in valid_levels:
            raise ValueError("Invalid learning level.")

        self.database.execute(
            """
            UPDATE profiles
            SET current_level = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (level, user_id)
        )

    def record_attempt(
        self,
        user_id,
        sign,
        success,
        predicted_sign=None,
        confidence=None,
        feedback=None
    ):
        if not isinstance(sign, str) or not sign.strip():
            raise ValueError("Sign name is required.")

        if not isinstance(success, bool):
            raise ValueError("Success must be True or False.")

        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise ValueError("Confidence must be numeric.")

            if not 0 <= confidence <= 1:
                raise ValueError("Confidence must be between 0 and 1.")

        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_attempts
                (user_id, sign, success, predicted_sign, confidence, feedback)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    sign.strip(),
                    int(success),
                    predicted_sign,
                    confidence,
                    feedback
                )
            )

            alpha_change = 1 if success else 0
            beta_change = 0 if success else 1

            conn.execute(
                """
                INSERT INTO sign_progress
                (user_id, sign, alpha, beta, attempts, successes, failures,
                 last_attempt_at)
                VALUES (?, ?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)

                ON CONFLICT(user_id, sign)
                DO UPDATE SET
                    alpha = alpha + excluded.alpha - 1,
                    beta = beta + excluded.beta - 1,
                    attempts = attempts + 1,
                    successes = successes + excluded.successes,
                    failures = failures + excluded.failures,
                    last_attempt_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    sign.strip(),
                    1 + alpha_change,
                    1 + beta_change,
                    int(success),
                    int(not success)
                )
            )

    def get_progress(self, user_id):
        rows = self.database.fetch_all(
            """
            SELECT sign, alpha, beta, attempts,
                   successes, failures, last_attempt_at
            FROM sign_progress
            WHERE user_id = ?
            ORDER BY sign
            """,
            (user_id,)
        )

        result = []

        for row in rows:
            mastery = row["alpha"] / (row["alpha"] + row["beta"])

            result.append({
                "sign": row["sign"],
                "mastery": mastery,
                "attempts": row["attempts"],
                "successes": row["successes"],
                "failures": row["failures"],
                "last_attempt_at": row["last_attempt_at"]
            })

        return result

    def get_history(self, user_id, limit=50):
        if not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("Limit must be between 1 and 500.")

        return self.database.fetch_all(
            """
            SELECT attempt_id, sign, success, predicted_sign,
                   confidence, feedback, attempted_at
            FROM learning_attempts
            WHERE user_id = ?
            ORDER BY attempt_id DESC
            LIMIT ?
            """,
            (user_id, limit)
        )
