
import hashlib
import hmac
import secrets
import base64
import sqlite3

from database import Database


class Auth:
    ITERATIONS = 310000

    def __init__(self, database):
        self.database = database

    def validate_password(self, password):
        if not isinstance(password, str) or len(password) < 8:
            raise ValueError("Password must contain at least 8 characters.")

    def hash_password(self, password):
        salt = secrets.token_bytes(16)

        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.ITERATIONS
        )

        salt_text = base64.b64encode(salt).decode("utf-8")
        key_text = base64.b64encode(key).decode("utf-8")

        return f"pbkdf2_sha256${self.ITERATIONS}${salt_text}${key_text}"

    def verify_password(self, password, stored_hash):
        try:
            algorithm, iterations, salt_text, key_text = stored_hash.split("$")

            if algorithm != "pbkdf2_sha256":
                return False

            salt = base64.b64decode(salt_text, validate=True)
            expected_key = base64.b64decode(key_text, validate=True)

            actual_key = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                int(iterations)
            )

            return hmac.compare_digest(actual_key, expected_key)

        except (ValueError, TypeError):
            return False

    def register(self, username, password, display_name=None):
        username = username.strip() if isinstance(username, str) else ""

        if len(username) < 3 or len(username) > 30:
            raise ValueError("Username must be 3 to 30 characters long.")

        if not all(c.isalnum() or c in "_-" for c in username):
            raise ValueError("Username can contain letters, numbers, _ and - only.")

        self.validate_password(password)

        display_name = display_name or username

        if not isinstance(display_name, str):
            raise ValueError("Display name must be text.")

        display_name = display_name.strip()

        if not display_name or len(display_name) > 60:
            raise ValueError("Display name must be 1 to 60 characters long.")

        password_hash = self.hash_password(password)

        try:
            with self.database.connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, password_hash)
                    VALUES (?, ?)
                    """,
                    (username, password_hash)
                )

                user_id = cursor.lastrowid

                conn.execute(
                    """
                    INSERT INTO profiles (user_id, display_name)
                    VALUES (?, ?)
                    """,
                    (user_id, display_name)
                )

                return user_id

        except sqlite3.IntegrityError as error:
            raise ValueError("Username already exists.") from error

    def login(self, username, password):
        if not isinstance(username, str) or not isinstance(password, str):
            return None

        user = self.database.fetch_one(
            """
            SELECT user_id, username, password_hash
            FROM users
            WHERE username = ?
            """,
            (username.strip(),)
        )

        if user is None:
            return None

        if not self.verify_password(password, user["password_hash"]):
            return None

        return {
            "user_id": user["user_id"],
            "username": user["username"]
        }
