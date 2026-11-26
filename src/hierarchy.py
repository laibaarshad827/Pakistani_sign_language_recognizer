
"""Defines the PSL learning hierarchy and level unlocking rules."""

import math


LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced"]


class LearningHierarchy:
    def __init__(self, classes, sign_order=None):
        classes = list(dict.fromkeys(str(c) for c in classes))

        if not classes:
            raise ValueError("At least one sign is required.")

        if sign_order is None:
            ordered = classes
        else:
            ordered = [
                sign for sign in sign_order
                if sign in classes
            ]

            remaining = [
                sign for sign in classes
                if sign not in ordered
            ]

            ordered.extend(remaining)

        self.classes = ordered

        base_size, extra = divmod(len(ordered), len(LEVEL_ORDER))

        self.levels = {}
        start = 0

        for index, level in enumerate(LEVEL_ORDER):
            size = base_size + (1 if index < extra else 0)

            self.levels[level] = ordered[start:start + size]
            start += size

    def get_signs(self, level):
        if level not in self.levels:
            raise ValueError(f"Unknown level: {level}")

        return list(self.levels[level])

    def get_level(self, sign):
        for level, signs in self.levels.items():
            if sign in signs:
                return level

        raise ValueError(f"Unknown sign: {sign}")

    def is_unlocked(
        self,
        level,
        mastery_lookup,
        attempts_lookup,
        mastery_threshold=0.65,
        required_ratio=0.70,
        minimum_attempts=2
    ):
        """Unlock a level when enough signs in the previous level are mastered."""

        if level not in self.levels:
            raise ValueError(f"Unknown level: {level}")

        index = LEVEL_ORDER.index(level)

        if index == 0:
            return True

        previous_level = LEVEL_ORDER[index - 1]
        previous_signs = self.levels[previous_level]

        if not previous_signs:
            return True

        required_count = math.ceil(
            len(previous_signs) * required_ratio
        )

        mastered_count = sum(
            1
            for sign in previous_signs
            if attempts_lookup(sign) >= minimum_attempts
            and mastery_lookup(sign) >= mastery_threshold
        )

        return mastered_count >= required_count

    def get_all_levels(self):
        return {
            level: list(signs)
            for level, signs in self.levels.items()
        }
