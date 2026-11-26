
"""Constraint Satisfaction Problem for personalized practice scheduling."""

import math

from hierarchy import LEVEL_ORDER


class PracticeSequenceGenerator:
    """Generates practice sequences using CSP and backtracking."""

    def __init__(self, agent):
        self.agent = agent

    def generate(
        self,
        length=5,
        recent_signs=None,
        max_per_level=None
    ):
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError("Length must be an integer.")

        if length < 1:
            raise ValueError("Length must be at least 1.")

        if max_per_level is not None:
            if (
                not isinstance(max_per_level, int)
                or isinstance(max_per_level, bool)
                or max_per_level < 1
            ):
                raise ValueError("max_per_level must be a positive integer.")

        recent_signs = set(recent_signs or [])

        # Only signs from unlocked levels are valid CSP domain values.
        candidates = []

        for level_index, level in enumerate(LEVEL_ORDER):
            level_info = next(
                item
                for item in self.agent.get_level_progress()
                if item["level"] == level
            )

            if not level_info["unlocked"]:
                continue

            for sign in self.agent.hierarchy.get_signs(level):
                if sign in recent_signs:
                    continue

                mastery = self.agent.learner.mastery(sign)

                if mastery >= self.agent.MASTERY_TARGET:
                    continue

                candidates.append({
                    "sign": sign,
                    "level": level,
                    "level_index": level_index,
                    "mastery": mastery
                })

        if not candidates:
            return []

        if length > len(candidates):
            raise ValueError(
                f"Requested {length} unique signs, but only "
                f"{len(candidates)} eligible signs are available."
            )

        # Variables represent practice slots: 0, 1, ..., length - 1.
        variables = list(range(length))

        # Weakest signs are considered first.
        candidates.sort(
            key=lambda item: (
                item["mastery"],
                item["level_index"],
                item["sign"]
            )
        )

        assignment = {}
        level_counts = {}

        def legal_values(slot):
            values = []

            previous = assignment.get(slot - 1)
            following = assignment.get(slot + 1)

            for candidate in candidates:
                sign = candidate["sign"]
                level_index = candidate["level_index"]

                # Constraint 1: no repeated signs.
                if any(
                    assigned["sign"] == sign
                    for assigned in assignment.values()
                ):
                    continue

                # Constraint 2: maintain curriculum level order.
                if previous is not None:
                    if level_index < previous["level_index"]:
                        continue

                if following is not None:
                    if level_index > following["level_index"]:
                        continue

                # Constraint 3: optional maximum per level.
                if max_per_level is not None:
                    count = level_counts.get(candidate["level"], 0)

                    if count >= max_per_level:
                        continue

                values.append(candidate)

            return values

        def backtrack():
            if len(assignment) == length:
                return True

            # MRV: choose the unassigned slot with the fewest legal values.
            unassigned = [
                slot for slot in variables
                if slot not in assignment
            ]

            options = {
                slot: legal_values(slot)
                for slot in unassigned
            }

            slot = min(
                unassigned,
                key=lambda item: len(options[item])
            )

            # Forward checking: fail immediately if no value is legal.
            if not options[slot]:
                return False

            for candidate in options[slot]:
                assignment[slot] = candidate

                level = candidate["level"]
                level_counts[level] = level_counts.get(level, 0) + 1

                if backtrack():
                    return True

                del assignment[slot]
                level_counts[level] -= 1

            return False

        if not backtrack():
            raise ValueError(
                "No practice sequence satisfies the selected constraints."
            )

        return [
            {
                "position": slot + 1,
                "sign": assignment[slot]["sign"],
                "level": assignment[slot]["level"],
                "mastery": assignment[slot]["mastery"],
                "reason": (
                    "Prioritized because this sign has comparatively "
                    "low mastery."
                )
            }
            for slot in variables
        ]
