
"""Bayesian learner model with Thompson Sampling."""

import json
import os

import numpy as np

from classifiers import SEED


def practice_level(mastery):
    if mastery < 0.40:
        return "Practise a lot"
    if mastery < 0.60:
        return "Practise more"
    if mastery < 0.80:
        return "Almost there"
    return "Well done"


class LearnerModel:
    """Tracks Beta beliefs about mastery and recommends signs."""

    def __init__(self, classes, seed=SEED):
        self.classes = list(classes)
        self.alpha = {c: 1.0 for c in self.classes}
        self.beta = {c: 1.0 for c in self.classes}
        self.rng = np.random.default_rng(seed)

    def mastery(self, sign):
        return self.alpha[sign] / (self.alpha[sign] + self.beta[sign])

    def attempts(self, sign):
        return int(self.alpha[sign] + self.beta[sign] - 2)

    def update(self, sign, success):
        if sign not in self.alpha:
            raise ValueError(f"Unknown sign: {sign}")

        if not isinstance(success, bool):
            raise ValueError("Success must be True or False.")

        if success:
            self.alpha[sign] += 1.0
        else:
            self.beta[sign] += 1.0

    def next_sign(self, avoid=None, available=None):
        """Choose the least promising sign using Thompson Sampling."""

        candidates = list(self.classes if available is None else available)

        candidates = list(dict.fromkeys(
            sign for sign in candidates if sign in self.alpha
        ))

        if not candidates:
            raise ValueError("No signs are available for practice.")

        if avoid in candidates and len(candidates) > 1:
            candidates.remove(avoid)

        draws = {
            sign: self.rng.beta(self.alpha[sign], self.beta[sign])
            for sign in candidates
        }

        return min(draws, key=draws.get)

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w") as file:
            json.dump(
                {"alpha": self.alpha, "beta": self.beta},
                file,
                indent=2
            )

    def load(self, path):
        try:
            with open(path) as file:
                saved = json.load(file)
        except FileNotFoundError:
            return self
        except (json.JSONDecodeError, OSError):
            print(f"Could not read {path}; starting with fresh progress.")
            return self

        for sign in self.classes:
            self.alpha[sign] = float(saved["alpha"].get(sign, 1.0))
            self.beta[sign] = float(saved["beta"].get(sign, 1.0))

        return self
