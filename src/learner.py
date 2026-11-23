"""Progress tracker: remembers how well each sign is mastered and chooses what to practise next."""
import json
import os

import numpy as np

from classifiers import SEED



def practice_level(mastery):
    """Plain-words advice for a mastery value between 0 and 1."""
    if mastery < 0.40:
        return "Practise a lot"
    if mastery < 0.60:
        return "Practise more"
    if mastery < 0.80:
        return "Almost there"
    return "Well done"

class LearnerModel:
    """Keeps a Beta(alpha, beta) belief about the learner's mastery of every sign.

    Every sign starts at Beta(1, 1) ("no idea yet"). A success adds 1 to alpha and a
    failure adds 1 to beta (Bayes' rule for a coin-flip style outcome). To choose the next
    sign we use Thompson sampling: draw one random mastery value from each sign's belief
    and practise the sign with the LOWEST draw. Weak signs are drawn low most of the time,
    and signs we know little about are sometimes drawn low too, so they get explored.
    """

    def __init__(self, classes, seed=SEED):
        self.classes = list(classes)
        self.alpha = {c: 1.0 for c in self.classes}
        self.beta = {c: 1.0 for c in self.classes}
        self.rng = np.random.default_rng(seed)

    def mastery(self, sign):
        """Average belief that the learner has mastered the sign (0 to 1)."""
        return self.alpha[sign] / (self.alpha[sign] + self.beta[sign])

    def update(self, sign, success):
        """Update the belief after one attempt."""
        if success:
            self.alpha[sign] += 1.0
        else:
            self.beta[sign] += 1.0

    def next_sign(self, avoid=None):
        """Thompson sampling: pick the sign whose random mastery draw is lowest."""
        draws = {c: self.rng.beta(self.alpha[c], self.beta[c]) for c in self.classes
                 if c != avoid or len(self.classes) == 1}
        return min(draws, key=draws.get)

    def save(self, path):
        """Store the beliefs so progress survives between sessions."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            json.dump({"alpha": self.alpha, "beta": self.beta}, file, indent=1)

    def load(self, path):
        """Load saved beliefs (signs that are not in the file keep their fresh start)."""
        try:
            with open(path) as file:
                saved = json.load(file)
        except FileNotFoundError:
            return self
        except json.JSONDecodeError:
            print(f"Could not read {path}; starting with a fresh progress record.")
            return self
        for sign in self.classes:
            self.alpha[sign] = float(saved["alpha"].get(sign, 1.0))
            self.beta[sign] = float(saved["beta"].get(sign, 1.0))
        return self