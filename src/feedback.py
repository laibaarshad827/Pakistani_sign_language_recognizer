"""Feedback engine: compares an attempt with the typical hand shape of a sign and names the wrong finger."""
import numpy as np

from features import FINGER_JOINTS, angle_columns

ANGLES = angle_columns()                 # thumb_a1..a3, index_a1..a3, ... (finger by finger)
FINGERS = list(FINGER_JOINTS)            # thumb, index, middle, ring, pinky


class FeedbackEngine:
    """Learns the typical joint angles of every sign, then judges new attempts.

    For each sign we store the mean and the standard deviation (spread) of the 15 finger
    angles. For a new attempt we measure how many standard deviations each angle is away
    from the typical value (a z-score), average the three joints of every finger, and
    report the finger that is furthest away.
    """

    def __init__(self, min_std=5.0, flag_threshold=4.0, worst_scale=6.0):
        self.min_std = min_std                # spread is never smaller than 5 degrees
        self.flag_threshold = flag_threshold  # a finger further than this is "wrong"
        self.worst_scale = worst_scale        # deviation at which the score reaches 0
        self.mean = None
        self.std = None

    def fit(self, feats, labels):
        """Learn mean and spread of each angle for each sign (labels = sign names)."""
        table = feats[ANGLES].copy()
        table["label"] = np.asarray(labels)
        grouped = table.groupby("label")
        self.mean = grouped.mean()
        self.std = grouped.std().clip(lower=self.min_std)
        return self

    def finger_deviation(self, labels, angle_matrix):
        """Return (size, direction) of the deviation of each finger, both shaped (attempts, 5).

        size      = average |z| over the finger's three joints (0 = perfect)
        direction = average z: positive = straighter than typical, negative = more bent
        """
        mean = self.mean.loc[list(labels), ANGLES].to_numpy()
        std = self.std.loc[list(labels), ANGLES].to_numpy()
        z = ((np.asarray(angle_matrix, dtype=float) - mean) / std).reshape(-1, 5, 3)
        return np.abs(z).mean(axis=2), z.mean(axis=2)

    def score(self, size):
        """Score from 0 to 100, decided by the worst finger."""
        worst = size.max(axis=1)
        return 100.0 * np.clip(1.0 - worst / self.worst_scale, 0.0, 1.0)

    def explain(self, label, angles):
        """Judge one attempt of sign `label`; return the score and a message for the learner."""
        size, direction = self.finger_deviation([label], np.asarray(angles).reshape(1, -1))
        worst = int(size[0].argmax())
        score = float(self.score(size)[0])
        if size[0, worst] <= self.flag_threshold:
            return {"score": score, "finger": None, "message": "Great, that matches the sign!"}
        advice = "straighten" if direction[0, worst] < 0 else "bend a little more"
        finger = FINGERS[worst]
        return {"score": score, "finger": finger, "message": f"Your {finger} finger: {advice}."}