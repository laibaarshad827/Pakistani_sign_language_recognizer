"""Live trainer logic: smoothing, recognition, feedback and progress (no camera code in here)."""
from collections import deque

import numpy as np
import pandas as pd

from features import LANDMARK_COLS, META_COLS, angle_columns, build_features, coord_columns
from smoothing import AXES, KalmanSmoother

FEATURE_COLUMNS = coord_columns() + angle_columns()


class LivePipeline:
    """Turns one frame of 63 landmark numbers into smoothed features, frame after frame."""

    def __init__(self, noise, process_ratio=0.01, history=5):
        self.smoothers = {axis: KalmanSmoother(process_ratio * noise[axis], noise[axis]) for axis in AXES}
        self.recent_angles = deque(maxlen=history)

    def reset(self):
        """Forget the past (call when the hand disappears or a new round starts)."""
        for smoother in self.smoothers.values():
            smoother.reset()
        self.recent_angles.clear()

    def process(self, landmarks):
        """Return (feature vector for the classifier, finger angles averaged over recent frames)."""
        values = np.asarray(landmarks, dtype=float)
        smoothed = np.empty_like(values)
        for offset, axis in enumerate(AXES):
            smoothed[offset::3] = self.smoothers[axis].update(values[offset::3])
        row = pd.DataFrame([smoothed], columns=LANDMARK_COLS)
        for column, value in zip(META_COLS, ("live", "live", 0, 0)):
            row[column] = value
        feats = build_features(row)
        vector = feats[FEATURE_COLUMNS].to_numpy(dtype=float)[0]
        self.recent_angles.append(feats[angle_columns()].to_numpy(dtype=float)[0])
        return vector, np.mean(self.recent_angles, axis=0)


class TrainerSession:
    """One practice session: show a target sign, watch the learner, give feedback, move on."""

    def __init__(self, model, engine, learner, pipeline, rounds=10, hold_frames=15,
                 round_seconds=15.0, result_seconds=3.0):
        self.model, self.engine, self.learner, self.pipeline = model, engine, learner, pipeline
        self.rounds, self.hold_frames = rounds, hold_frames
        self.round_seconds, self.result_seconds = round_seconds, result_seconds
        self.state = "waiting"          # waiting -> playing -> result -> ... -> finished
        self.round = 0
        self.target = None
        self.predicted = None
        self.score = 0.0
        self.message = ""
        self.hold = 0
        self.last_success = None
        self.history = []               # (target, success) for every finished round
        self.round_start = 0.0
        self.result_start = 0.0
        self.summary = None

    def build_summary(self):
        """Collect what the end-of-session screen needs: results per sign and the weakest signs overall."""
        tries, wins = {}, {}
        for sign, success in self.history:
            tries[sign] = tries.get(sign, 0) + 1
            wins[sign] = wins.get(sign, 0) + int(success)
        rows = sorted(((sign, wins[sign], tries[sign], self.learner.mastery(sign)) for sign in tries),
                      key=lambda row: row[3])
        weakest = sorted(self.learner.classes, key=self.learner.mastery)[:5]
        return {"correct": sum(wins.values()), "rounds": len(self.history), "rows": rows,
                "weakest": [(sign, self.learner.mastery(sign)) for sign in weakest]}

    def start_round(self, now):
        """Pick the next sign (the weakest one) and start the clock."""
        previous = self.target
        self.target = self.learner.next_sign(avoid=previous)
        self.round += 1
        self.state, self.round_start = "playing", now
        self.predicted, self.score, self.hold = None, 0.0, 0
        self.message = "Make the sign shown on the screen"
        self.pipeline.reset()

    def finish_round(self, success, now):
        """Record the result and update the learner's progress."""
        self.learner.update(self.target, success)
        self.history.append((self.target, success))
        self.last_success = success
        self.state, self.result_start = "result", now
        self.message = "Correct!" if success else f"Not this time. {self.message}"

    def step(self, landmarks, now):
        """Advance by one camera frame (landmarks = 63 numbers, or None if no hand) and return the info to show."""
        if self.state == "waiting":
            self.start_round(now)
        elif self.state == "result":
            if now - self.result_start >= self.result_seconds:
                if self.round >= self.rounds:
                    self.state = "finished"
                    self.summary = self.build_summary()
                else:
                    self.start_round(now)
        elif self.state == "playing":
            if landmarks is None:
                self.pipeline.reset()
                self.hold, self.predicted = 0, None
                self.message = "Show your right hand to the camera"
            else:
                vector, angles = self.pipeline.process(landmarks)
                self.predicted = self.model.predict(vector.reshape(1, -1))[0]
                report = self.engine.explain(self.target, angles)
                self.score, self.message = report["score"], report["message"]
                self.hold = self.hold + 1 if self.predicted == self.target else 0
                if self.hold >= self.hold_frames:
                    self.finish_round(True, now)
            if self.state == "playing" and now - self.round_start >= self.round_seconds:
                self.finish_round(False, now)
        return self.info(now)

    def info(self, now):
        """Everything the screen needs to know."""
        left = max(0.0, self.round_seconds - (now - self.round_start)) if self.state == "playing" else 0.0
        return {"state": self.state, "target": self.target, "round": self.round, "rounds": self.rounds,
                "predicted": self.predicted, "score": self.score, "message": self.message,
                "hold_fraction": min(1.0, self.hold / self.hold_frames), "time_left": left,
                "time_fraction": left / self.round_seconds, "last_success": self.last_success,
                "results": [success for _, success in self.history], "summary": self.summary}