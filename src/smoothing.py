"""Kalman filter that smooths the shaky hand landmarks coming from the webcam."""
import numpy as np

AXES = "xyz"


class KalmanSmoother:
    """A constant-velocity Kalman filter that smooths many numbers in parallel.

    Every number (for example the x position of one fingertip) has a hidden state
    [position, velocity]. Each new frame the filter
      1. PREDICTS where the number should be now (old position + velocity), then
      2. UPDATES that guess with the new noisy measurement.
    How much it trusts the measurement is decided by the Kalman gain K, which is
    calculated from the two noise settings:
      process_noise     (Q): how much the true motion may change from frame to frame
      measurement_noise (R): how noisy the camera/tracker readings are
    """

    def __init__(self, process_noise, measurement_noise, dt=1.0):
        self.F = np.array([[1.0, dt], [0.0, 1.0]])           # how the state moves in one frame
        self.H = np.array([[1.0, 0.0]])                       # we only measure the position
        self.Q = process_noise * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                           [dt ** 3 / 2, dt ** 2]])
        self.R = measurement_noise
        self.measurement_noise = measurement_noise
        self.reset()

    def reset(self):
        """Forget everything (call this at the start of a new take)."""
        self.x = None
        self.P = np.eye(2) * self.measurement_noise * 10.0    # at first we are unsure

    def update(self, z):
        """Take one new measurement vector z and return the smoothed vector."""
        z = np.asarray(z, dtype=float)
        if self.x is None:                                    # first frame: trust the measurement
            self.x = np.zeros((z.size, 2))
            self.x[:, 0] = z
            return z.copy()
        # predict
        self.x = self.x @ self.F.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        # update
        S = (self.H @ self.P @ self.H.T)[0, 0] + self.R       # uncertainty of the innovation
        K = (self.P @ self.H.T) / S                           # Kalman gain, shape (2, 1)
        innovation = z - self.x[:, 0]                         # measurement minus prediction
        self.x = self.x + innovation[:, None] * K.T
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[:, 0].copy()


def axis_columns(axis):
    """Column names of one axis for all 21 landmarks, e.g. x0, x1, ... x20."""
    return [f"{axis}{i}" for i in range(21)]


def estimate_noise(df):
    """Estimate the measurement noise R for each axis from the data itself.

    While a hand is held still, the change from one frame to the next is mostly
    tracker noise. Var(difference) / 2 is therefore an estimate of the noise variance.
    """
    keys = ["sign", "session", "take"]
    noise = {}
    for axis in AXES:
        cols = axis_columns(axis)
        diffs = df.groupby(keys)[cols].diff().dropna()
        noise[axis] = float(diffs.to_numpy().var() / 2)
    return noise


def smooth_dataframe(df, noise, process_ratio):
    """Return a copy of df whose landmark columns are Kalman-smoothed, take by take.

    process_ratio sets Q = process_ratio * R. Small values smooth strongly (but lag
    behind real movement); large values follow the raw measurements closely.
    """
    out = df.copy()
    smoothers = {axis: KalmanSmoother(process_ratio * noise[axis], noise[axis]) for axis in AXES}
    for _, take in df.groupby(["sign", "session", "take"], sort=False):
        take = take.sort_values("frame")
        for axis in AXES:
            cols = axis_columns(axis)
            smoothers[axis].reset()
            smoothed = np.vstack([smoothers[axis].update(row) for row in take[cols].to_numpy(dtype=float)])
            out.loc[take.index, cols] = smoothed
    return out