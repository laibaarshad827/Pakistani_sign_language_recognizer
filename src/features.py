"""Turn raw landmark rows into normalized features (coordinates and finger angles)."""
import numpy as np
import pandas as pd

LANDMARK_COLS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
META_COLS = ["sign", "session", "take", "frame"]

# The camera frame is wider than it is tall, so MediaPipe's y values are
# "fractions of the height" and x values "fractions of the width".
# Multiplying y by height/width puts both axes in the same units.
# CHECK your camera size (see instructions) and change this if it is not 640x480.
ASPECT = 480 / 640

# For every finger: three joints, each given as (a, b, c) = the angle at b
# between the bones b->a and b->c.
FINGER_JOINTS = {
    "thumb":  [(0, 1, 2), (1, 2, 3), (2, 3, 4)],
    "index":  [(0, 5, 6), (5, 6, 7), (6, 7, 8)],
    "middle": [(0, 9, 10), (9, 10, 11), (10, 11, 12)],
    "ring":   [(0, 13, 14), (13, 14, 15), (14, 15, 16)],
    "pinky":  [(0, 17, 18), (17, 18, 19), (18, 19, 20)],
}


def to_points(df):
    """Return an array of shape (rows, 21, 3) from the raw landmark columns."""
    return df[LANDMARK_COLS].to_numpy(dtype=float).reshape(-1, 21, 3)

KNUCKLES = (5, 9, 13, 17)
def normalize(points, aspect=ASPECT):
    """Make the hand independent of its position and size.

    1. y is rescaled so x and y have the same units.
    2. The wrist (landmark 0) is moved to the origin.
    3. Everything is divided by the hand size: the average 3D distance from the
       wrist to the four finger-base knuckles. Using 3D (x, y and depth z) keeps
       the size steady when the hand points towards the camera.
    """
    pts = points.copy()
    pts[:, :, 1] *= aspect
    pts = pts - pts[:, 0:1, :]
    scale = np.linalg.norm(pts[:, KNUCKLES, :], axis=2).mean(axis=1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return pts / scale[:, None, None]


def joint_angle(pts, a, b, c):
    """Angle in degrees at landmark b between the bones b->a and b->c."""
    v1 = pts[:, a, :] - pts[:, b, :]
    v2 = pts[:, c, :] - pts[:, b, :]
    cos = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angle_columns():
    """Names of the 15 angle features, e.g. index_a1, index_a2, index_a3."""
    return [f"{finger}_a{k + 1}" for finger in FINGER_JOINTS for k in range(3)]


def coord_columns():
    """Names of the 60 normalized coordinate features (landmarks 1-20; the wrist is always 0)."""
    return [f"n{axis}{i}" for i in range(1, 21) for axis in "xyz"]


def build_features(df):
    """Return a DataFrame: meta columns + 60 normalized coordinates + 15 angles."""
    pts = normalize(to_points(df))
    coord_names = coord_columns()
    coords = pd.DataFrame(pts[:, 1:, :].reshape(len(df), -1), columns=coord_names, index=df.index)

    angles = {}
    for finger, joints in FINGER_JOINTS.items():
        for k, (a, b, c) in enumerate(joints):
            angles[f"{finger}_a{k + 1}"] = joint_angle(pts, a, b, c)
    angles = pd.DataFrame(angles, index=df.index)

    return pd.concat([df[META_COLS], coords, angles], axis=1)