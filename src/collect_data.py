
import argparse
import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
CSV_PATH = "data/raw/landmarks.csv"
FRAMES_PER_TAKE = 30      
COUNTDOWN_SECONDS = 2.0    
SIGNS = ["alif", "bay", "pay","tay","say","chay","hay","khay","dal","ddal","zal","ray","zay","zzay","seen","sheen",
         "suad","zuad","tuey","zuey","ain","ghain","fay","qauf","kaaf","gaaf","laam","meem","noon","wow",
        "haa","hamza","chotiyay","bariyay"]

HEADER = ["sign", "session", "take", "frame"] + [
    f"{axis}{i}" for i in range(21) for axis in "xyz"
]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def load_take_counts(path, session):
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["session"] == session:
                counts[row["sign"]] = max(counts.get(row["sign"], 0), int(row["take"]))
    return counts


def append_rows(path, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        writer.writerows(rows)


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="session name, e.g. s1")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Try VideoCapture(1).")

    takes = load_take_counts(CSV_PATH, args.session)
    index, state, t0, buffer = 0, "idle", 0.0, []
    start = time.time()

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.time() - start) * 1000))
            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            if hand is not None:
                draw_hand(frame, hand)

            sign = SIGNS[index]
            message = f"{sign}  |  session {args.session}  |  takes saved: {takes.get(sign, 0)}"

            if state == "countdown":
                remaining = COUNTDOWN_SECONDS - (time.time() - t0)
                if remaining <= 0:
                    state, buffer = "recording", []
                else:
                    message += f"  |  get ready {remaining:.0f}"
            elif state == "recording":
                if hand is not None:
                    buffer.append([v for lm in hand for v in (lm.x, lm.y, lm.z)])
                message += f"  |  RECORDING {len(buffer)}/{FRAMES_PER_TAKE}"
                if len(buffer) >= FRAMES_PER_TAKE:
                    takes[sign] = takes.get(sign, 0) + 1
                    rows = [[sign, args.session, takes[sign], i] + values
                            for i, values in enumerate(buffer)]
                    try:
                        append_rows(CSV_PATH, rows)
                    except OSError as error:
                        print(f"Could not save the take: {error}")
                        takes[sign] -= 1
                    state = "idle"
            else:
                message += "  |  SPACE = record"

            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
            cv2.imshow("collect data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                index = (index + 1) % len(SIGNS)
            if key == ord("b"):
                index = (index - 1) % len(SIGNS)
            if key == 32 and state == "idle":
                state, t0 = "countdown", time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()