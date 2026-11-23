"""Save a picture of YOUR OWN hand for each sign; the trainer shows it as the reference picture."""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from collect_data import MODEL_PATH, SIGNS, draw_hand

REFERENCE_DIR = "data/reference"
MARGIN = 0.35                      # extra space around the hand, as a share of its size


def crop_around_hand(frame, hand):
    """Return a square picture cropped around the hand landmarks."""
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand]
    ys = [lm.y * h for lm in hand]
    centre_x, centre_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + MARGIN) / 2
    left, right = int(max(0, centre_x - half)), int(min(w, centre_x + half))
    top, bottom = int(max(0, centre_y - half)), int(min(h, centre_y + half))
    return frame[top:bottom, left:right]


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Try VideoCapture(1).")

    index, note, start = 0, "", time.time()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            clean = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((time.time() - start) * 1000))
            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            if hand is not None:
                draw_hand(frame, hand)

            sign = SIGNS[index]
            done = os.path.exists(os.path.join(REFERENCE_DIR, f"{sign}.png"))
            text = f"{sign} ({'saved' if done else 'missing'})  SPACE=save  n/b=next/back  q=quit  {note}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.imshow("reference pictures", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                index, note = (index + 1) % len(SIGNS), ""
            if key == ord("b"):
                index, note = (index - 1) % len(SIGNS), ""
            if key == 32:
                if hand is None:
                    note = "no hand found"
                else:
                    cv2.imwrite(os.path.join(REFERENCE_DIR, f"{sign}.png"), crop_around_hand(clean, hand))
                    note = "saved!"
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()