import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# The "bones" of the hand: pairs of landmark numbers to join with a line
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # edge of the palm
]

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

start = time.time()
with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)                      # mirror image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # model wants RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]             # the 21 landmarks
            h, w = frame.shape[:2]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for a, b in CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
            for p in points:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)

            wrist = hand[0]
            print(f"wrist: x={wrist.x:.3f} y={wrist.y:.3f} z={wrist.z:.3f}")

        cv2.imshow("hello hand", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()