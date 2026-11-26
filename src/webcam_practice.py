
"""Account-independent webcam practice with the original PSL trainer UI."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collect_data import draw_hand, MODEL_PATH as HAND_MODEL_PATH
from trainer import load_everything
from session import LivePipeline
from features import FINGER_JOINTS
from ui import Celebration, load_reference, render


def draw_finger_panel(canvas, fingers, x):
    """Display individual finger corrections below the original trainer panel."""

    y = canvas.shape[0] - 140

    cv2.putText(
        canvas,
        "FINGER-BY-FINGER FEEDBACK",
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.48,
        (180, 205, 0),
        1,
        cv2.LINE_AA
    )

    for index, item in enumerate(fingers):
        row_y = y + 27 + index * 23

        color = (
            (110, 210, 70)
            if item["correct"]
            else (0, 150, 255)
        )

        status = (
            "CORRECT"
            if item["correct"]
            else item["advice"].upper()
        )

        cv2.putText(
            canvas,
            f"{item['finger'].capitalize()}: {status}",
            (x, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA
        )


def run_practice(sign, duration=15, hold_frames=15):

    if duration <= 0 or hold_frames <= 0:
        raise ValueError("Duration and hold frames must be positive.")

    model, classes, engine, noise = load_everything([])

    if sign not in classes:
        raise ValueError(
            f"Sign '{sign}' is not supported by the trained model."
        )

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(ROOT / HAND_MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    pipeline = LivePipeline(noise)
    celebration = Celebration()
    reference = load_reference(sign)

    start = time.monotonic()
    hold = 0
    predicted = None
    score = 0.0
    message = "Show your hand to the camera."

    success = False
    completed = False
    state = "playing"
    result_started = None

    finger_feedback = [
        {
            "finger": finger,
            "correct": False,
            "advice": "waiting"
        }
        for finger in FINGER_JOINTS
    ]

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:

            while True:
                ok, frame = camera.read()

                if not ok:
                    raise RuntimeError(
                        "Could not read a frame from the webcam."
                    )

                frame = cv2.flip(frame, 1)

                elapsed = time.monotonic() - start
                remaining = max(0, duration - elapsed)

                if state == "playing":

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    result = landmarker.detect_for_video(
                        mp_image,
                        int(elapsed * 1000)
                    )

                    hand = (
                        result.hand_landmarks[0]
                        if result.hand_landmarks
                        else None
                    )

                    if hand is not None:
                        draw_hand(frame, hand)

                        landmarks = [
                            value
                            for point in hand
                            for value in (point.x, point.y, point.z)
                        ]

                        vector, angles = pipeline.process(landmarks)

                        predicted = str(
                            model.predict(
                                vector.reshape(1, -1)
                            )[0]
                        )

                        # Compare all five fingers against the target sign.
                        angle_array = np.asarray(angles).reshape(1, -1)

                        deviations, directions = engine.finger_deviation(
                            [sign],
                            angle_array
                        )

                        score = float(engine.score(deviations)[0])

                        finger_feedback = []

                        for index, finger in enumerate(FINGER_JOINTS):
                            deviation = float(deviations[0, index])
                            direction = float(directions[0, index])

                            correct = (
                                deviation <= engine.flag_threshold
                            )

                            if correct:
                                advice = "correct position"
                            elif direction < 0:
                                advice = "straighten"
                            else:
                                advice = "bend a little more"

                            finger_feedback.append({
                                "finger": finger,
                                "correct": correct,
                                "advice": advice
                            })

                        incorrect = [
                            item for item in finger_feedback
                            if not item["correct"]
                        ]

                        if predicted == sign:
                            hold += 1

                            if incorrect:
                                message = (
                                    "Sign detected! Adjust the highlighted "
                                    "fingers for a better hand shape."
                                )
                            else:
                                message = "Correct sign! Keep holding."
                        else:
                            hold = 0
                            message = (
                                f"Detected: {predicted}. "
                                f"Try making {sign}."
                            )

                        if hold >= hold_frames:
                            success = True
                            completed = True
                            state = "result"
                            message = "Excellent! Sign completed!"
                            result_started = time.monotonic()

                    else:
                        pipeline.reset()
                        hold = 0
                        predicted = None
                        score = 0.0
                        message = "Show your hand to the camera."

                        finger_feedback = [
                            {
                                "finger": finger,
                                "correct": False,
                                "advice": "waiting"
                            }
                            for finger in FINGER_JOINTS
                        ]

                    if remaining <= 0 and state == "playing":
                        completed = True
                        success = False
                        state = "result"
                        message = "Time is up. Keep practising!"
                        result_started = time.monotonic()

                info = {
                    "state": state,
                    "rounds": 1,
                    "results": [success] if state == "result" else [],
                    "last_success": success,
                    "target": sign,
                    "predicted": predicted,
                    "score": score,
                    "hold_fraction": min(1.0, hold / hold_frames),
                    "time_fraction": remaining / duration,
                    "message": message
                }

                # Reuse the original colourful PSL trainer interface.
                canvas = render(frame, info, reference)

                # Make room for all five finger instructions.
                canvas = cv2.copyMakeBorder(
                    canvas,
                    0,
                    160,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(52, 34, 26)
                )

                draw_finger_panel(
                    canvas,
                    finger_feedback,
                    frame.shape[1] + 15
                )

                # The original UI's animated confetti and sparkles.
                if success:
                    celebration.draw(
                        canvas,
                        info,
                        time.monotonic()
                    )
                else:
                    celebration.draw(
                        canvas,
                        info,
                        time.monotonic()
                    )

                cv2.imshow("PSL Webcam Practice", canvas)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                # Keep the celebration visible before closing the camera.
                if (
                    state == "result"
                    and result_started is not None
                    and time.monotonic() - result_started >= 2.5
                ):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    finger_lines = [
        (
            f"{item['finger'].capitalize()}: "
            f"{'Correct' if item['correct'] else item['advice']}"
        )
        for item in finger_feedback
    ]

    feedback_text = message

    if finger_lines:
        feedback_text += "\n\nFinger guidance:\n"
        feedback_text += "\n".join(finger_lines)

    return {
        "completed": completed,
        "success": success,
        "sign": sign,
        "predicted": predicted,
        "score": float(score),
        "feedback": feedback_text,
        "finger_feedback": finger_feedback
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sign", required=True)
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--hold-frames", type=int, default=15)

    args = parser.parse_args()

    try:
        result = run_practice(
            args.sign,
            duration=args.duration,
            hold_frames=args.hold_frames
        )

        print("PRACTICE_RESULT:" + json.dumps(result))

    except Exception as error:
        print(f"PRACTICE_ERROR:{error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
