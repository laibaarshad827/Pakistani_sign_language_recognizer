"""The live PSL trainer: shows a sign, watches your hand, gives feedback and picks what to practise next."""
import argparse
import os
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from collect_data import CONNECTIONS, MODEL_PATH as HAND_MODEL_PATH, draw_hand
from feedback import FeedbackEngine
from learner import LearnerModel
from session import LivePipeline, TrainerSession
from smoothing import estimate_noise
from train_eval import FEATURES_PATH, RAW_PATH, merge_labels
from train_final import MODEL_PATH as CLASSIFIER_PATH
from ui import Celebration, load_reference, render

PROGRESS_PATH = "data/progress.json"


def load_everything(merge):
    """Load the saved classifier and the data needed by the filter and the feedback engine."""
    try:
        bundle = joblib.load(CLASSIFIER_PATH)
        raw = pd.read_csv(RAW_PATH)
        feats = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError as error:
        raise SystemExit(f"Missing file: {error.filename}. Run eda.py and train_final.py first.")
    labels = merge_labels(feats["sign"].to_numpy(), merge)
    engine = FeedbackEngine().fit(feats, labels)
    return bundle["model"], bundle["classes"], engine, estimate_noise(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="append", default=[], help="same merges as train_final.py")
    parser.add_argument("--rounds", type=int, default=10, help="signs to practise in one session")
    parser.add_argument("--seconds", type=float, default=15.0, help="time allowed for each sign")
    parser.add_argument("--hold", type=int, default=15, help="frames you must hold the right sign (30 = 1 second)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model, classes, engine, noise = load_everything(args.merge)
    learner = LearnerModel(classes).load(PROGRESS_PATH)
    session = TrainerSession(model, engine, learner, LivePipeline(noise), rounds=args.rounds,
                             hold_frames=args.hold, round_seconds=args.seconds)
    celebration = Celebration()
    references = {c: load_reference(c) for c in classes}

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Try --camera 1.")

    start = time.time()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time() - start
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now * 1000))
            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            landmarks = None
            if hand is not None:
                draw_hand(frame, hand)
                landmarks = [v for lm in hand for v in (lm.x, lm.y, lm.z)]

            info = session.step(landmarks, now)
            screen = render(frame, info, references.get(info["target"]))
            celebration.draw(screen, info, now)
            cv2.imshow("PSL Trainer", screen)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()

    learner.save(PROGRESS_PATH)
    done = len(session.history)
    won = sum(1 for _, success in session.history if success)
    print(f"\nSession: {won} of {done} rounds correct.")
    print("Weakest signs so far (mastery 0-1):")
    for sign in sorted(classes, key=learner.mastery)[:5]:
        print(f"  {sign:10s} {learner.mastery(sign):.2f}")


if __name__ == "__main__":
    main()