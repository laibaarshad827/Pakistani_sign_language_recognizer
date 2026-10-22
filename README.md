# PSL Trainer

A webcam-based tool for learning Pakistan Sign Language (PSL). It recognizes PSL alphabet signs from your hand, checks your attempt against the sign you were asked to make, and tells you which finger to fix. It also keeps track of the signs you find hard and gives you those more often.

> Work in progress. Setup and usage instructions will be added once the project is done.

## What it does

- Reads hand landmarks from the webcam and recognizes 10-15 PSL alphabet signs
- Scores each attempt and gives feedback, for example "index finger needs to be straighter"
- Smooths shaky hand tracking with a Kalman filter
- Compares three classifiers (KNN, SVM, Random Forest) against a simple baseline
- Tracks how well each sign is learned and picks the next sign to practise

## Built with

Python, OpenCV, MediaPipe (hand landmarks, used only for feature extraction), scikit-learn, NumPy, Matplotlib

## Data

Self-recorded webcam sessions. Only hand-landmark coordinates are stored, no face images. Signs are checked against [PSL reference].

## Limitations

- Static signs only, no words or sentences yet
- Small self-collected dataset, so accuracy may drop for other people, lighting or camera angles
- Similar signs can be confused
- Needs one clearly visible hand in front of the camera
- A practice aid, not a replacement for a PSL teacher****
