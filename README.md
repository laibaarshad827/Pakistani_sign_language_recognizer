# PSL Trainer: Pakistan Sign Language Alphabet Recognizer with Feedback

PSL Trainer is a webcam-based learning tool for Pakistan Sign Language (PSL). It reads the learner's hand with MediaPipe landmarks, recognizes which of 34 PSL alphabet signs is being shown, compares the attempt with the sign the learner was asked to make, and names the finger that needs fixing. A Kalman filter steadies the noisy hand tracking, an SVM (compared with KNN, Random Forest and a template baseline) does the recognition, and a Bayesian progress tracker decides which sign to practise next. The live screen shows the target sign, a score, feedback, and at the end of a session how much practice each sign still needs.

## Author
Laiba 

## Features

- Recorder that saves labelled hand landmarks of 34 PSL signs (3 recording sessions, 6,120 frames)
- Features that do not depend on hand position or size: wrist-relative coordinates scaled by 3D palm size, plus 15 finger-joint angles
- Model comparison with honest testing on a recording session the model never saw: template baseline, KNN, SVM, Random Forest
- Kalman filter (own code) that smooths the landmarks and reduces label flicker
- Feedback engine (own code) that scores an attempt and says which finger to straighten or bend
- Progress tracker (own code): a Beta belief per sign, Thompson sampling to choose the next sign
- Live trainer window with reference pictures, confetti on success, and an end-of-session summary
- One command that reproduces every table and chart with a fixed seed (42)

## Setup

Requires Python 3.11 and a webcam (tested on Windows). From the project folder:

```bash
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Run the project

```bash
python src/trainer.py --merge haa,hamza
```

A window opens. Make the sign shown, hold it, and read the feedback. Press `q` to quit. Useful options: `--rounds 3` (fewer signs per session), `--seconds 25` (more time per sign), `--camera 1` (another camera).

The trained model (`models/psl_svm.joblib`) and the recorded data are included, so this works straight after setup.

## Reproduce the experiments

```bash
python src/run_all.py
```

Takes about two minutes. It runs data checks, the model comparison (all signs, and with the look-alike pair haa/hamza merged), the Kalman experiment, the feedback experiments and retrains the final model. Tables go to `outputs/results/`, charts to `outputs/figures/`, and everything printed is saved in `outputs/results/run_log.txt`.

## Results (tested on unseen recording sessions)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (nearest template, raw coordinates) | 0.528 | 0.480 |
| SVM on raw coordinates | 0.717 | 0.690 |
| KNN | 0.833 | 0.804 |
| Random Forest | 0.821 | 0.788 |
| **SVM** | **0.869** | **0.839** |
| SVM, haa and hamza merged | 0.900 | 0.869 |

- Kalman filter (Q/R = 0.01): label flicker falls from 0.61 to 0.27 changes per second; accuracy is unchanged.
- Feedback engine: 9.4% of correct attempts are wrongly flagged; a deliberate 45-degree finger mistake is named correctly 89% of the time.

## Project structure

```
src/        all code (trainer.py, run_all.py, collect_data.py, features.py, smoothing.py, classifiers.py,
            train_eval.py, train_final.py, kalman_experiment.py, feedback*.py, learner.py, session.py, ui.py, ...)
data/       raw/landmarks.csv, processed/features.csv, reference/ (pictures of the hand for each sign)
models/     hand_landmarker.task (MediaPipe), psl_svm.joblib (trained classifier)
outputs/    figures/ and results/
docs/       proposal and report (PDF and Markdown)
```

Other commands: `python src/collect_data.py --session s4` records a new session; `python src/make_references.py` saves hand pictures for the trainer.

## Data

Self-recorded webcam sessions; only hand-landmark coordinates are stored, no face images. The signs were checked against the PSL Dictionary of the Family Educational Services Foundation (https://psl.org.pk) and the alphabet chart in: Urdu alphabet dataset for Pakistan Sign Language (UAlpha40), https://pmc.ncbi.nlm.nih.gov/articles/PMC11848795/. The program works offline.

## Known limitations

- Static hand shapes only; signs that need movement, words and sentences are not supported
- Small dataset (3 sessions), so accuracy may drop for other people, lighting or cameras
- Some signs look identical as a still hand shape (haa and hamza) and cannot be told apart; others are often confused (for example zuey and zuad)
- The feedback engine was validated with deliberate mistakes created by computer, not with real wrong hands
- Needs one clearly visible right hand; this is a practice aid, not a replacement for a PSL teacher

## Declarations

- **Pre-trained tool:** MediaPipe hand landmarks are used only for feature extraction. No deep-learning framework is used.
- **Own code:** Kalman filter, feedback engine, progress tracker, baseline classifier, live trainer. **Libraries:** scikit-learn (KNN, SVM, Random Forest), OpenCV, NumPy, pandas, Matplotlib.