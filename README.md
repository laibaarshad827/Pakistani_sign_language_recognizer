# PSL Trainer: Pakistan Sign Language Alphabet Recognizer with Feedback

AL-2002 Artificial Intelligence Lab, Fall 2026 · FAST-NUCES, Karachi · Project idea #17

PSL Trainer is a webcam-based learning tool for Pakistan Sign Language (PSL). It reads the learner's hand with MediaPipe landmarks, recognizes which of 10–15 PSL alphabet signs is being shown, compares it with the sign the learner was asked to make, and gives a score with feedback naming the finger that is wrong. A Kalman filter steadies the noisy landmarks, three classifiers (KNN, SVM, Random Forest) are compared against a simple baseline, and a Bayesian mastery tracker chooses which sign to practise next.

## Author

| Name | Roll number | What I built |
|---|---|---|
| [Your full name] | [24K-xxxx] | Everything: data collection, features, classifiers, Kalman filter, feedback engine, learner model, experiments and report (individual project) |

## Features

- Self-recorded dataset collector for labelled hand-landmark samples
- Landmark normalization and finger joint-angle features
- KNN, SVM and Random Forest compared on the same session-based split, plus a nearest-template baseline
- Kalman filter smoothing of landmarks (own implementation)
- Real-time webcam recognition with on-screen label, score and feedback message
- Finger-level feedback, e.g. "index finger needs to be straighter"
- Bayesian (Beta) mastery tracking per sign to choose the next sign to practise
- One command to reproduce all experiments with a fixed seed

## Project structure

```
psl-trainer/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                 proposal and report, each as PDF and MD
├── src/
│   ├── main.py           runs the trainer (webcam session)
│   ├── collect_data.py   records labelled samples to CSV
│   ├── features.py       HandFeatureExtractor: landmarks, normalization, angles
│   ├── smoothing.py      KalmanSmoother
│   ├── classifiers.py    BaseClassifier, KNNModel, SVMModel, RandomForestModel, TemplateBaseline
│   ├── feedback.py       FeedbackEngine
│   ├── learner.py        LearnerModel (Beta mastery tracking)
│   ├── session.py        TrainerSession (camera loop)
│   └── experiments.py    reproduces every table and chart
├── data/
│   ├── raw/              recorded sessions
│   └── processed/        cached landmark CSVs (used offline)
└── outputs/
    ├── figures/          charts
    └── results/          result tables and logs
```

## Setup

Requires Python 3.10–3.12 and a working webcam.

```bash
git clone https://github.com/<your-username>/psl-trainer.git
cd psl-trainer

python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate

pip install -r requirements.txt
```

## Run the project

```bash
python src/main.py
```

A window opens showing your hand, the target sign, the recognized sign, a score and feedback. Press `q` to quit. If the wrong camera opens, run `python src/main.py --camera 1`.

## Reproduce the experiments

```bash
python src/experiments.py --seed 42
```

This trains and compares all models on the held-out session split, runs the raw vs Kalman-smoothed experiment and the feedback validation test, and writes the results table to `outputs/results/` and all charts to `outputs/figures/`. It reads the cached CSVs in `data/processed/`, so no webcam or internet connection is needed.

## Collect your own data (optional)

```bash
python src/collect_data.py --sign A --samples 200 --session s1
```

Records labelled landmark samples for one sign into `data/raw/`. Record on different days, lighting and distances, and use a new `--session` name each time so the test split stays honest.

## Data

- **Source:** self-recorded webcam sessions; only landmark coordinates are stored, no face images
- **Signs:** 10–15 PSL alphabet signs, checked against [name and link of the PSL reference used]
- **Size:** roughly 2,000–3,000 rows of landmark features
- **Offline:** all data is cached locally in `data/processed/`

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (nearest template, raw) | [fill in] | [fill in] |
| KNN | [fill in] | [fill in] |
| SVM | [fill in] | [fill in] |
| Random Forest | [fill in] | [fill in] |

Full tables and charts are in `outputs/` and explained in the report in `docs/`.

## Known limitations

- Static signs only; dynamic signs, full words and sentences are not supported
- Trained on a small, self-collected dataset from a limited number of signers, so accuracy may drop for new people, hand sizes, lighting or camera angles
- Similar signs can be confused; confused pairs are reported in the results
- Needs one clearly visible hand in front of the webcam; occlusion lowers accuracy
- This is a practice aid, not a replacement for a PSL teacher

## Ethical considerations

Reference signs must be correct, because a trainer that teaches wrong signs causes harm. Signs were taken from [PSL reference] and [checked by ...]. Volunteers gave consent for their recordings, and no face images or personal data are stored.

## Declarations

- **Pre-trained tools:** MediaPipe hand landmarks are used only for feature extraction. No deep-learning framework is used.
- **Libraries:** OpenCV, MediaPipe, scikit-learn, NumPy, pandas, Matplotlib
- **Own code:** the Kalman filter, feedback engine, Bayesian mastery tracker and baseline are written by me; scikit-learn provides KNN, SVM and Random Forest
- **AI assistance:** Claude (Anthropic) was used for planning and for drafting the proposal and this README. [List any other use.]
