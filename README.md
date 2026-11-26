# 🤟 PSL Learning Platform
### An AI-Powered Personalized Pakistani Sign Language Learning System

A webcam-based Pakistani Sign Language (PSL) learning platform that combines real-time hand-sign recognition, intelligent feedback, and adaptive learning to help users learn and practise the PSL alphabet.

The platform recognizes **34 PSL alphabet signs** using MediaPipe hand landmarks and a trained Support Vector Machine (SVM) classifier. It provides a structured learning hierarchy, personalized practice recommendations, finger-level feedback, and progress tracking for individual learners.

## ✨ Key Features

### 1. User Authentication and Profiles
- Create a personal learner account and log in securely.
- Passwords are stored as salted PBKDF2-SHA256 hashes.
- Maintain separate learning progress and practice history for each user.
- Store learner information and progress in a local SQLite database.

### 2. Structured Learning Hierarchy
- Learn PSL signs through three progressive levels:
  - **Beginner**
  - **Intermediate**
  - **Advanced**
- View reference images for individual signs.
- Track mastery, attempts, and learning status for each sign.
- Unlock subsequent levels after meeting the required mastery and practice criteria.

### 3. Real-Time Webcam Recognition
- Recognize PSL alphabet signs through a webcam.
- Detect and track hand landmarks using MediaPipe.
- Extract normalized hand coordinates and finger-joint angles.
- Apply Kalman filtering to smooth noisy landmark measurements.
- Compare the predicted sign against the target sign.
- Require the learner to hold the correct sign for a specified number of frames before marking it successful.

### 4. Intelligent Finger-Level Feedback
- Compare the learner's hand configuration against learned reference patterns.
- Evaluate the five fingers individually:
  - Thumb
  - Index
  - Middle
  - Ring
  - Pinky
- Identify incorrect finger positions and provide corrective instructions.
- Calculate a feedback score from 0 to 100.
- Display a celebratory animation when the learner successfully completes a sign.

### 5. Personalized Adaptive Learning
The platform adapts its recommendations according to each learner's performance.

- **Bayesian Learning:** Maintain a Beta distribution for each sign to estimate the learner's mastery.
- **Thompson Sampling:** Balance practice of weaker signs with exploration of other signs.
- **Q-Learning:** Learn which sign to recommend based on the learner's current level, mastery, and recent success rate.
- **Adaptive Recommendations:** Select the next sign using the learner's recorded performance rather than following a fixed practice sequence.

### 6. Learning Dashboard and Analytics
Each learner receives a personalized dashboard containing:

- Total number of signs.
- Signs attempted and mastered.
- Overall mastery percentage.
- Learning-level progress and unlocking status.
- Next recommended sign.
- Recent practice success rate.
- Practice history and performance trends.
- Learned Q-values from the adaptive recommendation system.

### 7. Additional AI and Algorithmic Components
The repository also contains implementations of:

- Genetic Algorithm for SVM feature selection and hyperparameter optimization.
- K-Means clustering for identifying similar PSL signs.
- Temporal analysis of hand-landmark recordings.
- Constraint Satisfaction Problem (CSP) with backtracking for practice-sequence generation.
- Breadth-First Search (BFS), Depth-First Search (DFS), and A* for learning-path navigation.
- Minimax with Alpha-Beta pruning for a Tic-Tac-Toe game.

---

## 🧠 Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| User Interface | Streamlit |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | Scikit-learn, SVM |
| Numerical Processing | NumPy, pandas, SciPy |
| Data Visualization | Matplotlib, Streamlit charts |
| Model Persistence | Joblib |
| Database | SQLite |
| Authentication | PBKDF2-SHA256 password hashing |
| Version Control | Git and GitHub |

## 🔬 Machine Learning Pipeline

The recognition pipeline transforms webcam input into numerical features that can be processed by the trained classifier.

1. **Hand Detection:** MediaPipe detects 21 landmarks on the hand.
2. **Landmark Extraction:** Extract the x, y, and z coordinates of each landmark.
3. **Normalization:** Adjust coordinates to reduce the effect of hand position, orientation, and size.
4. **Feature Engineering:** Generate 60 normalized coordinate features and 15 finger-joint angle features.
5. **Feature Smoothing:** Apply a Kalman filter to reduce tracking noise.
6. **Classification:** Use the trained SVM model to predict the PSL sign.
7. **Feedback Generation:** Compare the learner's finger angles with the learned patterns for the target sign.
8. **Progress Update:** Record the attempt and update the learner's personalized learning model.

**Final feature representation:** 75 features per hand sample.

## 📊 Experimental Evaluation

A separate Phase 4 experiment evaluates the baseline SVM against a Genetic Algorithm-optimized SVM.

### Dataset

| Property | Value |
|---|---:|
| Total samples | 6,120 |
| Number of PSL signs | 34 |
| Total features | 75 |
| Training samples | 4,080 |
| Testing samples | 2,040 |
| Training sessions | s1 and s2 |
| Held-out test session | s3 |

The final test set uses a separate recording session that is not included in training.

### Model Comparison

| Metric | Baseline SVM | Genetic Algorithm SVM |
|---|---:|---:|
| Accuracy | **93.33%** | 91.52% |
| Macro F1-score | **91.71%** | 90.19% |
| Weighted F1-score | **91.71%** | 90.19% |
| Features selected | 75 | 35 |

The Genetic Algorithm selected 35 of the original 75 features and optimized the SVM hyperparameters.

**Observation:** The baseline SVM achieved higher test accuracy in this experiment, while the optimized model used fewer features. Feature reduction therefore did not translate into better test performance in this particular run.

### Additional Experimental Analysis

The Phase 4 experiment also includes:

- Classification reports.
- Confusion matrices.
- Genetic Algorithm optimization history.
- Selected feature analysis.
- Similar-sign identification.
- K-Means sign clusters.
- Temporal recording analysis.
- Individual test predictions.

All generated experiment outputs are stored in `reports/phase4/`.

## 📁 Project Structure

```text
Pakistani_sign_language_recognizer/
│
├── app.py
├── requirements.txt
├── diagnostic.py
├── test_streamlit.py
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── landmarks.csv
│   ├── processed/
│   │   └── features.csv
│   └── reference/
│       └── PSL sign reference images
│
├── models/
│   ├── hand_landmarker.task
│   └── psl_svm.joblib
│
├── reports/
│   └── phase4/
│       ├── experiment_summary.json
│       ├── model_comparison.csv
│       ├── classification reports
│       ├── confusion matrices
│       ├── selected_features.csv
│       └── other experiment outputs
│
├── src/
│   ├── auth.py
│   ├── database.py
│   ├── features.py
│   ├── collect_data.py
│   ├── trainer.py
│   ├── webcam_practice.py
│   ├── feedback.py
│   ├── learner.py
│   ├── hierarchy.py
│   ├── profiles.py
│   ├── recommendations.py
│   ├── q_learning.py
│   ├── clustering.py
│   ├── genetic_algorithm.py
│   ├── temporal_analysis.py
│   ├── csp.py
│   ├── search_algorithms.py
│   └── minimax_game.py
│
└── tests/
    ├── test_auth.py
    ├── test_database.py
    ├── test_learning_agent.py
    ├── test_minimax_game.py
    ├── test_phase4.py
    └── test_search_algorithms.py
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11
- A working webcam
- Windows, Linux, or macOS
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/laibaarshad827/Pakistani_sign_language_recognizer.git
```

Navigate into the project directory:

```bash
cd Pakistani_sign_language_recognizer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install Streamlit, which is required by the application:

```bash
pip install streamlit
```

### 4. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

### 5. Create an Account and Start Learning

1. Open the application in your browser.
2. Select **Create Account**.
3. Enter your name, username, and password.
4. Log in using your new account.
5. Explore your personalized learning dashboard.
6. Select an unlocked learning level.
7. Study a sign's reference image.
8. Select **Practice** to start webcam recognition.
9. Follow the feedback and practise until you successfully complete the sign.

The application creates its SQLite database automatically when it starts. Learner accounts, practice attempts, mastery estimates, and Q-learning values are stored locally.

**Note:** Webcam recognition uses a local OpenCV camera window. Run the application on a machine with webcam access; a remote browser deployment does not automatically provide access to the user's webcam through this implementation.

---

## 🧪 Running Tests

The project includes automated tests for authentication, database operations, learning logic, search algorithms, game logic, and Phase 4 components.

Run the test suite from the project root:

```bash
python -m unittest discover -s tests -v
```

---

## 📚 Dataset and References

The project uses self-recorded webcam data containing hand-landmark coordinates for 34 PSL alphabet signs.

The dataset consists of three recording sessions and stores hand landmarks rather than face images.

PSL sign references were checked against:

- [Family Educational Services Foundation – PSL Dictionary](https://psl.org.pk/)
- [UAlpha40: Urdu Alphabet Dataset for Pakistan Sign Language](https://pmc.ncbi.nlm.nih.gov/articles/PMC11848795/)

MediaPipe is used for hand-landmark detection. The classification, feature engineering, feedback, learning, and adaptive recommendation components are implemented as part of this project.

## ⚠️ Limitations

- The current recognition system focuses on static alphabet hand configurations.
- Dynamic signs involving movement, complete words, and sentences are not supported.
- The training dataset contains a limited number of recording sessions and may not represent every user, camera, or lighting condition.
- Visually similar hand configurations can be difficult to distinguish.
- Finger-level feedback is based on learned joint-angle patterns and should be treated as practice guidance rather than a substitute for instruction from a qualified PSL teacher.
- The webcam practice implementation requires local camera access.

## 🚀 Future Improvements

- Expand the dataset with more signers, recording environments, and variations.
- Support dynamic PSL gestures and word-level recognition.
- Improve recognition of visually similar signs.
- Evaluate finger-level feedback with real learner attempts.
- Explore mobile-friendly webcam practice.
- Extend adaptive recommendations using longer-term learner performance.

## 👩‍💻 Author

**Laiba**

Developed as an AI and machine-learning project focused on making Pakistani Sign Language alphabet learning more interactive, accessible, and personalized.

---

**Repository:** [Pakistani Sign Language Learning Platform](https://github.com/laibaarshad827/Pakistani_sign_language_recognizer)
