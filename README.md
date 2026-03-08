# Human Activity Recognition using Hidden Markov Models

ML Techniques II - Formative 2

**Team Members**
- Kerie Izere Uwonkunda — iPhone 11, Sensor Logger, 100 Hz
- Relebohile Pheko — iPhone 12 Pro Max, Sensor Logger, 100 Hz

**Activities:** Standing, Walking, Jumping, Still
**Approach:** Gaussian HMM implemented from scratch using NumPy
**Result:** 98.78% accuracy on 14 unseen test recordings (245 windows)

---

## Overview

This project builds a Hidden Markov Model from scratch to classify human physical activities from smartphone inertial sensor data. Accelerometer and gyroscope readings were collected at 100 Hz, segmented into 1-second windows, and converted into 38 time- and frequency-domain features. A Gaussian HMM was trained using the Baum-Welch algorithm and decoded at inference time using the Viterbi algorithm.

---

## Repository Structure

```
FORMATIVE-2_HIDDEN-MARKOV-MODELS/
│
├── data/
│   ├── jumping/                      # Raw CSV recordings for jumping
│   ├── standing/                     # Raw CSV recordings for standing
│   ├── still/                        # Raw CSV recordings for still
│   ├── walking/                      # Raw CSV recordings for walking
│   ├── all_activities_combined.csv   # Merged dataset (62,743 rows)
│   ├── extracted_features.csv        # 1,162 feature windows (38 features each)
│   ├── train.csv                     # Training split
│   └── test.csv                      # Test split
│
├── unprocessed/                      # Raw zip files from Sensor Logger
│
├── confusion_matrix.png              # Confusion matrix heatmap
├── decoded_sequences.png             # Viterbi decoded sequence plot
├── emission_means.png                # Learned emission means heatmap
├── hmm_activity_recognition.ipynb   # Main notebook
├── process_data.py                   # Data preprocessing script
├── .gitignore
└── README.md
```

---

## Dataset

| Property | Value |
|----------|-------|
| Total recordings | 64 (16 per activity, 32 per person) |
| Raw data points | 62,743 rows |
| Sampling rate | 100 Hz — both devices |
| Sensors | Accelerometer (x, y, z) + Gyroscope (x, y, z) |
| Feature windows | 1,162 (100 samples/window, 50% overlap) |
| Features per window | 38 |
| Train / Test split | 50 recordings (917 windows) / 14 recordings (245 windows) |

---

## Feature Extraction

Each 1-second window produces 38 features across six sensor axes.

**Time-domain (26 features)**
- Mean, Variance, Standard Deviation — per axis x 6 = 18 features
- Signal Magnitude Area (SMA) — accelerometer + gyroscope = 2 features
- Pearson correlation between axis pairs (XY, XZ, YZ) x 2 sensors = 6 features

**Frequency-domain (12 features)**
- Dominant frequency (FFT peak, excluding DC) — per axis x 6 = 6 features
- Spectral energy (sum of squared FFT magnitudes) — per axis x 6 = 6 features

All features are Z-score normalised, fit on the training set only.

---

## Model

**HMM Components**

| Symbol | Component | Description |
|--------|-----------|-------------|
| Z | Hidden states | {standing, walking, jumping, still} |
| X | Observations | 38-dimensional normalised feature vectors |
| A | Transitions | 4x4 matrix, initialised with 0.95 self-transition bias |
| B | Emissions | Multivariate Gaussian per state, regularised with 1e-3 x I |
| pi | Initial probs | Uniform (0.25 each), updated by Baum-Welch |

**Training — Baum-Welch**
- Mixed training sequences: each sequence concatenates one recording per activity in random order, forcing Baum-Welch to learn non-zero off-diagonal transitions
- Convergence criterion: change in log-likelihood < 1e-4
- Converged in 8 iterations, final log-likelihood: 24,436.94

**Decoding — Viterbi**
- Test recordings concatenated into one mixed sequence before decoding
- All computations in log-space to prevent numerical underflow

---

## Results

**Learned Transition Matrix**

| From \ To | Standing | Walking | Jumping | Still  |
|-----------|----------|---------|---------|--------|
| Standing  | 0.9543   | 0.0137  | 0.0137  | 0.0183 |
| Walking   | 0.0208   | 0.9458  | 0.0167  | 0.0167 |
| Jumping   | 0.0050   | 0.0201  | 0.9648  | 0.0101 |
| Still     | 0.0138   | 0.0092  | 0.0184  | 0.9585 |

**Test Performance**

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|----------|---------|-------------|-------------|----------|
| Standing | 45      | 0.9333      | 1.0000      | 0.9878   |
| Walking  | 53      | 1.0000      | 1.0000      | 0.9878   |
| Jumping  | 67      | 1.0000      | 1.0000      | 0.9878   |
| Still    | 80      | 1.0000      | 0.9818      | 0.9878   |
| Overall  | 245     | —           | —           | 0.9878   |

3 standing windows were misclassified as still — the only errors in the test set.

---

## Setup

**Requirements**

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

No external HMM library is used. The model is implemented entirely from scratch in NumPy.

**Running the Notebook**

```bash
git clone https://github.com/<your-username>/FORMATIVE-2_HIDDEN-MARKOV-MODELS.git
cd FORMATIVE-2_HIDDEN-MARKOV-MODELS
jupyter notebook hmm_activity_recognition.ipynb
```

Run all cells top to bottom. Make sure the `data/` folder is in the same directory as the notebook before running.

---

## Task Division

https://docs.google.com/spreadsheets/d/1zzRBIszNLMgdMrloOYYJhxNXoTzA1CbA1dsA0CIhHYU/edit?usp=sharing

---

