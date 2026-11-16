# 🧠 MemoryMap

### Personalized memory-retention prediction and adaptive revision in Streamlit

[![Live app](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://memorymap.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Random%20Forest-7C5CFF)](#model-training)

**MemoryMap** is a Streamlit-only educational research application that predicts recall, recommends revision times, records delayed-recall quiz results, and compares predictions with real outcomes.

🌐 **Live application:** [memorymap.streamlit.app](https://memorymap.streamlit.app/)

> **Research question:** Can learning-history data predict memory retention and recommend a better revision interval than a fixed forgetting-curve schedule?

## What is implemented

- A polished, multipage Streamlit interface
- Student profiles, topics, study sessions, notes, and confidence tracking
- A mathematical forgetting-curve baseline
- A trained public-data bootstrap model
- Fixed, model-assisted, and adaptive revision strategies
- Prospective predictions saved before quiz outcomes are known
- Delayed recall checkpoints at 1, 3, and 7 days
- Quiz scoring and observed half-life estimation
- Analytics for study behaviour, retention, calibration, and prediction error
- Side-by-side model comparison with MAE, RMSE, bias, R², and accuracy bands
- A normalized SQLite data layer with validation, migrations, and indexes
- A reproducible preprocessing, training, and evaluation pipeline
- Streamlit Community Cloud deployment

## Application workflow

```text
Create student profile
        ↓
Record topic and study session
        ↓
Save prospective retention prediction
        ↓
Generate threshold-based revision schedule
        ↓
Complete a delayed recall quiz
        ↓
Compare predicted retention with actual recall
        ↓
Use genuine recall evidence for later adaptive schedules
```

The important experimental rule is that a prediction is stored **before** the corresponding quiz is taken. Current-session outcomes such as `actual_score`, `recall_accuracy`, and `p_recall` are explicitly blocked from model inputs to prevent target leakage.

## Pages

| Page | Purpose |
|---|---|
| Home | Explains the experiment, runtime state, model artifact, and workflow |
| Dashboard | Creates/selects a learner and summarizes current learning activity |
| Study Session | Records a topic, duration, difficulty, confidence, revision history, and notes |
| Retention Predictor | Produces fixed and model-assisted retention curves and checkpoint estimates |
| Revision Scheduler | Schedules reviews at a selected retention threshold and tracks completion |
| Quiz and Recall | Records delayed-recall evidence and connects it to the original study session |
| Analytics | Visualizes study behaviour, retention, errors, trends, and calibration |
| Model Comparison | Compares candidate models and real prospective prediction performance |

## Dataset

The bootstrap model uses the public **Duolingo Spaced Repetition / Half-Life Regression learning traces** released with Settles and Meeder (ACL 2016).

- Source project: [Duolingo half-life-regression](https://github.com/duolingo/halflife-regression)
- Dataset: [Harvard Dataverse, DOI 10.7910/DVN/N8XJME](https://doi.org/10.7910/DVN/N8XJME)
- Paper: [A Trainable Spaced Repetition Model for Language Learning](https://aclanthology.org/P16-1174/)
- Size: approximately 13 million anonymized learning traces in a 361 MB compressed CSV

### Raw fields used by preprocessing

| Field | Meaning |
|---|---|
| `p_recall` | Proportion recalled correctly in the current practice event; training target |
| `timestamp` | UNIX time of the event |
| `delta` | Seconds since the previous relevant practice |
| `history_seen` | Number of previous exposures |
| `history_correct` | Number of previous correct responses |
| `session_seen` | Number of exercises in the current event |
| `session_correct` | Number answered correctly in the current event |

Identifiers and language fields are read to validate the official schema, but the deployed base model does not train on identity information.

### Preprocessing

`ml/preprocess.py` reads the compressed file in chunks of 250,000 rows so the full dataset does not need to be loaded into memory. For every chunk it:

1. converts required columns to numeric values;
2. removes missing or non-finite required values;
3. keeps `p_recall` within `[0, 1]`;
4. rejects negative timestamps/delays and impossible count combinations;
5. checks `p_recall` against `session_correct / session_seen`;
6. converts delay from seconds to days;
7. derives historical errors and historical accuracy;
8. downcasts numeric types and writes a clean CSV incrementally.

Engineered preprocessing fields are:

```text
delta_days          = delta / 86,400
history_incorrect   = history_seen - history_correct
history_accuracy    = history_correct / history_seen  (0 when history_seen = 0)
```

Raw and processed datasets are intentionally excluded from Git because of their size. The validated trained artifact and evaluation metadata are committed, so the live app does not need the 13-million-row dataset at runtime.

## Algorithms

### 1. Fixed forgetting curve

The baseline follows the half-life form used in spaced-repetition research:

```text
p(Δ) = 2 ^ (-Δ / h)
```

where:

- `p(Δ)` is the probability of recall after elapsed time `Δ`;
- `Δ` is measured in days;
- `h` is the memory half-life in days.

At `Δ = h`, predicted recall is 50%. MemoryMap uses a one-day default half-life when no learner evidence exists.

An observed delayed-recall result can be inverted to estimate half-life:

```text
h = -Δ / log₂(p)
```

The estimate is accepted only when elapsed time is positive and observed recall lies strictly between 0 and 1.

### 2. Public bootstrap ML predictor

The deployed model is a `RandomForestRegressor` trained to predict `p_recall` from three pre-outcome features:

| Feature | Description |
|---|---|
| `delta_days` | Days since the previous practice event |
| `history_seen` | Previous exposures before the current event |
| `history_accuracy` | Previous correct responses divided by previous exposures |

Random Forest fits many decision trees to bootstrapped samples and averages their outputs. This captures nonlinear interactions such as the way a long delay may matter differently for a learner with many accurate prior exposures. Final outputs are validated, finite, and clipped to `[0, 1]`.

The model is a **public-data bootstrap model**, not a claim that a fully personalized MemoryMap model has already learned from this app's users. MemoryMap-specific variables—study duration, topic difficulty, confidence, revision count, previous quiz score, and time since revision—are stored for future longitudinal training once enough genuine outcomes exist.

### 3. Adaptive scheduler

For a desired retention threshold `τ`, the review interval is calculated from half-life:

```text
interval = -h × log₂(τ)
```

The default threshold is 70%. Intervals are clamped between one hour and 30 days. The scheduler:

1. uses a valid half-life obtained from the selected prediction strategy;
2. prefers genuine delayed-recall evidence for adaptive scheduling;
3. falls back to the safe fixed half-life when learner evidence is insufficient;
4. assigns urgency from current retention and overdue state;
5. stores the recommended time, source, threshold, and estimated half-life.

This makes the fallback explicit instead of presenting guessed personalization as learned behaviour.

### 4. Quiz and prospective evaluation

Quiz recall is calculated as:

```text
recall_accuracy = correct_answers / total_questions
```

For a fair comparison, predictions are matched to later quiz results by student, topic/session, and prediction horizon. The analytics layer calculates:

- **MAE:** average absolute prediction error;
- **RMSE:** square-root of mean squared error, which penalizes large misses more strongly;
- **Bias:** mean signed error, revealing systematic over- or under-prediction;
- **R²:** variance explained relative to predicting the mean;
- **Within-band accuracy:** fraction within 5, 10, and 15 percentage points;
- **Calibration:** average predicted recall versus average observed recall in probability bins.

## Model training

### Sampling and split strategy

The processed dataset is streamed and each row receives an independent random key. Keeping the one million smallest keys creates a bounded, approximately uniform sample without loading the full file at once (`random_state=42`).

The sampled rows are then sorted by timestamp:

```text
1,000,000 usable sampled rows
├── Development: earliest 90% (900,000)
│   ├── Candidate training: 810,000
│   └── Candidate validation: latest 90,000 within development
└── Final test: latest 10% (100,000, untouched during selection)
```

Three candidates are trained:

| Candidate | Main configuration |
|---|---|
| Linear Regression | Standardized inputs |
| Random Forest | 40 trees, max depth 12, minimum leaf size 8, all CPU cores |
| Histogram Gradient Boosting | learning rate 0.08, 200 iterations, 31 leaves, L2 = 0.1 |

Candidates are ranked by validation MAE. The winner is refit on all 900,000 development rows and evaluated once on the 100,000-row test set.

### Saved model results

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| **Random Forest** | **16.949 pp** | 26.802 pp | 0.0264 |
| Histogram Gradient Boosting | 16.976 pp | **26.784 pp** | **0.0278** |
| Linear Regression | 17.125 pp | 26.961 pp | 0.0148 |

Random Forest won under the predefined MAE ranking rule.

Final test results for the saved model:

| Metric | Value |
|---|---:|
| Test observations | 100,000 |
| MAE | 17.026 percentage points |
| RMSE | 26.918 percentage points |
| R² | 0.0241 |
| Mean bias | +0.107 percentage points |
| Within 10 percentage points | 49.720% |
| Within 15 percentage points | 80.047% |
| Expected calibration error | 0.238 percentage points |

The evaluation script reloads the artifact, reconstructs the saved split, verifies feature order/ranges, and reproduces the recorded metrics. All 11 artifact sanity checks pass.

The deployment model deliberately uses a bounded forest. Reducing the forest from 100 depth-16 trees to 40 depth-12 trees reduced the serialized artifact from approximately 100.3 MiB to 8.5 MiB while slightly improving final-test RMSE and R². This lowers both startup memory and download time on Streamlit Community Cloud without changing the dataset, split, features, candidate-selection rule, or evaluation protocol.

### Interpretation and limitations

The small R² means the three-feature public model explains little of the event-level variance even though aggregate calibration is strong in the dominant high-recall region. It should be treated as a transparent bootstrap baseline, not a high-confidence individualized forecast.

Current limitations are recorded in `models/model_metadata.json` and `models/evaluation_report.json`:

- the chronological split is applied after uniform sampling, not across every original row;
- user and lexeme identifiers are not retained in the processed training frame, so group overlap cannot be ruled out;
- the final report reconstructs the original held-out test split; it is not a second external dataset;
- low-retention calibration bins contain far fewer observations than high-retention bins;
- a real personalized MemoryMap model requires substantially more longitudinal in-app data.

## Database

SQLite stores the complete application workflow locally:

```text
students ──< topics ──< study_sessions ──< predictions
    │           │              │
    │           ├──────────────┴──< revision_schedule
    │           └─────────────────< quiz_results
    └─────────────────────────────< quiz_results
```

Foreign keys, range checks, migrations, transaction handling, and indexes protect data quality. The local `memorymap.db` is ignored by Git.

> Streamlit Community Cloud has ephemeral local storage. Data created in the public demo can be reset when the app restarts or redeploys. For durable multi-user production storage, replace SQLite with a hosted database while keeping the Streamlit interface.

## Project structure

```text
MemoryMap/
├── app.py                         # Streamlit entry point
├── pages/                         # Seven Streamlit pages
├── core/                          # Curves, prediction, scheduling, quiz, metrics
├── services/                      # Student, study, revision, analytics workflows
├── ml/                            # Preprocessing, features, training, evaluation, loading
├── models/                        # Model artifact, metadata, evaluation report
├── assets/styles.css              # Custom application design system
├── .streamlit/config.toml         # Streamlit theme/runtime configuration
├── database.py                    # SQLite schema, migrations, and queries
├── config.py                      # Paths, limits, thresholds, feature definitions
├── runtime.txt                    # Streamlit Cloud Python runtime
└── requirements.txt               # Reproducible dependencies
```

## Run locally

Python 3.12 is recommended.

```bash
git clone https://github.com/DSGAMING98/MemoryMap.git
cd MemoryMap
python -m venv .venv
```

Activate the virtual environment, then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

### Rebuild the dataset and model

Download `settles.acl16.learning_traces.13m.csv.gz` from the Dataverse record and place it in `data/raw/`, then run:

```bash
python -m ml.preprocess
python -m ml.train_model
python -m ml.evaluate_model
```

These commands are needed only to reproduce training. Normal application use loads the committed validated model artifact.

## Streamlit deployment

The repository is configured for Streamlit Community Cloud:

1. repository: `DSGAMING98/MemoryMap`;
2. branch: `master`;
3. entry point: `app.py`;
4. Python runtime: Python 3.12, selected in Streamlit Cloud advanced settings (`runtime.txt` documents the preferred local/cloud runtime);
5. packages: `requirements.txt`;
6. trained artifact: `models/retention_model.pkl`.

Pushing to `master` triggers the existing Streamlit deployment to rebuild. No Flask, Django, React, or separate backend is required; the product remains Streamlit-only.

## Results and conclusion

The implementation answers the engineering part of the research question: learning-history variables can be converted into prospective recall estimates, those estimates can drive threshold-based schedules, and later delayed-recall outcomes can be matched back to the original predictions without outcome leakage.

On the public Duolingo test split, Random Forest produced the lowest validation MAE and achieved a final MAE of 17.026 percentage points. Its low test R² of 0.0241 shows that the three public-history features do not explain most individual event variation. Therefore, the honest conclusion is not that personalization has already been proven superior. The application now provides the experimental infrastructure needed to test that claim as genuine MemoryMap longitudinal data accumulates.

The model-comparison dashboard reports a winner only when at least two methods have predictions for the same strict recall observations. Until that condition is met, the interface reports insufficient evidence instead of manufacturing personalized values.

### Future research scope

- collect repeated 1-, 3-, and 7-day recall observations from consenting learners;
- train a separate MemoryMap model using study duration, difficulty, confidence, revision count, previous genuine quiz score, and elapsed time;
- use learner- or topic-grouped validation to prevent identity overlap;
- compare fixed, public bootstrap, and personalized methods on a common prospective test set;
- replace ephemeral demo SQLite storage with a hosted database for durable multi-user research;
- study uncertainty estimates and threshold sensitivity before making stronger scheduling claims.

## Project status

- [x] Project idea finalized
- [x] Research problem finalized
- [x] Real-world dataset selected
- [x] Program architecture designed
- [x] Streamlit application initialized
- [x] Dataset preprocessing
- [x] Feature engineering
- [x] Database implementation
- [x] Forgetting curve engine
- [x] ML model training and artifact validation
- [x] Adaptive scheduler
- [x] Quiz engine
- [x] Analytics dashboard
- [x] Model comparison
- [x] Streamlit deployment

## Responsible use

MemoryMap is an educational research project, not a medical or cognitive diagnostic tool. Predictions are uncertain estimates and should guide study planning rather than label a learner's ability.

## Citation

If you use the public Duolingo data, cite:

```bibtex
@inproceedings{settles2016trainable,
  title     = {A Trainable Spaced Repetition Model for Language Learning},
  author    = {Settles, Burr and Meeder, Brendan},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics},
  pages     = {1848--1858},
  year      = {2016},
  doi       = {10.18653/v1/P16-1174}
}
```

---

**Project:** MemoryMap

**Domain:** Educational technology, machine learning, personalized learning

**Interface:** Streamlit only
