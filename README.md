# README.md

# 🧠 MemoryMap

### Personalized Memory Retention Prediction and Adaptive Revision System

**MemoryMap** is a Python-based experiential learning project designed to predict how much information a student is likely to remember over time and determine better revision intervals using personalized learning data.

Instead of relying only on a fixed forgetting-curve schedule for every student, MemoryMap compares multiple approaches and studies whether revision timing can be improved using individual learning behaviour.

---

## 🎯 Research Question

> **Can personalized learning data be used to predict a student's retention and determine a better revision interval than a fixed forgetting-curve schedule?**

MemoryMap compares three approaches:

1. **Fixed Forgetting Curve**
2. **Personalized Retention Prediction Model**
3. **Adaptive Revision Schedule**

The predictions are later compared with actual quiz or recall performance.

---

## 🚀 Project Objectives

* Predict student memory retention over time.
* Analyze individual learning behaviour.
* Compare fixed and personalized retention models.
* Recommend suitable revision intervals.
* Record actual recall performance after revision.
* Compare predicted retention with actual quiz results.
* Use real-world educational data for initial model training.
* Collect new MemoryMap user data for future personalization.
* Demonstrate Python concepts, libraries, machine learning, data analysis, databases, and visualization in a complete application.

---

## 🧩 Learning Data Used

MemoryMap can use information such as:

* Study duration
* Topic difficulty
* Previous revision time
* Number of revisions
* Previous quiz/test performance
* Time since last revision
* Confidence level
* Recall accuracy

These features are used to understand how different students retain information differently.

---

## 📊 Real-World Dataset

MemoryMap initially uses the **Duolingo Spaced Repetition / Half-Life Regression learning-trace dataset**.

The raw dataset contains millions of real student learning interactions and includes information related to:

* learner history
* time between practice sessions
* previous exposures
* previous correct recalls
* session performance
* actual recall rate

The original dataset is stored without modification inside:

```text
data/raw/
```

MemoryMap preprocessing scripts create a separate cleaned dataset for model training.

---

## 🧠 How MemoryMap Works

```text
Student Records Study Session
            ↓
Learning Data Stored
            ↓
Fixed Forgetting Curve Prediction
            ↓
Personalized ML Prediction
            ↓
Adaptive Revision Scheduler
            ↓
Recommended Revision Time
            ↓
Student Takes Recall Quiz
            ↓
Actual Recall Score Recorded
            ↓
Predicted vs Actual Performance
            ↓
Model Evaluation and Personalization
```

---

## 🔬 Prediction Methods

### 1. Fixed Forgetting Curve

MemoryMap uses a mathematical forgetting-curve model as the baseline prediction method.

A simplified form is:

```text
p = 2^(-Δ / h)
```

Where:

* `p` = predicted probability of recall
* `Δ` = time since previous study/revision
* `h` = estimated memory half-life

This approach provides the same general memory-decay principle without fully considering individual learning behaviour.

---

### 2. Personalized Prediction Model

The personalized system uses machine learning to estimate retention based on features such as:

```text
study duration
topic difficulty
revision count
previous performance
confidence
time since revision
previous learning history
```

As more information becomes available for a student, MemoryMap can generate increasingly personalized predictions.

---

### 3. Adaptive Revision Scheduler

The adaptive scheduler uses predicted retention to determine when a topic should be revised again.

For example:

```text
Current predicted retention: 84%
Minimum desired retention: 70%

Recommended revision:
Approximately 2 days from now
```

Instead of assigning the same revision interval to every topic and student, MemoryMap adjusts recommendations according to predicted memory strength.

---

## 📝 Recall Testing

Prediction alone is not enough.

MemoryMap records actual student performance through delayed recall tests.

Examples:

```text
After 1 Day
After 3 Days
After 7 Days
```

A result may look like:

```text
Predicted Retention: 76%
Actual Quiz Score: 81%
Prediction Error: 5%
```

These results allow the application to evaluate whether personalized predictions are more accurate than the fixed forgetting curve.

---

## 📈 Model Evaluation

MemoryMap can use evaluation metrics such as:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Prediction Error
* Average Recall Accuracy
* Retention Improvement

The final project will compare:

```text
Fixed Forgetting Curve
          VS
Personalized Prediction Model
          VS
Adaptive Revision Schedule
```

---

## 🖥️ Main Application Features

### Dashboard

Displays:

* active study topics
* upcoming revisions
* average predicted retention
* actual recall performance
* study statistics
* recent activity

### Study Session Logger

Allows students to record:

* topic
* study duration
* difficulty
* confidence
* revision count
* previous performance

### Retention Predictor

Displays estimated retention over different time intervals.

### Revision Scheduler

Recommends when each topic should be studied again.

### Quiz and Recall Testing

Records actual recall performance after a delay.

### Analytics

Visualizes relationships such as:

* retention vs time
* study duration vs recall
* difficulty vs recall
* revision count vs retention
* predicted vs actual performance

### Model Comparison

Shows the performance difference between fixed, personalized, and adaptive methods.

---

## 🛠️ Technologies Used

### Python

Main programming language used throughout the application.

### Streamlit

Used to build the interactive web application.

### Pandas

Used for:

* dataset processing
* table operations
* data analysis
* filtering and transformation

### NumPy

Used for:

* mathematical operations
* numerical processing
* retention calculations

### Scikit-learn

Used for:

* machine learning
* model training
* prediction
* model evaluation

### Plotly

Used for interactive charts and dashboards.

### Matplotlib

Used for research and analytical visualizations.

### SQLite

Used for storing:

* student information
* study sessions
* predictions
* revision schedules
* quiz results

### Joblib

Used to save and load trained machine-learning models.

### Python Standard Library

MemoryMap also uses built-in Python modules such as:

```text
datetime
math
pathlib
statistics
json
os
```

---

## 📁 Project Architecture

```text
MemoryMap/
│
├── app.py
├── requirements.txt
├── README.md
├── config.py
├── database.py
├── utils.py
│
├── data/
│   ├── raw/
│   │   └── settles.acl16.learning_traces.13m.csv.gz
│   │
│   └── processed/
│       └── memorymap_dataset.csv
│
├── models/
│   ├── retention_model.pkl
│   └── scaler.pkl
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Study_Session.py
│   ├── 3_Retention_Predictor.py
│   ├── 4_Revision_Scheduler.py
│   ├── 5_Quiz_and_Recall.py
│   ├── 6_Analytics.py
│   └── 7_Model_Comparison.py
│
├── core/
│   ├── forgetting_curve.py
│   ├── predictor.py
│   ├── scheduler.py
│   ├── quiz_engine.py
│   └── metrics.py
│
├── ml/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── load_model.py
│
├── services/
│   ├── student_service.py
│   ├── study_service.py
│   ├── revision_service.py
│   └── analytics_service.py
│
└── assets/
    ├── logo.png
    └── styles.css
```

---

## ⚙️ Installation

### 1. Clone or download the project

Open the project in **PyCharm**.

### 2. Create a virtual environment

PyCharm can automatically create one when opening the project.

### 3. Install required libraries

Open the PyCharm terminal and run:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running MemoryMap

From the project directory:

```bash
streamlit run app.py
```

Streamlit will normally open:

```text
http://localhost:8501
```

in the browser.

---

## 📦 Dataset Location

Place the downloaded Duolingo dataset inside:

```text
MemoryMap/data/raw/
```

The expected filename is:

```text
settles.acl16.learning_traces.13m.csv.gz
```

The `.gz` file does **not** need to be manually converted to Excel or extracted.

Pandas can directly read compressed CSV files.

---

## 🧪 Development Strategy

MemoryMap follows this development pipeline:

```text
Raw Real-World Dataset
        ↓
Data Exploration
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Training Dataset
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Saved Prediction Model
        ↓
MemoryMap Application
        ↓
Real Student Usage
        ↓
New Recall Data
        ↓
Personalized Predictions
```

---

## 🏗️ Application Architecture

MemoryMap follows a simple three-layer architecture:

```text
Presentation Layer
        ↓
Streamlit User Interface

Logic Layer
        ↓
Retention Prediction
Forgetting Curve
Machine Learning
Adaptive Scheduling
Quiz Evaluation

Data Layer
        ↓
SQLite Database
Duolingo Dataset
Saved ML Models
```

This modular structure keeps individual parts of the project easier to understand, test, maintain, and explain.

---

## 🔮 Expected Outcome

After successful completion, MemoryMap should be able to determine whether personalized learning information provides more accurate memory-retention predictions than a fixed forgetting-curve approach.

The application is expected to:

* predict future retention
* detect topics likely to be forgotten
* recommend revision intervals
* record real recall performance
* analyze student learning behaviour
* evaluate prediction accuracy
* adapt recommendations based on previous performance

The long-term goal is to help students revise **when revision is actually useful**, rather than following the same fixed schedule for every topic.

---

## ⚠️ Important Research Note

The public dataset is used to develop and initially evaluate MemoryMap's prediction system.

Information collected directly through MemoryMap will be kept logically separate from the original dataset.

The application will not present simulated or generated information as real student experimental data.

Actual MemoryMap recall performance will be recorded only from real user study sessions and subsequent tests.

---

## 📌 Project Status

```text
[✓] Project idea finalized
[✓] Research problem finalized
[✓] Real-world dataset selected
[✓] Program architecture designed
[✓] Streamlit application initialized
[ ] Dataset preprocessing
[ ] Feature engineering
[ ] Database implementation
[ ] Forgetting curve engine
[ ] ML model training
[ ] Adaptive scheduler
[ ] Quiz engine
[ ] Analytics dashboard
[ ] Model comparison
[ ] Deployment
```

---

## 🎓 Project Type

**Python Experiential Learning Project**

**Project Name:** MemoryMap

**Domain:** Educational Technology / Machine Learning / Personalized Learning

**Primary Language:** Python
