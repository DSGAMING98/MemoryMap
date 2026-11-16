from pathlib import Path

# APPLICATION INFORMATION


APP_NAME = "MemoryMap"

APP_VERSION = "1.1.0"

APP_TITLE = (
    "MemoryMap - Personalized Memory Retention "
    "and Adaptive Revision System"
)

APP_ICON = "🧠"

# PROJECT PATHS


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"

ASSETS_DIR = BASE_DIR / "assets"

# DATASET PATHS


RAW_DATASET_PATH = (
    RAW_DATA_DIR
    / "settles.acl16.learning_traces.13m.csv.gz"
)

PROCESSED_DATASET_PATH = (
    PROCESSED_DATA_DIR
    / "memorymap_dataset.csv"
)

# DATABASE


DATABASE_PATH = BASE_DIR / "memorymap.db"


# MACHINE LEARNING MODEL PATHS


RETENTION_MODEL_PATH = (
    MODELS_DIR
    / "retention_model.pkl"
)

SCALER_PATH = (
    MODELS_DIR
    / "scaler.pkl"
)

MODEL_METADATA_PATH = (
    MODELS_DIR
    / "model_metadata.json"
)



# ASSET PATHS


CSS_PATH = (
    ASSETS_DIR
    / "styles.css"
)

LOGO_PATH = (
    ASSETS_DIR
    / "logo.png"
)



# DUOLINGO RAW DATASET COLUMNS


DUOLINGO_COLUMNS = [
    "p_recall",
    "timestamp",
    "delta",
    "user_id",
    "learning_language",
    "ui_language",
    "lexeme_id",
    "lexeme_string",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct"
]


# DATASET COLUMN GROUPS


TARGET_COLUMN = "p_recall"

IDENTIFIER_COLUMNS = [
    "user_id",
    "lexeme_id",
    "lexeme_string",
    "learning_language",
    "ui_language"
]

RAW_NUMERIC_COLUMNS = [
    "timestamp",
    "delta",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct"
]



# ENGINEERED MACHINE LEARNING FEATURES


MODEL_FEATURE_COLUMNS = [
    "delta_days",
    "history_seen",
    "history_correct",
    "history_incorrect",
    "history_accuracy"
]



# MEMORYMAP USER-COLLECTED FEATURES


MEMORYMAP_USER_FEATURES = [
    "study_duration",
    "topic_difficulty",
    "confidence_level",
    "revision_count",
    "previous_quiz_score",
    "time_since_revision",
    "actual_recall"
]


# TIME CONSTANTS


SECONDS_PER_MINUTE = 60

SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE

SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR



# RETENTION SETTINGS


MIN_RETENTION = 0.0

MAX_RETENTION = 1.0

DEFAULT_HALF_LIFE_DAYS = 1.0

TARGET_RETENTION_THRESHOLD = 0.70



# RECALL TEST CHECKPOINTS


RECALL_CHECKPOINT_DAYS = [
    1,
    3,
    7
]

# ADAPTIVE REVISION SETTINGS


MIN_REVISION_INTERVAL_HOURS = 1

MAX_REVISION_INTERVAL_DAYS = 30

SCHEDULER_STEP_HOURS = 1


# PERSONALIZATION SETTINGS


MIN_PERSONAL_RECORDS = 20

PERSONALIZATION_LEVELS = {
    "fixed": 0,
    "general": 5,
    "personalized": 20
}

# DATA PREPROCESSING SETTINGS


PREPROCESS_CHUNK_SIZE = 250_000

MAX_TRAINING_ROWS = 1_000_000



# MACHINE LEARNING SETTINGS


RANDOM_STATE = 42

TRAIN_FRACTION = 0.90

TEST_FRACTION = 0.10



# MODEL CANDIDATES


MODEL_CANDIDATES = [
    "linear_regression",
    "random_forest",
    "gradient_boosting"
]

# MODEL EVALUATION METRICS


EVALUATION_METRICS = [
    "mae",
    "rmse",
    "r2"
]



# INPUT LIMITS


MIN_STUDY_DURATION = 1

MAX_STUDY_DURATION = 600

MIN_DIFFICULTY = 1

MAX_DIFFICULTY = 5

MIN_CONFIDENCE = 1

MAX_CONFIDENCE = 5

MIN_QUIZ_SCORE = 0

MAX_QUIZ_SCORE = 100

# DATABASE TABLE NAMES

STUDENTS_TABLE = "students"

TOPICS_TABLE = "topics"

STUDY_SESSIONS_TABLE = "study_sessions"

PREDICTIONS_TABLE = "predictions"

REVISION_SCHEDULE_TABLE = "revision_schedule"

QUIZ_RESULTS_TABLE = "quiz_results"

# PREDICTION MODES


PREDICTION_MODES = [
    "Fixed Forgetting Curve",
    "Personalized Model",
    "Adaptive Schedule"
]


DEFAULT_PREDICTION_MODE = (
    "Fixed Forgetting Curve"
)

# DIRECTORY INITIALIZATION

DIRECTORIES_TO_CREATE = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ASSETS_DIR
]


def create_project_directories():
    """
    Create required MemoryMap directories if they
    do not already exist.
    """

    for directory in DIRECTORIES_TO_CREATE:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

# CONFIGURATION VALIDATION

def get_configuration_status():
    """
    Return the availability status of important
    MemoryMap files.
    """

    return {
        "dataset_available":
            RAW_DATASET_PATH.exists(),

        "processed_dataset_available":
            PROCESSED_DATASET_PATH.exists(),

        "model_available":
            RETENTION_MODEL_PATH.exists(),

        "database_available":
            DATABASE_PATH.exists(),

        "css_available":
            CSS_PATH.exists(),

        "logo_available":
            LOGO_PATH.exists()
    }

# DIRECT EXECUTION


if __name__ == "__main__":

    create_project_directories()

    status = get_configuration_status()

    print("\nMemoryMap Configuration")
    print("=" * 40)

    print(f"Application: {APP_NAME}")
    print(f"Version: {APP_VERSION}")
    print(f"Base Directory: {BASE_DIR}")

    print("\nSystem Status")
    print("-" * 40)

    for item, available in status.items():

        state = "Available" if available else "Not Available"

        print(
            f"{item}: {state}"
        )
