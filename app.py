import json
from html import escape
from textwrap import dedent

import streamlit as st

from config import (
    APP_NAME,
    APP_VERSION,
    APP_TITLE,
    APP_ICON,
    RAW_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    DATABASE_PATH,
    SCALER_PATH,
    MODEL_METADATA_PATH,
    CSS_PATH,
    LOGO_PATH,
    TARGET_RETENTION_THRESHOLD,
    RECALL_CHECKPOINT_DAYS,
    create_project_directories,
)

from database import (
    initialize_database,
)

from ml.load_model import (
    get_model_information,
    get_artifact_status,
    load_metadata,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# initialization

create_project_directories()
initialize_database()


def initialize_session_state():
    defaults = {
        "student_id":
            None,

        "student_name":
            None,

        "selected_topic":
            None,

        "selected_session_id":
            None,

        "prediction_mode":
            "Fixed Forgetting Curve",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[
                key
            ] = value


initialize_session_state()


# styling

def load_css():
    if not CSS_PATH.exists():
        return

    try:
        css = CSS_PATH.read_text(
            encoding="utf-8"
        )

        st.html(
            f"<style>{css}</style>"
        )

    except OSError:
        pass


def render_html(
    content
):
    html = dedent(
        content
    ).strip()

    st.html(
        html
    )


load_css()


# file helpers

def file_exists(
    path
):
    try:
        return path.exists()

    except AttributeError:
        return False


def load_json_file(
    path
):
    if not path.exists():
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return None


# model state

def get_model_state():
    information = (
        get_model_information()
    )

    artifact_status = (
        get_artifact_status()
    )

    metadata = {}

    if information.get(
        "available",
        False
    ):
        try:
            metadata = load_metadata()

        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
            TypeError,
        ):
            metadata = {}

    model_ready = bool(
        information.get(
            "available",
            False
        )
    )

    personal_model = (
        model_ready
        and bool(
            metadata.get(
                "memorymap_personal_model",
                False
            )
        )
    )

    uses_scaler = bool(
        information.get(
            "uses_scaler",
            False
        )
    )

    scaler_exists = bool(
        artifact_status.get(
            "scaler_exists",
            False
        )
    )

    if uses_scaler:
        scaler_state = (
            "Ready"
            if scaler_exists
            else "Missing"
        )

    else:
        scaler_state = (
            "Not Required"
        )

    evaluation_report_path = (
        MODEL_METADATA_PATH.parent
        / "evaluation_report.json"
    )

    evaluation_report = (
        load_json_file(
            evaluation_report_path
        )
        or {}
    )

    evaluation_reproduced = (
        model_ready
        and evaluation_report.get(
            "evaluation_reproduced"
        )
        is True
    )

    independent_new_test_set = (
        evaluation_report.get(
            "independent_new_test_set"
        )
    )

    return {
        "information":
            information,

        "metadata":
            metadata,

        "artifact_status":
            artifact_status,

        "model_ready":
            model_ready,

        "personal_model":
            personal_model,

        "uses_scaler":
            uses_scaler,

        "scaler_state":
            scaler_state,

        "evaluation_report":
            evaluation_report,

        "evaluation_report_path":
            evaluation_report_path,

        "evaluation_reproduced":
            evaluation_reproduced,

        "independent_new_test_set":
            independent_new_test_set,
    }


MODEL_STATE = get_model_state()


# status helpers

def status_badge(
    label,
    status_type="info"
):
    css_class = {
        "success":
            "mm-badge-success",

        "warning":
            "mm-badge-warning",

        "info":
            "mm-badge-info",
    }.get(
        status_type,
        "mm-badge-info"
    )

    return (
        '<span class="mm-badge '
        f'{css_class}">'
        f'● {escape(str(label))}'
        '</span>'
    )


def show_status_card(
    title,
    badge,
    description
):
    render_html(
        f"""
        <div class="mm-card">
            <div class="mm-card-title">
                {escape(str(title))}
            </div>

            {badge}

            <div
                class="mm-card-text"
                style="margin-top: 10px;"
            >
                {escape(str(description))}
            </div>
        </div>
        """
    )


# sidebar

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(
            str(
                LOGO_PATH
            ),
            width=90
        )

    st.title(
        APP_NAME
    )

    st.caption(
        f"Version {APP_VERSION}"
    )

    st.markdown(
        """
Personalized memory retention,
recall measurement and adaptive
revision research.
        """
    )

    st.divider()

    if st.session_state.get(
            "student_name"
    ) is not None:

        st.success(
            "Active student: "
            f"{st.session_state.student_name}"
        )

        if st.button(
                "↪ Switch Profile",
                use_container_width=True
        ):
            st.session_state.student_id = None
            st.session_state.student_name = None
            st.session_state.selected_topic = None
            st.session_state.selected_session_id = None

            if (
                    "dashboard_student_selector"
                    in st.session_state
            ):
                del st.session_state[
                    "dashboard_student_selector"
                ]

            st.rerun()

    else:
        st.info(
            "No student profile selected yet."
        )

    st.divider()

    st.markdown(
        "### Research checkpoints"
    )

    for day in RECALL_CHECKPOINT_DAYS:
        st.markdown(
            f"**Day {day}** recall test"
        )

    st.markdown(
        "Target retention: "
        f"**{TARGET_RETENTION_THRESHOLD * 100:.0f}%**"
    )

    st.divider()

    if MODEL_STATE[
        "model_ready"
    ]:
        st.success(
            "ML artifact validated"
        )

    else:
        st.warning(
            "ML artifact unavailable"
        )

    if MODEL_STATE[
        "personal_model"
    ]:
        st.success(
            "Personalized model available"
        )

    else:
        st.info(
            "Personalized model not trained yet"
        )

    st.divider()

    st.caption(
        "MemoryMap Research System"
    )


# hero

render_html(
    """
    <div class="mm-hero">
        <div class="mm-section-label">
            PERSONALIZED LEARNING INTELLIGENCE
        </div>

        <div class="mm-hero-title">
            MemoryMap
        </div>

        <div class="mm-hero-subtitle">
            A memory retention prediction and adaptive
            revision research system combining forgetting
            curves, machine learning and real delayed
            recall measurements.
        </div>
    </div>
    """
)


# research question

render_html(
    """
    <div class="mm-card">
        <div class="mm-section-label">
            RESEARCH QUESTION
        </div>

        <div class="mm-card-title">
            Can personalized learning data predict
            memory retention more accurately and
            recommend better revision intervals than
            a fixed forgetting-curve schedule?
        </div>

        <div class="mm-card-text">
            MemoryMap does not judge a model from a
            convincing graph alone. Predictions are
            recorded before testing and later compared
            with actual delayed recall measurements.
        </div>
    </div>
    """
)


# system summary

runtime_database_ready = (
    file_exists(
        DATABASE_PATH
    )
)

model_ready = MODEL_STATE[
    "model_ready"
]

evaluation_reproduced = (
    MODEL_STATE[
        "evaluation_reproduced"
    ]
)

personal_model = MODEL_STATE[
    "personal_model"
]


col1, col2, col3, col4 = (
    st.columns(
        4
    )
)

with col1:
    st.metric(
        "Runtime Core",
        (
            "Ready"
            if runtime_database_ready
            else "Pending"
        )
    )

with col2:
    st.metric(
        "ML Artifact",
        (
            "Validated"
            if model_ready
            else "Unavailable"
        )
    )

with col3:
    st.metric(
        "Evaluation",
        (
            "Reproduced"
            if evaluation_reproduced
            else "Pending"
        )
    )

with col4:
    st.metric(
        "Personalized Model",
        (
            "Available"
            if personal_model
            else "Not Yet"
        )
    )


# workflow

st.markdown(
    "## 🧭 Research Workflow"
)

render_html(
    """
    <div class="mm-pipeline">
        <div class="mm-pipeline-step">
            📚 Study Session
        </div>

        <div class="mm-pipeline-arrow">
            →
        </div>

        <div class="mm-pipeline-step">
            📉 Prospective Prediction
        </div>

        <div class="mm-pipeline-arrow">
            →
        </div>

        <div class="mm-pipeline-step">
            ⏰ Revision Schedule
        </div>

        <div class="mm-pipeline-arrow">
            →
        </div>

        <div class="mm-pipeline-step">
            🧪 Delayed Recall Test
        </div>

        <div class="mm-pipeline-arrow">
            →
        </div>

        <div class="mm-pipeline-step">
            📊 Strict Evaluation
        </div>
    </div>
    """
)


# feature overview

st.markdown(
    "## ⚡ What MemoryMap Does"
)

col1, col2, col3 = (
    st.columns(
        3
    )
)


with col1:
    render_html(
        """
        <div class="mm-card">
            <div class="mm-section-label">
                PREDICT
            </div>

            <div class="mm-card-title">
                🧠 Retention Intelligence
            </div>

            <div class="mm-card-text">
                Generate prospective retention estimates
                and preserve them before the student's
                delayed recall result becomes known.
            </div>
        </div>
        """
    )


with col2:
    render_html(
        """
        <div class="mm-card">
            <div class="mm-section-label">
                ADAPT
            </div>

            <div class="mm-card-title">
                ⏰ Revision Scheduling
            </div>

            <div class="mm-card-text">
                Estimate when retention may reach a
                selected threshold and use genuine
                learner evidence when it becomes
                available.
            </div>
        </div>
        """
    )


with col3:
    render_html(
        """
        <div class="mm-card">
            <div class="mm-section-label">
                VERIFY
            </div>

            <div class="mm-card-title">
                🔬 Real Recall Validation
            </div>

            <div class="mm-card-text">
                Compare predictions with actual delayed
                recall using strict time matching, MAE,
                RMSE, bias, R² and accuracy bands.
            </div>
        </div>
        """
    )


# navigation

st.markdown(
    "## 🚀 Enter MemoryMap"
)

nav1, nav2, nav3 = (
    st.columns(
        3
    )
)


with nav1:
    st.page_link(
        "pages/1_Dashboard.py",
        label="📊 Open Dashboard",
        use_container_width=True
    )

    st.page_link(
        "pages/2_Study_Session.py",
        label="📚 Record Study Session",
        use_container_width=True
    )


with nav2:
    st.page_link(
        "pages/3_Retention_Predictor.py",
        label="📉 Retention Predictor",
        use_container_width=True
    )

    st.page_link(
        "pages/4_Revision_Scheduler.py",
        label="⏰ Revision Scheduler",
        use_container_width=True
    )


with nav3:
    st.page_link(
        "pages/5_Quiz_and_Recall.py",
        label="🧪 Quiz & Recall",
        use_container_width=True
    )

    st.page_link(
        "pages/6_Analytics.py",
        label="📊 Analytics",
        use_container_width=True
    )


st.page_link(
    "pages/7_Model_Comparison.py",
    label="⚖️ Compare Prediction Models",
    use_container_width=True
)


# machine learning state

st.markdown(
    "## 🤖 Machine Learning State"
)

information = MODEL_STATE[
    "information"
]

metadata = MODEL_STATE[
    "metadata"
]


if model_ready:
    model_name = information.get(
        "model_name",
        "Unknown"
    )

    model_class = information.get(
        "model_class",
        "Unknown"
    )

    feature_set = information.get(
        "feature_set",
        "Unknown"
    )

    feature_columns = information.get(
        "feature_columns",
        []
    )

    training_source = (
        information.get(
            "training_source"
        )
        or metadata.get(
            "training_source"
        )
        or "Unknown"
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    with col1:
        st.metric(
            "Loaded Model",
            model_name
        )

    with col2:
        st.metric(
            "Feature Set",
            feature_set
        )

    with col3:
        st.metric(
            "Scaler",
            MODEL_STATE[
                "scaler_state"
            ]
        )

    st.caption(
        f"Model class: {model_class}"
    )

    if feature_columns:
        st.caption(
            "Validated model inputs: "
            + ", ".join(
                feature_columns
            )
        )

    st.caption(
        "Training source: "
        f"{training_source}"
    )

    if personal_model:
        st.success(
            "The loaded artifact is marked as "
            "a MemoryMap personalized model."
        )

    else:
        st.info(
            "The loaded artifact is the public "
            "bootstrap retention model. It is "
            "validated and usable for its genuine "
            "training feature contract, but it is "
            "not being presented as a personalized "
            "MemoryMap model."
        )

else:
    st.warning(
        "The trained retention model could not "
        "be validated."
    )

    error_message = information.get(
        "message"
    )

    if error_message:
        st.caption(
            error_message
        )

    st.info(
        "The fixed forgetting-curve baseline "
        "remains available."
    )


# evaluation state

st.markdown(
    "## ✅ Model Evaluation State"
)

evaluation_report = MODEL_STATE[
    "evaluation_report"
]


if evaluation_reproduced:
    st.success(
        "The saved model evaluation was "
        "successfully reproduced."
    )

    metrics = (
        evaluation_report.get(
            "metrics"
        )
        or {}
    )

    mae = metrics.get(
        "mae"
    )

    rmse = metrics.get(
        "rmse"
    )

    r2 = metrics.get(
        "r2"
    )

    test_rows = (
        evaluation_report.get(
            "test_rows"
        )
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Evaluation Reproduced",
            "YES"
        )

    with col2:
        st.metric(
            "Test Rows",
            (
                f"{int(test_rows):,}"
                if test_rows
                is not None
                else "N/A"
            )
        )

    with col3:
        st.metric(
            "Test MAE",
            (
                f"{float(mae) * 100:.3f} pp"
                if mae
                is not None
                else "N/A"
            )
        )

    with col4:
        st.metric(
            "Test R²",
            (
                f"{float(r2):.4f}"
                if r2
                is not None
                else "N/A"
            )
        )

    if rmse is not None:
        st.caption(
            "Test RMSE: "
            f"{float(rmse) * 100:.3f} "
            "percentage points."
        )

    if (
        MODEL_STATE[
            "independent_new_test_set"
        ]
        is False
    ):
        st.caption(
            "This evaluation reconstructs the "
            "untouched test split used during "
            "training for reproducibility. It is "
            "not being described as a new external "
            "holdout dataset."
        )

else:
    st.info(
        "A reproduced saved-model evaluation "
        "is not currently available."
    )


# prediction strategies

st.markdown(
    "## 🧪 Prediction & Scheduling Strategies"
)

mode1, mode2, mode3 = (
    st.columns(
        3
    )
)


with mode1:
    render_html(
        """
        <div class="mm-model-card">
            <div class="mm-section-label">
                BASELINE
            </div>

            <div class="mm-card-title">
                Fixed Forgetting Curve
            </div>

            <div class="mm-card-text">
                Uses the configured baseline memory
                half-life and provides the reference
                schedule against which later methods
                can be compared.
            </div>
        </div>
        """
    )


with mode2:
    render_html(
        """
        <div class="mm-model-card">
            <div class="mm-section-label">
                PERSONALIZATION
            </div>

            <div class="mm-card-title">
                Personalized Model
            </div>

            <div class="mm-card-text">
                A future MemoryMap model trained from
                genuine longitudinal learner variables
                such as study behaviour, confidence,
                previous performance and recall outcomes.
            </div>
        </div>
        """
    )


with mode3:
    render_html(
        """
        <div class="mm-model-card">
            <div class="mm-section-label">
                ADAPTIVE
            </div>

            <div class="mm-card-title">
                Adaptive Schedule
            </div>

            <div class="mm-card-text">
                Uses valid observed delayed recall
                when possible and falls back safely
                when there is not enough learner
                evidence for an adaptive estimate.
            </div>
        </div>
        """
    )


# research assets

st.markdown(
    "## 🧰 Research & Training Assets"
)

raw_available = file_exists(
    RAW_DATASET_PATH
)

processed_available = file_exists(
    PROCESSED_DATASET_PATH
)

evaluation_available = (
    MODEL_STATE[
        "evaluation_report_path"
    ].exists()
)

asset1, asset2, asset3 = (
    st.columns(
        3
    )
)


with asset1:
    show_status_card(
        title="Raw Duolingo Dataset",
        badge=status_badge(
            (
                "Available"
                if raw_available
                else "Not Local"
            ),
            (
                "success"
                if raw_available
                else "info"
            )
        ),
        description=(
            "Training asset. It is not required "
            "for ordinary runtime prediction after "
            "a validated model has been produced."
        )
    )


with asset2:
    show_status_card(
        title="Processed Dataset",
        badge=status_badge(
            (
                "Available"
                if processed_available
                else "Not Local"
            ),
            (
                "success"
                if processed_available
                else "info"
            )
        ),
        description=(
            "Prepared research dataset used by "
            "the model-training and evaluation "
            "pipeline."
        )
    )


with asset3:
    show_status_card(
        title="Evaluation Report",
        badge=status_badge(
            (
                "Reproduced"
                if evaluation_reproduced
                else (
                    "Available"
                    if evaluation_available
                    else "Not Available"
                )
            ),
            (
                "success"
                if evaluation_reproduced
                else "info"
            )
        ),
        description=(
            "Stores diagnostic and reproducibility "
            "results for the trained public "
            "bootstrap model."
        )
    )


# experiment explanation

with st.expander(
    "🔬 How the MemoryMap experiment works"
):
    st.markdown(
        """
### 1. Record a study session

MemoryMap records genuine learning context such as study duration, topic difficulty, confidence, previous study history and previous recall when available.

### 2. Generate a prospective prediction

Retention predictions are generated **before** the delayed recall result is known.

### 3. Schedule revision

The fixed schedule provides the baseline. Personalized and adaptive scheduling are used only when their required genuine evidence exists.

### 4. Wait for delayed recall

Recall is measured around planned checkpoints such as approximately **1 day, 3 days and 7 days**.

### 5. Measure actual performance

A completed recall test produces the observed recall value.

### 6. Match prediction and outcome

Only predictions created before the quiz are eligible. Their prediction horizon must also be close to the actual recall delay for strict evaluation.

### 7. Compare prediction methods

MAE, RMSE, R², bias and accuracy bands are calculated from real prediction-recall pairs.

When competing methods are compared directly, MemoryMap uses common observations so one model is not rewarded merely because it happened to have an easier evaluation subset.

### 8. Build personalization over time

MemoryMap-specific variables become useful for personalized modelling only after genuine longitudinal observations have been collected. The public Duolingo dataset bootstraps the system but does not substitute for real MemoryMap learner data.
        """
    )


# development status

with st.expander(
    "⚙️ Development Status"
):
    st.markdown(
        f"""
**Application:** {APP_NAME}

**Current version:** {APP_VERSION}

**Interface:** Streamlit

**Database:** SQLite

**Machine Learning:** scikit-learn

**Data processing:** pandas + NumPy

**Visualization:** Plotly

### Current ML state

**Artifact validated:** {"Yes" if model_ready else "No"}

**Evaluation reproduced:** {"Yes" if evaluation_reproduced else "No"}

**MemoryMap personalized model:** {"Available" if personal_model else "Not trained yet"}

### Project Team

- Prajwal C Pradhan
- Namhay D Wangchuk
- M. Sanjana Reddy
- Malavika Vinod

The Duolingo spaced-repetition learning traces are used as public bootstrap data. MemoryMap-specific variables such as study duration, topic difficulty, confidence and delayed recall come from genuine MemoryMap usage.
        """
    )


# footer

render_html(
    f"""
    <div class="mm-footer">
        {escape(str(APP_ICON))}
        <strong>{escape(str(APP_NAME))}</strong>

        &nbsp;•&nbsp;

        Version {escape(str(APP_VERSION))}

        <br>

        Personalized Memory Retention
        &amp; Adaptive Revision Research System

        <br><br>

        <strong>Project Team</strong><br>
        Prajwal C Pradhan •
        Namhay D Wangchuk •
        M. Sanjana Reddy •
        Malavika Vinod
    </div>
    """
)