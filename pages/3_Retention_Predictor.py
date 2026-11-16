from datetime import datetime, timedelta
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    DEFAULT_HALF_LIFE_DAYS,
    TARGET_RETENTION_THRESHOLD,
    RECALL_CHECKPOINT_DAYS,
)

from core.forgetting_curve import (
    calculate_retention,
)

from core.predictor import (
    FIXED_MODE,
    PERSONALIZED_MODE,
    predict_retention,
)

from database import (
    initialize_database,
    add_prediction,
    get_predictions_for_session,
)

from ml.load_model import (
    get_model_information,
    load_metadata,
)

from services.study_service import (
    get_study_page_data,
    list_topic_sessions,
    get_topic_profile,
    create_memorymap_feature_values,
    parse_datetime,
)

from utils import (
    format_datetime,
    format_percentage,
    format_study_duration,
    get_confidence_label,
    get_difficulty_label,
    get_retention_status,
)


st.set_page_config(
    page_title="Retention Predictor | MemoryMap",
    page_icon="📉",
    layout="wide",
)


# initialization

initialize_database()

if "student_id" not in st.session_state:
    st.session_state.student_id = None

if "student_name" not in st.session_state:
    st.session_state.student_name = None

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = None


# numeric helpers

def safe_float(value, default=None):
    if value is None:
        return default

    try:
        value = float(value)

    except (
        TypeError,
        ValueError
    ):
        return default

    if not math.isfinite(value):
        return default

    return value


def values_match(
    first,
    second,
    tolerance=1e-9
):
    if (
        first is None
        and second is None
    ):
        return True

    if (
        first is None
        or second is None
    ):
        return False

    first = safe_float(first)
    second = safe_float(second)

    if (
        first is None
        or second is None
    ):
        return False

    return math.isclose(
        first,
        second,
        rel_tol=tolerance,
        abs_tol=tolerance
    )


# record helpers

def get_record_id(
    record,
    canonical_name
):
    if not record:
        return None

    value = record.get(
        canonical_name
    )

    if value is None:
        value = record.get(
            "id"
        )

    return value


def get_subject_name(topic):
    if not topic:
        return "No Subject"

    subject = topic.get(
        "subject_name"
    )

    if not subject:
        subject = topic.get(
            "subject"
        )

    return (
        str(subject)
        if subject
        else "No Subject"
    )


# prediction result helpers

def get_prediction_value(result):
    if not isinstance(
        result,
        dict
    ):
        return None

    possible_keys = [
        "retention",
        "predicted_retention",
    ]

    for key in possible_keys:
        value = safe_float(
            result.get(
                key
            )
        )

        if value is None:
            continue

        if 0 <= value <= 1:
            return value

    return None


def get_prediction_half_life(result):
    if not isinstance(
        result,
        dict
    ):
        return None

    possible_keys = [
        "half_life",
        "half_life_days",
        "estimated_half_life",
        "predicted_half_life",
    ]

    for key in possible_keys:
        value = safe_float(
            result.get(
                key
            )
        )

        if (
            value is not None
            and value > 0
        ):
            return value

    return None


def get_prediction_source(
    result,
    default="Unknown"
):
    if not isinstance(
        result,
        dict
    ):
        return default

    source = result.get(
        "source"
    )

    if not source:
        return default

    return str(source)


def get_prediction_message(
    result,
    default=None
):
    if not isinstance(
        result,
        dict
    ):
        return default

    for key in [
        "message",
        "error",
        "reason",
    ]:
        message = result.get(
            key
        )

        if message:
            return str(
                message
            )

    return default


# flash messages

def show_flash_message():
    flash = st.session_state.pop(
        "prediction_flash",
        None
    )

    if not flash:
        return

    message = flash.get(
        "message",
        ""
    )

    message_type = flash.get(
        "type",
        "info"
    )

    if message_type == "success":
        st.success(message)

    elif message_type == "warning":
        st.warning(message)

    elif message_type == "error":
        st.error(message)

    else:
        st.info(message)

    prediction_id = flash.get(
        "prediction_id"
    )

    if prediction_id is not None:
        st.caption(
            f"Prediction ID: "
            f"{prediction_id}"
        )

        st.page_link(
            "pages/4_Revision_Scheduler.py",
            label=(
                "🗓️ Continue to "
                "Revision Scheduler"
            )
        )


# active student

def get_active_student():
    student_id = st.session_state.get(
        "student_id"
    )

    if student_id is None:
        st.warning(
            "No student profile is "
            "currently selected."
        )

        st.info(
            "Open the Dashboard and create "
            "or select a student profile first."
        )

        return None

    page_data = (
        get_study_page_data(
            student_id
        )
    )

    if not page_data.get(
        "success",
        False
    ):
        st.error(
            page_data.get(
                "message",
                (
                    "The selected student "
                    "could not be loaded."
                )
            )
        )

        return None

    student = page_data.get(
        "student"
    )

    if not student:
        st.error(
            "Student profile could "
            "not be found."
        )

        return None

    return {
        "student_id":
            student_id,

        "student":
            student,

        "topics":
            page_data.get(
                "topics",
                []
            ),
    }


# topic selection

def select_topic(
    student_id,
    topics
):
    st.subheader(
        "Select Study Data"
    )

    usable_topics = []

    for topic in topics:
        topic_id = get_record_id(
            topic,
            "topic_id"
        )

        if topic_id is not None:
            usable_topics.append(
                topic
            )

    if not usable_topics:
        st.warning(
            "No topics have been "
            "created yet."
        )

        st.page_link(
            "pages/2_Study_Session.py",
            label=(
                "📚 Create a Topic "
                "in Study Session"
            )
        )

        return None

    topic_map = {}

    for topic in usable_topics:
        topic_id = get_record_id(
            topic,
            "topic_id"
        )

        topic_name = topic.get(
            "topic_name",
            "Unnamed Topic"
        )

        subject_name = (
            get_subject_name(
                topic
            )
        )

        topic_map[
            topic_id
        ] = (
            f"{topic_name} • "
            f"{subject_name}"
        )

    topic_ids = list(
        topic_map.keys()
    )

    default_topic = (
        st.session_state.get(
            "selected_topic"
        )
    )

    if default_topic not in topic_ids:
        default_topic = (
            topic_ids[0]
        )

    selected_index = (
        topic_ids.index(
            default_topic
        )
    )

    selected_topic_id = (
        st.selectbox(
            "Topic",
            options=topic_ids,
            index=selected_index,
            format_func=lambda topic_id:
                topic_map[
                    topic_id
                ],
            key="predictor_topic"
        )
    )

    st.session_state[
        "selected_topic"
    ] = selected_topic_id

    return selected_topic_id


# session selection

def session_sort_key(session):
    parsed = parse_datetime(
        session.get(
            "studied_at"
        )
    )

    if parsed is None:
        return datetime.min

    return parsed


def select_study_session(
    student_id,
    topic_id
):
    sessions = list_topic_sessions(
        student_id,
        topic_id
    )

    if not sessions:
        st.warning(
            "This topic does not have "
            "a study session yet."
        )

        st.page_link(
            "pages/2_Study_Session.py",
            label=(
                "📚 Record a Study Session"
            )
        )

        return None

    sessions = sorted(
        sessions,
        key=session_sort_key,
        reverse=True
    )

    session_map = {}

    for session in sessions:
        session_id = get_record_id(
            session,
            "session_id"
        )

        if session_id is None:
            continue

        session_map[
            session_id
        ] = (
            f"Session {session_id} • "
            f"{format_datetime(session.get('studied_at'))}"
        )

    session_ids = list(
        session_map.keys()
    )

    if not session_ids:
        return None

    default_session = (
        st.session_state.get(
            "selected_session_id"
        )
    )

    if default_session not in session_ids:
        default_session = (
            session_ids[0]
        )

    selected_index = (
        session_ids.index(
            default_session
        )
    )

    selected_session_id = (
        st.selectbox(
            "Study Session",
            options=session_ids,
            index=selected_index,
            format_func=lambda session_id:
                session_map[
                    session_id
                ],
            key="predictor_session"
        )
    )

    st.session_state[
        "selected_session_id"
    ] = selected_session_id

    for session in sessions:
        session_id = get_record_id(
            session,
            "session_id"
        )

        if (
            session_id
            == selected_session_id
        ):
            return session

    return None


# session information

def show_session_information(
    session,
    topic
):
    st.subheader(
        "Selected Study Session"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    study_duration = safe_float(
        session.get(
            "study_duration"
        ),
        0
    )

    difficulty = safe_float(
        topic.get(
            "difficulty"
        ),
        3
    )

    confidence = safe_float(
        session.get(
            "confidence_level"
        ),
        3
    )

    revision_count = session.get(
        "revision_count"
    )

    with col1:
        st.metric(
            "Study Duration",
            format_study_duration(
                study_duration
            )
        )

    with col2:
        st.metric(
            "Difficulty",
            f"{int(difficulty)}/5"
        )

        st.caption(
            get_difficulty_label(
                int(
                    difficulty
                )
            )
        )

    with col3:
        st.metric(
            "Confidence",
            f"{int(confidence)}/5"
        )

        st.caption(
            get_confidence_label(
                int(
                    confidence
                )
            )
        )

    with col4:
        st.metric(
            "Previous Sessions",
            (
                revision_count
                if revision_count
                is not None
                else "N/A"
            )
        )

    previous_score = safe_float(
        session.get(
            "previous_quiz_score"
        )
    )

    if previous_score is None:
        previous_score_text = (
            "No previous recall available"
        )

    else:
        previous_score_text = (
            f"{previous_score:.1f}%"
        )

    st.caption(
        "Studied at: "
        f"{format_datetime(session.get('studied_at'))}"
    )

    st.caption(
        "Previous recall used when this "
        f"session was saved: {previous_score_text}"
    )


# prediction interval

def select_prediction_interval():
    st.subheader(
        "Prediction Interval"
    )

    checkpoint_days = []

    for value in (
        RECALL_CHECKPOINT_DAYS
    ):
        value = safe_float(
            value
        )

        if (
            value is not None
            and value > 0
        ):
            checkpoint_days.append(
                value
            )

    checkpoint_days = sorted(
        set(
            checkpoint_days
        )
    )

    option_map = {
        f"{day:g} Day":
            day

        for day in checkpoint_days
    }

    option_labels = list(
        option_map.keys()
    )

    option_labels.append(
        "Custom"
    )

    selected = st.radio(
        "Predict retention after:",
        option_labels,
        horizontal=True,
        key="prediction_interval_mode"
    )

    if selected != "Custom":
        return option_map[
            selected
        ]

    return float(
        st.number_input(
            "Days after study session",
            min_value=0.25,
            max_value=365.0,
            value=2.0,
            step=0.25,
            key="custom_prediction_days"
        )
    )


# prediction timing

def get_prediction_target_time(
    session,
    prediction_days
):
    studied_at = parse_datetime(
        session.get(
            "studied_at"
        )
    )

    if studied_at is None:
        return None

    return (
        studied_at
        + timedelta(
            days=float(
                prediction_days
            )
        )
    )


def get_matching_now(
    target_datetime
):
    if (
        target_datetime is not None
        and target_datetime.tzinfo
        is not None
    ):
        return datetime.now(
            target_datetime.tzinfo
        )

    return datetime.now()


def prediction_is_prospective(
    session,
    prediction_days
):
    target_time = (
        get_prediction_target_time(
            session,
            prediction_days
        )
    )

    if target_time is None:
        return False

    return (
        target_time
        > get_matching_now(
            target_time
        )
    )


def show_prediction_timing(
    session,
    prediction_days
):
    target_time = (
        get_prediction_target_time(
            session,
            prediction_days
        )
    )

    if target_time is None:
        st.error(
            "The study timestamp is invalid."
        )

        return False

    future_prediction = (
        prediction_is_prospective(
            session,
            prediction_days
        )
    )

    st.caption(
        "Prediction target: "
        f"{format_datetime(target_time)}"
    )

    if future_prediction:
        st.success(
            "This prediction target is still "
            "in the future and can be saved "
            "as a prospective prediction."
        )

    else:
        st.warning(
            "This prediction target has already "
            "passed. You can inspect the estimate, "
            "but MemoryMap will not save it as a "
            "new prospective research prediction. "
            "Choose a longer interval or a more "
            "recent study session."
        )

    return future_prediction


# core prediction execution

def run_core_prediction(
    mode,
    session,
    topic,
    prediction_days,
    extra_features=None
):
    try:
        result = predict_retention(
            mode=mode,
            session=session,
            topic=topic,
            prediction_days=(
                prediction_days
            ),
            extra_features=(
                extra_features
            )
        )

    except Exception as error:
        return None, str(
            error
        )

    if not isinstance(
        result,
        dict
    ):
        return (
            None,
            (
                "Prediction engine returned "
                "an unexpected result."
            )
        )

    retention = (
        get_prediction_value(
            result
        )
    )

    if retention is None:
        return (
            result,
            get_prediction_message(
                result,
                (
                    "No valid retention "
                    "prediction was produced."
                )
            )
        )

    return result, None


# fixed prediction

def show_fixed_prediction(
    session,
    topic,
    prediction_days
):
    st.subheader(
        "Fixed Forgetting Curve"
    )

    result, error = (
        run_core_prediction(
            mode=FIXED_MODE,
            session=session,
            topic=topic,
            prediction_days=(
                prediction_days
            )
        )
    )

    if error:
        st.error(
            "Fixed prediction could "
            "not be generated."
        )

        st.caption(
            error
        )

        return None

    retention = (
        get_prediction_value(
            result
        )
    )

    half_life = (
        get_prediction_half_life(
            result
        )
    )

    if half_life is None:
        half_life = (
            DEFAULT_HALF_LIFE_DAYS
        )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Predicted Retention",
            format_percentage(
                retention
            )
        )

    with col2:
        st.metric(
            "Retention Status",
            get_retention_status(
                retention
            )
        )

    with col3:
        st.metric(
            "Half-Life",
            f"{half_life:.2f} days"
        )

    if (
        retention
        < TARGET_RETENTION_THRESHOLD
    ):
        st.warning(
            "Predicted retention is below "
            f"the MemoryMap threshold of "
            f"{TARGET_RETENTION_THRESHOLD * 100:.0f}%."
        )

    else:
        st.success(
            "Predicted retention remains "
            "above the current revision threshold."
        )

    return result


# forgetting curve graph

def show_forgetting_curve(
    prediction_days
):
    st.subheader(
        "Fixed Retention Curve"
    )

    maximum_days = max(
        7.0,
        float(
            prediction_days
        )
    )

    number_of_points = 150

    days = [
        maximum_days
        * index
        / (
            number_of_points
            - 1
        )

        for index
        in range(
            number_of_points
        )
    ]

    retention_values = [
        calculate_retention(
            day,
            DEFAULT_HALF_LIFE_DAYS
        )
        * 100

        for day
        in days
    ]

    dataframe = pd.DataFrame(
        {
            "Days":
                days,

            "Retention":
                retention_values,
        }
    )

    figure = px.line(
        dataframe,
        x="Days",
        y="Retention",
        title=(
            "Fixed Forgetting Curve"
        )
    )

    figure.add_hline(
        y=(
            TARGET_RETENTION_THRESHOLD
            * 100
        ),
        line_dash="dash",
        annotation_text=(
            "Revision Threshold"
        )
    )

    selected_retention = (
        calculate_retention(
            prediction_days,
            DEFAULT_HALF_LIFE_DAYS
        )
        * 100
    )

    figure.add_scatter(
        x=[
            prediction_days
        ],
        y=[
            selected_retention
        ],
        mode="markers",
        name="Selected Prediction"
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title=(
            "Predicted Retention (%)"
        )
    )

    figure.update_xaxes(
        title="Days Since Study"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )


# model status

def get_model_context(
    session,
    prediction_days
):
    information = (
        get_model_information()
    )

    try:
        metadata = load_metadata()

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError,
    ):
        metadata = {}

    session_id = get_record_id(
        session,
        "session_id"
    )

    feature_result = (
        create_memorymap_feature_values(
            session_id,
            prediction_days
        )
    )

    features = {}

    if feature_result.get(
        "success",
        False
    ):
        features = (
            feature_result.get(
                "features"
            )
            or {}
        )

    expected_features = (
        information.get(
            "feature_columns",
            []
        )
    )

    missing_features = []

    for feature in expected_features:
        if feature not in features:
            missing_features.append(
                feature
            )

            continue

        if features[
            feature
        ] is None:
            missing_features.append(
                feature
            )

    return {
        "information":
            information,

        "metadata":
            metadata,

        "features":
            features,

        "missing_features":
            missing_features,

        "memorymap_personal_model":
            bool(
                metadata.get(
                    "memorymap_personal_model",
                    False
                )
            ),
    }


def show_model_status(
    model_context
):
    st.subheader(
        "Machine-Learning Model"
    )

    information = (
        model_context[
            "information"
        ]
    )

    metadata = (
        model_context[
            "metadata"
        ]
    )

    if not information.get(
        "available",
        False
    ):
        st.warning(
            "The trained model artifact "
            "is not currently available."
        )

        st.caption(
            information.get(
                "message",
                (
                    "Train the model before "
                    "using ML prediction."
                )
            )
        )

        return

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Model",
            information.get(
                "model_name",
                "Unknown"
            )
        )

    with col2:
        st.metric(
            "Feature Set",
            information.get(
                "feature_set",
                "Unknown"
            )
        )

    with col3:
        st.metric(
            "Role",
            (
                "Personalized"
                if model_context[
                    "memorymap_personal_model"
                ]
                else "Public Bootstrap"
            )
        )

    expected_features = (
        information.get(
            "feature_columns",
            []
        )
    )

    if expected_features:
        st.caption(
            "Model inputs: "
            + ", ".join(
                expected_features
            )
        )

    if not model_context[
        "memorymap_personal_model"
    ]:
        st.info(
            "The saved ML artifact is the "
            "public bootstrap retention model. "
            "It is not being labelled as a "
            "personalized MemoryMap model."
        )

        model_role = metadata.get(
            "model_role"
        )

        if model_role:
            st.caption(
                model_role
            )

    if model_context[
        "missing_features"
    ]:
        st.warning(
            "This study session does not "
            "contain all inputs expected by "
            "the saved model."
        )

        st.caption(
            "Unavailable model features: "
            + ", ".join(
                model_context[
                    "missing_features"
                ]
            )
        )


# personalized prediction

def show_personalized_prediction(
    session,
    topic,
    prediction_days,
    model_context
):
    st.subheader(
        "Personalized Prediction"
    )

    information = (
        model_context[
            "information"
        ]
    )

    if not information.get(
        "available",
        False
    ):
        st.info(
            "Personalized prediction is "
            "not available yet."
        )

        return None

    if not model_context[
        "memorymap_personal_model"
    ]:
        st.info(
            "No MemoryMap-personalized model "
            "has been trained yet."
        )

        st.caption(
            "The current Random Forest artifact "
            "is a public Duolingo bootstrap model. "
            "MemoryMap will not pretend that its "
            "Duolingo history features are the same "
            "as study duration, confidence, topic "
            "difficulty or revision count."
        )

        return None

    missing_features = (
        model_context[
            "missing_features"
        ]
    )

    if missing_features:
        st.info(
            "Personalized prediction cannot "
            "be generated for this session."
        )

        st.caption(
            "Missing genuine inputs: "
            + ", ".join(
                missing_features
            )
        )

        return None

    result, error = (
        run_core_prediction(
            mode=PERSONALIZED_MODE,
            session=session,
            topic=topic,
            prediction_days=(
                prediction_days
            ),
            extra_features=(
                model_context[
                    "features"
                ]
            )
        )
    )

    if error:
        st.info(
            "Personalized prediction is "
            "not available for this session."
        )

        st.caption(
            error
        )

        return None

    retention = (
        get_prediction_value(
            result
        )
    )

    half_life = (
        get_prediction_half_life(
            result
        )
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Personalized Retention",
            format_percentage(
                retention
            )
        )

    with col2:
        st.metric(
            "Retention Status",
            get_retention_status(
                retention
            )
        )

    with col3:
        st.metric(
            "Estimated Half-Life",
            (
                f"{half_life:.2f} days"
                if half_life
                is not None
                else "N/A"
            )
        )

    st.caption(
        "Source: "
        + get_prediction_source(
            result,
            "MemoryMap personalized model"
        )
    )

    return result


# model comparison

def show_prediction_comparison(
    fixed_result,
    personalized_result
):
    fixed_value = (
        get_prediction_value(
            fixed_result
        )
    )

    personalized_value = (
        get_prediction_value(
            personalized_result
        )
    )

    if (
        fixed_value is None
        or personalized_value is None
    ):
        return

    st.subheader(
        "Prediction Comparison"
    )

    comparison_data = (
        pd.DataFrame(
            [
                {
                    "Model":
                        (
                            "Fixed "
                            "Forgetting Curve"
                        ),

                    "Predicted Retention":
                        fixed_value
                        * 100,
                },
                {
                    "Model":
                        (
                            "Personalized "
                            "MemoryMap Model"
                        ),

                    "Predicted Retention":
                        personalized_value
                        * 100,
                },
            ]
        )
    )

    figure = px.bar(
        comparison_data,
        x="Model",
        y="Predicted Retention",
        text_auto=".1f",
        title=(
            "Fixed vs Personalized "
            "Retention Prediction"
        )
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title=(
            "Predicted Retention (%)"
        )
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )

    difference = abs(
        fixed_value
        - personalized_value
    )

    st.write(
        "Difference between predictions: "
        f"**{difference * 100:.1f} "
        "percentage points**"
    )


# duplicate protection

def prediction_already_exists(
    session_id,
    prediction_days,
    fixed_prediction,
    personalized_prediction
):
    predictions = (
        get_predictions_for_session(
            session_id
        )
    )

    for prediction in predictions:
        stored_days = safe_float(
            prediction.get(
                "prediction_for_days"
            )
        )

        if stored_days is None:
            continue

        if not math.isclose(
            stored_days,
            float(
                prediction_days
            ),
            rel_tol=1e-9,
            abs_tol=1e-6
        ):
            continue

        if not values_match(
            prediction.get(
                "fixed_prediction"
            ),
            fixed_prediction
        ):
            continue

        if not values_match(
            prediction.get(
                "personalized_prediction"
            ),
            personalized_prediction
        ):
            continue

        return True

    return False


# prediction storage

def save_prediction(
    session,
    prediction_days,
    fixed_result,
    personalized_result
):
    session_id = get_record_id(
        session,
        "session_id"
    )

    if session_id is None:
        st.error(
            "Study Session ID is missing."
        )

        return

    fixed_prediction = (
        get_prediction_value(
            fixed_result
        )
    )

    personalized_prediction = (
        get_prediction_value(
            personalized_result
        )
    )

    if fixed_prediction is None:
        st.error(
            "A valid fixed prediction "
            "is required before saving."
        )

        return

    if prediction_already_exists(
        session_id=session_id,
        prediction_days=prediction_days,
        fixed_prediction=fixed_prediction,
        personalized_prediction=(
            personalized_prediction
        )
    ):
        st.warning(
            "This exact prediction has "
            "already been saved."
        )

        return

    personalized_half_life = (
        get_prediction_half_life(
            personalized_result
        )
    )

    fixed_half_life = (
        get_prediction_half_life(
            fixed_result
        )
    )

    predicted_half_life = (
        personalized_half_life
        or fixed_half_life
        or DEFAULT_HALF_LIFE_DAYS
    )

    if personalized_prediction is not None:
        prediction_mode = (
            "Fixed + Personalized"
        )

    else:
        prediction_mode = (
            "Fixed Forgetting Curve"
        )

    try:
        prediction_id = add_prediction(
            session_id=session_id,
            fixed_prediction=(
                fixed_prediction
            ),
            personalized_prediction=(
                personalized_prediction
            ),
            adaptive_prediction=None,
            predicted_half_life=(
                predicted_half_life
            ),
            prediction_for_days=(
                float(
                    prediction_days
                )
            ),
            prediction_mode=(
                prediction_mode
            )
        )

    except Exception as error:
        st.error(
            "Prediction could not be saved."
        )

        st.caption(
            str(
                error
            )
        )

        return

    st.session_state[
        "prediction_flash"
    ] = {
        "type":
            "success",

        "message":
            (
                f"Prediction for "
                f"{prediction_days:g} day(s) "
                "saved successfully."
            ),

        "prediction_id":
            prediction_id,
    }

    st.rerun()


# prediction history

def show_prediction_history(
    session_id
):
    predictions = (
        get_predictions_for_session(
            session_id
        )
    )

    if not predictions:
        return

    st.subheader(
        "Saved Predictions"
    )

    rows = []

    for prediction in predictions:
        fixed_value = safe_float(
            prediction.get(
                "fixed_prediction"
            )
        )

        personalized_value = safe_float(
            prediction.get(
                "personalized_prediction"
            )
        )

        adaptive_value = safe_float(
            prediction.get(
                "adaptive_prediction"
            )
        )

        half_life = safe_float(
            prediction.get(
                "predicted_half_life"
            )
        )

        rows.append(
            {
                "Prediction ID":
                    get_record_id(
                        prediction,
                        "prediction_id"
                    ),

                "Days":
                    prediction.get(
                        "prediction_for_days"
                    ),

                "Fixed":
                    (
                        f"{fixed_value * 100:.1f}%"
                        if fixed_value
                        is not None
                        else "N/A"
                    ),

                "Personalized":
                    (
                        f"{personalized_value * 100:.1f}%"
                        if personalized_value
                        is not None
                        else "N/A"
                    ),

                "Adaptive":
                    (
                        f"{adaptive_value * 100:.1f}%"
                        if adaptive_value
                        is not None
                        else "N/A"
                    ),

                "Half-Life":
                    (
                        f"{half_life:.2f} d"
                        if half_life
                        is not None
                        else "N/A"
                    ),

                "Mode":
                    prediction.get(
                        "prediction_mode",
                        "Unknown"
                    ),

                "Created":
                    format_datetime(
                        prediction.get(
                            "created_at"
                        )
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        width="stretch",
        hide_index=True
    )


# experiment explanation

def show_experiment_information():
    with st.expander(
        "How will MemoryMap evaluate these predictions?"
    ):
        st.markdown(
            """
A saved prediction is **not treated as proof that a model is correct**.

For the research experiment:

1. A prediction is generated before the recall test.
2. The student completes a delayed recall test near the selected interval.
3. MemoryMap matches that test with a prediction created before the quiz.
4. Predicted retention is compared with actual recall.
5. MAE, RMSE, bias and other metrics are calculated across repeated observations.

The current public Random Forest is a **bootstrap model**, not a personalized MemoryMap model. It was trained using Duolingo history variables. MemoryMap does not fabricate those variables from confidence, revision count or study duration.

The personalized model becomes scientifically meaningful only after genuine longitudinal MemoryMap observations have been collected and used for training.
            """
        )


# page

def show_page():
    show_flash_message()

    active = get_active_student()

    if active is None:
        return

    student_id = active[
        "student_id"
    ]

    student = active[
        "student"
    ]

    topics = active[
        "topics"
    ]

    student_name = (
        student.get(
            "name"
        )
        or st.session_state.get(
            "student_name"
        )
        or "Student"
    )

    st.session_state[
        "student_name"
    ] = student_name

    st.title(
        "📉 Retention Predictor"
    )

    st.write(
        "Estimate future retention for "
        f"**{student_name}**."
    )

    st.caption(
        "Predictions saved here can later "
        "be matched with delayed recall tests."
    )

    st.divider()

    topic_id = select_topic(
        student_id,
        topics
    )

    if topic_id is None:
        return

    session = select_study_session(
        student_id,
        topic_id
    )

    if session is None:
        return

    topic = get_topic_profile(
        topic_id
    )

    if topic is None:
        st.error(
            "Topic information could "
            "not be loaded."
        )

        return

    st.divider()

    show_session_information(
        session,
        topic
    )

    st.divider()

    prediction_days = (
        select_prediction_interval()
    )

    prospective = (
        show_prediction_timing(
            session,
            prediction_days
        )
    )

    st.divider()

    fixed_result = (
        show_fixed_prediction(
            session,
            topic,
            prediction_days
        )
    )

    if fixed_result is None:
        return

    show_forgetting_curve(
        prediction_days
    )

    st.divider()

    model_context = (
        get_model_context(
            session,
            prediction_days
        )
    )

    show_model_status(
        model_context
    )

    personalized_result = (
        show_personalized_prediction(
            session,
            topic,
            prediction_days,
            model_context
        )
    )

    show_prediction_comparison(
        fixed_result,
        personalized_result
    )

    st.divider()

    st.subheader(
        "Save Prediction"
    )

    if not prospective:
        st.caption(
            "Saving is disabled because "
            "the selected prediction target "
            "has already passed."
        )

    save_clicked = st.button(
        "Save Prospective Prediction",
        width="stretch",
        disabled=(
            not prospective
        )
    )

    if save_clicked:
        save_prediction(
            session=session,
            prediction_days=(
                prediction_days
            ),
            fixed_result=(
                fixed_result
            ),
            personalized_result=(
                personalized_result
            )
        )

    session_id = get_record_id(
        session,
        "session_id"
    )

    if session_id is not None:
        show_prediction_history(
            session_id
        )

    st.divider()

    show_experiment_information()


show_page()
