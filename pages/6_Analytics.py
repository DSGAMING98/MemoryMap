import math

import pandas as pd
import plotly.express as px
import streamlit as st

from database import (
    initialize_database,
)

from services.student_service import (
    list_students,
)

from services.analytics_service import (
    MODEL_COLUMNS,
    get_student_analytics,
)


st.set_page_config(
    page_title="Analytics | MemoryMap",
    page_icon="📊",
    layout="wide",
)


# initialization

initialize_database()

STRICT_TOLERANCE_DAYS = 0.5

if "student_id" not in st.session_state:
    st.session_state.student_id = None

if "student_name" not in st.session_state:
    st.session_state.student_name = None

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = None


# helpers

def safe_float(
    value,
    default=None
):
    if value is None:
        return default

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(value):
        return default

    return value


def safe_int(
    value,
    default=0
):
    if value is None:
        return default

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def ensure_dataframe(
    value
):
    if isinstance(
        value,
        pd.DataFrame
    ):
        return value.copy()

    if value is None:
        return pd.DataFrame()

    try:
        return pd.DataFrame(
            value
        )

    except Exception:
        return pd.DataFrame()


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


def format_percentage(
    value,
    decimals=1
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    return (
        f"{value * 100:.{decimals}f}%"
    )


def format_percentage_points(
    value,
    decimals=2
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    return (
        f"{value:.{decimals}f} pp"
    )


def format_number(
    value,
    decimals=3
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    return (
        f"{value:.{decimals}f}"
    )


def numeric_series(
    dataframe,
    column
):
    if (
        dataframe.empty
        or column
        not in dataframe.columns
    ):
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        dataframe[
            column
        ],
        errors="coerce"
    )


# student selection

def select_student():
    students = list_students()

    if not students:
        st.warning(
            "No student profiles exist yet."
        )

        st.info(
            "Create a profile from the "
            "Dashboard first."
        )

        st.page_link(
            "pages/1_Dashboard.py",
            label="🧠 Open Dashboard"
        )

        return None

    student_map = {}

    for student in students:
        student_id = get_record_id(
            student,
            "student_id"
        )

        if student_id is None:
            continue

        name = (
            student.get(
                "name"
            )
            or "Unnamed Student"
        )

        email = student.get(
            "email"
        )

        label = (
            f"{name} • {email}"
            if email
            else str(name)
        )

        student_map[
            student_id
        ] = {
            "label":
                label,

            "name":
                str(name),
        }

    student_ids = list(
        student_map.keys()
    )

    if not student_ids:
        st.error(
            "No valid student profiles "
            "could be loaded."
        )

        return None

    current_student = (
        st.session_state.get(
            "student_id"
        )
    )

    if current_student in student_ids:
        selected_index = (
            student_ids.index(
                current_student
            )
        )

    else:
        selected_index = 0

    selected_student = (
        st.sidebar.selectbox(
            "Student Profile",
            options=student_ids,
            index=selected_index,
            format_func=lambda student_id:
                student_map[
                    student_id
                ][
                    "label"
                ],
            key="analytics_student_selector"
        )
    )

    previous_student = (
        st.session_state.get(
            "student_id"
        )
    )

    if (
        previous_student is not None
        and previous_student
        != selected_student
    ):
        st.session_state.selected_topic = None
        st.session_state.selected_session_id = None

    st.session_state.student_id = (
        selected_student
    )

    st.session_state.student_name = (
        student_map[
            selected_student
        ][
            "name"
        ]
    )

    return selected_student


# analytics loading

def load_analytics(
    student_id
):
    result = get_student_analytics(
        student_id,
        tolerance_days=(
            STRICT_TOLERANCE_DAYS
        )
    )

    if not result.get(
        "success",
        False
    ):
        st.error(
            result.get(
                "message",
                "Analytics could not be loaded."
            )
        )

        return None

    return result


# header

def show_header(
    analytics
):
    student = (
        analytics.get(
            "student"
        )
        or {}
    )

    name = (
        student.get(
            "name"
        )
        or st.session_state.get(
            "student_name"
        )
        or "Student"
    )

    st.session_state.student_name = name

    st.title(
        "📊 Analytics"
    )

    st.write(
        f"Learning and recall analytics "
        f"for **{name}**."
    )

    st.caption(
        "Descriptive learning charts use genuine "
        "stored observations. Prediction evaluation "
        "uses only predictions made before the recall "
        "test and matched to the testing delay."
    )


# summary

def show_summary(
    analytics
):
    sessions = ensure_dataframe(
        analytics.get(
            "sessions"
        )
    )

    quizzes = ensure_dataframe(
        analytics.get(
            "quizzes"
        )
    )

    evaluation = ensure_dataframe(
        analytics.get(
            "evaluation_data"
        )
    )

    research_status = (
        analytics.get(
            "research_status"
        )
        or {}
    )

    learning = ensure_dataframe(
        analytics.get(
            "learning_data"
        )
    )

    average_recall = None
    average_delay = None

    if (
        not quizzes.empty
        and "recall_accuracy"
        in quizzes.columns
    ):
        recall = pd.to_numeric(
            quizzes[
                "recall_accuracy"
            ],
            errors="coerce"
        ).dropna()

        if not recall.empty:
            average_recall = float(
                recall.mean()
            )

    if (
        not learning.empty
        and "delay_days"
        in learning.columns
    ):
        delay = pd.to_numeric(
            learning[
                "delay_days"
            ],
            errors="coerce"
        ).dropna()

        if not delay.empty:
            average_delay = float(
                delay.mean()
            )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:
        st.metric(
            "Study Sessions",
            len(sessions)
        )

    with col2:
        st.metric(
            "Recall Tests",
            len(quizzes)
        )

    with col3:
        st.metric(
            "Strict Matches",
            len(evaluation)
        )

    with col4:
        st.metric(
            "Average Recall",
            (
                format_percentage(
                    average_recall
                )
                if average_recall
                is not None
                else "N/A"
            )
        )

    with col5:
        st.metric(
            "Average Delay",
            (
                f"{average_delay:.2f} days"
                if average_delay
                is not None
                else "N/A"
            )
        )

    stage = research_status.get(
        "research_stage"
    )

    if stage:
        st.caption(
            f"Current research stage: **{stage}**"
        )


# data readiness

def show_data_readiness(
    analytics
):
    sessions = ensure_dataframe(
        analytics.get(
            "sessions"
        )
    )

    quizzes = ensure_dataframe(
        analytics.get(
            "quizzes"
        )
    )

    evaluation = ensure_dataframe(
        analytics.get(
            "evaluation_data"
        )
    )

    st.subheader(
        "Data Readiness"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        if sessions.empty:
            st.warning(
                "No study session yet."
            )

        else:
            st.success(
                f"{len(sessions)} study "
                "session(s) available."
            )

    with col2:
        if quizzes.empty:
            st.info(
                "No actual recall test yet."
            )

        else:
            st.success(
                f"{len(quizzes)} actual recall "
                "test(s) available."
            )

    with col3:
        if evaluation.empty:
            st.info(
                "No strict prediction-recall "
                "match yet."
            )

        else:
            st.success(
                f"{len(evaluation)} strict "
                "prediction match(es) available."
            )

    if (
        not sessions.empty
        and quizzes.empty
    ):
        st.info(
            "Your study-session graphs can already "
            "be displayed. Recall and prediction "
            "accuracy graphs appear only after an "
            "actual recall test is recorded."
        )


# study session timeline

def show_study_session_timeline(
    analytics
):
    st.subheader(
        "Study Session Timeline"
    )

    sessions = ensure_dataframe(
        analytics.get(
            "sessions"
        )
    )

    if sessions.empty:
        st.info(
            "Record a study session to create "
            "the first analytics graph."
        )

        return

    if (
        "studied_at"
        not in sessions.columns
    ):
        st.info(
            "Study timestamps are not available."
        )

        return

    dataframe = sessions.copy()

    dataframe[
        "studied_at"
    ] = pd.to_datetime(
        dataframe[
            "studied_at"
        ],
        errors="coerce"
    )

    if (
        "study_duration"
        in dataframe.columns
    ):
        dataframe[
            "study_duration"
        ] = pd.to_numeric(
            dataframe[
                "study_duration"
            ],
            errors="coerce"
        )

    dataframe = dataframe.dropna(
        subset=[
            "studied_at",
            "study_duration",
        ]
    )

    if dataframe.empty:
        st.info(
            "No valid study-duration records "
            "are available to plot."
        )

        return

    hover_columns = []

    for column in [
        "topic_name",
        "subject",
        "confidence_level",
        "difficulty",
        "revision_count",
    ]:
        if column in dataframe.columns:
            hover_columns.append(
                column
            )

    figure = px.scatter(
        dataframe,
        x="studied_at",
        y="study_duration",
        hover_data=hover_columns,
        title=(
            "Recorded Study Duration "
            "Across Sessions"
        )
    )

    figure.update_traces(
        marker=dict(
            size=12
        )
    )

    figure.update_xaxes(
        title="Study Date"
    )

    figure.update_yaxes(
        title="Study Duration (minutes)"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )

    st.caption(
        "This chart appears immediately from "
        "stored study-session data. It does not "
        "require a recall test."
    )


# actual recall timeline

def show_recall_timeline(
    analytics
):
    st.subheader(
        "Actual Recall Over Time"
    )

    quizzes = ensure_dataframe(
        analytics.get(
            "quizzes"
        )
    )

    if quizzes.empty:
        st.info(
            "No actual recall result exists yet. "
            "Complete a real Quiz & Recall test "
            "to unlock this graph."
        )

        return

    if (
        "quiz_date"
        not in quizzes.columns
        or "recall_accuracy"
        not in quizzes.columns
    ):
        st.info(
            "Recall timeline fields are "
            "not available."
        )

        return

    dataframe = quizzes.copy()

    dataframe[
        "quiz_date"
    ] = pd.to_datetime(
        dataframe[
            "quiz_date"
        ],
        errors="coerce"
    )

    dataframe[
        "recall_accuracy"
    ] = (
        pd.to_numeric(
            dataframe[
                "recall_accuracy"
            ],
            errors="coerce"
        )
        * 100
    )

    dataframe = dataframe.dropna(
        subset=[
            "quiz_date",
            "recall_accuracy",
        ]
    )

    if dataframe.empty:
        st.info(
            "No valid recall values "
            "are available to plot."
        )

        return

    hover_columns = []

    for column in [
        "topic_name",
        "delay_days",
        "total_questions",
        "correct_answers",
    ]:
        if column in dataframe.columns:
            hover_columns.append(
                column
            )

    figure = px.scatter(
        dataframe,
        x="quiz_date",
        y="recall_accuracy",
        hover_data=hover_columns,
        title=(
            "Observed Recall Performance"
        )
    )

    figure.update_traces(
        marker=dict(
            size=12
        )
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Actual Recall (%)"
    )

    figure.update_xaxes(
        title="Recall Test Date"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )

    st.caption(
        "These points may represent different "
        "topics and study sessions. They should "
        "not be interpreted as one continuous "
        "forgetting curve."
    )


# delay vs recall

def show_delay_vs_recall(
    analytics
):
    st.subheader(
        "Recall Delay vs Actual Recall"
    )

    learning = ensure_dataframe(
        analytics.get(
            "learning_data"
        )
    )

    if learning.empty:
        st.info(
            "This graph requires at least one "
            "study session linked to an actual "
            "recall test."
        )

        return

    required = [
        "delay_days",
        "recall_accuracy",
    ]

    if any(
        column not in learning.columns
        for column in required
    ):
        st.info(
            "Delay and recall fields are "
            "not available."
        )

        return

    dataframe = learning.copy()

    dataframe[
        "delay_days"
    ] = pd.to_numeric(
        dataframe[
            "delay_days"
        ],
        errors="coerce"
    )

    dataframe[
        "recall_accuracy"
    ] = (
        pd.to_numeric(
            dataframe[
                "recall_accuracy"
            ],
            errors="coerce"
        )
        * 100
    )

    dataframe = dataframe.dropna(
        subset=required
    )

    if dataframe.empty:
        st.info(
            "No valid delay-recall pairs "
            "are available."
        )

        return

    hover_columns = []

    for column in [
        "topic_name",
        "study_duration",
        "confidence_level",
        "difficulty",
    ]:
        if column in dataframe.columns:
            hover_columns.append(
                column
            )

    figure = px.scatter(
        dataframe,
        x="delay_days",
        y="recall_accuracy",
        hover_data=hover_columns,
        title=(
            "Time Since Study vs "
            "Observed Recall"
        )
    )

    figure.update_traces(
        marker=dict(
            size=12
        )
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Actual Recall (%)"
    )

    figure.update_xaxes(
        title="Delay Since Study (days)"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )


# learning relationships

def relationship_graph(
    learning,
    column,
    title,
    x_label
):
    if (
        learning.empty
        or column not in learning.columns
        or "recall_accuracy"
        not in learning.columns
    ):
        st.info(
            "Not enough data for this graph."
        )

        return

    dataframe = learning.copy()

    dataframe[
        column
    ] = pd.to_numeric(
        dataframe[
            column
        ],
        errors="coerce"
    )

    dataframe[
        "recall_accuracy"
    ] = (
        pd.to_numeric(
            dataframe[
                "recall_accuracy"
            ],
            errors="coerce"
        )
        * 100
    )

    dataframe = dataframe.dropna(
        subset=[
            column,
            "recall_accuracy",
        ]
    )

    if dataframe.empty:
        st.info(
            "No valid paired observations "
            "are available."
        )

        return

    hover_columns = []

    if "topic_name" in dataframe.columns:
        hover_columns.append(
            "topic_name"
        )

    figure = px.scatter(
        dataframe,
        x=column,
        y="recall_accuracy",
        hover_data=hover_columns,
        title=title
    )

    figure.update_traces(
        marker=dict(
            size=12
        )
    )

    figure.update_xaxes(
        title=x_label
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Actual Recall (%)"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )


def show_learning_relationships(
    analytics
):
    st.subheader(
        "Learning Behaviour Relationships"
    )

    learning = ensure_dataframe(
        analytics.get(
            "learning_data"
        )
    )

    if learning.empty:
        st.info(
            "Relationship graphs appear after "
            "study sessions have corresponding "
            "actual recall tests."
        )

        return

    tabs = st.tabs(
        [
            "Study Duration",
            "Difficulty",
            "Confidence",
            "Revision Count",
            "Previous Recall",
        ]
    )

    with tabs[0]:
        relationship_graph(
            learning,
            "study_duration",
            (
                "Study Duration vs "
                "Actual Recall"
            ),
            "Study Duration (minutes)"
        )

    with tabs[1]:
        relationship_graph(
            learning,
            "difficulty",
            (
                "Topic Difficulty vs "
                "Actual Recall"
            ),
            "Difficulty"
        )

    with tabs[2]:
        relationship_graph(
            learning,
            "confidence_level",
            (
                "Confidence vs "
                "Actual Recall"
            ),
            "Confidence"
        )

    with tabs[3]:
        relationship_graph(
            learning,
            "revision_count",
            (
                "Revision Count vs "
                "Actual Recall"
            ),
            "Previous Revision Count"
        )

    with tabs[4]:
        relationship_graph(
            learning,
            "previous_quiz_score",
            (
                "Previous Recall vs "
                "Later Recall"
            ),
            "Previous Quiz Score"
        )

    st.caption(
        "These relationships are descriptive. "
        "They do not establish causation."
    )


# correlations

def show_correlations(
    analytics
):
    st.subheader(
        "Correlation Summary"
    )

    correlations = (
        analytics.get(
            "correlations"
        )
        or {}
    )

    if not correlations:
        st.info(
            "Correlation estimates require "
            "additional paired learning and "
            "recall observations."
        )

        return

    rows = []

    for label, result in (
        correlations.items()
    ):
        correlation = safe_float(
            result.get(
                "correlation"
            )
        )

        rows.append(
            {
                "Variable":
                    label,

                "Correlation":
                    (
                        f"{correlation:.3f}"
                        if correlation
                        is not None
                        else "N/A"
                    ),

                "Classification":
                    result.get(
                        "classification",
                        "N/A"
                    ),

                "Interpretation":
                    result.get(
                        "interpretation",
                        (
                            "Descriptive association only."
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


# topic analytics

def show_topic_analytics(
    analytics
):
    st.subheader(
        "Topic Analytics"
    )

    dataframe = ensure_dataframe(
        analytics.get(
            "topic_analytics"
        )
    )

    if dataframe.empty:
        st.info(
            "Topic recall analytics will appear "
            "after actual recall measurements exist."
        )

        return

    display = dataframe.copy()

    percentage_columns = [
        "average_recall",
        "best_recall",
        "lowest_recall",
    ]

    for column in percentage_columns:
        if column in display.columns:
            numeric = pd.to_numeric(
                display[
                    column
                ],
                errors="coerce"
            )

            display[
                column
            ] = numeric.map(
                lambda value:
                    (
                        f"{value * 100:.1f}%"
                        if pd.notna(value)
                        else "N/A"
                    )
            )

    for column in [
        "average_delay_days",
        "average_study_minutes",
    ]:
        if column in display.columns:
            numeric = pd.to_numeric(
                display[
                    column
                ],
                errors="coerce"
            )

            display[
                column
            ] = numeric.map(
                lambda value:
                    (
                        f"{value:.2f}"
                        if pd.notna(value)
                        else "N/A"
                    )
            )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True
    )

    if (
        "topic_name"
        in dataframe.columns
        and "average_recall"
        in dataframe.columns
    ):
        chart = dataframe[
            [
                "topic_name",
                "average_recall",
            ]
        ].copy()

        chart[
            "average_recall"
        ] = (
            pd.to_numeric(
                chart[
                    "average_recall"
                ],
                errors="coerce"
            )
            * 100
        )

        chart = chart.dropna()

        if not chart.empty:
            figure = px.bar(
                chart,
                x="topic_name",
                y="average_recall",
                title=(
                    "Average Observed Recall "
                    "by Topic"
                )
            )

            figure.update_yaxes(
                range=[
                    0,
                    100
                ],
                title="Average Recall (%)"
            )

            figure.update_xaxes(
                title="Topic"
            )

            st.plotly_chart(
                figure,
                width="stretch"
            )


# checkpoint analytics

def show_checkpoint_analytics(
    analytics
):
    st.subheader(
        "Recall Checkpoints"
    )

    dataframe = ensure_dataframe(
        analytics.get(
            "checkpoint_analytics"
        )
    )

    if dataframe.empty:
        st.info(
            "Checkpoint analytics require "
            "actual recall measurements."
        )

        return

    display = dataframe.copy()

    for column in [
        "average_recall",
        "median_recall",
        "minimum_recall",
        "maximum_recall",
    ]:
        if column not in display.columns:
            continue

        numeric = pd.to_numeric(
            display[
                column
            ],
            errors="coerce"
        )

        display[
            column
        ] = numeric.map(
            lambda value:
                (
                    f"{value * 100:.1f}%"
                    if pd.notna(value)
                    else "N/A"
                )
        )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True
    )

    if (
        "checkpoint"
        in dataframe.columns
        and "average_recall"
        in dataframe.columns
    ):
        chart = dataframe[
            [
                "checkpoint",
                "average_recall",
            ]
        ].copy()

        chart[
            "average_recall"
        ] = (
            pd.to_numeric(
                chart[
                    "average_recall"
                ],
                errors="coerce"
            )
            * 100
        )

        chart = chart.dropna()

        if not chart.empty:
            figure = px.bar(
                chart,
                x="checkpoint",
                y="average_recall",
                title=(
                    "Average Recall by "
                    "Checkpoint"
                )
            )

            figure.update_yaxes(
                range=[
                    0,
                    100
                ],
                title="Average Recall (%)"
            )

            figure.update_xaxes(
                title="Checkpoint"
            )

            st.plotly_chart(
                figure,
                width="stretch"
            )


# model availability

def show_model_availability(
    analytics
):
    st.subheader(
        "Prediction Model Availability"
    )

    evaluation = ensure_dataframe(
        analytics.get(
            "evaluation_data"
        )
    )

    rows = []

    for model_name, column in (
        MODEL_COLUMNS.items()
    ):
        observations = 0

        if (
            not evaluation.empty
            and column in evaluation.columns
        ):
            values = pd.to_numeric(
                evaluation[
                    column
                ],
                errors="coerce"
            )

            observations = int(
                values.notna().sum()
            )

        rows.append(
            {
                "Model":
                    model_name,

                "Usable Strict Predictions":
                    observations,

                "Status":
                    (
                        "Available"
                        if observations > 0
                        else "Not Yet Available"
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


# model metrics dataframe

def model_metrics_dataframe(
    metrics
):
    rows = []

    for result in metrics:
        mae_pp = safe_float(
            result.get(
                "mae_pp"
            )
        )

        rmse_pp = safe_float(
            result.get(
                "rmse_pp"
            )
        )

        r2 = safe_float(
            result.get(
                "r2"
            )
        )

        bias_pp = safe_float(
            result.get(
                "bias_pp"
            )
        )

        within_5 = safe_float(
            result.get(
                "within_5pp"
            )
        )

        within_10 = safe_float(
            result.get(
                "within_10pp"
            )
        )

        within_15 = safe_float(
            result.get(
                "within_15pp"
            )
        )

        rows.append(
            {
                "Model":
                    result.get(
                        "model",
                        "Unknown"
                    ),

                "Observations":
                    safe_int(
                        result.get(
                            "observations"
                        ),
                        0
                    ),

                "MAE":
                    (
                        f"{mae_pp:.2f} pp"
                        if mae_pp is not None
                        else "N/A"
                    ),

                "RMSE":
                    (
                        f"{rmse_pp:.2f} pp"
                        if rmse_pp is not None
                        else "N/A"
                    ),

                "R²":
                    (
                        f"{r2:.3f}"
                        if r2 is not None
                        else "N/A"
                    ),

                "Bias":
                    (
                        f"{bias_pp:.2f} pp"
                        if bias_pp is not None
                        else "N/A"
                    ),

                "Bias Direction":
                    result.get(
                        "bias_direction",
                        "N/A"
                    ),

                "Within 5pp":
                    (
                        f"{within_5 * 100:.1f}%"
                        if within_5
                        is not None
                        else "N/A"
                    ),

                "Within 10pp":
                    (
                        f"{within_10 * 100:.1f}%"
                        if within_10
                        is not None
                        else "N/A"
                    ),

                "Within 15pp":
                    (
                        f"{within_15 * 100:.1f}%"
                        if within_15
                        is not None
                        else "N/A"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# individual diagnostics

def show_individual_model_metrics(
    analytics
):
    st.subheader(
        "Available Model Diagnostics"
    )

    metrics = (
        analytics.get(
            "individual_model_metrics"
        )
        or []
    )

    if not metrics:
        st.info(
            "No model has enough matched "
            "prediction-recall data for "
            "diagnostics yet."
        )

        return

    dataframe = model_metrics_dataframe(
        metrics
    )

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "These are model-specific diagnostics. "
        "Different models may have different "
        "numbers of available observations, so "
        "this table must not be used alone to "
        "declare a comparative winner."
    )


# common subset comparison

def show_common_subset_comparison(
    analytics
):
    st.subheader(
        "Fair Common-Observation Comparison"
    )

    common = (
        analytics.get(
            "common_subset"
        )
        or {}
    )

    observations = safe_int(
        common.get(
            "observations"
        ),
        0
    )

    models = (
        common.get(
            "models"
        )
        or []
    )

    ranking = (
        common.get(
            "ranking"
        )
        or []
    )

    metrics = (
        common.get(
            "metrics"
        )
        or []
    )

    if not metrics:
        st.info(
            "No common comparison subset "
            "is available yet."
        )

        return

    display_metrics = (
        ranking
        if ranking
        else metrics
    )

    dataframe = (
        model_metrics_dataframe(
            display_metrics
        )
    )

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True
    )

    st.caption(
        f"Common observations: {observations}"
    )

    if len(models) < 2:
        st.warning(
            "Only one prediction method currently "
            "has usable strict observations."
        )

        st.info(
            "MemoryMap will not report a comparative "
            "winner until at least two methods are "
            "evaluated on the same recall outcomes."
        )

        return

    winner = (
        display_metrics[
            0
        ]
    )

    winner_name = winner.get(
        "model",
        "Unknown"
    )

    winner_mae = safe_float(
        winner.get(
            "mae_pp"
        )
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:
        st.metric(
            "Current Numerical Leader",
            winner_name
        )

    with col2:
        st.metric(
            "Leader MAE",
            (
                f"{winner_mae:.2f} pp"
                if winner_mae is not None
                else "N/A"
            )
        )

    if observations < 5:
        st.warning(
            "The comparison currently contains "
            "fewer than five common observations. "
            "Do not treat this as a final result."
        )

    chart_rows = []

    for result in display_metrics:
        mae = safe_float(
            result.get(
                "mae_pp"
            )
        )

        if mae is None:
            continue

        chart_rows.append(
            {
                "Model":
                    result.get(
                        "model",
                        "Unknown"
                    ),

                "MAE":
                    mae,
            }
        )

    if chart_rows:
        figure = px.bar(
            pd.DataFrame(
                chart_rows
            ),
            x="Model",
            y="MAE",
            text_auto=".2f",
            title=(
                "MAE on the Same "
                "Recall Observations"
            )
        )

        figure.update_yaxes(
            title=(
                "Mean Absolute Error "
                "(percentage points)"
            )
        )

        st.plotly_chart(
            figure,
            width="stretch"
        )


# predicted vs actual

def create_prediction_plot_dataframe(
    analytics
):
    evaluation = ensure_dataframe(
        analytics.get(
            "evaluation_data"
        )
    )

    if evaluation.empty:
        return pd.DataFrame()

    if (
        "actual_recall"
        not in evaluation.columns
    ):
        return pd.DataFrame()

    rows = []

    for _, record in (
        evaluation.iterrows()
    ):
        actual = safe_float(
            record.get(
                "actual_recall"
            )
        )

        if actual is None:
            continue

        for model_name, column in (
            MODEL_COLUMNS.items()
        ):
            if column not in evaluation.columns:
                continue

            prediction = safe_float(
                record.get(
                    column
                )
            )

            if prediction is None:
                continue

            if not (
                0 <= actual <= 1
                and 0 <= prediction <= 1
            ):
                continue

            rows.append(
                {
                    "Model":
                        model_name,

                    "Topic":
                        record.get(
                            "topic_name",
                            "Unknown"
                        ),

                    "Actual Recall":
                        actual * 100,

                    "Predicted Recall":
                        prediction * 100,

                    "Absolute Error":
                        abs(
                            prediction
                            - actual
                        )
                        * 100,

                    "Delay Days":
                        safe_float(
                            record.get(
                                "delay_days"
                            )
                        ),

                    "Prediction Horizon":
                        safe_float(
                            record.get(
                                "prediction_for_days"
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def show_predicted_vs_actual(
    analytics
):
    st.subheader(
        "Predicted vs Actual Recall"
    )

    dataframe = (
        create_prediction_plot_dataframe(
            analytics
        )
    )

    if dataframe.empty:
        st.info(
            "This graph appears after a saved "
            "prediction is followed by an actual "
            "recall test that satisfies strict "
            "time matching."
        )

        return

    figure = px.scatter(
        dataframe,
        x="Predicted Recall",
        y="Actual Recall",
        color="Model",
        hover_data=[
            "Topic",
            "Delay Days",
            "Prediction Horizon",
            "Absolute Error",
        ],
        title=(
            "Strict Predictions Compared "
            "with Actual Recall"
        )
    )

    figure.update_traces(
        marker=dict(
            size=12
        )
    )

    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        line=dict(
            dash="dash"
        )
    )

    figure.update_xaxes(
        range=[
            0,
            100
        ],
        title="Predicted Recall (%)"
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Actual Recall (%)"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )

    st.caption(
        "Points closer to the diagonal indicate "
        "smaller prediction errors."
    )


# baseline comparisons

def show_baseline_comparisons(
    analytics
):
    st.subheader(
        "Fixed Baseline Comparisons"
    )

    comparisons = (
        analytics.get(
            "baseline_comparisons"
        )
        or {}
    )

    if not comparisons:
        st.info(
            "Personalized or adaptive predictions "
            "do not yet share enough strict recall "
            "observations with the fixed baseline."
        )

        return

    rows = []

    for candidate, result in (
        comparisons.items()
    ):
        pairwise = (
            result.get(
                "pairwise"
            )
            or {}
        )

        first = (
            pairwise.get(
                "first_model"
            )
            or {}
        )

        second = (
            pairwise.get(
                "second_model"
            )
            or {}
        )

        rows.append(
            {
                "Comparison":
                    (
                        "Fixed Forgetting Curve "
                        f"vs {candidate}"
                    ),

                "Shared Observations":
                    safe_int(
                        pairwise.get(
                            "observations"
                        ),
                        0
                    ),

                "Fixed MAE":
                    format_percentage_points(
                        first.get(
                            "mae_pp"
                        )
                    ),

                "Candidate MAE":
                    format_percentage_points(
                        second.get(
                            "mae_pp"
                        )
                    ),

                "Numerical Winner":
                    pairwise.get(
                        "winner",
                        "N/A"
                    ),

                "MAE Difference":
                    format_percentage_points(
                        pairwise.get(
                            "mae_difference_pp"
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


# research verdict

def show_research_verdict(
    analytics
):
    st.subheader(
        "Current Research Finding"
    )

    verdict = (
        analytics.get(
            "research_verdict"
        )
        or {}
    )

    common = (
        analytics.get(
            "common_subset"
        )
        or {}
    )

    models = (
        common.get(
            "models"
        )
        or []
    )

    observations = safe_int(
        common.get(
            "observations"
        ),
        0
    )

    message = verdict.get(
        "message",
        (
            "No research conclusion "
            "is currently available."
        )
    )

    if len(models) < 2:
        st.info(
            message
        )

        st.warning(
            "A comparative winner cannot be "
            "reported until at least two methods "
            "have predictions for the same "
            "strict recall observations."
        )

        return

    winner = verdict.get(
        "winner"
    )

    if observations < 5:
        st.warning(
            message
        )

        st.caption(
            "Any current ranking is preliminary "
            "because the common sample is tiny."
        )

        return

    if winner:
        st.success(
            message
        )

    else:
        st.info(
            message
        )

    if observations < 20:
        st.caption(
            "This result remains preliminary "
            "until more genuine delayed-recall "
            "observations are collected."
        )


# raw data

def probability_display(
    dataframe
):
    dataframe = dataframe.copy()

    for column in [
        "recall_accuracy",
        "actual_recall",
        "fixed_prediction",
        "personalized_prediction",
        "adaptive_prediction",
    ]:
        if column not in dataframe.columns:
            continue

        numeric = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce"
        )

        dataframe[
            column
        ] = numeric.map(
            lambda value:
                (
                    f"{value * 100:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                )
        )

    return dataframe


def show_raw_data(
    analytics
):
    with st.expander(
        "View Analytics Data"
    ):
        sessions = ensure_dataframe(
            analytics.get(
                "sessions"
            )
        )

        quizzes = ensure_dataframe(
            analytics.get(
                "quizzes"
            )
        )

        learning = ensure_dataframe(
            analytics.get(
                "learning_data"
            )
        )

        evaluation = ensure_dataframe(
            analytics.get(
                "evaluation_data"
            )
        )

        common = (
            analytics.get(
                "common_subset"
            )
            or {}
        )

        common_dataframe = ensure_dataframe(
            common.get(
                "dataframe"
            )
        )

        tab1, tab2, tab3, tab4, tab5 = (
            st.tabs(
                [
                    "Study Sessions",
                    "Recall Tests",
                    "Learning Data",
                    "Strict Matches",
                    "Common Subset",
                ]
            )
        )

        with tab1:
            if sessions.empty:
                st.info(
                    "No study sessions."
                )

            else:
                st.dataframe(
                    sessions,
                    width="stretch",
                    hide_index=True
                )

        with tab2:
            if quizzes.empty:
                st.info(
                    "No recall tests."
                )

            else:
                st.dataframe(
                    probability_display(
                        quizzes
                    ),
                    width="stretch",
                    hide_index=True
                )

        with tab3:
            if learning.empty:
                st.info(
                    "No linked learning-recall data."
                )

            else:
                st.dataframe(
                    probability_display(
                        learning
                    ),
                    width="stretch",
                    hide_index=True
                )

        with tab4:
            if evaluation.empty:
                st.info(
                    "No strict prediction matches."
                )

            else:
                st.dataframe(
                    probability_display(
                        evaluation
                    ),
                    width="stretch",
                    hide_index=True
                )

        with tab5:
            if common_dataframe.empty:
                st.info(
                    "No common model subset."
                )

            else:
                st.dataframe(
                    probability_display(
                        common_dataframe
                    ),
                    width="stretch",
                    hide_index=True
                )


# explanation

def show_explanation():
    with st.expander(
        "How should these analytics be interpreted?"
    ):
        st.markdown(
            """
### Study graphs

Study-session graphs appear as soon as genuine study-session data is stored.

### Recall graphs

Recall graphs require an actual completed recall test. Saving only a study session or prediction is not an observed memory outcome.

### Prediction evaluation

A prediction is evaluated only when it existed before the recall test. MemoryMap also checks whether the prediction horizon closely matches the actual testing delay.

### Strict matching

MemoryMap currently uses a ±0.5-day interval tolerance for formal evaluation.

### Individual model diagnostics

These show how each model performs on the observations available to that specific model. Sample counts can differ.

### Fair model comparison

A direct comparison should use the same actual recall observations for every competing method.

### R² with tiny samples

R² is not meaningful with only one observation. MemoryMap therefore displays `N/A` instead of crashing or fabricating a value.

### Correlation

Correlation describes association. It does not prove that study duration, confidence, difficulty or another variable caused the recall result.
            """
        )


# page

def show_page():
    student_id = select_student()

    if student_id is None:
        return

    analytics = load_analytics(
        student_id
    )

    if analytics is None:
        return

    show_header(
        analytics
    )

    st.divider()

    show_summary(
        analytics
    )

    st.divider()

    show_data_readiness(
        analytics
    )

    st.divider()

    show_study_session_timeline(
        analytics
    )

    st.divider()

    show_recall_timeline(
        analytics
    )

    st.divider()

    show_delay_vs_recall(
        analytics
    )

    st.divider()

    show_learning_relationships(
        analytics
    )

    st.divider()

    show_correlations(
        analytics
    )

    st.divider()

    show_topic_analytics(
        analytics
    )

    st.divider()

    show_checkpoint_analytics(
        analytics
    )

    st.divider()

    show_model_availability(
        analytics
    )

    st.divider()

    show_individual_model_metrics(
        analytics
    )

    st.divider()

    show_common_subset_comparison(
        analytics
    )

    st.divider()

    show_predicted_vs_actual(
        analytics
    )

    st.divider()

    show_baseline_comparisons(
        analytics
    )

    st.divider()

    show_research_verdict(
        analytics
    )

    st.divider()

    st.page_link(
        "pages/7_Model_Comparison.py",
        label="⚖️ Open Model Comparison"
    )

    show_raw_data(
        analytics
    )

    show_explanation()


show_page()
