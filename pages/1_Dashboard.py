from datetime import datetime
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    APP_NAME,
    APP_VERSION,
)

from database import (
    initialize_database,
)

from services.student_service import (
    create_student,
    list_students,
    get_student_dashboard_data,
    calculate_student_progress,
)

from services.study_service import (
    parse_datetime,
)

from utils import (
    format_datetime,
    format_study_duration,
    format_percentage,
    get_difficulty_label,
)


st.set_page_config(
    page_title="Dashboard | MemoryMap",
    page_icon="🧠",
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


def get_subject_name(
    record
):
    if not record:
        return "Not specified"

    subject = record.get(
        "subject_name"
    )

    if not subject:
        subject = record.get(
            "subject"
        )

    if not subject:
        return "Not specified"

    return str(subject)


def get_topic_name(
    record
):
    if not record:
        return "Unknown Topic"

    topic_name = record.get(
        "topic_name"
    )

    if not topic_name:
        return "Unknown Topic"

    return str(topic_name)


# date helpers

def datetime_sort_value(
    record,
    field
):
    parsed = parse_datetime(
        record.get(field)
    )

    if parsed is None:
        return datetime.min

    return parsed


# flash messages

def show_dashboard_flash():
    flash = st.session_state.pop(
        "dashboard_flash",
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


# profile creation

def create_student_profile():
    st.subheader(
        "Create Student Profile"
    )

    st.write(
        "Create a profile before recording "
        "study sessions and recall data."
    )

    with st.form(
        "create_student_form"
    ):
        name = st.text_input(
            "Student Name",
            placeholder="Enter student name"
        )

        email = st.text_input(
            "Email (optional)",
            placeholder="student@example.com"
        )

        submitted = st.form_submit_button(
            "Create Profile",
            use_container_width=True
        )

    if not submitted:
        return

    result = create_student(
        name=name,
        email=email,
    )

    if not result.get(
        "success",
        False
    ):
        st.error(
            result.get(
                "message",
                "Student profile could not be created."
            )
        )

        existing_student = result.get(
            "student"
        )

        if existing_student:
            st.caption(
                "A matching profile already exists. "
                "Select it from the sidebar."
            )

        return

    student = (
        result.get(
            "student"
        )
        or {}
    )

    student_id = get_record_id(
        student,
        "student_id"
    )

    student_name = (
        student.get(
            "name"
        )
        or name.strip()
    )

    if student_id is None:
        st.error(
            "The profile was created, but its "
            "student ID could not be loaded."
        )

        return

    st.session_state.student_id = (
        student_id
    )

    st.session_state.student_name = (
        student_name
    )

    st.session_state.selected_topic = None
    st.session_state.selected_session_id = None

    # Important:
    # do NOT modify dashboard_student_selector here.
    # The widget already exists during this run.
    # Instead, request a reset for the next rerun.

    st.session_state[
        "reset_dashboard_student_selector"
    ] = True

    st.session_state[
        "dashboard_flash"
    ] = {
        "type":
            "success",

        "message":
            (
                f"Profile created for "
                f"{student_name}."
            ),
    }

    st.rerun()


# profile selection

def select_student():
    students = list_students()

    if not students:
        st.title(
            "🧠 MemoryMap Dashboard"
        )

        st.caption(
            f"{APP_NAME} v{APP_VERSION}"
        )

        st.info(
            "No student profiles exist yet."
        )

        create_student_profile()

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

        if email:
            label = (
                f"{name} • {email}"
            )

        else:
            label = str(name)

        student_map[
            student_id
        ] = {
            "label":
                label,

            "name":
                str(name),

            "student":
                student,
        }

    student_ids = list(
        student_map.keys()
    )

    if not student_ids:
        st.error(
            "Student profiles exist, but no "
            "valid student IDs were found."
        )

        return None

    SELECT_PROFILE = (
        "__SELECT_PROFILE__"
    )

    CREATE_PROFILE = (
        "__CREATE_PROFILE__"
    )

    options = [
        SELECT_PROFILE,
        *student_ids,
        CREATE_PROFILE,
    ]

    # This executes before the selectbox is created.
    # Therefore deleting the widget state here is safe.

    reset_selector = (
        st.session_state.pop(
            "reset_dashboard_student_selector",
            False
        )
    )

    if reset_selector:
        if (
            "dashboard_student_selector"
            in st.session_state
        ):
            del st.session_state[
                "dashboard_student_selector"
            ]

    active_student = (
        st.session_state.get(
            "student_id"
        )
    )

    if active_student in student_ids:
        default_index = options.index(
            active_student
        )

    else:
        default_index = 0

    def format_profile_option(
        value
    ):
        if value == SELECT_PROFILE:
            return "Select a profile"

        if value == CREATE_PROFILE:
            return "➕ Create New Profile"

        return student_map[
            value
        ][
            "label"
        ]

    selected = st.sidebar.selectbox(
        "Student Profile",
        options=options,
        index=default_index,
        format_func=format_profile_option,
        key="dashboard_student_selector",
    )

    # Create profile option

    if selected == CREATE_PROFILE:
        st.session_state.student_id = None
        st.session_state.student_name = None
        st.session_state.selected_topic = None
        st.session_state.selected_session_id = None

        st.title(
            "🧠 MemoryMap Dashboard"
        )

        st.info(
            "Create a new student profile below."
        )

        st.divider()

        create_student_profile()

        return None

    # No profile selected

    if selected == SELECT_PROFILE:
        st.session_state.student_id = None
        st.session_state.student_name = None
        st.session_state.selected_topic = None
        st.session_state.selected_session_id = None

        st.title(
            "🧠 MemoryMap Dashboard"
        )

        st.caption(
            f"{APP_NAME} v{APP_VERSION}"
        )

        st.info(
            "Select an existing profile "
            "from the sidebar or create a new one."
        )

        st.divider()

        create_student_profile()

        return None

    previous_student = (
        st.session_state.get(
            "student_id"
        )
    )

    if (
        previous_student is not None
        and previous_student != selected
    ):
        st.session_state.selected_topic = None
        st.session_state.selected_session_id = None

    st.session_state.student_id = (
        selected
    )

    st.session_state.student_name = (
        student_map[
            selected
        ][
            "name"
        ]
    )

    return selected


# dashboard loading

def load_dashboard_data(
    student_id
):
    result = get_student_dashboard_data(
        student_id
    )

    if not result.get(
        "success",
        False
    ):
        st.error(
            result.get(
                "message",
                "Dashboard data could not be loaded."
            )
        )

        return None

    return result


# header

def show_header(
    dashboard
):
    student = (
        dashboard.get(
            "student"
        )
        or {}
    )

    student_name = (
        student.get(
            "name"
        )
        or st.session_state.get(
            "student_name"
        )
        or "Student"
    )

    st.session_state.student_name = (
        student_name
    )

    st.title(
        "🧠 MemoryMap Dashboard"
    )

    st.write(
        f"Welcome, **{student_name}**. "
        "Here is your current learning "
        "and research overview."
    )

    st.caption(
        f"{APP_NAME} v{APP_VERSION}"
    )


# summary metrics

def show_summary_metrics(
    dashboard,
    progress
):
    statistics = (
        dashboard.get(
            "statistics"
        )
        or {}
    )

    due_revisions = (
        dashboard.get(
            "due_revisions"
        )
        or []
    )

    upcoming_revisions = (
        dashboard.get(
            "upcoming_revisions"
        )
        or []
    )

    topics = safe_int(
        statistics.get(
            "topics"
        ),
        0
    )

    study_sessions = safe_int(
        statistics.get(
            "study_sessions"
        ),
        len(
            dashboard.get(
                "study_sessions"
            )
            or []
        )
    )

    quiz_tests = safe_int(
        progress.get(
            "quiz_tests"
        ),
        len(
            dashboard.get(
                "quiz_results"
            )
            or []
        )
    )

    average_recall = safe_float(
        progress.get(
            "average_recall"
        )
    )

    pending_revisions = (
        len(due_revisions)
        + len(upcoming_revisions)
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:
        st.metric(
            "Topics",
            topics
        )

    with col2:
        st.metric(
            "Study Sessions",
            study_sessions
        )

    with col3:
        st.metric(
            "Recall Tests",
            quiz_tests
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
            "Pending Revisions",
            pending_revisions
        )


# research progress

def show_learning_status(
    progress
):
    st.subheader(
        "Research Data Progress"
    )

    study_sessions = safe_int(
        progress.get(
            "study_sessions"
        ),
        0
    )

    quiz_tests = safe_int(
        progress.get(
            "quiz_tests"
        ),
        0
    )

    valid_recall_tests = safe_int(
        progress.get(
            "valid_recall_tests"
        ),
        0
    )

    average_recall = safe_float(
        progress.get(
            "average_recall"
        )
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Study Observations",
            study_sessions
        )

    with col2:
        st.metric(
            "Valid Recall Records",
            valid_recall_tests
        )

    with col3:
        st.metric(
            "Observed Mean Recall",
            (
                format_percentage(
                    average_recall
                )
                if average_recall
                is not None
                else "N/A"
            )
        )

    if study_sessions == 0:
        st.info(
            "Start by recording a study session. "
            "MemoryMap needs a study timestamp "
            "before retention can be measured."
        )

        st.page_link(
            "pages/2_Study_Session.py",
            label="📚 Record First Study Session"
        )

        return

    if quiz_tests == 0:
        st.info(
            "Study data exists. The next important "
            "step is collecting real delayed recall."
        )

        st.page_link(
            "pages/3_Retention_Predictor.py",
            label="📉 Create a Prospective Prediction"
        )

        st.page_link(
            "pages/5_Quiz_and_Recall.py",
            label="🧪 Record Actual Recall"
        )

        return

    if valid_recall_tests < 5:
        st.info(
            "MemoryMap is in early longitudinal "
            "data collection. More real delayed "
            "recall observations are needed."
        )

    elif valid_recall_tests < 20:
        st.info(
            "The recall history is growing, "
            "but the dataset is still preliminary."
        )

    else:
        st.success(
            "A larger longitudinal recall history "
            "is now available for personalization "
            "experiments."
        )

    st.caption(
        "Record count alone does not prove model "
        "quality. Formal conclusions depend on "
        "strict prediction-recall evaluation."
    )


# next action

def show_next_action(
    dashboard
):
    st.subheader(
        "What Needs Attention?"
    )

    due_revisions = (
        dashboard.get(
            "due_revisions"
        )
        or []
    )

    upcoming_revisions = (
        dashboard.get(
            "upcoming_revisions"
        )
        or []
    )

    quiz_results = (
        dashboard.get(
            "quiz_results"
        )
        or []
    )

    study_sessions = (
        dashboard.get(
            "study_sessions"
        )
        or []
    )

    if due_revisions:
        first_due = sorted(
            due_revisions,
            key=lambda revision:
                datetime_sort_value(
                    revision,
                    "scheduled_for"
                )
        )[0]

        st.warning(
            "A revision is due: "
            f"**{get_topic_name(first_due)}**"
        )

        st.caption(
            "Scheduled for "
            f"{format_datetime(first_due.get('scheduled_for'))}"
        )

        st.page_link(
            "pages/4_Revision_Scheduler.py",
            label="⏰ Open Revision Scheduler"
        )

        return

    if upcoming_revisions:
        first_upcoming = sorted(
            upcoming_revisions,
            key=lambda revision:
                datetime_sort_value(
                    revision,
                    "scheduled_for"
                )
        )[0]

        st.info(
            "Next scheduled revision: "
            f"**{get_topic_name(first_upcoming)}**"
        )

        st.caption(
            format_datetime(
                first_upcoming.get(
                    "scheduled_for"
                )
            )
        )

        return

    if study_sessions and not quiz_results:
        st.info(
            "You have study data but no "
            "recorded recall outcome yet."
        )

        st.page_link(
            "pages/5_Quiz_and_Recall.py",
            label="🧪 Open Quiz & Recall"
        )

        return

    if not study_sessions:
        st.info(
            "Record a study session to begin "
            "the MemoryMap workflow."
        )

        return

    st.success(
        "No revision currently requires "
        "immediate attention."
    )


# revision table

def revision_table_rows(
    revisions
):
    rows = []

    for revision in revisions:
        predicted_retention = safe_float(
            revision.get(
                "predicted_retention"
            )
        )

        threshold = safe_float(
            revision.get(
                "threshold"
            )
        )

        rows.append(
            {
                "Topic":
                    get_topic_name(
                        revision
                    ),

                "Subject":
                    get_subject_name(
                        revision
                    ),

                "Scheduled For":
                    format_datetime(
                        revision.get(
                            "scheduled_for"
                        )
                    ),

                "Predicted Retention":
                    (
                        format_percentage(
                            predicted_retention
                        )
                        if predicted_retention
                        is not None
                        else "N/A"
                    ),

                "Threshold":
                    (
                        format_percentage(
                            threshold
                        )
                        if threshold
                        is not None
                        else "N/A"
                    ),

                "Priority":
                    (
                        revision.get(
                            "priority"
                        )
                        or "Normal"
                    ),

                "Mode":
                    (
                        revision.get(
                            "schedule_mode"
                        )
                        or "N/A"
                    ),
            }
        )

    return rows


# revision overview

def show_revision_overview(
    dashboard
):
    st.subheader(
        "Revision Overview"
    )

    due_revisions = (
        dashboard.get(
            "due_revisions"
        )
        or []
    )

    upcoming_revisions = (
        dashboard.get(
            "upcoming_revisions"
        )
        or []
    )

    due_revisions = sorted(
        due_revisions,
        key=lambda revision:
            datetime_sort_value(
                revision,
                "scheduled_for"
            )
    )

    upcoming_revisions = sorted(
        upcoming_revisions,
        key=lambda revision:
            datetime_sort_value(
                revision,
                "scheduled_for"
            )
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Due Now",
            len(due_revisions)
        )

    with col2:
        st.metric(
            "Upcoming",
            len(upcoming_revisions)
        )

    if due_revisions:
        st.warning(
            f"{len(due_revisions)} revision(s) "
            "currently require attention."
        )

        st.dataframe(
            pd.DataFrame(
                revision_table_rows(
                    due_revisions
                )
            ),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success(
            "No revisions are currently due."
        )

    if upcoming_revisions:
        with st.expander(
            "View Upcoming Revisions"
        ):
            st.dataframe(
                pd.DataFrame(
                    revision_table_rows(
                        upcoming_revisions
                    )
                ),
                use_container_width=True,
                hide_index=True
            )

    st.page_link(
        "pages/4_Revision_Scheduler.py",
        label="⏰ Manage Revision Schedule"
    )


# recent sessions

def show_recent_sessions(
    dashboard
):
    st.subheader(
        "Recent Study Sessions"
    )

    sessions = (
        dashboard.get(
            "study_sessions"
        )
        or []
    )

    if not sessions:
        st.info(
            "No study sessions have "
            "been recorded yet."
        )

        return

    sessions = sorted(
        sessions,
        key=lambda session:
            datetime_sort_value(
                session,
                "studied_at"
            ),
        reverse=True
    )

    recent_sessions = (
        sessions[:5]
    )

    rows = []

    for session in recent_sessions:
        difficulty = safe_int(
            session.get(
                "difficulty"
            ),
            3
        )

        confidence = safe_int(
            session.get(
                "confidence_level"
            ),
            None
        )

        previous_quiz_score = safe_float(
            session.get(
                "previous_quiz_score"
            )
        )

        rows.append(
            {
                "Session":
                    get_record_id(
                        session,
                        "session_id"
                    ),

                "Topic":
                    get_topic_name(
                        session
                    ),

                "Subject":
                    get_subject_name(
                        session
                    ),

                "Duration":
                    format_study_duration(
                        session.get(
                            "study_duration",
                            0
                        )
                    ),

                "Difficulty":
                    (
                        f"{difficulty}/5 • "
                        f"{get_difficulty_label(difficulty)}"
                    ),

                "Confidence":
                    (
                        f"{confidence}/5"
                        if confidence is not None
                        else "N/A"
                    ),

                "Previous Sessions":
                    (
                        session.get(
                            "revision_count"
                        )
                        if session.get(
                            "revision_count"
                        )
                        is not None
                        else "N/A"
                    ),

                "Previous Recall":
                    (
                        f"{previous_quiz_score:.1f}%"
                        if previous_quiz_score
                        is not None
                        else "N/A"
                    ),

                "Studied At":
                    format_datetime(
                        session.get(
                            "studied_at"
                        )
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.page_link(
        "pages/2_Study_Session.py",
        label="📚 Record Another Study Session"
    )


# recall dataframe

def create_recall_dataframe(
    dashboard
):
    quiz_results = (
        dashboard.get(
            "quiz_results"
        )
        or []
    )

    rows = []

    for quiz in quiz_results:
        quiz_datetime = parse_datetime(
            quiz.get(
                "quiz_date"
            )
        )

        if quiz_datetime is None:
            continue

        actual_score = safe_float(
            quiz.get(
                "actual_score"
            )
        )

        if actual_score is None:
            recall_accuracy = safe_float(
                quiz.get(
                    "recall_accuracy"
                )
            )

            if recall_accuracy is not None:
                actual_score = (
                    recall_accuracy
                    * 100
                )

        if actual_score is None:
            continue

        if not (
            0
            <= actual_score
            <= 100
        ):
            continue

        rows.append(
            {
                "Date":
                    quiz_datetime,

                "Topic":
                    get_topic_name(
                        quiz
                    ),

                "Subject":
                    get_subject_name(
                        quiz
                    ),

                "Recall Score":
                    actual_score,

                "Delay Days":
                    safe_float(
                        quiz.get(
                            "delay_days"
                        )
                    ),

                "Quiz ID":
                    get_record_id(
                        quiz,
                        "quiz_id"
                    ),
            }
        )

    if not rows:
        return pd.DataFrame()

    dataframe = pd.DataFrame(
        rows
    )

    return dataframe.sort_values(
        "Date"
    )


# recall performance

def show_recall_performance(
    dashboard
):
    st.subheader(
        "Actual Recall Performance"
    )

    dataframe = (
        create_recall_dataframe(
            dashboard
        )
    )

    if dataframe.empty:
        st.info(
            "Actual recall performance will "
            "appear after the first recall test."
        )

        return

    figure = px.line(
        dataframe,
        x="Date",
        y="Recall Score",
        color="Topic",
        markers=True,
        hover_data=[
            "Subject",
            "Delay Days",
            "Quiz ID",
        ],
        title=(
            "Observed Recall Scores "
            "Across Test Dates"
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
        use_container_width=True
    )

    st.caption(
        "This chart shows chronological "
        "quiz performance. It is not one "
        "continuous forgetting curve."
    )


# topic performance

def show_topic_performance(
    dashboard
):
    dataframe = (
        create_recall_dataframe(
            dashboard
        )
    )

    if dataframe.empty:
        return

    dataframe[
        "Topic Label"
    ] = (
        dataframe[
            "Subject"
        ].astype(str)
        + " • "
        + dataframe[
            "Topic"
        ].astype(str)
    )

    topic_summary = (
        dataframe
        .groupby(
            "Topic Label",
            as_index=False
        )
        .agg(
            average_recall=(
                "Recall Score",
                "mean"
            ),
            tests_completed=(
                "Quiz ID",
                "count"
            ),
        )
    )

    topic_summary[
        "average_recall"
    ] = (
        topic_summary[
            "average_recall"
        ].round(2)
    )

    st.subheader(
        "Recall by Topic"
    )

    figure = px.bar(
        topic_summary,
        x="Topic Label",
        y="average_recall",
        hover_data=[
            "tests_completed"
        ],
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
        use_container_width=True
    )


# latest activity

def show_latest_activity(
    dashboard
):
    st.subheader(
        "Latest Activity"
    )

    latest_session = (
        dashboard.get(
            "latest_session"
        )
    )

    latest_quiz = (
        dashboard.get(
            "latest_quiz"
        )
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:
        st.write(
            "#### Latest Study"
        )

        if not latest_session:
            st.info(
                "No study session yet."
            )

        else:
            st.write(
                "**"
                + get_topic_name(
                    latest_session
                )
                + "**"
            )

            st.write(
                format_datetime(
                    latest_session.get(
                        "studied_at"
                    )
                )
            )

            st.caption(
                "Duration: "
                + format_study_duration(
                    latest_session.get(
                        "study_duration",
                        0
                    )
                )
            )

    with col2:
        st.write(
            "#### Latest Recall"
        )

        if not latest_quiz:
            st.info(
                "No recall test yet."
            )

        else:
            score = safe_float(
                latest_quiz.get(
                    "actual_score"
                )
            )

            if score is None:
                recall = safe_float(
                    latest_quiz.get(
                        "recall_accuracy"
                    )
                )

                if recall is not None:
                    score = (
                        recall * 100
                    )

            st.write(
                "**"
                + get_topic_name(
                    latest_quiz
                )
                + "**"
            )

            st.write(
                format_datetime(
                    latest_quiz.get(
                        "quiz_date"
                    )
                )
            )

            st.caption(
                (
                    f"Actual recall: "
                    f"{score:.1f}%"
                )
                if score is not None
                else "Actual recall: N/A"
            )


# quick actions

def show_quick_actions():
    st.subheader(
        "Quick Actions"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:
        st.page_link(
            "pages/2_Study_Session.py",
            label="📚 Study Session",
            use_container_width=True
        )

    with col2:
        st.page_link(
            "pages/3_Retention_Predictor.py",
            label="📉 Predict Retention",
            use_container_width=True
        )

    with col3:
        st.page_link(
            "pages/4_Revision_Scheduler.py",
            label="⏰ Revision Scheduler",
            use_container_width=True
        )

    with col4:
        st.page_link(
            "pages/5_Quiz_and_Recall.py",
            label="🧪 Quiz & Recall",
            use_container_width=True
        )


# explanation

def show_dashboard_explanation():
    with st.expander(
        "How should I read this dashboard?"
    ):
        st.markdown(
            """
The Dashboard is a descriptive learning overview, not a final research conclusion.

**Study Sessions**

Count the learning observations recorded in MemoryMap.

**Recall Tests**

Represent real delayed-recall outcomes.

**Average Recall**

Is calculated from genuine recorded recall measurements. If no recall measurement exists, MemoryMap shows `N/A` instead of treating missing data as 0%.

**Revision Overview**

Shows recommendations generated by the scheduling system.

**Recall Charts**

Describe observed performance. They do not prove that study duration, confidence, difficulty or any other variable caused the result.

Formal prediction evaluation belongs in Analytics and Model Comparison.
            """
        )


# dashboard

def show_dashboard(
    student_id
):
    dashboard = (
        load_dashboard_data(
            student_id
        )
    )

    if dashboard is None:
        return

    progress = (
        calculate_student_progress(
            student_id
        )
    )

    show_header(
        dashboard
    )

    st.divider()

    show_summary_metrics(
        dashboard,
        progress
    )

    st.divider()

    show_next_action(
        dashboard
    )

    st.divider()

    show_learning_status(
        progress
    )

    st.divider()

    show_quick_actions()

    st.divider()

    show_latest_activity(
        dashboard
    )

    st.divider()

    show_revision_overview(
        dashboard
    )

    st.divider()

    show_recent_sessions(
        dashboard
    )

    st.divider()

    show_recall_performance(
        dashboard
    )

    show_topic_performance(
        dashboard
    )

    st.divider()

    st.page_link(
        "pages/6_Analytics.py",
        label="📊 Open Detailed Analytics"
    )

    st.page_link(
        "pages/7_Model_Comparison.py",
        label="⚖️ Open Model Comparison"
    )

    show_dashboard_explanation()


# page

def show_page():
    show_dashboard_flash()

    student_id = (
        select_student()
    )

    if student_id is None:
        return

    show_dashboard(
        student_id
    )


show_page()