from datetime import datetime

import pandas as pd
import streamlit as st

from database import initialize_database

from services.study_service import (
    create_topic,
    create_study_session,
    get_study_page_data,
    get_topic_study_summary,
    get_previous_quiz_score,
    parse_datetime,
)

from utils import (
    format_datetime,
    format_study_duration,
    get_difficulty_label,
    get_confidence_label,
)


st.set_page_config(
    page_title="Study Session | MemoryMap",
    page_icon="📚",
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
    topic
):
    if not topic:
        return "No Subject"

    subject = topic.get(
        "subject_name"
    )

    if not subject:
        subject = topic.get(
            "subject"
        )

    if not subject:
        return "No Subject"

    return str(
        subject
    )


def get_quiz_score(
    quiz
):
    if not quiz:
        return None

    score = quiz.get(
        "actual_score"
    )

    if score is not None:
        try:
            score = float(
                score
            )

            if (
                0
                <= score
                <= 100
            ):
                return score

        except (
            TypeError,
            ValueError
        ):
            pass

    recall = quiz.get(
        "recall_accuracy"
    )

    if recall is not None:
        try:
            recall = float(
                recall
            )

            if (
                0
                <= recall
                <= 1
            ):
                return (
                    recall
                    * 100
                )

        except (
            TypeError,
            ValueError
        ):
            pass

    return None


# flash messages

def show_flash_message():
    flash = st.session_state.pop(
        "study_flash",
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
        st.success(
            message
        )

    elif message_type == "warning":
        st.warning(
            message
        )

    elif message_type == "error":
        st.error(
            message
        )

    else:
        st.info(
            message
        )

    session_id = flash.get(
        "session_id"
    )

    if session_id is not None:
        st.caption(
            f"Study Session ID: "
            f"{session_id}"
        )

        st.page_link(
            "pages/3_Retention_Predictor.py",
            label=(
                "🧠 Continue to "
                "Retention Predictor"
            )
        )


# student

def get_active_student():
    student_id = (
        st.session_state.get(
            "student_id"
        )
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
                    "profile could not be loaded."
                )
            )
        )

        return None

    student = page_data.get(
        "student"
    )

    if not student:
        st.error(
            "The selected student profile "
            "could not be found."
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

        "sessions":
            page_data.get(
                "sessions",
                []
            ),
    }


# topic creation

def create_new_topic(
    student_id
):
    st.subheader(
        "Create New Topic"
    )

    with st.form(
        "new_topic_form"
    ):
        subject = st.text_input(
            "Subject *",
            placeholder=(
                "Example: "
                "Discrete Mathematics"
            )
        )

        topic_name = st.text_input(
            "Topic *",
            placeholder=(
                "Example: Graph Theory"
            )
        )

        difficulty = st.slider(
            "Topic Difficulty",
            min_value=1,
            max_value=5,
            value=3
        )

        st.caption(
            "Difficulty: "
            f"{get_difficulty_label(difficulty)}"
        )

        submitted = (
            st.form_submit_button(
                "Create Topic",
                width="stretch"
            )
        )

    if not submitted:
        return

    result = create_topic(
        student_id=student_id,
        topic_name=topic_name,
        subject=subject,
        difficulty=difficulty
    )

    if not result.get(
        "success",
        False
    ):
        existing_topic = (
            result.get(
                "topic"
            )
        )

        if existing_topic:
            existing_id = (
                get_record_id(
                    existing_topic,
                    "topic_id"
                )
            )

            if existing_id is not None:
                st.session_state[
                    "selected_topic"
                ] = existing_id

        st.error(
            result.get(
                "message",
                "Topic could not be created."
            )
        )

        return

    topic = result.get(
        "topic"
    )

    topic_id = get_record_id(
        topic,
        "topic_id"
    )

    if topic_id is not None:
        st.session_state[
            "selected_topic"
        ] = topic_id

    st.session_state[
        "study_flash"
    ] = {
        "type":
            "success",

        "message":
            (
                f"Topic '{topic_name.strip()}' "
                "created successfully."
            ),
    }

    st.rerun()


# topic selection

def select_topic(
    student_id,
    topics
):
    st.subheader(
        "Select Topic"
    )

    usable_topics = []

    for topic in topics:
        topic_id = get_record_id(
            topic,
            "topic_id"
        )

        if topic_id is None:
            continue

        usable_topics.append(
            topic
        )

    if not usable_topics:
        st.info(
            "You do not have any topics yet."
        )

        create_new_topic(
            student_id
        )

        return None

    option_map = {}

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

        option_map[
            topic_id
        ] = (
            f"{topic_name} • "
            f"{subject_name}"
        )

    topic_ids = list(
        option_map.keys()
    )

    default_topic = (
        st.session_state.get(
            "selected_topic"
        )
    )

    if default_topic not in topic_ids:
        default_topic = (
            topic_ids[
                0
            ]
        )

    default_index = (
        topic_ids.index(
            default_topic
        )
    )

    selected_topic_id = (
        st.selectbox(
            "Study Topic",
            options=topic_ids,
            index=default_index,
            format_func=lambda topic_id:
                option_map[
                    topic_id
                ]
        )
    )

    st.session_state[
        "selected_topic"
    ] = selected_topic_id

    with st.expander(
        "Create Another Topic"
    ):
        create_new_topic(
            student_id
        )

    return selected_topic_id


# topic summary

def load_topic_summary(
    student_id,
    topic_id
):
    summary = (
        get_topic_study_summary(
            student_id,
            topic_id
        )
    )

    if not summary.get(
        "success",
        False
    ):
        st.error(
            summary.get(
                "message",
                "Topic data could not be loaded."
            )
        )

        return None

    return summary


def show_topic_overview(
    summary
):
    topic = summary[
        "topic"
    ]

    session_count = summary.get(
        "study_session_count",
        0
    )

    latest_quiz = summary.get(
        "latest_quiz"
    )

    previous_score = get_quiz_score(
        latest_quiz
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        difficulty = topic.get(
            "difficulty",
            3
        )

        st.metric(
            "Difficulty",
            f"{difficulty}/5"
        )

        st.caption(
            get_difficulty_label(
                difficulty
            )
        )

    with col2:
        st.metric(
            "Previous Sessions",
            session_count
        )

    with col3:
        if previous_score is None:
            score_text = (
                "No Test Yet"
            )

        else:
            score_text = (
                f"{previous_score:.1f}%"
            )

        st.metric(
            "Latest Recall",
            score_text
        )

    with col4:
        total_minutes = float(
            summary.get(
                "total_study_minutes",
                0
            )
        )

        st.metric(
            "Total Study Time",
            format_study_duration(
                total_minutes
            )
        )


# study form

def log_study_session(
    student_id,
    topic_id,
    summary
):
    topic = summary[
        "topic"
    ]

    now = datetime.now().replace(
        second=0,
        microsecond=0
    )

    latest_stored_score = (
        get_previous_quiz_score(
            topic_id
        )
    )

    st.subheader(
        "Record Study Session"
    )

    st.caption(
        "Revision count and previous MemoryMap "
        "recall are calculated using only data "
        "that existed before the study time you "
        "record below."
    )

    with st.form(
        "study_session_form"
    ):
        col1, col2 = st.columns(
            2
        )

        with col1:
            study_duration = (
                st.number_input(
                    "Study Duration (minutes)",
                    min_value=1,
                    max_value=600,
                    value=30,
                    step=5
                )
            )

        with col2:
            confidence_level = (
                st.slider(
                    "Current Confidence",
                    min_value=1,
                    max_value=5,
                    value=3
                )
            )

            st.caption(
                "Confidence: "
                f"{get_confidence_label(confidence_level)}"
            )

        study_date = st.date_input(
            "Study Date",
            value=now.date()
        )

        study_time = st.time_input(
            "Study Time",
            value=now.time()
        )

        st.markdown(
            "**Previous recall information**"
        )

        if latest_stored_score is not None:
            st.info(
                "MemoryMap currently has a recall "
                f"score of {latest_stored_score:.1f}% "
                "in this topic's history. When the "
                "session is saved, MemoryMap will "
                "only use it if that quiz occurred "
                "before the study time above."
            )

        has_external_score = (
            st.checkbox(
                "Add an external previous "
                "quiz/test score"
            )
        )

        external_previous_score = None

        if has_external_score:
            external_previous_score = (
                st.number_input(
                    "External Previous Score (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0
                )
            )

            st.caption(
                "A valid earlier MemoryMap recall "
                "record takes priority over this "
                "external score."
            )

        notes = st.text_area(
            "Notes (optional)",
            placeholder=(
                "Example: Revised BFS and DFS. "
                "Still unsure about traversal order."
            )
        )

        submitted = (
            st.form_submit_button(
                "Save Study Session",
                width="stretch"
            )
        )

    if not submitted:
        return

    studied_at = datetime.combine(
        study_date,
        study_time
    )

    result = (
        create_study_session(
            student_id=student_id,
            topic_id=topic_id,
            study_duration=study_duration,
            confidence_level=confidence_level,
            studied_at=studied_at,
            notes=notes,
            external_previous_score=(
                external_previous_score
            )
        )
    )

    if not result.get(
        "success",
        False
    ):
        st.error(
            result.get(
                "message",
                (
                    "Study session could "
                    "not be recorded."
                )
            )
        )

        return

    session = result.get(
        "session"
    )

    session_id = get_record_id(
        session,
        "session_id"
    )

    if session_id is not None:
        st.session_state[
            "selected_session_id"
        ] = session_id

    revision_count = result.get(
        "revision_count"
    )

    previous_score = result.get(
        "previous_quiz_score"
    )

    previous_score_source = (
        result.get(
            "previous_score_source",
            "Unavailable"
        )
    )

    topic_name = topic.get(
        "topic_name",
        "Topic"
    )

    summary_lines = [
        (
            f"Study session for "
            f"'{topic_name}' saved successfully."
        )
    ]

    if revision_count is not None:
        summary_lines.append(
            (
                f"Previous study sessions: "
                f"{revision_count}."
            )
        )

    if previous_score is not None:
        summary_lines.append(
            (
                f"Previous recall used: "
                f"{float(previous_score):.1f}% "
                f"({previous_score_source})."
            )
        )

    else:
        summary_lines.append(
            (
                "No previous recall score "
                "was available for this session."
            )
        )

    st.session_state[
        "study_flash"
    ] = {
        "type":
            "success",

        "message":
            " ".join(
                summary_lines
            ),

        "session_id":
            session_id,
    }

    st.rerun()


# study history

def show_study_history(
    summary
):
    st.subheader(
        "Topic Study History"
    )

    sessions = summary.get(
        "study_sessions",
        []
    )

    if not sessions:
        st.info(
            "No previous study sessions "
            "for this topic."
        )

        return

    def session_sort_key(
        session
    ):
        parsed = parse_datetime(
            session.get(
                "studied_at"
            )
        )

        if parsed is None:
            return datetime.min

        return parsed

    sessions = sorted(
        sessions,
        key=session_sort_key,
        reverse=True
    )

    rows = []

    for session in sessions:
        previous_score = session.get(
            "previous_quiz_score"
        )

        if previous_score is None:
            score_text = (
                "Not available"
            )

        else:
            try:
                score_text = (
                    f"{float(previous_score):.1f}%"
                )

            except (
                TypeError,
                ValueError
            ):
                score_text = (
                    "Not available"
                )

        confidence = session.get(
            "confidence_level"
        )

        revision_count = session.get(
            "revision_count"
        )

        rows.append(
            {
                "Session ID":
                    get_record_id(
                        session,
                        "session_id"
                    ),

                "Date":
                    format_datetime(
                        session.get(
                            "studied_at"
                        )
                    ),

                "Duration":
                    format_study_duration(
                        session.get(
                            "study_duration",
                            0
                        )
                    ),

                "Confidence":
                    (
                        f"{confidence}/5"
                        if confidence
                        is not None
                        else "N/A"
                    ),

                "Previous Sessions":
                    (
                        revision_count
                        if revision_count
                        is not None
                        else "N/A"
                    ),

                "Previous Recall":
                    score_text,
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        width="stretch",
        hide_index=True
    )


# research explanation

def show_data_explanation():
    with st.expander(
        "Why does MemoryMap collect this data?"
    ):
        st.markdown(
            """
**Study duration**

Records how long the student studied the topic.

**Topic difficulty**

Stores the student's difficulty level for the topic.

**Confidence**

Records how confident the student feels immediately after studying.

**Revision count**

Counts only study sessions that occurred before the current session.

**Previous quiz score**

Uses the most recent valid recall measurement that occurred before the study session. A missing previous score remains missing rather than being treated as 0%.

**Study time**

Provides the timestamp from which delayed recall can later be measured.

These MemoryMap-specific observations are being collected for the longitudinal personalization experiment. The current public bootstrap model was trained using safe Duolingo history features, not fabricated versions of these personal variables.
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

    student_name = student.get(
        "name"
    )

    if not student_name:
        student_name = (
            st.session_state.get(
                "student_name"
            )
            or "Student"
        )

    st.session_state[
        "student_name"
    ] = student_name

    st.title(
        "📚 Study Session"
    )

    st.write(
        "Record learning data for "
        f"**{student_name}**."
    )

    st.caption(
        "Each saved session becomes a dated "
        "observation that can later be compared "
        "with actual delayed recall."
    )

    st.divider()

    topic_id = select_topic(
        student_id,
        topics
    )

    if topic_id is None:
        return

    summary = load_topic_summary(
        student_id,
        topic_id
    )

    if summary is None:
        return

    st.divider()

    show_topic_overview(
        summary
    )

    st.divider()

    log_study_session(
        student_id,
        topic_id,
        summary
    )

    st.divider()

    show_study_history(
        summary
    )

    st.divider()

    show_data_explanation()


show_page()
