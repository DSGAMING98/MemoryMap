from datetime import datetime
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    TARGET_RETENTION_THRESHOLD,
    MAX_REVISION_INTERVAL_DAYS,
)

from core.forgetting_curve import (
    calculate_retention,
)

from core.predictor import (
    FIXED_MODE,
    PERSONALIZED_MODE,
    ADAPTIVE_MODE,
)

from database import (
    initialize_database,
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

from services.revision_service import (
    generate_revision_schedule,
    preview_all_schedules,
    save_revision_schedule,
    get_revision_overview,
    complete_revision,
    get_next_revision,
)

from utils import (
    format_datetime,
    format_percentage,
)


st.set_page_config(
    page_title="Revision Scheduler | MemoryMap",
    page_icon="⏰",
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
        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return default

    if not math.isfinite(
        value
    ):
        return default

    return value


def format_days_value(
    value
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    if value < 1:
        hours = (
            value
            * 24
        )

        return (
            f"{hours:.1f} hr"
            if not math.isclose(
                hours,
                1.0
            )
            else "1.0 hr"
        )

    if math.isclose(
        value,
        1.0
    ):
        return "1.00 day"

    return (
        f"{value:.2f} days"
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


# flash messages

def show_flash_message():
    flash = st.session_state.pop(
        "revision_flash",
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

    revision_id = flash.get(
        "revision_id"
    )

    if revision_id is not None:
        st.caption(
            f"Revision ID: "
            f"{revision_id}"
        )


# active student

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
            "or select a student first."
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
                    "Student information "
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
                "📚 Create a Topic"
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
            topic_ids[
                0
            ]
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
            key="revision_topic"
        )
    )

    st.session_state[
        "selected_topic"
    ] = selected_topic_id

    topic = get_topic_profile(
        selected_topic_id
    )

    if topic is None:
        st.error(
            "Topic information could "
            "not be loaded."
        )

        return None

    return topic


# session selection

def session_sort_key(
    session
):
    studied_at = parse_datetime(
        session.get(
            "studied_at"
        )
    )

    if studied_at is None:
        return datetime.min

    return studied_at


def select_study_session(
    student_id,
    topic_id
):
    sessions = (
        list_topic_sessions(
            student_id,
            topic_id
        )
    )

    if not sessions:
        st.warning(
            "This topic does not have "
            "any study sessions yet."
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
        session_id = (
            get_record_id(
                session,
                "session_id"
            )
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
            session_ids[
                0
            ]
        )

    selected_index = (
        session_ids.index(
            default_session
        )
    )

    session_id = (
        st.selectbox(
            "Study Session",
            options=session_ids,
            index=selected_index,
            format_func=lambda item:
                session_map[
                    item
                ],
            key="revision_session"
        )
    )

    st.session_state[
        "selected_session_id"
    ] = session_id

    for session in sessions:
        current_id = (
            get_record_id(
                session,
                "session_id"
            )
        )

        if current_id == session_id:
            return session

    return None


# selected session

def show_session_information(
    session
):
    st.subheader(
        "Selected Study Session"
    )

    studied_at = parse_datetime(
        session.get(
            "studied_at"
        )
    )

    elapsed_days = None

    if studied_at is not None:
        elapsed_seconds = (
            datetime.now()
            - studied_at
        ).total_seconds()

        elapsed_days = max(
            elapsed_seconds
            / 86400,
            0.0
        )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Previous Sessions",
            session.get(
                "revision_count",
                "N/A"
            )
        )

    with col2:
        confidence = session.get(
            "confidence_level"
        )

        st.metric(
            "Confidence",
            (
                f"{confidence}/5"
                if confidence
                is not None
                else "N/A"
            )
        )

    with col3:
        duration = safe_float(
            session.get(
                "study_duration"
            )
        )

        st.metric(
            "Study Duration",
            (
                f"{duration:g} min"
                if duration
                is not None
                else "N/A"
            )
        )

    with col4:
        st.metric(
            "Time Since Study",
            format_days_value(
                elapsed_days
            )
        )

    st.caption(
        "Studied at: "
        f"{format_datetime(session.get('studied_at'))}"
    )

    previous_score = safe_float(
        session.get(
            "previous_quiz_score"
        )
    )

    st.caption(
        "Previous recall stored with session: "
        + (
            f"{previous_score:.1f}%"
            if previous_score
            is not None
            else "Not available"
        )
    )


# scheduler settings

def select_scheduler_settings():
    st.subheader(
        "Scheduling Method"
    )

    label_map = {
        FIXED_MODE:
            "Fixed Schedule",

        PERSONALIZED_MODE:
            "Personalized Schedule",

        ADAPTIVE_MODE:
            "Adaptive Schedule",
    }

    modes = [
        FIXED_MODE,
        PERSONALIZED_MODE,
        ADAPTIVE_MODE,
    ]

    selected_mode = (
        st.radio(
            "Choose revision strategy:",
            modes,
            format_func=lambda mode:
                label_map.get(
                    mode,
                    str(
                        mode
                    )
                ),
            horizontal=True,
            key="revision_strategy"
        )
    )

    threshold_percentage = (
        st.slider(
            "Desired minimum retention",
            min_value=50,
            max_value=90,
            value=int(
                TARGET_RETENTION_THRESHOLD
                * 100
            ),
            step=5,
            key="revision_threshold"
        )
    )

    threshold = (
        threshold_percentage
        / 100
    )

    return (
        selected_mode,
        threshold,
    )


# personal feature context

def get_live_features(
    session_id,
    reference_days=1.0
):
    result = (
        create_memorymap_feature_values(
            session_id,
            reference_days
        )
    )

    if not result.get(
        "success",
        False
    ):
        return {}

    return (
        result.get(
            "features"
        )
        or {}
    )


# ML model context

def get_model_context():
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

    return {
        "information":
            information,

        "metadata":
            metadata,

        "personal_model":
            bool(
                metadata.get(
                    "memorymap_personal_model",
                    False
                )
            ),
    }


def show_model_context(
    selected_mode,
    model_context
):
    if selected_mode == FIXED_MODE:
        return

    information = (
        model_context[
            "information"
        ]
    )

    if not information.get(
        "available",
        False
    ):
        st.warning(
            "No trained ML artifact is "
            "currently available."
        )

        st.caption(
            "MemoryMap can still use the "
            "fixed forgetting-curve fallback."
        )

        return

    model_name = information.get(
        "model_name",
        "Unknown"
    )

    if model_context[
        "personal_model"
    ]:
        st.success(
            f"{model_name} is available as "
            "a MemoryMap personalized model."
        )

    else:
        st.info(
            f"{model_name} is currently a "
            "public bootstrap model, not a "
            "MemoryMap personalized model."
        )

        st.caption(
            "If personalized inputs required "
            "by the model are unavailable, "
            "the scheduler must fall back "
            "rather than fabricate them."
        )


# schedule generation

def generate_schedule(
    student_id,
    session_id,
    mode,
    threshold,
    extra_features
):
    current_time = datetime.now()

    result = (
        generate_revision_schedule(
            student_id=student_id,
            session_id=session_id,
            mode=mode,
            threshold=threshold,
            current_time=current_time,
            reference_prediction_days=1.0,
            extra_features=(
                extra_features
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
                    "Revision schedule could "
                    "not be generated."
                )
            )
        )

        return None

    return result


# adaptive recall context

def show_recall_context(
    mode,
    schedule
):
    if mode != ADAPTIVE_MODE:
        return

    context = schedule.get(
        "observed_recall_context",
        {}
    )

    st.subheader(
        "Adaptive Recall Evidence"
    )

    if not context.get(
        "available",
        False
    ):
        st.info(
            "No usable delayed recall "
            "measurement is available yet."
        )

        st.caption(
            "Adaptive scheduling will use "
            "the next safe fallback defined "
            "by the scheduling engine."
        )

        return

    recall = safe_float(
        context.get(
            "recall"
        )
    )

    delay_days = safe_float(
        context.get(
            "delay_days"
        )
    )

    col1, col2 = st.columns(
        2
    )

    with col1:
        st.metric(
            "Observed Recall",
            (
                format_percentage(
                    recall
                )
                if recall
                is not None
                else "N/A"
            )
        )

    with col2:
        st.metric(
            "Observed Delay",
            format_days_value(
                delay_days
            )
        )

    quiz = context.get(
        "quiz"
    )

    if quiz:
        st.caption(
            "Recall measurement: "
            f"{format_datetime(quiz.get('quiz_date'))}"
        )


# recommendation

def show_recommendation(
    topic,
    schedule
):
    st.subheader(
        "Recommended Revision"
    )

    half_life = safe_float(
        schedule.get(
            "estimated_half_life"
        )
    )

    if half_life is None:
        half_life = safe_float(
            schedule.get(
                "half_life_days"
            )
        )

    current_retention = safe_float(
        schedule.get(
            "current_retention"
        )
    )

    threshold = safe_float(
        schedule.get(
            "threshold"
        )
    )

    interval_days = safe_float(
        schedule.get(
            "interval_days"
        )
    )

    scheduled_for = (
        parse_datetime(
            schedule.get(
                "scheduled_for"
            )
        )
    )

    status = schedule.get(
        "status",
        "Unknown"
    )

    priority = schedule.get(
        "priority",
        "Unknown"
    )

    source = schedule.get(
        "source",
        "Unknown"
    )

    fallback_used = bool(
        schedule.get(
            "fallback_used",
            False
        )
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Current Retention",
            (
                format_percentage(
                    current_retention
                )
                if current_retention
                is not None
                else "N/A"
            )
        )

    with col2:
        st.metric(
            "Memory Half-Life",
            format_days_value(
                half_life
            )
        )

    with col3:
        st.metric(
            "Revision Interval",
            format_days_value(
                interval_days
            )
        )

    with col4:
        st.metric(
            "Status",
            status
        )

    st.write(
        "**Topic:** "
        f"{topic.get('topic_name', 'Unknown')}"
    )

    st.write(
        "**Scheduling mode:** "
        f"{schedule.get('schedule_mode', schedule.get('mode', 'Unknown'))}"
    )

    st.write(
        "**Evidence source:** "
        f"{source}"
    )

    st.write(
        "**Priority:** "
        f"{priority}"
    )

    st.write(
        "**Minimum desired retention:** "
        + (
            f"{threshold * 100:.0f}%"
            if threshold
            is not None
            else "N/A"
        )
    )

    st.write(
        "**Recommended revision time:** "
        + (
            format_datetime(
                scheduled_for
            )
            if scheduled_for
            is not None
            else "N/A"
        )
    )

    recommendation = schedule.get(
        "recommendation"
    )

    if recommendation:
        st.info(
            recommendation
        )

    if fallback_used:
        st.warning(
            "Fallback scheduling was used."
        )

        st.caption(
            "The requested strategy did not "
            "have enough genuine information "
            "to estimate its own half-life, "
            "so MemoryMap used a safer "
            "fallback instead."
        )

    if status == "Due":
        st.error(
            "This revision is already due."
        )

    elif status == "Due Soon":
        st.warning(
            "This revision is due soon."
        )

    else:
        st.success(
            "The revision is currently "
            "scheduled for the future."
        )


# retention forecast

def show_retention_forecast(
    schedule
):
    half_life = safe_float(
        schedule.get(
            "estimated_half_life"
        )
    )

    if half_life is None:
        half_life = safe_float(
            schedule.get(
                "half_life_days"
            )
        )

    threshold = safe_float(
        schedule.get(
            "threshold"
        )
    )

    interval_days = safe_float(
        schedule.get(
            "interval_days"
        )
    )

    if (
        half_life is None
        or half_life <= 0
        or threshold is None
        or interval_days is None
    ):
        return

    st.subheader(
        "Retention Forecast"
    )

    maximum_days = max(
        7.0,
        interval_days
        * 1.5
    )

    maximum_days = min(
        maximum_days,
        float(
            MAX_REVISION_INTERVAL_DAYS
        )
    )

    points = 160

    rows = []

    for index in range(
        points
    ):
        day = (
            maximum_days
            * index
            / (
                points
                - 1
            )
        )

        retention = (
            calculate_retention(
                day,
                half_life
            )
        )

        rows.append(
            {
                "Days Since Study":
                    day,

                "Retention":
                    retention
                    * 100,
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    figure = px.line(
        dataframe,
        x="Days Since Study",
        y="Retention",
        title=(
            "Estimated Retention "
            "and Revision Point"
        )
    )

    figure.add_hline(
        y=threshold * 100,
        line_dash="dash",
        annotation_text=(
            "Retention Threshold"
        )
    )

    figure.add_vline(
        x=interval_days,
        line_dash="dash",
        annotation_text=(
            "Recommended Revision"
        )
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Retention (%)"
    )

    figure.update_xaxes(
        title="Days Since Study"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )


# strategy comparison

def show_strategy_comparison(
    student_id,
    session_id,
    threshold,
    extra_features
):
    with st.expander(
        "Compare all three scheduling strategies"
    ):
        result = (
            preview_all_schedules(
                student_id=student_id,
                session_id=session_id,
                threshold=threshold,
                current_time=datetime.now(),
                extra_features=(
                    extra_features
                )
            )
        )

        if not result.get(
            "success",
            False
        ):
            st.warning(
                result.get(
                    "message",
                    (
                        "Schedule comparison "
                        "could not be generated."
                    )
                )
            )

            return

        schedules = (
            result.get(
                "schedules"
            )
            or {}
        )

        rows = []

        for name, schedule in (
            schedules.items()
        ):
            half_life = safe_float(
                schedule.get(
                    "estimated_half_life"
                )
            )

            if half_life is None:
                half_life = safe_float(
                    schedule.get(
                        "half_life_days"
                    )
                )

            rows.append(
                {
                    "Strategy":
                        schedule.get(
                            "mode",
                            name
                        ),

                    "Interval":
                        format_days_value(
                            schedule.get(
                                "interval_days"
                            )
                        ),

                    "Recommended":
                        format_datetime(
                            schedule.get(
                                "scheduled_for"
                            )
                        ),

                    "Half-Life":
                        format_days_value(
                            half_life
                        ),

                    "Current Retention":
                        (
                            format_percentage(
                                schedule.get(
                                    "current_retention"
                                )
                            )
                            if schedule.get(
                                "current_retention"
                            )
                            is not None
                            else "N/A"
                        ),

                    "Status":
                        schedule.get(
                            "status",
                            "Unknown"
                        ),

                    "Source":
                        schedule.get(
                            "source",
                            "Unknown"
                        ),

                    "Fallback":
                        (
                            "Yes"
                            if schedule.get(
                                "fallback_used",
                                False
                            )
                            else "No"
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

        recall_context = (
            result.get(
                "recall_context"
            )
            or {}
        )

        if recall_context.get(
            "available",
            False
        ):
            st.caption(
                "Adaptive comparison includes "
                "a valid previous delayed recall "
                "measurement."
            )

        else:
            st.caption(
                "No usable delayed recall is "
                "currently available, so the "
                "adaptive strategy may use a fallback."
            )


# save schedule

def save_generated_schedule(
    schedule
):
    result = (
        save_revision_schedule(
            schedule
        )
    )

    if result.get(
        "success",
        False
    ):
        revision = (
            result.get(
                "revision"
            )
            or {}
        )

        revision_id = (
            get_record_id(
                revision,
                "revision_id"
            )
        )

        st.session_state[
            "revision_flash"
        ] = {
            "type":
                "success",

            "message":
                result.get(
                    "message",
                    (
                        "Revision schedule "
                        "saved successfully."
                    )
                ),

            "revision_id":
                revision_id,
        }

        st.rerun()

        return

    if result.get(
        "duplicate",
        False
    ):
        st.warning(
            result.get(
                "message",
                (
                    "A matching revision "
                    "schedule already exists."
                )
            )
        )

        return

    st.error(
        result.get(
            "message",
            (
                "Revision schedule "
                "could not be saved."
            )
        )
    )


# next revision

def show_next_revision(
    student_id
):
    next_revision = (
        get_next_revision(
            student_id
        )
    )

    if not next_revision:
        return

    revision = next_revision.get(
        "revision"
    )

    if not revision:
        return

    status = next_revision.get(
        "status",
        "Unknown"
    )

    topic_name = revision.get(
        "topic_name",
        "Revision"
    )

    scheduled_for = (
        revision.get(
            "scheduled_for"
        )
    )

    st.info(
        f"Next revision: **{topic_name}** "
        f"• {status} • "
        f"{format_datetime(scheduled_for)}"
    )


# completion action

def mark_revision_done(
    revision
):
    revision_id = (
        get_record_id(
            revision,
            "revision_id"
        )
    )

    if revision_id is None:
        st.error(
            "Revision ID is missing."
        )

        return

    result = (
        complete_revision(
            revision_id
        )
    )

    if result.get(
        "success",
        False
    ):
        st.session_state[
            "revision_flash"
        ] = {
            "type":
                "success",

            "message":
                result.get(
                    "message",
                    (
                        "Revision marked "
                        "as completed."
                    )
                ),
        }

        st.rerun()

        return

    st.error(
        result.get(
            "message",
            (
                "Revision could not "
                "be completed."
            )
        )
    )


# due revisions

def show_due_revisions(
    overview
):
    st.subheader(
        "Revisions Due"
    )

    due = overview.get(
        "due",
        []
    )

    if not due:
        st.success(
            "No revisions are due right now."
        )

        return

    for revision in due:
        revision_id = (
            get_record_id(
                revision,
                "revision_id"
            )
        )

        with st.container(
            border=True
        ):
            col1, col2 = (
                st.columns(
                    [
                        4,
                        1
                    ]
                )
            )

            with col1:
                st.write(
                    "### "
                    + str(
                        revision.get(
                            "topic_name",
                            "Revision"
                        )
                    )
                )

                st.write(
                    "Subject: "
                    + str(
                        revision.get(
                            "subject_name"
                        )
                        or "Not specified"
                    )
                )

                st.write(
                    "Scheduled: "
                    f"{format_datetime(revision.get('scheduled_for'))}"
                )

                st.write(
                    "Priority: "
                    f"**{revision.get('priority', 'Unknown')}**"
                )

                schedule_mode = (
                    revision.get(
                        "schedule_mode"
                    )
                )

                if schedule_mode:
                    st.caption(
                        "Mode: "
                        f"{schedule_mode}"
                    )

            with col2:
                if st.button(
                    "Mark Complete",
                    key=(
                        f"complete_revision_"
                        f"{revision_id}"
                    ),
                    width="stretch"
                ):
                    mark_revision_done(
                        revision
                    )


# upcoming revisions

def show_upcoming_revisions(
    overview
):
    st.subheader(
        "Upcoming Revisions"
    )

    upcoming = overview.get(
        "upcoming",
        []
    )

    if not upcoming:
        st.info(
            "No upcoming revisions "
            "are scheduled."
        )

        return

    rows = []

    for revision in upcoming:
        threshold = safe_float(
            revision.get(
                "threshold"
            )
        )

        rows.append(
            {
                "Revision ID":
                    get_record_id(
                        revision,
                        "revision_id"
                    ),

                "Topic":
                    revision.get(
                        "topic_name",
                        "Unknown"
                    ),

                "Subject":
                    (
                        revision.get(
                            "subject_name"
                        )
                        or "Not specified"
                    ),

                "Scheduled For":
                    format_datetime(
                        revision.get(
                            "scheduled_for"
                        )
                    ),

                "Threshold":
                    (
                        f"{threshold * 100:.0f}%"
                        if threshold
                        is not None
                        else "N/A"
                    ),

                "Priority":
                    revision.get(
                        "priority",
                        "Unknown"
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

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        width="stretch",
        hide_index=True
    )


# revision overview

def show_revision_overview(
    student_id
):
    overview = (
        get_revision_overview(
            student_id
        )
    )

    if not overview.get(
        "success",
        False
    ):
        st.error(
            overview.get(
                "message",
                (
                    "Revision overview "
                    "could not be loaded."
                )
            )
        )

        return

    st.subheader(
        "Revision Queue"
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    with col1:
        st.metric(
            "Due",
            overview.get(
                "due_count",
                0
            )
        )

    with col2:
        st.metric(
            "Upcoming",
            overview.get(
                "upcoming_count",
                0
            )
        )

    with col3:
        st.metric(
            "Total Pending",
            overview.get(
                "total_pending",
                0
            )
        )

    show_next_revision(
        student_id
    )

    st.divider()

    show_due_revisions(
        overview
    )

    st.divider()

    show_upcoming_revisions(
        overview
    )


# explanation

def show_scheduler_explanation():
    with st.expander(
        "How does the revision scheduler work?"
    ):
        st.markdown(
            """
**Fixed Schedule**

Uses the fixed forgetting-curve assumption and the configured default memory half-life.

**Personalized Schedule**

Attempts to use a model derived from genuine MemoryMap personalization data. If the required information is unavailable, the scheduling engine reports and uses its fallback rather than inventing missing features.

**Adaptive Schedule**

Uses a valid previous delayed-recall observation when one exists. Recall values of exactly 0% or 100% cannot be inverted into a finite single-observation half-life, so they are retained as quiz outcomes but are not used for this half-life calculation.

**Retention threshold**

The selected percentage represents the minimum desired retention. The scheduler estimates the point at which retention reaches that threshold and recommends revision around that time.

A schedule that is already in the past is not moved to the present. It remains marked **Due**, preserving what the model originally recommended.
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
        "⏰ Revision Scheduler"
    )

    st.write(
        "Determine when "
        f"**{student_name}** "
        "should revisit a topic."
    )

    st.caption(
        "The scheduler begins with a fixed "
        "forgetting curve and becomes genuinely "
        "adaptive as delayed recall data accumulates."
    )

    st.divider()

    topic = select_topic(
        student_id,
        topics
    )

    if topic is None:
        return

    topic_id = get_record_id(
        topic,
        "topic_id"
    )

    session = (
        select_study_session(
            student_id,
            topic_id
        )
    )

    if session is None:
        return

    session_id = get_record_id(
        session,
        "session_id"
    )

    st.divider()

    show_session_information(
        session
    )

    st.divider()

    mode, threshold = (
        select_scheduler_settings()
    )

    model_context = (
        get_model_context()
    )

    show_model_context(
        mode,
        model_context
    )

    extra_features = (
        get_live_features(
            session_id,
            reference_days=1.0
        )
    )

    result = generate_schedule(
        student_id=student_id,
        session_id=session_id,
        mode=mode,
        threshold=threshold,
        extra_features=(
            extra_features
        )
    )

    if result is None:
        return

    schedule = result[
        "schedule"
    ]

    st.divider()

    show_recall_context(
        mode,
        schedule
    )

    show_recommendation(
        topic,
        schedule
    )

    show_retention_forecast(
        schedule
    )

    show_strategy_comparison(
        student_id=student_id,
        session_id=session_id,
        threshold=threshold,
        extra_features=(
            extra_features
        )
    )

    st.divider()

    st.subheader(
        "Save Recommendation"
    )

    if st.button(
        "Save Revision Schedule",
        width="stretch"
    ):
        save_generated_schedule(
            schedule
        )

    st.caption(
        "A revision that is already due can "
        "still be saved. MemoryMap preserves "
        "the originally recommended time."
    )

    st.divider()

    show_revision_overview(
        student_id
    )

    st.divider()

    st.page_link(
        "pages/5_Quiz_and_Recall.py",
        label=(
            "🧪 Continue to Quiz & Recall"
        )
    )

    show_scheduler_explanation()


show_page()
