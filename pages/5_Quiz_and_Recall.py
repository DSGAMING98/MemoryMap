from datetime import datetime, timedelta
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from database import (
    initialize_database,
    add_quiz_result,
    get_predictions_for_session,
)

from core.quiz_engine import (
    evaluate_recall_test,
    create_database_payload,
    summarize_recall_results,
    classify_checkpoint,
)

from services.study_service import (
    get_study_page_data,
    list_topic_sessions,
    get_topic_profile,
    list_topic_quizzes,
    parse_datetime,
)

from utils import (
    format_datetime,
    format_study_duration,
    format_percentage,
    get_confidence_label,
)


st.set_page_config(
    page_title="Quiz & Recall | MemoryMap",
    page_icon="🧪",
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


def safe_int(
    value,
    default=None
):
    if value is None:
        return default

    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError
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


# flash result

def show_flash_message():
    flash = st.session_state.pop(
        "quiz_flash",
        None
    )

    if not flash:
        return None

    message = flash.get(
        "message",
        ""
    )

    message_type = flash.get(
        "type",
        "success"
    )

    if message_type == "error":
        st.error(
            message
        )

    elif message_type == "warning":
        st.warning(
            message
        )

    elif message_type == "info":
        st.info(
            message
        )

    else:
        st.success(
            message
        )

    quiz_id = flash.get(
        "quiz_id"
    )

    if quiz_id is not None:
        st.caption(
            f"Quiz ID: {quiz_id}"
        )

    return flash


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
        "Select Recall Test Data"
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

    topic_id = st.selectbox(
        "Topic",
        options=topic_ids,
        index=selected_index,
        format_func=lambda item:
            topic_map[
                item
            ],
        key="quiz_topic"
    )

    st.session_state[
        "selected_topic"
    ] = topic_id

    topic = get_topic_profile(
        topic_id
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
            "No study sessions exist "
            "for this topic."
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

    session_id = st.selectbox(
        "Study Session Being Tested",
        options=session_ids,
        index=selected_index,
        format_func=lambda item:
            session_map[
                item
            ],
        key="quiz_session"
    )

    st.session_state[
        "selected_session_id"
    ] = session_id

    for session in sessions:
        current_id = get_record_id(
            session,
            "session_id"
        )

        if current_id == session_id:
            return session

    return None


# session information

def show_session_information(
    session,
    topic
):
    st.subheader(
        "Study Session Information"
    )

    study_duration = safe_float(
        session.get(
            "study_duration"
        )
    )

    confidence = safe_int(
        session.get(
            "confidence_level"
        )
    )

    revision_count = (
        session.get(
            "revision_count"
        )
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Topic",
            topic.get(
                "topic_name",
                "Unknown"
            )
        )

    with col2:
        st.metric(
            "Study Duration",
            (
                format_study_duration(
                    study_duration
                )
                if study_duration
                is not None
                else "N/A"
            )
        )

    with col3:
        st.metric(
            "Confidence Before Test",
            (
                f"{confidence}/5"
                if confidence
                is not None
                else "N/A"
            )
        )

        if confidence is not None:
            st.caption(
                get_confidence_label(
                    confidence
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

    st.caption(
        "Study session recorded at: "
        f"{format_datetime(session.get('studied_at'))}"
    )


# saved predictions

def show_saved_predictions(
    session_id
):
    predictions = (
        get_predictions_for_session(
            session_id
        )
    )

    st.subheader(
        "Predictions Made Before Testing"
    )

    if not predictions:
        st.warning(
            "No retention prediction has "
            "been saved for this study session."
        )

        st.info(
            "You may still record an actual recall "
            "result, but it cannot become a clean "
            "predicted-vs-actual research observation "
            "unless a prediction existed before the test."
        )

        st.page_link(
            "pages/3_Retention_Predictor.py",
            label=(
                "📉 Open Retention Predictor"
            )
        )

        return

    rows = []

    for prediction in predictions:
        fixed_prediction = (
            safe_float(
                prediction.get(
                    "fixed_prediction"
                )
            )
        )

        personalized_prediction = (
            safe_float(
                prediction.get(
                    "personalized_prediction"
                )
            )
        )

        adaptive_prediction = (
            safe_float(
                prediction.get(
                    "adaptive_prediction"
                )
            )
        )

        prediction_days = safe_float(
            prediction.get(
                "prediction_for_days"
            )
        )

        rows.append(
            {
                "Prediction ID":
                    get_record_id(
                        prediction,
                        "prediction_id"
                    ),

                "Interval":
                    (
                        f"{prediction_days:.2f} days"
                        if prediction_days
                        is not None
                        else "N/A"
                    ),

                "Fixed":
                    (
                        format_percentage(
                            fixed_prediction
                        )
                        if fixed_prediction
                        is not None
                        else "N/A"
                    ),

                "Personalized":
                    (
                        format_percentage(
                            personalized_prediction
                        )
                        if personalized_prediction
                        is not None
                        else "N/A"
                    ),

                "Adaptive":
                    (
                        format_percentage(
                            adaptive_prediction
                        )
                        if adaptive_prediction
                        is not None
                        else "N/A"
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
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Only predictions created on or before "
        "the recall-test timestamp are eligible "
        "for comparison."
    )


# duplicate protection

def find_duplicate_quiz(
    topic_id,
    session_id,
    quiz_datetime,
    tolerance_minutes=5
):
    quizzes = (
        list_topic_quizzes(
            topic_id
        )
    )

    tolerance_seconds = (
        tolerance_minutes
        * 60
    )

    for quiz in quizzes:
        quiz_session_id = (
            get_record_id(
                quiz,
                "session_id"
            )
        )

        try:
            same_session = (
                int(
                    quiz_session_id
                )
                == int(
                    session_id
                )
            )

        except (
            TypeError,
            ValueError
        ):
            same_session = False

        if not same_session:
            continue

        existing_datetime = (
            parse_datetime(
                quiz.get(
                    "quiz_date"
                )
            )
        )

        if existing_datetime is None:
            continue

        difference = abs(
            (
                existing_datetime
                - quiz_datetime
            ).total_seconds()
        )

        if (
            difference
            <= tolerance_seconds
        ):
            return quiz

    return None


# form validation

def validate_quiz_datetime(
    studied_at,
    quiz_datetime
):
    study_datetime = (
        parse_datetime(
            studied_at
        )
    )

    if study_datetime is None:
        return (
            False,
            (
                "The study-session timestamp "
                "is invalid."
            )
        )

    if quiz_datetime < study_datetime:
        return (
            False,
            (
                "The recall test cannot occur "
                "before the study session."
            )
        )

    current_time = datetime.now()

    if (
        quiz_datetime
        > current_time
        + timedelta(
            minutes=1
        )
    ):
        return (
            False,
            (
                "A future recall test cannot "
                "be recorded as an actual result."
            )
        )

    return (
        True,
        None
    )


# checkpoint information

def get_checkpoint_information(
    delay_days
):
    try:
        checkpoint = (
            classify_checkpoint(
                delay_days,
                tolerance_days=0.5
            )
        )

    except (
        TypeError,
        ValueError
    ):
        return {
            "matched":
                False,

            "label":
                "Custom Recall",

            "checkpoint_days":
                None,

            "difference_days":
                None,
        }

    return checkpoint


# save recall

def save_recall_result(
    student_id,
    topic_id,
    session,
    total_questions,
    correct_answers,
    confidence_after_quiz,
    quiz_datetime
):
    session_id = get_record_id(
        session,
        "session_id"
    )

    valid_time, error = (
        validate_quiz_datetime(
            session.get(
                "studied_at"
            ),
            quiz_datetime
        )
    )

    if not valid_time:
        st.error(
            error
        )

        return

    duplicate = (
        find_duplicate_quiz(
            topic_id=topic_id,
            session_id=session_id,
            quiz_datetime=(
                quiz_datetime
            )
        )
    )

    if duplicate is not None:
        st.warning(
            "A recall result already exists "
            "for this study session at "
            "approximately the same test time."
        )

        st.caption(
            "Existing Quiz ID: "
            f"{get_record_id(duplicate, 'quiz_id')}"
        )

        return

    predictions = (
        get_predictions_for_session(
            session_id
        )
    )

    try:
        evaluation = (
            evaluate_recall_test(
                predictions=predictions,
                studied_at=(
                    session.get(
                        "studied_at"
                    )
                ),
                quiz_at=(
                    quiz_datetime
                ),
                correct_answers=int(
                    correct_answers
                ),
                total_questions=int(
                    total_questions
                ),
                confidence_after_quiz=int(
                    confidence_after_quiz
                ),
                tolerance_days=0.5
            )
        )

    except Exception as error:
        st.error(
            "The recall result could "
            "not be evaluated."
        )

        st.caption(
            str(
                error
            )
        )

        return

    quiz_result = evaluation.get(
        "quiz_result"
    )

    if not quiz_result:
        st.error(
            "The quiz engine did not return "
            "a valid recall result."
        )

        return

    payload = (
        create_database_payload(
            student_id=student_id,
            topic_id=topic_id,
            session_id=session_id,
            quiz_result=(
                quiz_result
            )
        )
    )

    try:
        quiz_id = add_quiz_result(
            **payload
        )

    except Exception as error:
        st.error(
            "The recall result could "
            "not be saved."
        )

        st.caption(
            str(
                error
            )
        )

        return

    st.session_state[
        "quiz_flash"
    ] = {
        "type":
            "success",

        "message":
            (
                "Actual recall result "
                "saved successfully."
            ),

        "quiz_id":
            quiz_id,

        "session_id":
            session_id,

        "evaluation":
            evaluation,
    }

    st.rerun()


# recall form

def show_quiz_form(
    student_id,
    topic,
    session
):
    st.subheader(
        "Record Actual Recall Test"
    )

    st.write(
        "Enter the student's real performance "
        "from a completed recall test."
    )

    st.info(
        "For useful research data, use comparable "
        "question difficulty and scoring rules across "
        "the 1-day, 3-day and 7-day checkpoints."
    )

    with st.form(
        "recall_test_form"
    ):
        total_questions = (
            st.number_input(
                "Total Questions",
                min_value=1,
                max_value=100,
                value=10,
                step=1
            )
        )

        correct_answers = (
            st.number_input(
                "Correct Answers",
                min_value=0,
                max_value=int(
                    total_questions
                ),
                value=0,
                step=1
            )
        )

        confidence_after_quiz = (
            st.slider(
                "Confidence After Recall Test",
                min_value=1,
                max_value=5,
                value=3
            )
        )

        st.caption(
            "Confidence: "
            f"{get_confidence_label(confidence_after_quiz)}"
        )

        now = datetime.now()

        quiz_date = (
            st.date_input(
                "Recall Test Date",
                value=now.date()
            )
        )

        quiz_time = (
            st.time_input(
                "Recall Test Time",
                value=now.time()
            )
        )

        confirm_actual_test = (
            st.checkbox(
                "I confirm that this score "
                "comes from an actual completed "
                "recall test and is not a "
                "predicted, estimated or "
                "fabricated score."
            )
        )

        submitted = (
            st.form_submit_button(
                "Save Recall Result",
                use_container_width=True
            )
        )

    if not submitted:
        return

    if not confirm_actual_test:
        st.error(
            "Confirm that the result comes "
            "from an actual recall test."
        )

        return

    quiz_datetime = (
        datetime.combine(
            quiz_date,
            quiz_time
        )
    )

    save_recall_result(
        student_id=student_id,
        topic_id=get_record_id(
            topic,
            "topic_id"
        ),
        session=session,
        total_questions=(
            total_questions
        ),
        correct_answers=(
            correct_answers
        ),
        confidence_after_quiz=(
            confidence_after_quiz
        ),
        quiz_datetime=(
            quiz_datetime
        )
    )


# comparison result

def show_prediction_comparison(
    evaluation
):
    if not evaluation:
        return

    quiz_result = (
        evaluation.get(
            "quiz_result"
        )
        or {}
    )

    match = (
        evaluation.get(
            "prediction_match"
        )
        or {}
    )

    comparisons = (
        evaluation.get(
            "comparisons"
        )
        or []
    )

    actual_recall = safe_float(
        quiz_result.get(
            "recall_accuracy"
        )
    )

    delay_days = safe_float(
        quiz_result.get(
            "delay_days"
        )
    )

    checkpoint = (
        get_checkpoint_information(
            delay_days
        )
        if delay_days
        is not None
        else {}
    )

    st.subheader(
        "Predicted vs Actual Recall"
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    with col1:
        st.metric(
            "Actual Recall",
            (
                format_percentage(
                    actual_recall
                )
                if actual_recall
                is not None
                else "N/A"
            )
        )

    with col2:
        st.metric(
            "Delay Since Study",
            (
                f"{delay_days:.2f} days"
                if delay_days
                is not None
                else "N/A"
            )
        )

    with col3:
        st.metric(
            "Checkpoint",
            checkpoint.get(
                "label",
                "Custom Recall"
            )
        )

    prediction = match.get(
        "prediction"
    )

    if prediction is None:
        st.warning(
            "No usable prediction created "
            "before this recall test could "
            "be matched."
        )

        st.caption(
            "The actual recall result is still "
            "valuable learning data, but this "
            "observation cannot be used for "
            "predicted-vs-actual evaluation."
        )

        return

    prediction_days = safe_float(
        prediction.get(
            "prediction_for_days"
        )
    )

    interval_difference = (
        safe_float(
            match.get(
                "interval_difference"
            )
        )
    )

    strict_match = bool(
        evaluation.get(
            "valid_for_strict_evaluation",
            match.get(
                "strict_match",
                False
            )
        )
    )

    st.write(
        "**Matched prediction interval:** "
        + (
            f"{prediction_days:.2f} days"
            if prediction_days
            is not None
            else "N/A"
        )
    )

    st.write(
        "**Actual recall delay:** "
        + (
            f"{delay_days:.2f} days"
            if delay_days
            is not None
            else "N/A"
        )
    )

    if interval_difference is not None:
        st.write(
            "**Interval difference:** "
            f"{interval_difference:.2f} days"
        )

    if strict_match:
        st.success(
            "This prediction and recall test "
            "form a strict research-valid "
            "interval match."
        )

    else:
        st.warning(
            "A prediction was found, but its "
            "time horizon is too different from "
            "the actual recall delay for strict "
            "research evaluation."
        )

        st.caption(
            "The comparison below is exploratory "
            "only and will not be treated as a "
            "formal model winner."
        )

    if not comparisons:
        st.info(
            "The matched prediction contains "
            "no usable model values to compare."
        )

        return

    rows = []

    for comparison in comparisons:
        predicted_recall = safe_float(
            comparison.get(
                "predicted_recall"
            )
        )

        absolute_error_pp = (
            safe_float(
                comparison.get(
                    "absolute_error_pp"
                )
            )
        )

        rows.append(
            {
                "Method":
                    comparison.get(
                        "model",
                        "Unknown"
                    ),

                "Recall":
                    (
                        predicted_recall
                        * 100
                        if predicted_recall
                        is not None
                        else None
                    ),

                "Absolute Error":
                    absolute_error_pp,

                "Direction":
                    comparison.get(
                        "bias_direction",
                        "Unknown"
                    ),
            }
        )

    if actual_recall is not None:
        rows.append(
            {
                "Method":
                    "Actual Recall",

                "Recall":
                    actual_recall
                    * 100,

                "Absolute Error":
                    0.0,

                "Direction":
                    "Observed",
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    figure = px.bar(
        dataframe,
        x="Method",
        y="Recall",
        text_auto=".1f",
        title=(
            "Prediction vs Actual "
            "Recall Performance"
        )
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Retention / Recall (%)"
    )

    st.plotly_chart(
        figure,
        use_container_width=True
    )

    display_dataframe = (
        dataframe.copy()
    )

    display_dataframe[
        "Recall"
    ] = (
        display_dataframe[
            "Recall"
        ].map(
            lambda value:
                (
                    f"{value:.1f}%"
                    if pd.notna(
                        value
                    )
                    else "N/A"
                )
        )
    )

    display_dataframe[
        "Absolute Error"
    ] = (
        display_dataframe[
            "Absolute Error"
        ].map(
            lambda value:
                (
                    f"{value:.1f} pp"
                    if pd.notna(
                        value
                    )
                    else "N/A"
                )
        )
    )

    st.dataframe(
        display_dataframe,
        use_container_width=True,
        hide_index=True
    )

    exploratory_winner = (
        evaluation.get(
            "exploratory_winner"
        )
    )

    if exploratory_winner is None:
        exploratory_winner = (
            comparisons[
                0
            ]
            if comparisons
            else None
        )

    strict_winner = (
        evaluation.get(
            "winner"
        )
        if strict_match
        else None
    )

    if (
        strict_match
        and strict_winner
        is None
        and comparisons
    ):
        strict_winner = (
            comparisons[
                0
            ]
        )

    if strict_winner:
        model_name = (
            strict_winner.get(
                "model",
                "Unknown"
            )
        )

        error_pp = safe_float(
            strict_winner.get(
                "absolute_error_pp"
            )
        )

        st.success(
            f"{model_name} was the closest "
            "available prediction for this "
            "strictly matched recall test."
        )

        if error_pp is not None:
            st.caption(
                "Absolute error for this "
                f"observation: {error_pp:.1f} "
                "percentage points."
            )

        st.caption(
            "One recall test does not establish "
            "overall model superiority. Research "
            "conclusions require repeated strict "
            "matches across observations."
        )

    elif exploratory_winner:
        model_name = (
            exploratory_winner.get(
                "model",
                "Unknown"
            )
        )

        error_pp = safe_float(
            exploratory_winner.get(
                "absolute_error_pp"
            )
        )

        st.info(
            f"Exploratory closest method: "
            f"{model_name}"
        )

        if error_pp is not None:
            st.caption(
                f"Exploratory error: "
                f"{error_pp:.1f} percentage points."
            )


# recall history

def show_recall_history(
    topic_id,
    selected_session_id
):
    st.subheader(
        "Recall History"
    )

    quizzes = (
        list_topic_quizzes(
            topic_id
        )
    )

    if not quizzes:
        st.info(
            "No recall tests have been "
            "recorded for this topic yet."
        )

        return

    summary = (
        summarize_recall_results(
            quizzes
        )
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    with col1:
        st.metric(
            "Recall Tests",
            summary.get(
                "tests",
                0
            )
        )

    with col2:
        average_recall = safe_float(
            summary.get(
                "average_recall"
            )
        )

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

    with col3:
        average_delay = safe_float(
            summary.get(
                "average_delay"
            )
        )

        st.metric(
            "Average Delay",
            (
                f"{average_delay:.2f} days"
                if average_delay
                is not None
                else "N/A"
            )
        )

    rows = []

    selected_session_rows = []

    for quiz in quizzes:
        quiz_id = get_record_id(
            quiz,
            "quiz_id"
        )

        session_id = get_record_id(
            quiz,
            "session_id"
        )

        delay_days = safe_float(
            quiz.get(
                "delay_days"
            )
        )

        recall_accuracy = safe_float(
            quiz.get(
                "recall_accuracy"
            )
        )

        actual_score = safe_float(
            quiz.get(
                "actual_score"
            )
        )

        if (
            actual_score is None
            and recall_accuracy
            is not None
        ):
            actual_score = (
                recall_accuracy
                * 100
            )

        checkpoint = (
            get_checkpoint_information(
                delay_days
            )
            if delay_days
            is not None
            else {}
        )

        total_questions = safe_int(
            quiz.get(
                "total_questions"
            )
        )

        correct_answers = safe_int(
            quiz.get(
                "correct_answers"
            )
        )

        rows.append(
            {
                "Quiz ID":
                    quiz_id,

                "Session":
                    session_id,

                "Test Date":
                    format_datetime(
                        quiz.get(
                            "quiz_date"
                        )
                    ),

                "Delay":
                    (
                        f"{delay_days:.2f} days"
                        if delay_days
                        is not None
                        else "N/A"
                    ),

                "Checkpoint":
                    checkpoint.get(
                        "label",
                        "N/A"
                    ),

                "Correct":
                    (
                        f"{correct_answers}/"
                        f"{total_questions}"
                        if (
                            correct_answers
                            is not None
                            and total_questions
                            is not None
                        )
                        else "N/A"
                    ),

                "Recall":
                    (
                        f"{actual_score:.1f}%"
                        if actual_score
                        is not None
                        else "N/A"
                    ),

                "Confidence":
                    (
                        f"{quiz.get('confidence_after_quiz')}/5"
                        if quiz.get(
                            "confidence_after_quiz"
                        )
                        is not None
                        else "N/A"
                    ),
            }
        )

        try:
            same_session = (
                int(
                    session_id
                )
                == int(
                    selected_session_id
                )
            )

        except (
            TypeError,
            ValueError
        ):
            same_session = False

        if (
            same_session
            and delay_days
            is not None
            and actual_score
            is not None
        ):
            selected_session_rows.append(
                {
                    "Days Since Study":
                        delay_days,

                    "Actual Recall":
                        actual_score,

                    "Quiz ID":
                        quiz_id,
                }
            )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True
    )

    if selected_session_rows:
        st.subheader(
            "Selected Session Recall Curve"
        )

        chart_dataframe = (
            pd.DataFrame(
                selected_session_rows
            )
            .sort_values(
                "Days Since Study"
            )
        )

        figure = px.line(
            chart_dataframe,
            x="Days Since Study",
            y="Actual Recall",
            markers=True,
            title=(
                "Observed Recall for "
                "Selected Study Session"
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
            title="Days Since Study"
        )

        st.plotly_chart(
            figure,
            use_container_width=True
        )

        st.caption(
            "Only recall tests linked to the "
            "selected study session are plotted "
            "together. Tests from different study "
            "sessions have different time origins."
        )


# experiment explanation

def show_experiment_explanation():
    with st.expander(
        "Why is actual delayed recall important?"
    ):
        st.markdown(
            """
A retention graph by itself does **not** prove that a prediction model works.

MemoryMap therefore records the student's actual recall after a real delay.

The intended experiment is:

1. Record a study session.
2. Save a prediction before the recall test.
3. Conduct the real recall test near the chosen checkpoint.
4. Record the actual score.
5. Match the recall test only with predictions created before it.
6. Check whether the prediction horizon is close enough to the actual delay.
7. Use strict matches for formal error metrics.

Typical checkpoints are approximately **1 day, 3 days and 7 days**.

A prediction for 7 days should not be treated as an equally valid prediction for a test performed after 1 day simply because it is the closest prediction available.

Likewise, a prediction created after seeing the quiz result must never be used to evaluate that earlier result.

Recall scores of exactly 0% or 100% are still valid observations. They are stored normally. They simply cannot, by themselves, be inverted into a finite single-observation half-life for adaptive scheduling.

The same or equivalent quiz difficulty should be maintained across delayed checkpoints wherever practical. Otherwise changes in question difficulty could be confused with changes in memory retention.
            """
        )


# page

def show_page():
    flash = (
        show_flash_message()
    )

    active = (
        get_active_student()
    )

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
        "🧪 Quiz & Recall"
    )

    st.write(
        "Record actual delayed recall "
        f"performance for **{student_name}**."
    )

    st.caption(
        "This is the outcome data used to test "
        "whether MemoryMap's predictions actually "
        "match real recall."
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
        session,
        topic
    )

    st.divider()

    show_saved_predictions(
        session_id
    )

    st.divider()

    if flash:
        flash_session_id = (
            flash.get(
                "session_id"
            )
        )

        try:
            same_flash_session = (
                int(
                    flash_session_id
                )
                == int(
                    session_id
                )
            )

        except (
            TypeError,
            ValueError
        ):
            same_flash_session = False

        if same_flash_session:
            evaluation = (
                flash.get(
                    "evaluation"
                )
            )

            if evaluation:
                show_prediction_comparison(
                    evaluation
                )

                st.divider()

    show_quiz_form(
        student_id,
        topic,
        session
    )

    st.divider()

    show_recall_history(
        topic_id,
        session_id
    )

    st.divider()

    st.page_link(
        "pages/4_Revision_Scheduler.py",
        label=(
            "⏰ Recalculate Adaptive Schedule"
        )
    )

    st.page_link(
        "pages/6_Analytics.py",
        label=(
            "📊 Continue to Analytics"
        )
    )

    show_experiment_explanation()


show_page()