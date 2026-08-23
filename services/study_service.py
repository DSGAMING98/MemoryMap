from datetime import datetime

from database import (
    add_topic,
    get_topic,
    get_topics_for_student,
    add_study_session,
    get_study_session,
    get_study_sessions_for_student,
    get_latest_study_session,
    get_quiz_results_for_topic,
)

from services.student_service import (
    get_student_profile,
)

from utils import (
    clean_text,
    parse_datetime as parse_datetime_util,
    validate_topic_name as is_valid_topic_name,
    validate_difficulty as is_valid_difficulty,
    validate_study_duration as is_valid_study_duration,
    validate_confidence as is_valid_confidence,
    validate_quiz_score as is_valid_quiz_score,
)


# normalization

def normalize_record(
    record
):
    if record is None:
        return None

    if isinstance(
        record,
        dict
    ):
        return record.copy()

    try:
        return dict(
            record
        )

    except (
        TypeError,
        ValueError
    ):
        return None


def normalize_records(
    records
):
    if not records:
        return []

    normalized = []

    for record in records:
        item = normalize_record(
            record
        )

        if item is not None:
            normalized.append(
                item
            )

    return normalized


def normalize_id(
    value
):
    if value is None:
        return None

    try:
        value = int(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return None

    if value <= 0:
        return None

    return value


# datetime

def parse_datetime(
    value
):
    return parse_datetime_util(
        value
    )


def normalize_datetime(
    value=None
):
    if value is None:
        return datetime.now()

    parsed = parse_datetime(
        value
    )

    if parsed is None:
        raise ValueError(
            "Invalid date and time."
        )

    return parsed


def get_current_time_for(
    date_value
):
    if (
        date_value is not None
        and
        date_value.tzinfo is not None
    ):
        return datetime.now(
            tz=date_value.tzinfo
        )

    return datetime.now()


# validation

def validate_topic_name(
    topic_name
):
    topic_name = clean_text(
        topic_name
    )

    topic_name = " ".join(
        topic_name.split()
    )

    if not topic_name:
        raise ValueError(
            "Topic name is required."
        )

    if not is_valid_topic_name(
        topic_name
    ):
        raise ValueError(
            "Topic name must contain "
            "at least two characters."
        )

    if len(
        topic_name
    ) > 150:
        raise ValueError(
            "Topic name is too long."
        )

    return topic_name


def validate_subject(
    subject
):
    subject = clean_text(
        subject
    )

    subject = " ".join(
        subject.split()
    )

    if not subject:
        raise ValueError(
            "Subject name is required."
        )

    if len(
        subject
    ) > 150:
        raise ValueError(
            "Subject name is too long."
        )

    return subject


def validate_topic_difficulty(
    difficulty
):
    try:
        difficulty = int(
            difficulty
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Topic difficulty must be "
            "a whole number."
        )

    if not is_valid_difficulty(
        difficulty
    ):
        raise ValueError(
            "Topic difficulty must be "
            "between 1 and 5."
        )

    return difficulty


def validate_session_duration(
    study_duration
):
    try:
        study_duration = float(
            study_duration
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Study duration must be numeric."
        )

    if not is_valid_study_duration(
        study_duration
    ):
        raise ValueError(
            "Study duration must be between "
            "1 and 600 minutes."
        )

    return study_duration


def validate_session_confidence(
    confidence_level
):
    try:
        confidence_level = int(
            confidence_level
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Confidence level must be "
            "a whole number."
        )

    if not is_valid_confidence(
        confidence_level
    ):
        raise ValueError(
            "Confidence level must be "
            "between 1 and 5."
        )

    return confidence_level


def validate_previous_quiz_score(
    score
):
    if score is None:
        return None

    try:
        score = float(
            score
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Previous quiz score must "
            "be numeric."
        )

    if not is_valid_quiz_score(
        score
    ):
        raise ValueError(
            "Previous quiz score must be "
            "between 0 and 100."
        )

    return score


# topic retrieval

def get_topic_profile(
    topic_id
):
    topic_id = normalize_id(
        topic_id
    )

    if topic_id is None:
        return None

    try:
        topic = get_topic(
            topic_id
        )

    except Exception:
        return None

    return normalize_record(
        topic
    )


def list_student_topics(
    student_id
):
    student_id = normalize_id(
        student_id
    )

    if student_id is None:
        return []

    try:
        topics = (
            get_topics_for_student(
                student_id
            )
        )

    except Exception:
        return []

    return normalize_records(
        topics
    )


def topic_belongs_to_student(
    topic,
    student_id
):
    topic = normalize_record(
        topic
    )

    student_id = normalize_id(
        student_id
    )

    if (
        not topic
        or student_id is None
    ):
        return False

    topic_student_id = normalize_id(
        topic.get(
            "student_id"
        )
    )

    return (
        topic_student_id
        == student_id
    )


# duplicate topics

def find_existing_topic(
    student_id,
    topic_name,
    subject=None
):
    student_id = normalize_id(
        student_id
    )

    if student_id is None:
        return None

    topic_name = clean_text(
        topic_name
    ).lower()

    topic_name = " ".join(
        topic_name.split()
    )

    if subject:
        subject = clean_text(
            subject
        ).lower()

        subject = " ".join(
            subject.split()
        )

    else:
        subject = None

    topics = list_student_topics(
        student_id
    )

    for topic in topics:
        existing_name = clean_text(
            topic.get(
                "topic_name"
            )
        ).lower()

        existing_name = " ".join(
            existing_name.split()
        )

        existing_subject = clean_text(
            topic.get(
                "subject_name",
                topic.get(
                    "subject"
                )
            )
        ).lower()

        existing_subject = " ".join(
            existing_subject.split()
        )

        same_name = (
            existing_name
            == topic_name
        )

        same_subject = (
            subject is None
            or
            existing_subject
            == subject
        )

        if (
            same_name
            and same_subject
        ):
            return topic

    return None


# topic creation

def create_topic(
    student_id,
    topic_name,
    subject,
    difficulty
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "topic":
                None,

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    try:
        topic_name = (
            validate_topic_name(
                topic_name
            )
        )

        subject = validate_subject(
            subject
        )

        difficulty = (
            validate_topic_difficulty(
                difficulty
            )
        )

    except ValueError as error:
        return {
            "success":
                False,

            "topic":
                None,

            "message":
                str(
                    error
                ),
        }

    existing_topic = (
        find_existing_topic(
            student_id,
            topic_name,
            subject
        )
    )

    if existing_topic:
        return {
            "success":
                False,

            "topic":
                existing_topic,

            "message":
                (
                    "This topic already exists "
                    "for the selected student."
                ),
        }

    try:
        topic_id = add_topic(
            student_id=student_id,
            topic_name=topic_name,
            subject_name=subject,
            difficulty=difficulty
        )

    except Exception as error:
        return {
            "success":
                False,

            "topic":
                None,

            "message":
                (
                    "Topic could not be created: "
                    f"{error}"
                ),
        }

    topic = get_topic_profile(
        topic_id
    )

    if topic is None:
        return {
            "success":
                False,

            "topic":
                None,

            "message":
                (
                    "Topic was created but could "
                    "not be reloaded."
                ),
        }

    return {
        "success":
            True,

        "topic":
            topic,

        "message":
            "Topic created successfully.",
    }


# sessions

def get_session_profile(
    session_id
):
    session_id = normalize_id(
        session_id
    )

    if session_id is None:
        return None

    try:
        session = get_study_session(
            session_id
        )

    except Exception:
        return None

    return normalize_record(
        session
    )


def list_student_sessions(
    student_id
):
    student_id = normalize_id(
        student_id
    )

    if student_id is None:
        return []

    try:
        sessions = (
            get_study_sessions_for_student(
                student_id
            )
        )

    except Exception:
        return []

    return normalize_records(
        sessions
    )


def list_topic_sessions(
    student_id,
    topic_id
):
    student_id = normalize_id(
        student_id
    )

    topic_id = normalize_id(
        topic_id
    )

    if (
        student_id is None
        or topic_id is None
    ):
        return []

    sessions = list_student_sessions(
        student_id
    )

    result = []

    for session in sessions:
        session_topic_id = (
            normalize_id(
                session.get(
                    "topic_id"
                )
            )
        )

        if (
            session_topic_id
            == topic_id
        ):
            result.append(
                session
            )

    return result


def get_session_datetime(
    session
):
    session = normalize_record(
        session
    )

    if not session:
        return datetime.min

    parsed = parse_datetime(
        session.get(
            "studied_at"
        )
    )

    if parsed is None:
        return datetime.min

    return parsed


def get_quiz_datetime(
    quiz
):
    quiz = normalize_record(
        quiz
    )

    if not quiz:
        return datetime.min

    parsed = parse_datetime(
        quiz.get(
            "quiz_date"
        )
    )

    if parsed is None:
        return datetime.min

    return parsed


def get_latest_topic_session(
    student_id,
    topic_id
):
    student_id = normalize_id(
        student_id
    )

    topic_id = normalize_id(
        topic_id
    )

    if (
        student_id is None
        or topic_id is None
    ):
        return None

    try:
        session = (
            get_latest_study_session(
                student_id,
                topic_id
            )
        )

    except Exception:
        session = None

    return normalize_record(
        session
    )


# revision count

def calculate_revision_count(
    student_id,
    topic_id,
    studied_at=None
):
    studied_at = normalize_datetime(
        studied_at
    )

    sessions = list_topic_sessions(
        student_id,
        topic_id
    )

    previous_sessions = 0

    for session in sessions:
        session_time = (
            get_session_datetime(
                session
            )
        )

        try:
            if (
                session_time
                < studied_at
            ):
                previous_sessions += 1

        except TypeError:
            continue

    return previous_sessions


# quiz history

def list_topic_quizzes(
    topic_id,
    student_id=None
):
    topic_id = normalize_id(
        topic_id
    )

    if topic_id is None:
        return []

    if student_id is not None:
        student_id = normalize_id(
            student_id
        )

        if student_id is None:
            return []

    try:
        if student_id is None:
            quizzes = (
                get_quiz_results_for_topic(
                    topic_id
                )
            )

        else:
            quizzes = (
                get_quiz_results_for_topic(
                    student_id,
                    topic_id
                )
            )

    except Exception:
        return []

    return normalize_records(
        quizzes
    )


def get_latest_quiz_before(
    topic_id,
    before_datetime=None,
    student_id=None
):
    if before_datetime is None:
        before_datetime = datetime.now()

    before_datetime = (
        normalize_datetime(
            before_datetime
        )
    )

    quizzes = list_topic_quizzes(
        topic_id=topic_id,
        student_id=student_id
    )

    usable = []

    for quiz in quizzes:
        quiz_time = (
            get_quiz_datetime(
                quiz
            )
        )

        try:
            if (
                quiz_time
                < before_datetime
            ):
                usable.append(
                    quiz
                )

        except TypeError:
            continue

    if not usable:
        return None

    return max(
        usable,
        key=get_quiz_datetime
    )


def get_previous_quiz_score(
    topic_id,
    before_datetime=None,
    student_id=None
):
    quiz = get_latest_quiz_before(
        topic_id=topic_id,
        before_datetime=before_datetime,
        student_id=student_id
    )

    if quiz is None:
        return None

    score = quiz.get(
        "actual_score"
    )

    if score is None:
        return None

    try:
        score = float(
            score
        )

    except (
        TypeError,
        ValueError
    ):
        return None

    if not is_valid_quiz_score(
        score
    ):
        return None

    return score


# study session creation

def create_study_session(
    student_id,
    topic_id,
    study_duration,
    confidence_level,
    studied_at=None,
    notes=None,
    external_previous_score=None
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "session":
                None,

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    topic = get_topic_profile(
        topic_id
    )

    if topic is None:
        return {
            "success":
                False,

            "session":
                None,

            "message":
                "Topic not found.",
        }

    topic_id = topic[
        "topic_id"
    ]

    if not topic_belongs_to_student(
        topic,
        student_id
    ):
        return {
            "success":
                False,

            "session":
                None,

            "message":
                (
                    "The selected topic does not "
                    "belong to this student."
                ),
        }

    try:
        study_duration = (
            validate_session_duration(
                study_duration
            )
        )

        confidence_level = (
            validate_session_confidence(
                confidence_level
            )
        )

        studied_datetime = (
            normalize_datetime(
                studied_at
            )
        )

        external_previous_score = (
            validate_previous_quiz_score(
                external_previous_score
            )
        )

    except ValueError as error:
        return {
            "success":
                False,

            "session":
                None,

            "message":
                str(
                    error
                ),
        }

    current_time = (
        get_current_time_for(
            studied_datetime
        )
    )

    if (
        studied_datetime
        > current_time
    ):
        return {
            "success":
                False,

            "session":
                None,

            "message":
                (
                    "A study session cannot "
                    "be recorded in the future."
                ),
        }

    revision_count = (
        calculate_revision_count(
            student_id=student_id,
            topic_id=topic_id,
            studied_at=studied_datetime
        )
    )

    stored_previous_score = (
        get_previous_quiz_score(
            topic_id=topic_id,
            before_datetime=studied_datetime,
            student_id=student_id
        )
    )

    if (
        stored_previous_score
        is not None
    ):
        previous_quiz_score = (
            stored_previous_score
        )

        previous_score_source = (
            "MemoryMap recall history"
        )

    else:
        previous_quiz_score = (
            external_previous_score
        )

        if (
            external_previous_score
            is not None
        ):
            previous_score_source = (
                "External score"
            )

        else:
            previous_score_source = (
                "Unavailable"
            )

    notes = clean_text(
        notes
    )

    if not notes:
        notes = None

    try:
        session_id = add_study_session(
            student_id=student_id,
            topic_id=topic_id,
            study_duration=study_duration,
            confidence_level=confidence_level,
            revision_count=revision_count,
            previous_quiz_score=(
                previous_quiz_score
            ),
            studied_at=(
                studied_datetime.isoformat()
            ),
            notes=notes
        )

    except Exception as error:
        return {
            "success":
                False,

            "session":
                None,

            "message":
                (
                    "Study session could not "
                    f"be saved: {error}"
                ),
        }

    session = get_session_profile(
        session_id
    )

    if session is None:
        return {
            "success":
                False,

            "session":
                None,

            "message":
                (
                    "Study session was saved but "
                    "could not be reloaded."
                ),
        }

    return {
        "success":
            True,

        "session":
            session,

        "topic":
            topic,

        "revision_count":
            revision_count,

        "previous_quiz_score":
            previous_quiz_score,

        "previous_score_source":
            previous_score_source,

        "message":
            (
                "Study session recorded "
                "successfully."
            ),
    }


# prediction context

def create_prediction_context(
    session_id
):
    session = get_session_profile(
        session_id
    )

    if session is None:
        return {
            "success":
                False,

            "session":
                None,

            "topic":
                None,

            "message":
                "Study session not found.",
        }

    topic_id = session.get(
        "topic_id"
    )

    topic = get_topic_profile(
        topic_id
    )

    if topic is None:
        return {
            "success":
                False,

            "session":
                session,

            "topic":
                None,

            "message":
                (
                    "The topic associated with "
                    "this session was not found."
                ),
        }

    if not topic_belongs_to_student(
        topic,
        session.get(
            "student_id"
        )
    ):
        return {
            "success":
                False,

            "session":
                session,

            "topic":
                None,

            "message":
                (
                    "Session and topic ownership "
                    "could not be verified."
                ),
        }

    return {
        "success":
            True,

        "session":
            session,

        "topic":
            topic,

        "message":
            "Prediction context prepared.",
    }


# live MemoryMap features

def create_memorymap_feature_values(
    session_id,
    prediction_days
):
    context = create_prediction_context(
        session_id
    )

    if not context.get(
        "success",
        False
    ):
        return {
            "success":
                False,

            "features":
                None,

            "message":
                context.get(
                    "message",
                    "Prediction context unavailable."
                ),
        }

    session = context[
        "session"
    ]

    topic = context[
        "topic"
    ]

    try:
        prediction_days = float(
            prediction_days
        )

    except (
        TypeError,
        ValueError
    ):
        return {
            "success":
                False,

            "features":
                None,

            "message":
                (
                    "Prediction interval "
                    "must be numeric."
                ),
        }

    if prediction_days < 0:
        return {
            "success":
                False,

            "features":
                None,

            "message":
                (
                    "Prediction interval "
                    "cannot be negative."
                ),
        }

    features = {
        "study_duration":
            session.get(
                "study_duration"
            ),

        "topic_difficulty":
            topic.get(
                "difficulty"
            ),

        "confidence_level":
            session.get(
                "confidence_level"
            ),

        "revision_count":
            session.get(
                "revision_count"
            ),

        "previous_quiz_score":
            session.get(
                "previous_quiz_score"
            ),

        "time_since_revision":
            prediction_days,

        "delta_days":
            prediction_days,
    }

    return {
        "success":
            True,

        "features":
            features,

        "session":
            session,

        "topic":
            topic,

        "message":
            "Prediction features prepared.",
    }


# study page

def get_study_page_data(
    student_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "student":
                None,

            "topics":
                [],

            "sessions":
                [],

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    topics = list_student_topics(
        student_id
    )

    sessions = list_student_sessions(
        student_id
    )

    sessions = sorted(
        sessions,
        key=get_session_datetime,
        reverse=True
    )

    return {
        "success":
            True,

        "student":
            student,

        "topics":
            topics,

        "sessions":
            sessions,

        "topic_count":
            len(
                topics
            ),

        "session_count":
            len(
                sessions
            ),

        "message":
            "Study data loaded.",
    }


# topic summary

def get_topic_study_summary(
    student_id,
    topic_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    topic = get_topic_profile(
        topic_id
    )

    if topic is None:
        return {
            "success":
                False,

            "message":
                "Topic not found.",
        }

    topic_id = topic[
        "topic_id"
    ]

    if not topic_belongs_to_student(
        topic,
        student_id
    ):
        return {
            "success":
                False,

            "message":
                (
                    "Topic does not belong "
                    "to the selected student."
                ),
        }

    sessions = list_topic_sessions(
        student_id,
        topic_id
    )

    quizzes = list_topic_quizzes(
        topic_id=topic_id,
        student_id=student_id
    )

    total_study_minutes = 0.0

    for session in sessions:
        duration = session.get(
            "study_duration"
        )

        if duration is None:
            continue

        try:
            duration = float(
                duration
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        if duration >= 0:
            total_study_minutes += (
                duration
            )

    recall_values = []

    for quiz in quizzes:
        recall = quiz.get(
            "recall_accuracy"
        )

        if recall is None:
            continue

        try:
            recall = float(
                recall
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        if (
            0
            <= recall
            <= 1
        ):
            recall_values.append(
                recall
            )

    if recall_values:
        average_recall = (
            sum(
                recall_values
            )
            / len(
                recall_values
            )
        )

    else:
        average_recall = None

    latest_session = (
        max(
            sessions,
            key=get_session_datetime
        )
        if sessions
        else None
    )

    latest_quiz = (
        max(
            quizzes,
            key=get_quiz_datetime
        )
        if quizzes
        else None
    )

    return {
        "success":
            True,

        "topic":
            topic,

        "study_sessions":
            sessions,

        "quiz_results":
            quizzes,

        "study_session_count":
            len(
                sessions
            ),

        "quiz_count":
            len(
                quizzes
            ),

        "total_study_minutes":
            total_study_minutes,

        "average_recall":
            average_recall,

        "latest_session":
            latest_session,

        "latest_quiz":
            latest_quiz,

        "message":
            "Topic study summary loaded.",
    }


# direct execution

if __name__ == "__main__":

    from services.student_service import (
        list_students,
    )

    print(
        "MemoryMap Study Service"
    )

    print(
        "=" * 40
    )

    students = list_students()

    print(
        "Students:",
        len(
            students
        )
    )

    if students:
        student = students[
            0
        ]

        student_id = student[
            "student_id"
        ]

        data = get_study_page_data(
            student_id
        )

        print(
            "Student:",
            student.get(
                "name"
            )
        )

        print(
            "Student ID:",
            student_id
        )

        print(
            "Topics:",
            data.get(
                "topic_count",
                0
            )
        )

        print(
            "Study Sessions:",
            data.get(
                "session_count",
                0
            )
        )