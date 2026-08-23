from datetime import datetime

from config import (
    TARGET_RETENTION_THRESHOLD,
)

from database import (
    add_revision,
    get_revision,
    get_upcoming_revisions,
    get_due_revisions,
    mark_revision_completed,
    get_quiz_results_for_topic,
)

from core.scheduler import (
    create_revision_schedule,
    create_all_schedules,
    get_schedule_message,
    get_schedule_urgency,
)

from core.predictor import (
    FIXED_MODE,
    PERSONALIZED_MODE,
    ADAPTIVE_MODE,
)

from services.student_service import (
    get_student_profile,
)

from services.study_service import (
    get_session_profile,
    get_topic_profile,
    topic_belongs_to_student,
)

from utils import (
    parse_datetime,
)


SUPPORTED_MODES = [
    FIXED_MODE,
    PERSONALIZED_MODE,
    ADAPTIVE_MODE,
]


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


def safe_float(
    value,
    default=None
):
    if value is None:
        return default

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return default


# datetime

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


# mode validation

def validate_schedule_mode(
    mode
):
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unknown revision mode: {mode}"
        )

    return mode


# threshold validation

def validate_threshold(
    threshold
):
    try:
        threshold = float(
            threshold
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Retention threshold must "
            "be numeric."
        )

    if not (
        0 < threshold < 1
    ):
        raise ValueError(
            "Retention threshold must "
            "be between 0 and 1."
        )

    return threshold


# quiz retrieval

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


# adaptive recall

def get_latest_usable_recall(
    topic_id,
    before_datetime=None,
    student_id=None
):
    before_datetime = normalize_datetime(
        before_datetime
    )

    quizzes = list_topic_quizzes(
        topic_id=topic_id,
        student_id=student_id
    )

    usable = []

    for quiz in quizzes:
        recall = safe_float(
            quiz.get(
                "recall_accuracy"
            )
        )

        delay = safe_float(
            quiz.get(
                "delay_days"
            )
        )

        quiz_date = parse_datetime(
            quiz.get(
                "quiz_date"
            )
        )

        if (
            recall is None
            or delay is None
            or quiz_date is None
        ):
            continue

        try:
            occurs_before = (
                quiz_date
                <= before_datetime
            )

        except TypeError:
            continue

        if not occurs_before:
            continue

        # Exact 0 or 1 cannot be converted
        # into a finite single-point half-life.

        if not (
            0 < recall < 1
        ):
            continue

        if delay <= 0:
            continue

        usable.append(
            {
                "quiz":
                    quiz,

                "recall":
                    recall,

                "delay_days":
                    delay,

                "quiz_date":
                    quiz_date,
            }
        )

    if not usable:
        return {
            "available":
                False,

            "recall":
                None,

            "delay_days":
                None,

            "quiz":
                None,

            "message":
                (
                    "No usable previous recall "
                    "measurement is available."
                ),
        }

    latest = max(
        usable,
        key=lambda item:
            item[
                "quiz_date"
            ]
    )

    return {
        "available":
            True,

        "recall":
            latest[
                "recall"
            ],

        "delay_days":
            latest[
                "delay_days"
            ],

        "quiz":
            latest[
                "quiz"
            ],

        "message":
            (
                "Previous observed recall "
                "is available for adaptive "
                "scheduling."
            ),
    }


# revision context

def validate_revision_context(
    student_id,
    session_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "valid":
                False,

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    session = get_session_profile(
        session_id
    )

    if session is None:
        return {
            "valid":
                False,

            "message":
                "Study session not found.",
        }

    session_id = session[
        "session_id"
    ]

    session_student_id = normalize_id(
        session.get(
            "student_id"
        )
    )

    if (
        session_student_id
        != student_id
    ):
        return {
            "valid":
                False,

            "message":
                (
                    "The selected study session "
                    "does not belong to this student."
                ),
        }

    topic_id = normalize_id(
        session.get(
            "topic_id"
        )
    )

    topic = get_topic_profile(
        topic_id
    )

    if topic is None:
        return {
            "valid":
                False,

            "message":
                (
                    "The topic associated with "
                    "this session was not found."
                ),
        }

    if not topic_belongs_to_student(
        topic,
        student_id
    ):
        return {
            "valid":
                False,

            "message":
                (
                    "The selected topic does not "
                    "belong to this student."
                ),
        }

    return {
        "valid":
            True,

        "student":
            student,

        "session":
            session,

        "topic":
            topic,

        "student_id":
            student_id,

        "session_id":
            session_id,

        "topic_id":
            topic[
                "topic_id"
            ],

        "message":
            "Revision context is valid.",
    }


# schedule metadata

def extract_half_life(
    schedule
):
    possible_fields = [
        "estimated_half_life",
        "half_life",
        "predicted_half_life",
    ]

    for field in possible_fields:
        value = safe_float(
            schedule.get(
                field
            )
        )

        if (
            value is not None
            and value > 0
        ):
            return value

    return None


def extract_schedule_source(
    schedule
):
    source = schedule.get(
        "source"
    )

    if source:
        return str(
            source
        )

    prediction_source = schedule.get(
        "prediction_source"
    )

    if prediction_source:
        return str(
            prediction_source
        )

    return None


def extract_scheduled_retention(
    schedule
):
    possible_fields = [
        "scheduled_retention",
        "predicted_retention",
        "retention_at_revision",
        "threshold",
    ]

    for field in possible_fields:
        value = safe_float(
            schedule.get(
                field
            )
        )

        if (
            value is not None
            and
            0 <= value <= 1
        ):
            return value

    return None


# schedule generation

def generate_revision_schedule(
    student_id,
    session_id,
    mode=FIXED_MODE,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None,
    reference_prediction_days=1.0,
    extra_features=None
):
    try:
        mode = validate_schedule_mode(
            mode
        )

        threshold = validate_threshold(
            threshold
        )

        current_time = normalize_datetime(
            current_time
        )

        reference_prediction_days = float(
            reference_prediction_days
        )

        if reference_prediction_days < 0:
            raise ValueError(
                "Reference prediction interval "
                "cannot be negative."
            )

    except (
        TypeError,
        ValueError
    ) as error:
        return {
            "success":
                False,

            "schedule":
                None,

            "message":
                str(
                    error
                ),
        }

    context = validate_revision_context(
        student_id,
        session_id
    )

    if not context.get(
        "valid",
        False
    ):
        return {
            "success":
                False,

            "schedule":
                None,

            "message":
                context.get(
                    "message",
                    "Invalid revision context."
                ),
        }

    student_id = context[
        "student_id"
    ]

    session_id = context[
        "session_id"
    ]

    topic_id = context[
        "topic_id"
    ]

    session = context[
        "session"
    ]

    topic = context[
        "topic"
    ]

    recall_context = {
        "available":
            False,

        "recall":
            None,

        "delay_days":
            None,

        "quiz":
            None,

        "message":
            (
                "Adaptive recall was not "
                "requested."
            ),
    }

    observed_recall = None
    observed_delay_days = None

    if mode == ADAPTIVE_MODE:
        recall_context = (
            get_latest_usable_recall(
                topic_id=topic_id,
                student_id=student_id,
                before_datetime=current_time
            )
        )

        if recall_context.get(
            "available",
            False
        ):
            observed_recall = (
                recall_context[
                    "recall"
                ]
            )

            observed_delay_days = (
                recall_context[
                    "delay_days"
                ]
            )

    try:
        schedule = (
            create_revision_schedule(
                mode=mode,
                session=session,
                topic=topic,
                threshold=threshold,
                current_time=current_time,
                reference_prediction_days=(
                    reference_prediction_days
                ),
                observed_recall=(
                    observed_recall
                ),
                observed_delay_days=(
                    observed_delay_days
                ),
                extra_features=(
                    extra_features
                )
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "schedule":
                None,

            "message":
                (
                    "Revision schedule could "
                    f"not be generated: {error}"
                ),
        }

    schedule = normalize_record(
        schedule
    )

    if schedule is None:
        return {
            "success":
                False,

            "schedule":
                None,

            "message":
                (
                    "Scheduler did not return "
                    "a valid schedule."
                ),
        }

    schedule[
        "student_id"
    ] = student_id

    schedule[
        "topic_id"
    ] = topic_id

    schedule[
        "session_id"
    ] = session_id

    schedule[
        "schedule_mode"
    ] = mode

    schedule[
        "generated_at"
    ] = current_time

    schedule[
        "estimated_half_life"
    ] = extract_half_life(
        schedule
    )

    schedule[
        "source"
    ] = extract_schedule_source(
        schedule
    )

    schedule[
        "recommendation"
    ] = get_schedule_message(
        schedule
    )

    schedule[
        "urgency"
    ] = get_schedule_urgency(
        schedule
    )

    schedule[
        "observed_recall_context"
    ] = recall_context

    scheduled_for = parse_datetime(
        schedule.get(
            "scheduled_for"
        )
    )

    if scheduled_for is None:
        return {
            "success":
                False,

            "schedule":
                None,

            "message":
                (
                    "Scheduler returned an invalid "
                    "revision date."
                ),
        }

    try:
        schedule[
            "currently_overdue"
        ] = (
            scheduled_for
            <= current_time
        )

    except TypeError:
        schedule[
            "currently_overdue"
        ] = False

    return {
        "success":
            True,

        "schedule":
            schedule,

        "student":
            context[
                "student"
            ],

        "session":
            session,

        "topic":
            topic,

        "message":
            "Revision schedule generated.",
    }


# all strategy preview

def preview_all_schedules(
    student_id,
    session_id,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None,
    extra_features=None
):
    context = validate_revision_context(
        student_id,
        session_id
    )

    if not context.get(
        "valid",
        False
    ):
        return {
            "success":
                False,

            "schedules":
                None,

            "message":
                context.get(
                    "message",
                    "Invalid revision context."
                ),
        }

    try:
        threshold = validate_threshold(
            threshold
        )

        current_time = normalize_datetime(
            current_time
        )

    except ValueError as error:
        return {
            "success":
                False,

            "schedules":
                None,

            "message":
                str(
                    error
                ),
        }

    student_id = context[
        "student_id"
    ]

    topic_id = context[
        "topic_id"
    ]

    session = context[
        "session"
    ]

    topic = context[
        "topic"
    ]

    recall_context = (
        get_latest_usable_recall(
            topic_id=topic_id,
            student_id=student_id,
            before_datetime=current_time
        )
    )

    if recall_context.get(
        "available",
        False
    ):
        observed_recall = (
            recall_context[
                "recall"
            ]
        )

        observed_delay_days = (
            recall_context[
                "delay_days"
            ]
        )

    else:
        observed_recall = None
        observed_delay_days = None

    try:
        schedules = (
            create_all_schedules(
                session=session,
                topic=topic,
                threshold=threshold,
                current_time=current_time,
                observed_recall=(
                    observed_recall
                ),
                observed_delay_days=(
                    observed_delay_days
                ),
                extra_features=(
                    extra_features
                )
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "schedules":
                None,

            "message":
                (
                    "Schedule comparison could "
                    f"not be generated: {error}"
                ),
        }

    cleaned_schedules = {}

    for (
        name,
        schedule
    ) in schedules.items():

        schedule = normalize_record(
            schedule
        )

        if schedule is None:
            continue

        schedule[
            "student_id"
        ] = context[
            "student_id"
        ]

        schedule[
            "topic_id"
        ] = context[
            "topic_id"
        ]

        schedule[
            "session_id"
        ] = context[
            "session_id"
        ]

        schedule[
            "schedule_mode"
        ] = schedule.get(
            "mode",
            name
        )

        schedule[
            "generated_at"
        ] = current_time

        schedule[
            "estimated_half_life"
        ] = extract_half_life(
            schedule
        )

        schedule[
            "source"
        ] = extract_schedule_source(
            schedule
        )

        schedule[
            "recommendation"
        ] = get_schedule_message(
            schedule
        )

        schedule[
            "urgency"
        ] = get_schedule_urgency(
            schedule
        )

        cleaned_schedules[
            name
        ] = schedule

    return {
        "success":
            True,

        "schedules":
            cleaned_schedules,

        "recall_context":
            recall_context,

        "message":
            (
                "Fixed, personalized and "
                "adaptive schedules generated."
            ),
    }


# stored revisions

def get_revision_overview(
    student_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "due":
                [],

            "upcoming":
                [],

            "due_count":
                0,

            "upcoming_count":
                0,

            "total_pending":
                0,

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    try:
        due = normalize_records(
            get_due_revisions(
                student_id
            )
        )

        upcoming = normalize_records(
            get_upcoming_revisions(
                student_id
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "due":
                [],

            "upcoming":
                [],

            "due_count":
                0,

            "upcoming_count":
                0,

            "total_pending":
                0,

            "message":
                (
                    "Revision data could not "
                    f"be loaded: {error}"
                ),
        }

    return {
        "success":
            True,

        "due":
            due,

        "upcoming":
            upcoming,

        "due_count":
            len(
                due
            ),

        "upcoming_count":
            len(
                upcoming
            ),

        "total_pending":
            (
                len(
                    due
                )
                +
                len(
                    upcoming
                )
            ),

        "message":
            "Revision overview loaded.",
    }


# duplicate detection

def revision_times_close(
    first_time,
    second_time,
    tolerance_minutes=5
):
    first_time = parse_datetime(
        first_time
    )

    second_time = parse_datetime(
        second_time
    )

    if (
        first_time is None
        or second_time is None
    ):
        return False

    try:
        difference = abs(
            (
                first_time
                - second_time
            ).total_seconds()
        )

    except TypeError:
        return False

    return (
        difference
        <= tolerance_minutes
        * 60
    )


def find_duplicate_revision(
    student_id,
    session_id,
    scheduled_for,
    schedule_mode=None,
    tolerance_minutes=5
):
    overview = get_revision_overview(
        student_id
    )

    if not overview.get(
        "success",
        False
    ):
        return None

    revisions = (
        overview[
            "due"
        ]
        +
        overview[
            "upcoming"
        ]
    )

    session_id = normalize_id(
        session_id
    )

    for revision in revisions:
        existing_session_id = normalize_id(
            revision.get(
                "session_id"
            )
        )

        if (
            existing_session_id
            != session_id
        ):
            continue

        if schedule_mode is not None:
            existing_mode = revision.get(
                "schedule_mode"
            )

            if (
                existing_mode is not None
                and
                existing_mode != schedule_mode
            ):
                continue

        if revision_times_close(
            revision.get(
                "scheduled_for"
            ),
            scheduled_for,
            tolerance_minutes
        ):
            return revision

    return None


# schedule persistence

def save_revision_schedule(
    schedule
):
    schedule = normalize_record(
        schedule
    )

    if not schedule:
        return {
            "success":
                False,

            "revision":
                None,

            "message":
                "Revision schedule is missing.",
        }

    required_fields = [
        "student_id",
        "topic_id",
        "session_id",
        "scheduled_for",
        "threshold",
        "priority",
    ]

    missing_fields = [
        field
        for field in required_fields
        if schedule.get(
            field
        ) is None
    ]

    if missing_fields:
        return {
            "success":
                False,

            "revision":
                None,

            "message":
                (
                    "Schedule is missing "
                    "required values: "
                    + ", ".join(
                        missing_fields
                    )
                ),
        }

    student_id = normalize_id(
        schedule.get(
            "student_id"
        )
    )

    topic_id = normalize_id(
        schedule.get(
            "topic_id"
        )
    )

    session_id = normalize_id(
        schedule.get(
            "session_id"
        )
    )

    if (
        student_id is None
        or topic_id is None
        or session_id is None
    ):
        return {
            "success":
                False,

            "revision":
                None,

            "message":
                (
                    "Schedule contains invalid "
                    "record identifiers."
                ),
        }

    scheduled_for = parse_datetime(
        schedule.get(
            "scheduled_for"
        )
    )

    if scheduled_for is None:
        return {
            "success":
                False,

            "revision":
                None,

            "message":
                "Scheduled time is invalid.",
        }

    generated_at = parse_datetime(
        schedule.get(
            "generated_at"
        )
    )

    if generated_at is None:
        generated_at = datetime.now()

    schedule_mode = schedule.get(
        "schedule_mode"
    )

    if schedule_mode is not None:
        try:
            schedule_mode = (
                validate_schedule_mode(
                    schedule_mode
                )
            )

        except ValueError as error:
            return {
                "success":
                    False,

                "revision":
                    None,

                "message":
                    str(
                        error
                    ),
            }

    threshold = safe_float(
        schedule.get(
            "threshold"
        )
    )

    try:
        threshold = validate_threshold(
            threshold
        )

    except ValueError as error:
        return {
            "success":
                False,

            "revision":
                None,

            "message":
                str(
                    error
                ),
        }

    predicted_retention = (
        extract_scheduled_retention(
            schedule
        )
    )

    estimated_half_life = (
        extract_half_life(
            schedule
        )
    )

    source = extract_schedule_source(
        schedule
    )

    priority = str(
        schedule.get(
            "priority"
        )
    )

    duplicate = (
        find_duplicate_revision(
            student_id=student_id,
            session_id=session_id,
            scheduled_for=scheduled_for,
            schedule_mode=schedule_mode
        )
    )

    if duplicate is not None:
        return {
            "success":
                False,

            "revision":
                duplicate,

            "duplicate":
                True,

            "message":
                (
                    "A matching revision "
                    "schedule already exists."
                ),
        }

    try:
        revision_id = add_revision(
            student_id=student_id,
            topic_id=topic_id,
            session_id=session_id,
            scheduled_for=(
                scheduled_for.isoformat()
            ),
            predicted_retention=(
                predicted_retention
            ),
            threshold=threshold,
            priority=priority,
            schedule_mode=schedule_mode,
            source=source,
            estimated_half_life=(
                estimated_half_life
            ),
            generated_at=(
                generated_at.isoformat()
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "revision":
                None,

            "duplicate":
                False,

            "message":
                (
                    "Revision schedule could "
                    f"not be saved: {error}"
                ),
        }

    revision = get_revision(
        revision_id
    )

    if revision is None:
        return {
            "success":
                False,

            "revision":
                None,

            "duplicate":
                False,

            "message":
                (
                    "Revision was saved but "
                    "could not be reloaded."
                ),
        }

    return {
        "success":
            True,

        "revision":
            revision,

        "duplicate":
            False,

        "message":
            (
                "Revision schedule saved "
                "successfully."
            ),
    }


# generate and save

def create_and_save_revision(
    student_id,
    session_id,
    mode=FIXED_MODE,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None,
    reference_prediction_days=1.0,
    extra_features=None
):
    generated = (
        generate_revision_schedule(
            student_id=student_id,
            session_id=session_id,
            mode=mode,
            threshold=threshold,
            current_time=current_time,
            reference_prediction_days=(
                reference_prediction_days
            ),
            extra_features=(
                extra_features
            )
        )
    )

    if not generated.get(
        "success",
        False
    ):
        return generated

    saved = save_revision_schedule(
        generated[
            "schedule"
        ]
    )

    return {
        "success":
            saved.get(
                "success",
                False
            ),

        "schedule":
            generated[
                "schedule"
            ],

        "revision":
            saved.get(
                "revision"
            ),

        "duplicate":
            saved.get(
                "duplicate",
                False
            ),

        "message":
            saved.get(
                "message"
            ),
    }


# completion

def complete_revision(
    revision_id
):
    revision_id = normalize_id(
        revision_id
    )

    if revision_id is None:
        return {
            "success":
                False,

            "message":
                "Revision ID is invalid.",
        }

    existing_revision = get_revision(
        revision_id
    )

    if existing_revision is None:
        return {
            "success":
                False,

            "message":
                "Revision schedule not found.",
        }

    if existing_revision.get(
        "completed"
    ) == 1:
        return {
            "success":
                False,

            "revision":
                existing_revision,

            "message":
                (
                    "Revision is already marked "
                    "as completed."
                ),
        }

    try:
        updated = (
            mark_revision_completed(
                revision_id
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "message":
                (
                    "Revision could not be "
                    "marked as completed: "
                    f"{error}"
                ),
        }

    if not updated:
        return {
            "success":
                False,

            "message":
                (
                    "Revision was not updated."
                ),
        }

    revision = get_revision(
        revision_id
    )

    return {
        "success":
            True,

        "revision":
            revision,

        "message":
            (
                "Revision marked as "
                "completed."
            ),
    }


# sorting

def revision_sort_key(
    revision
):
    scheduled_for = parse_datetime(
        revision.get(
            "scheduled_for"
        )
    )

    if scheduled_for is None:
        return datetime.max

    return scheduled_for


def sort_revisions(
    revisions
):
    revisions = normalize_records(
        revisions
    )

    return sorted(
        revisions,
        key=revision_sort_key
    )


# next revision

def get_next_revision(
    student_id
):
    overview = get_revision_overview(
        student_id
    )

    if not overview.get(
        "success",
        False
    ):
        return None

    due = sort_revisions(
        overview[
            "due"
        ]
    )

    if due:
        return {
            "revision":
                due[
                    0
                ],

            "status":
                "Due",
        }

    upcoming = sort_revisions(
        overview[
            "upcoming"
        ]
    )

    if upcoming:
        return {
            "revision":
                upcoming[
                    0
                ],

            "status":
                "Upcoming",
        }

    return None


# direct execution

if __name__ == "__main__":

    print(
        "MemoryMap Revision Service"
    )

    print(
        "=" * 40
    )

    print(
        "Supported Scheduling Modes:"
    )

    for mode in SUPPORTED_MODES:
        print(
            "-",
            mode
        )

    print()

    print(
        "Default Retention Threshold:",
        f"{TARGET_RETENTION_THRESHOLD * 100:.0f}%"
    )