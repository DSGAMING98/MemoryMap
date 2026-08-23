import math
from datetime import datetime, timedelta

from config import (
    DEFAULT_HALF_LIFE_DAYS,
    TARGET_RETENTION_THRESHOLD,
    MIN_REVISION_INTERVAL_HOURS,
    MAX_REVISION_INTERVAL_DAYS,
)

from core.forgetting_curve import (
    calculate_retention,
    calculate_revision_interval,
    estimate_half_life,
)

from core.predictor import (
    FIXED_MODE,
    PERSONALIZED_MODE,
    ADAPTIVE_MODE,
    predict_retention,
)


# datetime helpers

def parse_datetime(value):
    if isinstance(value, datetime):
        return value

    if value is None:
        return None

    try:
        return datetime.fromisoformat(
            str(value)
        )

    except ValueError:
        return None


def calculate_elapsed_days(
    studied_at,
    current_time=None
):
    studied_at = parse_datetime(
        studied_at
    )

    if studied_at is None:
        raise ValueError(
            "Invalid study datetime."
        )

    if current_time is None:
        current_time = datetime.now()

    else:
        current_time = parse_datetime(
            current_time
        )

    if current_time is None:
        raise ValueError(
            "Invalid current datetime."
        )

    difference = (
        current_time
        - studied_at
    )

    elapsed_days = (
        difference.total_seconds()
        / 86400
    )

    return max(
        elapsed_days,
        0.0
    )


# threshold validation

def validate_threshold(
    threshold
):
    threshold = float(
        threshold
    )

    if not (
        0 < threshold < 1
    ):
        raise ValueError(
            "Retention threshold must "
            "be between 0 and 1."
        )

    return threshold


# interval limits

def apply_interval_limits(
    interval_days
):
    interval_days = float(
        interval_days
    )

    minimum_days = (
        MIN_REVISION_INTERVAL_HOURS
        / 24
    )

    interval_days = max(
        interval_days,
        minimum_days
    )

    interval_days = min(
        interval_days,
        MAX_REVISION_INTERVAL_DAYS
    )

    return interval_days


# priority

def calculate_priority(
    current_retention,
    threshold=TARGET_RETENTION_THRESHOLD
):
    threshold = validate_threshold(
        threshold
    )

    current_retention = float(
        current_retention
    )

    if current_retention < 0.50:
        return "High"

    if current_retention < threshold:
        return "Medium"

    return "Low"


# revision status

def calculate_revision_status(
    scheduled_for,
    current_time=None
):
    scheduled_for = parse_datetime(
        scheduled_for
    )

    if scheduled_for is None:
        return "Unknown"

    if current_time is None:
        current_time = datetime.now()

    else:
        current_time = parse_datetime(
            current_time
        )

    if current_time is None:
        return "Unknown"

    if current_time >= scheduled_for:
        return "Due"

    difference = (
        scheduled_for
        - current_time
    )

    hours_remaining = (
        difference.total_seconds()
        / 3600
    )

    if hours_remaining <= 24:
        return "Due Soon"

    return "Upcoming"


# half-life extraction

def get_prediction_half_life(
    prediction_result
):
    if not prediction_result:
        return None

    half_life = prediction_result.get(
        "half_life"
    )

    if half_life is not None:
        try:
            half_life = float(
                half_life
            )

            if half_life > 0:
                return half_life

        except (
            TypeError,
            ValueError
        ):
            pass

    retention = prediction_result.get(
        "retention"
    )

    prediction_days = (
        prediction_result.get(
            "prediction_days"
        )
    )

    if (
        retention is None
        or prediction_days is None
    ):
        return None

    try:
        return estimate_half_life(
            prediction_days,
            retention
        )

    except (
        TypeError,
        ValueError
    ):
        return None


# fallback half-life

def get_safe_half_life(
    prediction_result
):
    half_life = (
        get_prediction_half_life(
            prediction_result
        )
    )

    if half_life is None:
        return (
            DEFAULT_HALF_LIFE_DAYS,
            True
        )

    return (
        half_life,
        False
    )


# schedule calculation

def calculate_schedule_from_half_life(
    studied_at,
    half_life_days,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None
):
    studied_at = parse_datetime(
        studied_at
    )

    if studied_at is None:
        raise ValueError(
            "Invalid study datetime."
        )

    threshold = validate_threshold(
        threshold
    )

    interval_days = (
        calculate_revision_interval(
            half_life_days,
            threshold
        )
    )

    interval_days = (
        apply_interval_limits(
            interval_days
        )
    )

    scheduled_for = (
        studied_at
        + timedelta(
            days=interval_days
        )
    )

    elapsed_days = (
        calculate_elapsed_days(
            studied_at,
            current_time
        )
    )

    current_retention = (
        calculate_retention(
            elapsed_days,
            half_life_days
        )
    )

    priority = calculate_priority(
        current_retention,
        threshold
    )

    status = (
        calculate_revision_status(
            scheduled_for,
            current_time
        )
    )

    remaining_days = max(
        (
            scheduled_for
            - (
                parse_datetime(
                    current_time
                )
                if current_time
                is not None
                else datetime.now()
            )
        ).total_seconds()
        / 86400,
        0.0
    )

    return {
        "studied_at":
            studied_at,

        "half_life_days":
            half_life_days,

        "estimated_half_life":
            half_life_days,

        "threshold":
            threshold,

        "interval_days":
            interval_days,

        "scheduled_for":
            scheduled_for,

        "elapsed_days":
            elapsed_days,

        "current_retention":
            current_retention,

        "predicted_retention":
            threshold,

        "remaining_days":
            remaining_days,

        "priority":
            priority,

        "status":
            status,
    }


# fixed schedule

def create_fixed_schedule(
    session,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None
):
    if session is None:
        raise ValueError(
            "Study session is required."
        )

    studied_at = session.get(
        "studied_at"
    )

    schedule = (
        calculate_schedule_from_half_life(
            studied_at=studied_at,
            half_life_days=(
                DEFAULT_HALF_LIFE_DAYS
            ),
            threshold=threshold,
            current_time=current_time
        )
    )

    schedule[
        "mode"
    ] = FIXED_MODE

    schedule[
        "source"
    ] = (
        "Default fixed forgetting curve"
    )

    schedule[
        "fallback_used"
    ] = False

    return schedule


# personalized schedule

def create_personalized_schedule(
    session,
    topic,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None,
    reference_prediction_days=1.0,
    extra_features=None
):
    if session is None:
        raise ValueError(
            "Study session is required."
        )

    prediction = (
        predict_retention(
            mode=PERSONALIZED_MODE,
            session=session,
            topic=topic,
            prediction_days=(
                reference_prediction_days
            ),
            extra_features=(
                extra_features
            )
        )
    )

    half_life, fallback_used = (
        get_safe_half_life(
            prediction
        )
    )

    schedule = (
        calculate_schedule_from_half_life(
            studied_at=session.get(
                "studied_at"
            ),
            half_life_days=half_life,
            threshold=threshold,
            current_time=current_time
        )
    )

    schedule[
        "mode"
    ] = PERSONALIZED_MODE

    schedule[
        "prediction"
    ] = prediction

    schedule[
        "fallback_used"
    ] = fallback_used

    if fallback_used:
        schedule[
            "source"
        ] = (
            "Fixed forgetting curve fallback"
        )

    else:
        schedule[
            "source"
        ] = prediction.get(
            "source",
            "Personalized model"
        )

    return schedule


# adaptive schedule

def create_adaptive_schedule(
    session,
    topic,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None,
    reference_prediction_days=1.0,
    observed_recall=None,
    observed_delay_days=None,
    extra_features=None
):
    if session is None:
        raise ValueError(
            "Study session is required."
        )

    prediction = (
        predict_retention(
            mode=ADAPTIVE_MODE,
            session=session,
            topic=topic,
            prediction_days=(
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

    half_life, fallback_used = (
        get_safe_half_life(
            prediction
        )
    )

    schedule = (
        calculate_schedule_from_half_life(
            studied_at=session.get(
                "studied_at"
            ),
            half_life_days=half_life,
            threshold=threshold,
            current_time=current_time
        )
    )

    schedule[
        "mode"
    ] = ADAPTIVE_MODE

    schedule[
        "prediction"
    ] = prediction

    schedule[
        "fallback_used"
    ] = (
        prediction.get(
            "fallback_used",
            fallback_used
        )
    )

    schedule[
        "source"
    ] = prediction.get(
        "source",
        "Adaptive prediction"
    )

    return schedule


# unified scheduler

def create_revision_schedule(
    mode,
    session,
    topic=None,
    threshold=TARGET_RETENTION_THRESHOLD,
    current_time=None,
    reference_prediction_days=1.0,
    observed_recall=None,
    observed_delay_days=None,
    extra_features=None
):
    if mode == FIXED_MODE:
        return create_fixed_schedule(
            session=session,
            threshold=threshold,
            current_time=current_time
        )

    if mode == PERSONALIZED_MODE:
        return (
            create_personalized_schedule(
                session=session,
                topic=topic,
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

    if mode == ADAPTIVE_MODE:
        return (
            create_adaptive_schedule(
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

    raise ValueError(
        f"Unknown scheduling mode: {mode}"
    )


# all schedules

def create_all_schedules(
        session,
        topic,
        threshold=TARGET_RETENTION_THRESHOLD,
        current_time=None,
        reference_prediction_days=1.0,
        observed_recall=None,
        observed_delay_days=None,
        extra_features=None
):


    fixed = create_fixed_schedule(
        session=session,
        threshold=threshold,
        current_time=current_time
    )

    personalized = (
        create_personalized_schedule(
            session=session,
            topic=topic,
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

    adaptive = (
        create_adaptive_schedule(
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

    return {
        "fixed":
            fixed,

        "personalized":
            personalized,

        "adaptive":
            adaptive,
    }


# schedule comparison

def compare_schedules(
    schedules
):
    rows = []

    for name, schedule in (
        schedules.items()
    ):
        rows.append(
            {
                "name":
                    name,

                "mode":
                    schedule.get(
                        "mode"
                    ),

                "interval_days":
                    schedule.get(
                        "interval_days"
                    ),

                "scheduled_for":
                    schedule.get(
                        "scheduled_for"
                    ),

                "half_life_days":
                    schedule.get(
                        "half_life_days"
                    ),

                "current_retention":
                    schedule.get(
                        "current_retention"
                    ),

                "priority":
                    schedule.get(
                        "priority"
                    ),

                "status":
                    schedule.get(
                        "status"
                    ),

                "source":
                    schedule.get(
                        "source"
                    ),
            }
        )

    rows.sort(
        key=lambda row:
            row[
                "interval_days"
            ]
    )

    return rows


# latest observed recall

def extract_latest_observed_recall(
    quiz_results
):
    if not quiz_results:
        return {
            "recall":
                None,

            "delay_days":
                None,

            "quiz":
                None,
        }

    usable_results = []

    for quiz in quiz_results:
        recall = quiz.get(
            "recall_accuracy"
        )

        delay = quiz.get(
            "delay_days"
        )

        if (
            recall is None
            or delay is None
        ):
            continue

        if not (
            0 < float(recall) < 1
        ):
            continue

        if float(delay) <= 0:
            continue

        usable_results.append(
            quiz
        )

    if not usable_results:
        return {
            "recall":
                None,

            "delay_days":
                None,

            "quiz":
                None,
        }

    def quiz_time(
        quiz
    ):
        value = quiz.get(
            "quiz_date"
        )

        parsed = parse_datetime(
            value
        )

        if parsed is None:
            return datetime.min

        return parsed

    latest_quiz = max(
        usable_results,
        key=quiz_time
    )

    return {
        "recall":
            float(
                latest_quiz[
                    "recall_accuracy"
                ]
            ),

        "delay_days":
            float(
                latest_quiz[
                    "delay_days"
                ]
            ),

        "quiz":
            latest_quiz,
    }


# schedule urgency

def get_schedule_urgency(
    schedule
):
    status = schedule.get(
        "status"
    )

    priority = schedule.get(
        "priority"
    )

    if status == "Due":
        return 3

    if (
        status == "Due Soon"
        or priority == "High"
    ):
        return 2

    if priority == "Medium":
        return 1

    return 0


# schedule recommendation text

def get_schedule_message(
    schedule
):
    status = schedule.get(
        "status"
    )

    priority = schedule.get(
        "priority"
    )

    interval_days = schedule.get(
        "interval_days",
        0
    )

    if status == "Due":
        return (
            "Revision is due now."
        )

    if status == "Due Soon":
        return (
            "Revision is recommended "
            "within the next 24 hours."
        )

    if priority == "Medium":
        return (
            "Retention is approaching "
            "the revision threshold."
        )

    return (
        f"Revision is recommended "
        f"approximately "
        f"{interval_days:.2f} days "
        f"after the study session."
    )


# direct execution

if __name__ == "__main__":
    print(
        "MemoryMap Scheduler"
    )

    print()

    example_session = {
        "session_id": 1,
        "study_duration": 60,
        "confidence_level": 4,
        "revision_count": 2,
        "previous_quiz_score": 75,
        "studied_at": (
            datetime.now()
            - timedelta(
                hours=12
            )
        ).isoformat(),
    }

    example_topic = {
        "topic_id": 1,
        "topic_name": "Graph Theory",
        "difficulty": 4,
    }

    fixed_schedule = (
        create_fixed_schedule(
            session=example_session
        )
    )

    print(
        "Fixed Schedule"
    )

    print(
        "Half-life:",
        f"{fixed_schedule['half_life_days']:.2f} days"
    )

    print(
        "Revision interval:",
        f"{fixed_schedule['interval_days']:.2f} days"
    )

    print(
        "Current retention:",
        f"{fixed_schedule['current_retention'] * 100:.2f}%"
    )

    print(
        "Priority:",
        fixed_schedule[
            "priority"
        ]
    )

    print(
        "Status:",
        fixed_schedule[
            "status"
        ]
    )

    print(
        "Recommended:",
        fixed_schedule[
            "scheduled_for"
        ]
    )

    print()

    adaptive_schedule = (
        create_adaptive_schedule(
            session=example_session,
            topic=example_topic,
            observed_recall=0.80,
            observed_delay_days=2
        )
    )

    print(
        "Adaptive Schedule"
    )

    print(
        "Half-life:",
        f"{adaptive_schedule['half_life_days']:.2f} days"
    )

    print(
        "Revision interval:",
        f"{adaptive_schedule['interval_days']:.2f} days"
    )

    print(
        "Source:",
        adaptive_schedule[
            "source"
        ]
    )