import math
import re
from datetime import datetime, timedelta

from config import (
    MIN_STUDY_DURATION,
    MAX_STUDY_DURATION,
    MIN_DIFFICULTY,
    MAX_DIFFICULTY,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    MIN_QUIZ_SCORE,
    MAX_QUIZ_SCORE,
    MIN_RETENTION,
    MAX_RETENTION,
    TARGET_RETENTION_THRESHOLD,
    SECONDS_PER_MINUTE,
    SECONDS_PER_HOUR,
    SECONDS_PER_DAY,
)


# numbers

def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(
            value,
            maximum
        )
    )


def safe_float(
    value,
    default=0.0
):
    try:
        result = float(
            value
        )

        if not math.isfinite(
            result
        ):
            return default

        return result

    except (
        TypeError,
        ValueError
    ):
        return default


def safe_int(
    value,
    default=0
):
    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return default


def round_value(
    value,
    decimals=2
):
    if value is None:
        return None

    return round(
        float(
            value
        ),
        decimals
    )


# percentages

def decimal_to_percentage(
    value,
    decimals=1
):
    if value is None:
        return 0.0

    value = clamp(
        safe_float(
            value
        ),
        MIN_RETENTION,
        MAX_RETENTION
    )

    return round(
        value * 100,
        decimals
    )


def percentage_to_decimal(
    value
):
    value = clamp(
        safe_float(
            value
        ),
        0,
        100
    )

    return (
        value / 100
    )


def format_percentage(
    value,
    decimals=1
):
    if value is None:
        return "Not available"

    percentage = (
        decimal_to_percentage(
            value,
            decimals
        )
    )

    return (
        f"{percentage:.{decimals}f}%"
    )


# study duration

def validate_study_duration(
    duration
):
    duration = safe_float(
        duration,
        default=-1
    )

    return (
        MIN_STUDY_DURATION
        <= duration
        <= MAX_STUDY_DURATION
    )


def format_study_duration(
    minutes
):
    minutes = safe_float(
        minutes
    )

    if minutes < 0:
        return "Unknown"

    if minutes < 60:
        return (
            f"{round(minutes)} min"
        )

    hours = int(
        minutes // 60
    )

    remaining_minutes = int(
        minutes % 60
    )

    if remaining_minutes == 0:
        if hours == 1:
            return "1 hr"

        return (
            f"{hours} hr"
        )

    return (
        f"{hours} hr "
        f"{remaining_minutes} min"
    )


# difficulty

def validate_difficulty(
    difficulty
):
    difficulty = safe_int(
        difficulty,
        default=-1
    )

    return (
        MIN_DIFFICULTY
        <= difficulty
        <= MAX_DIFFICULTY
    )


def get_difficulty_label(
    difficulty
):
    labels = {
        1: "Very Easy",
        2: "Easy",
        3: "Moderate",
        4: "Difficult",
        5: "Very Difficult",
    }

    return labels.get(
        safe_int(
            difficulty
        ),
        "Unknown"
    )


# confidence

def validate_confidence(
    confidence
):
    confidence = safe_int(
        confidence,
        default=-1
    )

    return (
        MIN_CONFIDENCE
        <= confidence
        <= MAX_CONFIDENCE
    )


def get_confidence_label(
    confidence
):
    labels = {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High",
    }

    return labels.get(
        safe_int(
            confidence
        ),
        "Unknown"
    )


# quiz scores

def validate_quiz_score(
    score
):
    score = safe_float(
        score,
        default=-1
    )

    return (
        MIN_QUIZ_SCORE
        <= score
        <= MAX_QUIZ_SCORE
    )


def calculate_quiz_score(
    correct_answers,
    total_questions
):
    correct_answers = safe_int(
        correct_answers
    )

    total_questions = safe_int(
        total_questions
    )

    if total_questions <= 0:
        raise ValueError(
            "Total questions must be "
            "greater than zero."
        )

    if correct_answers < 0:
        raise ValueError(
            "Correct answers cannot "
            "be negative."
        )

    if (
        correct_answers
        > total_questions
    ):
        raise ValueError(
            "Correct answers cannot "
            "exceed total questions."
        )

    return (
        correct_answers
        / total_questions
    ) * 100


def calculate_recall_accuracy(
    correct_answers,
    total_questions
):
    return (
        calculate_quiz_score(
            correct_answers,
            total_questions
        )
        / 100
    )


# retention

def validate_retention(
    value
):
    value = safe_float(
        value,
        default=-1
    )

    return (
        MIN_RETENTION
        <= value
        <= MAX_RETENTION
    )


def get_retention_status(
    retention,
    threshold=TARGET_RETENTION_THRESHOLD
):
    retention = clamp(
        safe_float(
            retention
        ),
        MIN_RETENTION,
        MAX_RETENTION
    )

    threshold = clamp(
        safe_float(
            threshold,
            TARGET_RETENTION_THRESHOLD
        ),
        MIN_RETENTION,
        MAX_RETENTION
    )

    if retention >= 0.85:
        return "Strong"

    if retention >= threshold:
        return "Stable"

    if retention >= 0.50:
        return "Needs Revision"

    return "High Forgetting Risk"


def get_retention_priority(
    retention,
    threshold=TARGET_RETENTION_THRESHOLD
):
    retention = clamp(
        safe_float(
            retention
        ),
        MIN_RETENTION,
        MAX_RETENTION
    )

    threshold = clamp(
        safe_float(
            threshold,
            TARGET_RETENTION_THRESHOLD
        ),
        MIN_RETENTION,
        MAX_RETENTION
    )

    if retention < 0.50:
        return "High"

    if retention < threshold:
        return "Medium"

    return "Low"


# prediction comparison

def calculate_prediction_error(
    predicted,
    actual
):
    if (
        predicted is None
        or actual is None
    ):
        return None

    predicted = safe_float(
        predicted,
        default=None
    )

    actual = safe_float(
        actual,
        default=None
    )

    if (
        predicted is None
        or actual is None
    ):
        return None

    return abs(
        predicted
        - actual
    )


def calculate_percentage_error(
    predicted,
    actual
):
    error = (
        calculate_prediction_error(
            predicted,
            actual
        )
    )

    if error is None:
        return None

    return (
        error * 100
    )


def compare_predictions(
    fixed_prediction,
    personalized_prediction,
    actual_recall
):
    fixed_error = (
        calculate_prediction_error(
            fixed_prediction,
            actual_recall
        )
    )

    personalized_error = (
        calculate_prediction_error(
            personalized_prediction,
            actual_recall
        )
    )

    if (
        fixed_error is None
        and personalized_error is None
    ):
        winner = "Unavailable"

    elif fixed_error is None:
        winner = "Personalized Model"

    elif personalized_error is None:
        winner = "Fixed Forgetting Curve"

    elif fixed_error < personalized_error:
        winner = "Fixed Forgetting Curve"

    elif personalized_error < fixed_error:
        winner = "Personalized Model"

    else:
        winner = "Tie"

    return {
        "fixed_error":
            fixed_error,

        "personalized_error":
            personalized_error,

        "winner":
            winner,
    }


# dates and time

def current_datetime():
    return datetime.now()


def current_datetime_iso():
    return (
        datetime.now().isoformat()
    )


def parse_datetime(
    value
):
    if isinstance(
        value,
        datetime
    ):
        return value

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    if (
        text.endswith(
            "Z"
        )
    ):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        return (
            datetime.fromisoformat(
                text
            )
        )

    except (
        ValueError,
        TypeError
    ):
        return None


def datetime_to_iso(
    value=None
):
    if value is None:
        return (
            current_datetime_iso()
        )

    parsed = parse_datetime(
        value
    )

    if parsed is None:
        return None

    return parsed.isoformat()


def format_datetime(
    value,
    date_format="%d %b %Y, %I:%M %p"
):
    date_value = (
        parse_datetime(
            value
        )
    )

    if date_value is None:
        return "Unknown"

    return date_value.strftime(
        date_format
    )


def add_days(
    value,
    days
):
    date_value = (
        parse_datetime(
            value
        )
    )

    if date_value is None:
        date_value = (
            datetime.now()
        )

    return (
        date_value
        + timedelta(
            days=safe_float(
                days
            )
        )
    )


def add_hours(
    value,
    hours
):
    date_value = (
        parse_datetime(
            value
        )
    )

    if date_value is None:
        date_value = (
            datetime.now()
        )

    return (
        date_value
        + timedelta(
            hours=safe_float(
                hours
            )
        )
    )


def calculate_time_difference(
    earlier,
    later=None
):
    earlier = parse_datetime(
        earlier
    )

    if earlier is None:
        return None

    if later is None:
        later = datetime.now()

    else:
        later = parse_datetime(
            later
        )

    if later is None:
        return None

    try:
        return (
            later - earlier
        )

    except TypeError:
        return None


def calculate_days_since(
    earlier,
    later=None
):
    difference = (
        calculate_time_difference(
            earlier,
            later
        )
    )

    if difference is None:
        return None

    return (
        difference.total_seconds()
        / SECONDS_PER_DAY
    )


def calculate_hours_since(
    earlier,
    later=None
):
    difference = (
        calculate_time_difference(
            earlier,
            later
        )
    )

    if difference is None:
        return None

    return (
        difference.total_seconds()
        / SECONDS_PER_HOUR
    )


def calculate_minutes_since(
    earlier,
    later=None
):
    difference = (
        calculate_time_difference(
            earlier,
            later
        )
    )

    if difference is None:
        return None

    return (
        difference.total_seconds()
        / SECONDS_PER_MINUTE
    )


# compatibility aliases

def days_since(
    earlier,
    later=None
):
    return calculate_days_since(
        earlier,
        later
    )


def hours_since(
    earlier,
    later=None
):
    return calculate_hours_since(
        earlier,
        later
    )


def minutes_since(
    earlier,
    later=None
):
    return calculate_minutes_since(
        earlier,
        later
    )


# revision intervals

def days_to_hours(
    days
):
    return (
        safe_float(
            days
        )
        * 24
    )


def hours_to_days(
    hours
):
    return (
        safe_float(
            hours
        )
        / 24
    )


def calculate_revision_datetime(
    start_time,
    interval_days
):
    start_time = (
        parse_datetime(
            start_time
        )
    )

    if start_time is None:
        start_time = (
            datetime.now()
        )

    return (
        start_time
        + timedelta(
            days=safe_float(
                interval_days
            )
        )
    )


def is_revision_due(
    scheduled_time,
    current_time=None
):
    scheduled_time = (
        parse_datetime(
            scheduled_time
        )
    )

    if scheduled_time is None:
        return False

    if current_time is None:
        current_time = (
            datetime.now()
        )

    else:
        current_time = (
            parse_datetime(
                current_time
            )
        )

    if current_time is None:
        return False

    try:
        return (
            current_time
            >= scheduled_time
        )

    except TypeError:
        return False


# text helpers

def clean_text(
    value
):
    if value is None:
        return ""

    return str(
        value
    ).strip()


def clean_name(
    name
):
    name = clean_text(
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name


def clean_email(
    email
):
    email = clean_text(
        email
    )

    if not email:
        return ""

    return email.lower()


def validate_name(
    name
):
    name = clean_name(
        name
    )

    if len(name) < 2:
        return False

    return any(
        character.isalpha()
        for character in name
    )


def validate_topic_name(
    topic_name
):
    topic_name = clean_text(
        topic_name
    )

    return (
        len(topic_name) >= 2
    )


def validate_email(
    email
):
    email = clean_email(
        email
    )

    if not email:
        return True

    pattern = (
        r"^[^@\s]+@"
        r"[^@\s]+\."
        r"[^@\s]+$"
    )

    return (
        re.fullmatch(
            pattern,
            email
        )
        is not None
    )


# revision count

def validate_revision_count(
    revision_count
):
    revision_count = safe_int(
        revision_count,
        default=-1
    )

    return (
        revision_count >= 0
    )


# study session input

def validate_study_session_input(
    topic_name,
    study_duration,
    difficulty,
    confidence,
    revision_count,
    previous_quiz_score=None
):
    errors = []

    if not validate_topic_name(
        topic_name
    ):
        errors.append(
            "Topic name must contain "
            "at least 2 characters."
        )

    if not validate_study_duration(
        study_duration
    ):
        errors.append(
            f"Study duration must be between "
            f"{MIN_STUDY_DURATION} and "
            f"{MAX_STUDY_DURATION} minutes."
        )

    if not validate_difficulty(
        difficulty
    ):
        errors.append(
            f"Difficulty must be between "
            f"{MIN_DIFFICULTY} and "
            f"{MAX_DIFFICULTY}."
        )

    if not validate_confidence(
        confidence
    ):
        errors.append(
            f"Confidence must be between "
            f"{MIN_CONFIDENCE} and "
            f"{MAX_CONFIDENCE}."
        )

    if not validate_revision_count(
        revision_count
    ):
        errors.append(
            "Revision count cannot "
            "be negative."
        )

    if (
        previous_quiz_score
        is not None
        and
        not validate_quiz_score(
            previous_quiz_score
        )
    ):
        errors.append(
            f"Previous quiz score must be "
            f"between {MIN_QUIZ_SCORE} and "
            f"{MAX_QUIZ_SCORE}."
        )

    return {
        "valid":
            len(
                errors
            ) == 0,

        "errors":
            errors,
    }


# display helpers

def format_prediction(
    value
):
    if value is None:
        return "Not available"

    return format_percentage(
        value,
        decimals=1
    )


def format_days(
    days
):
    if days is None:
        return "Not available"

    days = safe_float(
        days,
        default=None
    )

    if days is None:
        return "Not available"

    if days < 0:
        return "Unknown"

    if days < 1:
        hours = days_to_hours(
            days
        )

        if hours < 1:
            minutes = (
                hours
                * 60
            )

            return (
                f"{minutes:.0f} minutes"
            )

        return (
            f"{hours:.1f} hours"
        )

    if math.isclose(
        days,
        1.0
    ):
        return "1 day"

    return (
        f"{days:.1f} days"
    )


# direct execution

if __name__ == "__main__":

    print(
        "MemoryMap Utilities"
    )

    print(
        "=" * 40
    )

    print(
        "60 minutes:",
        format_study_duration(
            60
        )
    )

    print(
        "135 minutes:",
        format_study_duration(
            135
        )
    )

    print(
        "Retention:",
        format_percentage(
            0.78
        )
    )

    print(
        "Retention status:",
        get_retention_status(
            0.78
        )
    )

    print(
        "Difficulty 4:",
        get_difficulty_label(
            4
        )
    )

    print(
        "Confidence 3:",
        get_confidence_label(
            3
        )
    )

    print(
        "Quiz score 8/10:",
        calculate_quiz_score(
            8,
            10
        )
    )

    print(
        "Recall accuracy 8/10:",
        calculate_recall_accuracy(
            8,
            10
        )
    )

    print(
        "Prediction error:",
        calculate_prediction_error(
            0.72,
            0.80
        )
    )

    print(
        "Clean name:",
        clean_name(
            "  Prajwal   C Pradhan  "
        )
    )

    print(
        "Email valid:",
        validate_email(
            "student@example.com"
        )
    )