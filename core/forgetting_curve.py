import math

from config import (
    DEFAULT_HALF_LIFE_DAYS,
    TARGET_RETENTION_THRESHOLD,
    MIN_RETENTION,
    MAX_RETENTION,
    MIN_REVISION_INTERVAL_HOURS,
    MAX_REVISION_INTERVAL_DAYS,
)


# value validation

def validate_delta(delta_days):
    if delta_days is None:
        raise ValueError(
            "Time since study cannot be None."
        )

    delta_days = float(delta_days)

    if delta_days < 0:
        raise ValueError(
            "Time since study cannot be negative."
        )

    return delta_days


def validate_half_life(half_life_days):
    if half_life_days is None:
        raise ValueError(
            "Half-life cannot be None."
        )

    half_life_days = float(
        half_life_days
    )

    if half_life_days <= 0:
        raise ValueError(
            "Half-life must be greater than zero."
        )

    return half_life_days


def validate_retention(retention):
    if retention is None:
        raise ValueError(
            "Retention cannot be None."
        )

    retention = float(retention)

    if not (
        MIN_RETENTION
        <= retention
        <= MAX_RETENTION
    ):
        raise ValueError(
            "Retention must be between 0 and 1."
        )

    return retention


def validate_threshold(threshold):
    threshold = validate_retention(
        threshold
    )

    if threshold == 0:
        raise ValueError(
            "Retention threshold must "
            "be greater than zero."
        )

    if threshold == 1:
        raise ValueError(
            "Retention threshold must "
            "be less than one."
        )

    return threshold


# forgetting curve

def calculate_retention(
    delta_days,
    half_life_days=DEFAULT_HALF_LIFE_DAYS
):
    delta_days = validate_delta(
        delta_days
    )

    half_life_days = validate_half_life(
        half_life_days
    )

    retention = 2 ** (
        -delta_days
        / half_life_days
    )

    return max(
        MIN_RETENTION,
        min(
            retention,
            MAX_RETENTION
        )
    )


# forgetting percentage

def calculate_forgetting(
    delta_days,
    half_life_days=DEFAULT_HALF_LIFE_DAYS
):
    retention = calculate_retention(
        delta_days,
        half_life_days
    )

    return 1 - retention


# half-life estimation

def estimate_half_life(
    delta_days,
    observed_recall
):
    delta_days = validate_delta(
        delta_days
    )

    observed_recall = validate_retention(
        observed_recall
    )

    if delta_days == 0:
        return None

    if observed_recall <= 0:
        return None

    if observed_recall >= 1:
        return None

    denominator = math.log2(
        observed_recall
    )

    if denominator == 0:
        return None

    half_life = (
        -delta_days
        / denominator
    )

    if (
        not math.isfinite(
            half_life
        )
        or half_life <= 0
    ):
        return None

    return half_life


# revision interval

def calculate_revision_interval(
    half_life_days,
    threshold=TARGET_RETENTION_THRESHOLD
):
    half_life_days = validate_half_life(
        half_life_days
    )

    threshold = validate_threshold(
        threshold
    )

    interval_days = (
        -half_life_days
        * math.log2(
            threshold
        )
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


# time until threshold

def calculate_time_until_threshold(
    current_delta_days,
    half_life_days,
    threshold=TARGET_RETENTION_THRESHOLD
):
    current_delta_days = validate_delta(
        current_delta_days
    )

    threshold_interval = (
        calculate_revision_interval(
            half_life_days,
            threshold
        )
    )

    remaining_days = (
        threshold_interval
        - current_delta_days
    )

    return max(
        remaining_days,
        0.0
    )


# threshold check

def is_below_threshold(
    delta_days,
    half_life_days,
    threshold=TARGET_RETENTION_THRESHOLD
):
    threshold = validate_threshold(
        threshold
    )

    retention = calculate_retention(
        delta_days,
        half_life_days
    )

    return retention < threshold


def has_reached_threshold(
    delta_days,
    half_life_days,
    threshold=TARGET_RETENTION_THRESHOLD
):
    threshold = validate_threshold(
        threshold
    )

    retention = calculate_retention(
        delta_days,
        half_life_days
    )

    return retention <= threshold


# retention status

def classify_retention(
    retention,
    threshold=TARGET_RETENTION_THRESHOLD
):
    retention = validate_retention(
        retention
    )

    threshold = validate_threshold(
        threshold
    )

    if retention >= 0.85:
        return "Strong"

    if retention >= threshold:
        return "Stable"

    if retention >= 0.50:
        return "Needs Revision"

    return "High Forgetting Risk"


# memory strength

def calculate_memory_strength(
    half_life_days,
    baseline_half_life=DEFAULT_HALF_LIFE_DAYS
):
    half_life_days = validate_half_life(
        half_life_days
    )

    baseline_half_life = validate_half_life(
        baseline_half_life
    )

    return (
        half_life_days
        / baseline_half_life
    )


# half-life comparison

def compare_half_lives(
    fixed_half_life,
    personalized_half_life
):
    fixed_half_life = validate_half_life(
        fixed_half_life
    )

    personalized_half_life = (
        validate_half_life(
            personalized_half_life
        )
    )

    difference = (
        personalized_half_life
        - fixed_half_life
    )

    percentage_change = (
        difference
        / fixed_half_life
        * 100
    )

    if difference > 0:
        interpretation = (
            "Personalized memory strength "
            "is higher than the fixed baseline."
        )

    elif difference < 0:
        interpretation = (
            "Personalized memory strength "
            "is lower than the fixed baseline."
        )

    else:
        interpretation = (
            "Personalized and fixed memory "
            "strength are equal."
        )

    return {
        "fixed_half_life":
            fixed_half_life,

        "personalized_half_life":
            personalized_half_life,

        "difference":
            difference,

        "percentage_change":
            percentage_change,

        "interpretation":
            interpretation,
    }


# retention curve generation

def generate_retention_curve(
    maximum_days=7,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    points=100
):
    maximum_days = float(
        maximum_days
    )

    half_life_days = validate_half_life(
        half_life_days
    )

    points = int(
        points
    )

    if maximum_days <= 0:
        raise ValueError(
            "Maximum days must be "
            "greater than zero."
        )

    if points < 2:
        raise ValueError(
            "At least two curve points "
            "are required."
        )

    step = (
        maximum_days
        / (points - 1)
    )

    curve = []

    for index in range(
        points
    ):
        delta_days = (
            index
            * step
        )

        retention = (
            calculate_retention(
                delta_days,
                half_life_days
            )
        )

        curve.append(
            {
                "delta_days":
                    delta_days,

                "retention":
                    retention,

                "retention_percent":
                    retention * 100,

                "forgetting":
                    1 - retention,

                "forgetting_percent":
                    (
                        1 - retention
                    )
                    * 100,
            }
        )

    return curve


# checkpoint predictions

def predict_checkpoints(
    checkpoints=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS
):
    if checkpoints is None:
        checkpoints = [
            1,
            3,
            7
        ]

    results = {}

    for day in checkpoints:
        day = validate_delta(
            day
        )

        retention = (
            calculate_retention(
                day,
                half_life_days
            )
        )

        results[day] = {
            "retention":
                retention,

            "retention_percent":
                retention * 100,

            "forgetting":
                1 - retention,

            "status":
                classify_retention(
                    retention
                ),
        }

    return results


# observed half-life analysis

def analyze_observed_recall(
    delta_days,
    observed_recall,
    fixed_half_life=DEFAULT_HALF_LIFE_DAYS
):
    delta_days = validate_delta(
        delta_days
    )

    observed_recall = validate_retention(
        observed_recall
    )

    fixed_prediction = (
        calculate_retention(
            delta_days,
            fixed_half_life
        )
    )

    estimated_half_life = (
        estimate_half_life(
            delta_days,
            observed_recall
        )
    )

    absolute_error = abs(
        fixed_prediction
        - observed_recall
    )

    result = {
        "delta_days":
            delta_days,

        "observed_recall":
            observed_recall,

        "fixed_prediction":
            fixed_prediction,

        "absolute_error":
            absolute_error,

        "estimated_half_life":
            estimated_half_life,
    }

    if estimated_half_life is not None:
        result[
            "memory_strength_ratio"
        ] = calculate_memory_strength(
            estimated_half_life,
            fixed_half_life
        )

    else:
        result[
            "memory_strength_ratio"
        ] = None

    return result


# model prediction conversion

def retention_to_half_life(
    predicted_retention,
    prediction_days
):
    return estimate_half_life(
        prediction_days,
        predicted_retention
    )


def half_life_to_retention(
    half_life_days,
    prediction_days
):
    return calculate_retention(
        prediction_days,
        half_life_days
    )


# direct execution

if __name__ == "__main__":
    print(
        "MemoryMap Forgetting Curve"
    )

    print()

    print(
        "Default Half-Life:",
        DEFAULT_HALF_LIFE_DAYS,
        "day(s)"
    )

    print()

    checkpoints = (
        predict_checkpoints()
    )

    for day, result in (
        checkpoints.items()
    ):
        print(
            f"After {day:g} day(s): "
            f"{result['retention_percent']:.2f}% "
            f"retention "
            f"({result['status']})"
        )

    print()

    revision_interval = (
        calculate_revision_interval(
            DEFAULT_HALF_LIFE_DAYS,
            TARGET_RETENTION_THRESHOLD
        )
    )

    print(
        "Revision interval at "
        f"{TARGET_RETENTION_THRESHOLD * 100:.0f}% "
        "threshold:",
        f"{revision_interval:.2f} days"
    )

    print()

    example_half_life = (
        estimate_half_life(
            delta_days=3,
            observed_recall=0.75
        )
    )

    if example_half_life is not None:
        print(
            "Estimated half-life from "
            "75% recall after 3 days:",
            f"{example_half_life:.2f} days"
        )