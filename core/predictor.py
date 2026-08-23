import math

from config import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_PREDICTION_MODE,
)

from core.forgetting_curve import (
    calculate_retention,
    estimate_half_life,
)

from ml.load_model import (
    load_model_bundle,
    get_model_information,
    prepare_model_input,
    transform_model_input,
    clear_model_cache as clear_loaded_model_cache,
)


FIXED_MODE = "Fixed Forgetting Curve"

PERSONALIZED_MODE = "Personalized Model"

ADAPTIVE_MODE = "Adaptive Schedule"


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


def clamp_retention(
    retention
):
    retention = float(
        retention
    )

    return max(
        0.0,
        min(
            retention,
            1.0
        )
    )


def clear_model_cache():
    clear_loaded_model_cache()


def get_model_status():
    information = (
        get_model_information()
    )

    return {
        "model_available":
            information.get(
                "available",
                False
            ),

        "model_name":
            information.get(
                "model_name"
            ),

        "feature_set":
            information.get(
                "feature_set"
            ),

        "feature_columns":
            information.get(
                "feature_columns",
                []
            ),

        "uses_scaler":
            information.get(
                "uses_scaler",
                False
            ),

        "training_source":
            information.get(
                "training_source"
            ),

        "trained_at":
            information.get(
                "trained_at"
            ),

        "message":
            information.get(
                "message"
            ),
    }


def derive_history_features(
    features
):
    history_seen = safe_float(
        features.get(
            "history_seen"
        )
    )

    history_correct = safe_float(
        features.get(
            "history_correct"
        )
    )

    if (
        history_seen is None
        or history_correct is None
    ):
        return features

    features[
        "history_incorrect"
    ] = max(
        history_seen
        - history_correct,
        0.0
    )

    if history_seen > 0:
        features[
            "history_accuracy"
        ] = (
            history_correct
            / history_seen
        )

    else:
        features[
            "history_accuracy"
        ] = 0.0

    return features


def create_feature_map(
    session=None,
    topic=None,
    prediction_days=1.0,
    extra_features=None
):
    prediction_days = safe_float(
        prediction_days
    )

    if prediction_days is None:
        raise ValueError(
            "Prediction interval must "
            "be numeric."
        )

    if prediction_days < 0:
        raise ValueError(
            "Prediction interval cannot "
            "be negative."
        )

    features = {
        "delta_days":
            prediction_days,

        "time_since_revision":
            prediction_days,
    }

    if session:
        features[
            "study_duration"
        ] = safe_float(
            session.get(
                "study_duration"
            )
        )

        features[
            "confidence_level"
        ] = safe_float(
            session.get(
                "confidence_level"
            )
        )

        features[
            "revision_count"
        ] = safe_float(
            session.get(
                "revision_count"
            )
        )

        features[
            "previous_quiz_score"
        ] = safe_float(
            session.get(
                "previous_quiz_score"
            )
        )

    if topic:
        features[
            "topic_difficulty"
        ] = safe_float(
            topic.get(
                "difficulty"
            )
        )

    if extra_features:
        for key, value in (
            extra_features.items()
        ):
            features[
                key
            ] = value

    features = derive_history_features(
        features
    )

    return features


def normalize_model_prediction(
    prediction,
    metadata
):
    prediction = float(
        prediction
    )

    target_scale = metadata.get(
        "target_scale",
        "0-1"
    )

    if target_scale == "0-100":
        prediction = (
            prediction
            / 100
        )

    raw_prediction = prediction

    clipped_prediction = (
        clamp_retention(
            prediction
        )
    )

    was_clipped = not math.isclose(
        raw_prediction,
        clipped_prediction,
        rel_tol=1e-9,
        abs_tol=1e-9
    )

    return {
        "prediction":
            clipped_prediction,

        "raw_prediction":
            raw_prediction,

        "was_clipped":
            was_clipped,
    }


def predict_fixed_retention(
    prediction_days,
    half_life_days=DEFAULT_HALF_LIFE_DAYS
):
    prediction_days = safe_float(
        prediction_days
    )

    if prediction_days is None:
        raise ValueError(
            "Prediction interval must "
            "be numeric."
        )

    if prediction_days < 0:
        raise ValueError(
            "Prediction interval cannot "
            "be negative."
        )

    retention = calculate_retention(
        prediction_days,
        half_life_days
    )

    return {
        "available":
            True,

        "mode":
            FIXED_MODE,

        "retention":
            retention,

        "half_life":
            half_life_days,

        "prediction_days":
            prediction_days,

        "source":
            "Fixed forgetting curve",

        "fallback_used":
            False,

        "message":
            "Fixed prediction calculated.",
    }


def predict_personalized_retention(
    session,
    topic,
    prediction_days,
    extra_features=None
):
    prediction_days = safe_float(
        prediction_days
    )

    if prediction_days is None:
        return {
            "available":
                False,

            "mode":
                PERSONALIZED_MODE,

            "retention":
                None,

            "half_life":
                None,

            "prediction_days":
                None,

            "source":
                None,

            "message":
                (
                    "Prediction interval "
                    "must be numeric."
                ),
        }

    if prediction_days < 0:
        return {
            "available":
                False,

            "mode":
                PERSONALIZED_MODE,

            "retention":
                None,

            "half_life":
                None,

            "prediction_days":
                prediction_days,

            "source":
                None,

            "message":
                (
                    "Prediction interval "
                    "cannot be negative."
                ),
        }

    try:
        bundle = load_model_bundle()

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError
    ) as error:
        return {
            "available":
                False,

            "mode":
                PERSONALIZED_MODE,

            "retention":
                None,

            "half_life":
                None,

            "prediction_days":
                prediction_days,

            "source":
                None,

            "message":
                str(
                    error
                ),
        }

    model = bundle[
        "model"
    ]

    scaler = bundle[
        "scaler"
    ]

    metadata = bundle[
        "metadata"
    ]

    features = create_feature_map(
        session=session,
        topic=topic,
        prediction_days=(
            prediction_days
        ),
        extra_features=(
            extra_features
        )
    )

    required_features = metadata.get(
        "feature_columns",
        []
    )

    missing_features = [
        feature
        for feature in required_features
        if features.get(
            feature
        ) is None
    ]

    if missing_features:
        return {
            "available":
                False,

            "mode":
                PERSONALIZED_MODE,

            "retention":
                None,

            "half_life":
                None,

            "prediction_days":
                prediction_days,

            "source":
                None,

            "missing_features":
                missing_features,

            "message":
                (
                    "Personalized prediction "
                    "requires: "
                    + ", ".join(
                        missing_features
                    )
                ),
        }

    try:
        import pandas as pd

        dataframe = pd.DataFrame(
            [
                features
            ]
        )

        dataframe = prepare_model_input(
            dataframe,
            metadata
        )

        model_input = (
            transform_model_input(
                dataframe,
                scaler=scaler,
                metadata=metadata
            )
        )

        raw_prediction = (
            model.predict(
                model_input
            )[0]
        )

    except Exception as error:
        return {
            "available":
                False,

            "mode":
                PERSONALIZED_MODE,

            "retention":
                None,

            "half_life":
                None,

            "prediction_days":
                prediction_days,

            "source":
                None,

            "message":
                (
                    "Model prediction failed: "
                    f"{error}"
                ),
        }

    normalized = (
        normalize_model_prediction(
            raw_prediction,
            metadata
        )
    )

    retention = normalized[
        "prediction"
    ]

    half_life = None

    if (
        prediction_days > 0
        and
        0 < retention < 1
    ):
        half_life = (
            estimate_half_life(
                prediction_days,
                retention
            )
        )

    return {
        "available":
            True,

        "mode":
            PERSONALIZED_MODE,

        "retention":
            retention,

        "raw_prediction":
            normalized[
                "raw_prediction"
            ],

        "was_clipped":
            normalized[
                "was_clipped"
            ],

        "half_life":
            half_life,

        "prediction_days":
            prediction_days,

        "source":
            metadata.get(
                "model_display_name",
                metadata.get(
                    "model_name",
                    "Trained retention model"
                )
            ),

        "features":
            features,

        "fallback_used":
            False,

        "message":
            (
                "Personalized prediction "
                "calculated successfully."
            ),
    }


def calculate_observed_half_life(
    observed_recall,
    observed_delay_days
):
    observed_recall = safe_float(
        observed_recall
    )

    observed_delay_days = safe_float(
        observed_delay_days
    )

    if (
        observed_recall is None
        or observed_delay_days is None
    ):
        return None

    if not (
        0 < observed_recall < 1
    ):
        return None

    if observed_delay_days <= 0:
        return None

    try:
        return estimate_half_life(
            observed_delay_days,
            observed_recall
        )

    except (
        TypeError,
        ValueError
    ):
        return None


def predict_adaptive_retention(
    session,
    topic,
    prediction_days,
    observed_recall=None,
    observed_delay_days=None,
    extra_features=None
):
    prediction_days = safe_float(
        prediction_days
    )

    if prediction_days is None:
        raise ValueError(
            "Prediction interval must "
            "be numeric."
        )

    if prediction_days < 0:
        raise ValueError(
            "Prediction interval cannot "
            "be negative."
        )

    observed_half_life = (
        calculate_observed_half_life(
            observed_recall,
            observed_delay_days
        )
    )

    if observed_half_life is not None:
        retention = (
            calculate_retention(
                prediction_days,
                observed_half_life
            )
        )

        return {
            "available":
                True,

            "mode":
                ADAPTIVE_MODE,

            "retention":
                retention,

            "half_life":
                observed_half_life,

            "prediction_days":
                prediction_days,

            "source":
                "Observed recall history",

            "fallback_used":
                False,

            "message":
                (
                    "Adaptive prediction uses "
                    "observed recall history."
                ),
        }

    personalized = (
        predict_personalized_retention(
            session=session,
            topic=topic,
            prediction_days=(
                prediction_days
            ),
            extra_features=(
                extra_features
            )
        )
    )

    if personalized.get(
        "available",
        False
    ):
        result = (
            personalized.copy()
        )

        result[
            "mode"
        ] = ADAPTIVE_MODE

        result[
            "source"
        ] = (
            "Personalized model fallback"
        )

        result[
            "fallback_used"
        ] = True

        result[
            "message"
        ] = (
            "Observed recall was unavailable. "
            "Adaptive prediction therefore uses "
            "the personalized model."
        )

        return result

    fixed = predict_fixed_retention(
        prediction_days
    )

    fixed[
        "mode"
    ] = ADAPTIVE_MODE

    fixed[
        "source"
    ] = (
        "Fixed forgetting curve fallback"
    )

    fixed[
        "fallback_used"
    ] = True

    fixed[
        "message"
    ] = (
        "Observed recall and a usable "
        "personalized prediction were "
        "unavailable. The fixed baseline "
        "was used."
    )

    return fixed


def predict_retention(
    mode=DEFAULT_PREDICTION_MODE,
    session=None,
    topic=None,
    prediction_days=1.0,
    observed_recall=None,
    observed_delay_days=None,
    extra_features=None,
    fixed_half_life=DEFAULT_HALF_LIFE_DAYS
):
    if mode == FIXED_MODE:
        return predict_fixed_retention(
            prediction_days,
            fixed_half_life
        )

    if mode == PERSONALIZED_MODE:
        return (
            predict_personalized_retention(
                session=session,
                topic=topic,
                prediction_days=(
                    prediction_days
                ),
                extra_features=(
                    extra_features
                )
            )
        )

    if mode == ADAPTIVE_MODE:
        return (
            predict_adaptive_retention(
                session=session,
                topic=topic,
                prediction_days=(
                    prediction_days
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
        f"Unknown prediction mode: {mode}"
    )


def predict_all_models(
    session,
    topic,
    prediction_days,
    observed_recall=None,
    observed_delay_days=None,
    extra_features=None
):
    fixed = predict_fixed_retention(
        prediction_days
    )

    personalized = (
        predict_personalized_retention(
            session=session,
            topic=topic,
            prediction_days=(
                prediction_days
            ),
            extra_features=(
                extra_features
            )
        )
    )

    adaptive = (
        predict_adaptive_retention(
            session=session,
            topic=topic,
            prediction_days=(
                prediction_days
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


def compare_prediction_results(
    predictions,
    actual_recall
):
    actual_recall = safe_float(
        actual_recall
    )

    if (
        actual_recall is None
        or not (
            0 <= actual_recall <= 1
        )
    ):
        raise ValueError(
            "Actual recall must be "
            "between 0 and 1."
        )

    rows = []

    for name, result in (
        predictions.items()
    ):
        if not result.get(
            "available",
            False
        ):
            continue

        retention = safe_float(
            result.get(
                "retention"
            )
        )

        if retention is None:
            continue

        error = abs(
            retention
            - actual_recall
        )

        rows.append(
            {
                "model":
                    name,

                "mode":
                    result.get(
                        "mode"
                    ),

                "predicted_recall":
                    retention,

                "actual_recall":
                    actual_recall,

                "absolute_error":
                    error,

                "absolute_error_pp":
                    error * 100,
            }
        )

    rows.sort(
        key=lambda row:
            row[
                "absolute_error"
            ]
    )

    return rows


if __name__ == "__main__":

    print(
        "MemoryMap Predictor"
    )

    print()

    status = get_model_status()

    print(
        "Model Available:",
        status[
            "model_available"
        ]
    )

    print(
        "Model:",
        status[
            "model_name"
        ]
    )

    print()

    fixed = predict_fixed_retention(
        3
    )

    print(
        "Fixed retention after 3 days:",
        f"{fixed['retention'] * 100:.2f}%"
    )