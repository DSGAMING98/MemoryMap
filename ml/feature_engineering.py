import math

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATASET_PATH,
    TARGET_COLUMN,
    MAX_TRAINING_ROWS,
    RANDOM_STATE,
    PREPROCESS_CHUNK_SIZE,
)


# feature definitions

DUOLINGO_BASE_FEATURES = [
    "delta_days",
    "history_seen",
    "history_accuracy",
]


DUOLINGO_EXTENDED_FEATURES = [
    "delta_days",
    "log_delta_days",
    "history_seen",
    "log_history_seen",
    "history_correct",
    "history_incorrect",
    "history_accuracy",
]


MEMORYMAP_PERSONAL_FEATURES = [
    "study_duration",
    "topic_difficulty",
    "confidence_level",
    "revision_count",
    "previous_quiz_score",
    "time_since_revision",
]


# Values produced during the current
# recall/quiz event must never be used
# to predict that same event.

LEAKAGE_COLUMNS = {
    "p_recall",
    "session_seen",
    "session_correct",
    "session_accuracy",
    "actual_score",
    "recall_accuracy",
    "correct_answers",
    "incorrect_answers",
    "quiz_score",
}


CONTEXT_COLUMNS = [
    "timestamp",
]


FEATURE_SETS = {
    "base":
        DUOLINGO_BASE_FEATURES,

    "extended":
        DUOLINGO_EXTENDED_FEATURES,

    "memorymap":
        MEMORYMAP_PERSONAL_FEATURES,
}


# feature descriptions

FEATURE_DESCRIPTIONS = {
    "delta_days":
        (
            "Time since previous practice "
            "measured in days."
        ),

    "log_delta_days":
        (
            "Log-transformed time since "
            "previous practice."
        ),

    "history_seen":
        (
            "Number of previous exposures "
            "before the current session."
        ),

    "log_history_seen":
        (
            "Log-transformed number of "
            "previous exposures."
        ),

    "history_correct":
        (
            "Number of previous correct "
            "responses before the current "
            "session."
        ),

    "history_incorrect":
        (
            "Number of previous incorrect "
            "responses before the current "
            "session."
        ),

    "history_accuracy":
        (
            "Previous correct responses "
            "divided by previous exposures."
        ),

    "study_duration":
        (
            "Length of a MemoryMap study "
            "session in minutes."
        ),

    "topic_difficulty":
        (
            "Student-reported topic "
            "difficulty from 1 to 5."
        ),

    "confidence_level":
        (
            "Student confidence after "
            "studying from 1 to 5."
        ),

    "revision_count":
        (
            "Number of earlier study or "
            "revision sessions for the topic."
        ),

    "previous_quiz_score":
        (
            "Most recent genuine earlier "
            "recall score from 0 to 100."
        ),

    "time_since_revision":
        (
            "Elapsed time since the relevant "
            "previous study or revision event, "
            "measured in days."
        ),
}


# feature set helpers

def validate_feature_set(
    feature_set
):
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: "
            f"{feature_set}"
        )

    return feature_set


def get_feature_columns(
    feature_set="extended"
):
    feature_set = validate_feature_set(
        feature_set
    )

    return list(
        FEATURE_SETS[
            feature_set
        ]
    )


def get_feature_descriptions(
    feature_set="extended"
):
    feature_columns = (
        get_feature_columns(
            feature_set
        )
    )

    return {
        feature:
            FEATURE_DESCRIPTIONS.get(
                feature,
                "No description available."
            )

        for feature
        in feature_columns
    }


# leakage protection

def check_for_target_leakage(
    feature_columns
):
    if feature_columns is None:
        raise ValueError(
            "Feature columns are required."
        )

    feature_columns = set(
        feature_columns
    )

    leaked_columns = (
        feature_columns
        & LEAKAGE_COLUMNS
    )

    if leaked_columns:
        raise ValueError(
            "Target leakage detected. "
            "These columns cannot be used "
            "as prediction features: "
            + ", ".join(
                sorted(
                    leaked_columns
                )
            )
        )

    return True


# dataframe validation

def validate_required_columns(
    dataframe,
    required_columns
):
    if not isinstance(
        dataframe,
        pd.DataFrame
    ):
        raise ValueError(
            "Expected a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in required_columns

        if column
        not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required "
            "columns: "
            + ", ".join(
                missing_columns
            )
        )

    return True


# numeric helpers

def safe_numeric(
    series
):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def safe_log1p(
    values
):
    values = safe_numeric(
        values
    )

    result = pd.Series(
        np.nan,
        index=values.index,
        dtype=float
    )

    valid = (
        values.notna()
        & (values >= 0)
    )

    result.loc[
        valid
    ] = np.log1p(
        values.loc[
            valid
        ]
    )

    return result


# log features

def create_log_features(
    dataframe
):
    dataframe = (
        dataframe.copy()
    )

    if (
        "delta_days"
        in dataframe.columns
    ):
        dataframe[
            "log_delta_days"
        ] = safe_log1p(
            dataframe[
                "delta_days"
            ]
        )

    if (
        "history_seen"
        in dataframe.columns
    ):
        dataframe[
            "log_history_seen"
        ] = safe_log1p(
            dataframe[
                "history_seen"
            ]
        )

    return dataframe


# history features

def create_history_features(
    dataframe
):
    dataframe = (
        dataframe.copy()
    )

    if (
        "history_seen"
        not in dataframe.columns
    ):
        return dataframe

    dataframe[
        "history_seen"
    ] = safe_numeric(
        dataframe[
            "history_seen"
        ]
    )

    if (
        "history_correct"
        in dataframe.columns
    ):
        dataframe[
            "history_correct"
        ] = safe_numeric(
            dataframe[
                "history_correct"
            ]
        )

    if (
        "history_incorrect"
        in dataframe.columns
    ):
        dataframe[
            "history_incorrect"
        ] = safe_numeric(
            dataframe[
                "history_incorrect"
            ]
        )

    # Derive incorrect count only from
    # genuine historical values.

    if (
        "history_correct"
        in dataframe.columns
        and
        "history_incorrect"
        not in dataframe.columns
    ):
        dataframe[
            "history_incorrect"
        ] = (
            dataframe[
                "history_seen"
            ]
            - dataframe[
                "history_correct"
            ]
        )

    # Also support datasets containing
    # incorrect count but not correct count.

    if (
        "history_incorrect"
        in dataframe.columns
        and
        "history_correct"
        not in dataframe.columns
    ):
        dataframe[
            "history_correct"
        ] = (
            dataframe[
                "history_seen"
            ]
            - dataframe[
                "history_incorrect"
            ]
        )

    if (
        "history_accuracy"
        in dataframe.columns
    ):
        dataframe[
            "history_accuracy"
        ] = safe_numeric(
            dataframe[
                "history_accuracy"
            ]
        )

    elif (
        "history_correct"
        in dataframe.columns
    ):
        seen = dataframe[
            "history_seen"
        ]

        correct = dataframe[
            "history_correct"
        ]

        dataframe[
            "history_accuracy"
        ] = np.where(
            seen > 0,
            correct / seen,
            0.0
        )

    return dataframe


# duolingo engineering

def engineer_duolingo_features(
    dataframe
):
    dataframe = (
        dataframe.copy()
    )

    dataframe = (
        create_history_features(
            dataframe
        )
    )

    dataframe = (
        create_log_features(
            dataframe
        )
    )

    return dataframe


# memorymap engineering

def engineer_memorymap_features(
    dataframe
):
    dataframe = (
        dataframe.copy()
    )

    numeric_columns = [
        "study_duration",
        "topic_difficulty",
        "confidence_level",
        "revision_count",
        "previous_quiz_score",
        "time_since_revision",
    ]

    for column in numeric_columns:
        if (
            column
            not in dataframe.columns
        ):
            continue

        dataframe[
            column
        ] = safe_numeric(
            dataframe[
                column
            ]
        )

    # Important:
    #
    # Do not replace a missing previous
    # quiz score with zero.
    #
    # Missing means no earlier measurement.
    # Zero means the student genuinely
    # scored 0%.

    return dataframe


# general feature engineering

def engineer_features(
    dataframe,
    feature_set="extended"
):
    feature_set = validate_feature_set(
        feature_set
    )

    if feature_set in {
        "base",
        "extended",
    }:
        return (
            engineer_duolingo_features(
                dataframe
            )
        )

    if feature_set == "memorymap":
        return (
            engineer_memorymap_features(
                dataframe
            )
        )

    raise ValueError(
        f"Unsupported feature set: "
        f"{feature_set}"
    )


# numeric conversion

def convert_features_to_numeric(
    dataframe,
    feature_columns
):
    dataframe = (
        dataframe.copy()
    )

    validate_required_columns(
        dataframe,
        feature_columns
    )

    for column in feature_columns:
        dataframe[
            column
        ] = safe_numeric(
            dataframe[
                column
            ]
        )

    return dataframe


# missing or non-finite values

def replace_non_finite_values(
    dataframe,
    columns
):
    dataframe = (
        dataframe.copy()
    )

    for column in columns:
        if (
            column
            not in dataframe.columns
        ):
            continue

        dataframe[
            column
        ] = dataframe[
            column
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan
        )

    return dataframe


def remove_invalid_feature_rows(
    dataframe,
    feature_columns,
    target_column=None
):
    required_columns = list(
        feature_columns
    )

    if target_column is not None:
        required_columns.append(
            target_column
        )

    dataframe = (
        replace_non_finite_values(
            dataframe,
            required_columns
        )
    )

    dataframe = dataframe.dropna(
        subset=required_columns
    )

    return dataframe


# duolingo range validation

def remove_invalid_duolingo_rows(
    dataframe
):
    dataframe = (
        dataframe.copy()
    )

    if (
        "delta_days"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "delta_days"
            ] >= 0
        ]

    if (
        "history_seen"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "history_seen"
            ] >= 0
        ]

    if (
        "history_correct"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "history_correct"
            ] >= 0
        ]

    if (
        "history_incorrect"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "history_incorrect"
            ] >= 0
        ]

    if (
        "history_seen"
        in dataframe.columns
        and
        "history_correct"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "history_correct"
            ]
            <= dataframe[
                "history_seen"
            ]
        ]

    if (
        "history_seen"
        in dataframe.columns
        and
        "history_incorrect"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "history_incorrect"
            ]
            <= dataframe[
                "history_seen"
            ]
        ]

    if (
        "history_accuracy"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "history_accuracy"
            ].between(
                0,
                1
            )
        ]

    if (
        TARGET_COLUMN
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                TARGET_COLUMN
            ].between(
                0,
                1
            )
        ]

    return dataframe


# memorymap range validation

def remove_invalid_memorymap_rows(
    dataframe
):
    dataframe = (
        dataframe.copy()
    )

    if (
        "study_duration"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "study_duration"
            ] > 0
        ]

    if (
        "topic_difficulty"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "topic_difficulty"
            ].between(
                1,
                5
            )
        ]

    if (
        "confidence_level"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "confidence_level"
            ].between(
                1,
                5
            )
        ]

    if (
        "revision_count"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "revision_count"
            ] >= 0
        ]

    if (
        "previous_quiz_score"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "previous_quiz_score"
            ].between(
                0,
                100
            )
        ]

    if (
        "time_since_revision"
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                "time_since_revision"
            ] >= 0
        ]

    if (
        TARGET_COLUMN
        in dataframe.columns
    ):
        dataframe = dataframe[
            dataframe[
                TARGET_COLUMN
            ].between(
                0,
                1
            )
        ]

    return dataframe


# feature range dispatch

def remove_invalid_rows_for_feature_set(
    dataframe,
    feature_set
):
    feature_set = validate_feature_set(
        feature_set
    )

    if feature_set in {
        "base",
        "extended",
    }:
        return (
            remove_invalid_duolingo_rows(
                dataframe
            )
        )

    return (
        remove_invalid_memorymap_rows(
            dataframe
        )
    )


# training frame

def create_training_frame(
    dataframe,
    feature_set="extended",
    include_context=True
):
    feature_set = validate_feature_set(
        feature_set
    )

    feature_columns = (
        get_feature_columns(
            feature_set
        )
    )

    check_for_target_leakage(
        feature_columns
    )

    dataframe = (
        engineer_features(
            dataframe,
            feature_set
        )
    )

    required_columns = (
        feature_columns
        + [
            TARGET_COLUMN
        ]
    )

    validate_required_columns(
        dataframe,
        required_columns
    )

    dataframe = (
        convert_features_to_numeric(
            dataframe,
            feature_columns
        )
    )

    dataframe[
        TARGET_COLUMN
    ] = safe_numeric(
        dataframe[
            TARGET_COLUMN
        ]
    )

    dataframe = (
        remove_invalid_feature_rows(
            dataframe,
            feature_columns,
            TARGET_COLUMN
        )
    )

    dataframe = (
        remove_invalid_rows_for_feature_set(
            dataframe,
            feature_set
        )
    )

    output_columns = (
        feature_columns
        + [
            TARGET_COLUMN
        ]
    )

    if include_context:
        for column in CONTEXT_COLUMNS:
            if (
                column
                in dataframe.columns
                and
                column
                not in output_columns
            ):
                output_columns.append(
                    column
                )

    return dataframe[
        output_columns
    ].copy()


# x and y separation

def split_features_target(
    dataframe,
    feature_set="extended"
):
    feature_set = validate_feature_set(
        feature_set
    )

    feature_columns = (
        get_feature_columns(
            feature_set
        )
    )

    validate_required_columns(
        dataframe,
        feature_columns
        + [
            TARGET_COLUMN
        ]
    )

    check_for_target_leakage(
        feature_columns
    )

    x = dataframe[
        feature_columns
    ].copy()

    y = dataframe[
        TARGET_COLUMN
    ].copy()

    return (
        x,
        y
    )


# single feature row

def engineer_single_feature_row(
    feature_values,
    feature_set="extended"
):
    if not isinstance(
        feature_values,
        dict
    ):
        raise ValueError(
            "Feature values must be "
            "provided as a dictionary."
        )

    feature_set = validate_feature_set(
        feature_set
    )

    feature_columns = (
        get_feature_columns(
            feature_set
        )
    )

    check_for_target_leakage(
        feature_columns
    )

    supplied_leakage = (
        set(
            feature_values.keys()
        )
        & LEAKAGE_COLUMNS
    )

    if supplied_leakage:
        raise ValueError(
            "Outcome information cannot "
            "be supplied as model input: "
            + ", ".join(
                sorted(
                    supplied_leakage
                )
            )
        )

    dataframe = pd.DataFrame(
        [
            feature_values
        ]
    )

    dataframe = (
        engineer_features(
            dataframe,
            feature_set
        )
    )

    validate_required_columns(
        dataframe,
        feature_columns
    )

    dataframe = (
        convert_features_to_numeric(
            dataframe,
            feature_columns
        )
    )

    dataframe = (
        replace_non_finite_values(
            dataframe,
            feature_columns
        )
    )

    if dataframe[
        feature_columns
    ].isna().any().any():

        missing = [
            column
            for column
            in feature_columns

            if pd.isna(
                dataframe.iloc[
                    0
                ][
                    column
                ]
            )
        ]

        raise ValueError(
            "Missing feature values: "
            + ", ".join(
                missing
            )
        )

    validated = (
        remove_invalid_rows_for_feature_set(
            dataframe,
            feature_set
        )
    )

    if validated.empty:
        raise ValueError(
            "One or more feature values "
            "are outside the valid range."
        )

    return validated[
        feature_columns
    ].copy()


# training sample loader

def get_source_columns(
    feature_set,
    available_columns
):
    feature_set = validate_feature_set(
        feature_set
    )

    available_columns = set(
        available_columns
    )

    desired_columns = {
        TARGET_COLUMN,
        "timestamp",
    }

    if feature_set in {
        "base",
        "extended",
    }:
        desired_columns.update(
            {
                "delta_days",
                "history_seen",
                "history_correct",
                "history_incorrect",
                "history_accuracy",
            }
        )

    else:
        desired_columns.update(
            MEMORYMAP_PERSONAL_FEATURES
        )

    return [
        column
        for column in desired_columns
        if column in available_columns
    ]


def load_training_sample(
    dataset_path=PROCESSED_DATASET_PATH,
    feature_set="extended",
    max_rows=MAX_TRAINING_ROWS,
    random_state=RANDOM_STATE,
    chunk_size=PREPROCESS_CHUNK_SIZE
):
    feature_set = validate_feature_set(
        feature_set
    )

    header = pd.read_csv(
        dataset_path,
        nrows=0
    )

    available_columns = list(
        header.columns
    )

    use_columns = get_source_columns(
        feature_set,
        available_columns
    )

    if TARGET_COLUMN not in use_columns:
        raise ValueError(
            "Dataset does not contain "
            f"the target column: "
            f"{TARGET_COLUMN}"
        )

    if max_rows is not None:
        try:
            max_rows = int(
                max_rows
            )

        except (
            TypeError,
            ValueError
        ):
            raise ValueError(
                "Maximum training rows "
                "must be an integer."
            )

        if max_rows <= 0:
            raise ValueError(
                "Maximum training rows "
                "must be positive."
            )

    if chunk_size is None:
        chunk_size = 250_000

    try:
        chunk_size = int(
            chunk_size
        )

    except (
        TypeError,
        ValueError
    ):
        chunk_size = 250_000

    chunk_size = max(
        chunk_size,
        1
    )

    sampled = []

    total_sampled_rows = 0

    reader = pd.read_csv(
        dataset_path,
        usecols=use_columns,
        chunksize=chunk_size
    )

    for chunk_index, chunk in enumerate(
        reader
    ):
        try:
            frame = (
                create_training_frame(
                    chunk,
                    feature_set=feature_set,
                    include_context=True
                )
            )

        except ValueError:
            continue

        if frame.empty:
            continue

        sampled.append(
            frame
        )

        total_sampled_rows += len(
            frame
        )

        # Periodically shrink the working
        # sample so memory remains bounded.

        if (
            max_rows is not None
            and
            total_sampled_rows
            > max_rows * 2
        ):
            combined = pd.concat(
                sampled,
                ignore_index=True
            )

            combined = combined.sample(
                n=min(
                    max_rows,
                    len(
                        combined
                    )
                ),
                random_state=(
                    random_state
                    + chunk_index
                )
            )

            sampled = [
                combined
            ]

            total_sampled_rows = len(
                combined
            )

    if not sampled:
        return pd.DataFrame(
            columns=(
                get_feature_columns(
                    feature_set
                )
                + [
                    TARGET_COLUMN,
                ]
                + CONTEXT_COLUMNS
            )
        )

    dataframe = pd.concat(
        sampled,
        ignore_index=True
    )

    if (
        max_rows is not None
        and
        len(
            dataframe
        ) > max_rows
    ):
        dataframe = dataframe.sample(
            n=max_rows,
            random_state=random_state
        )

    if (
        "timestamp"
        in dataframe.columns
    ):
        dataframe[
            "timestamp"
        ] = pd.to_numeric(
            dataframe[
                "timestamp"
            ],
            errors="coerce"
        )

        dataframe = (
            dataframe
            .sort_values(
                "timestamp",
                na_position="last"
            )
            .reset_index(
                drop=True
            )
        )

    else:
        dataframe = dataframe.reset_index(
            drop=True
        )

    return dataframe


# feature statistics

def finite_statistic(
    value
):
    try:
        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return None

    if not math.isfinite(
        value
    ):
        return None

    return value


def get_feature_statistics(
    dataframe,
    feature_set="extended"
):
    feature_columns = (
        get_feature_columns(
            feature_set
        )
    )

    validate_required_columns(
        dataframe,
        feature_columns
    )

    statistics = {}

    for feature in feature_columns:
        series = safe_numeric(
            dataframe[
                feature
            ]
        )

        series = series.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan
        )

        statistics[
            feature
        ] = {
            "count":
                int(
                    series.count()
                ),

            "missing":
                int(
                    series.isna().sum()
                ),

            "mean":
                finite_statistic(
                    series.mean()
                ),

            "std":
                finite_statistic(
                    series.std()
                ),

            "minimum":
                finite_statistic(
                    series.min()
                ),

            "maximum":
                finite_statistic(
                    series.max()
                ),
        }

    return statistics


# metadata creation

def create_feature_metadata(
    feature_set="extended"
):
    feature_set = validate_feature_set(
        feature_set
    )

    feature_columns = (
        get_feature_columns(
            feature_set
        )
    )

    check_for_target_leakage(
        feature_columns
    )

    return {
        "feature_set":
            feature_set,

        "feature_columns":
            feature_columns,

        "target_column":
            TARGET_COLUMN,

        "target_scale":
            "0-1",

        "feature_descriptions":
            get_feature_descriptions(
                feature_set
            ),

        "blocked_leakage_columns":
            sorted(
                LEAKAGE_COLUMNS
            ),

        "context_columns":
            list(
                CONTEXT_COLUMNS
            ),

        "missing_value_policy":
            (
                "Missing feature values are "
                "not replaced with fabricated "
                "student measurements."
            ),

        "memorymap_public_model_note":
            (
                "MemoryMap personal features "
                "must not be converted into "
                "Duolingo history features."
            ),
    }


# direct execution

if __name__ == "__main__":

    print(
        "MemoryMap Feature Engineering"
    )

    print(
        "=" * 40
    )

    print(
        "Safe Base Features:"
    )

    for feature in (
        get_feature_columns(
            "base"
        )
    ):
        print(
            "-",
            feature
        )

    print()

    print(
        "Safe Extended Features:"
    )

    for feature in (
        get_feature_columns(
            "extended"
        )
    ):
        print(
            "-",
            feature
        )

    print()

    print(
        "MemoryMap Personal Features:"
    )

    for feature in (
        get_feature_columns(
            "memorymap"
        )
    ):
        print(
            "-",
            feature
        )

    print()

    print(
        "Blocked Leakage Columns:"
    )

    for column in sorted(
        LEAKAGE_COLUMNS
    ):
        print(
            "-",
            column
        )

    print()

    try:
        training_data = (
            load_training_sample(
                feature_set="base",
                max_rows=100_000
            )
        )

        print(
            "Training Sample:",
            f"{len(training_data):,}",
            "rows"
        )

        if not training_data.empty:
            print()

            print(
                training_data.head()
            )

            print()

            x, y = (
                split_features_target(
                    training_data,
                    feature_set="base"
                )
            )

            print(
                "X Shape:",
                x.shape
            )

            print(
                "Y Shape:",
                y.shape
            )

    except FileNotFoundError:
        print(
            "Processed dataset has not "
            "been created yet."
        )

    except ValueError as error:
        print(
            "Feature engineering test "
            "could not run:"
        )

        print(
            error
        )