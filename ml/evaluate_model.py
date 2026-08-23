import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATASET_PATH,
    MODELS_DIR,
    RETENTION_MODEL_PATH,
    SCALER_PATH,
    MODEL_METADATA_PATH,
    TARGET_COLUMN,
)

from core.metrics import (
    calculate_model_metrics,
    calculate_correlation,
)

from ml.feature_engineering import (
    create_training_frame,
    split_features_target,
    get_feature_columns,
)

from ml.load_model import (
    get_artifact_status,
    load_metadata as load_model_metadata,
    load_model_bundle,
    validate_metadata as validate_loader_metadata,
    transform_model_input,
)

from ml.train_model import (
    load_uniform_training_sample,
    normalize_predictions,
    validate_public_feature_set,
)


# evaluation output

EVALUATION_REPORT_PATH = (
    MODELS_DIR
    / "evaluation_report.json"
)


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


def validate_numeric_pair_arrays(
    actual,
    predicted
):
    actual = np.asarray(
        actual,
        dtype=float
    ).reshape(
        -1
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    ).reshape(
        -1
    )

    if len(
        actual
    ) != len(
        predicted
    ):
        raise ValueError(
            "Actual and predicted arrays "
            "must have the same length."
        )

    if len(
        actual
    ) == 0:
        raise ValueError(
            "Evaluation arrays cannot "
            "be empty."
        )

    if not np.isfinite(
        actual
    ).all():
        raise ValueError(
            "Actual recall values contain "
            "non-finite values."
        )

    if not np.isfinite(
        predicted
    ).all():
        raise ValueError(
            "Predictions contain "
            "non-finite values."
        )

    return (
        actual,
        predicted
    )


# artifact validation

def validate_model_artifacts():
    status = get_artifact_status()

    missing = []

    if not status[
        "model_exists"
    ]:
        missing.append(
            str(
                RETENTION_MODEL_PATH
            )
        )

    if not status[
        "metadata_exists"
    ]:
        missing.append(
            str(
                MODEL_METADATA_PATH
            )
        )

    if missing:
        raise FileNotFoundError(
            "Required trained model artifacts "
            "were not found: "
            + ", ".join(
                missing
            )
        )

    return True


# metadata loading

def load_metadata():
    validate_model_artifacts()

    return load_model_metadata()


# model loading

def load_saved_model():
    bundle = load_model_bundle()

    return bundle[
        "model"
    ]


def load_saved_scaler(
    metadata=None
):
    bundle = load_model_bundle()

    bundle_metadata = bundle[
        "metadata"
    ]

    if metadata is not None:
        validate_metadata(
            metadata
        )

        if (
            metadata.get(
                "model_name"
            )
            != bundle_metadata.get(
                "model_name"
            )
        ):
            raise ValueError(
                "Supplied metadata does not "
                "match the loaded model bundle."
            )

    return bundle[
        "scaler"
    ]


# metadata validation

def validate_metadata(
    metadata
):
    validate_loader_metadata(
        metadata
    )

    required_fields = [
        "feature_set",
        "feature_columns",
        "target_column",
        "target_scale",
        "sampled_rows",
        "random_state",
        "train_fraction",
        "test_fraction",
        "validation_fraction_within_development",
    ]

    missing_fields = [
        field

        for field
        in required_fields

        if field
        not in metadata
    ]

    if missing_fields:
        raise ValueError(
            "Model metadata is missing "
            "evaluation fields: "
            + ", ".join(
                missing_fields
            )
        )

    feature_set = metadata[
        "feature_set"
    ]

    validate_public_feature_set(
        feature_set
    )

    expected_features = (
        get_feature_columns(
            feature_set
        )
    )

    if metadata[
        "feature_columns"
    ] != expected_features:
        raise ValueError(
            "Metadata feature columns do "
            "not match the recorded "
            "feature set."
        )

    if metadata[
        "target_scale"
    ] != "0-1":
        raise ValueError(
            "This evaluation pipeline expects "
            "the retention target to use "
            "the 0-1 scale."
        )

    train_fraction = safe_float(
        metadata.get(
            "train_fraction"
        )
    )

    test_fraction = safe_float(
        metadata.get(
            "test_fraction"
        )
    )

    validation_fraction = safe_float(
        metadata.get(
            "validation_fraction_within_development"
        )
    )

    if (
        train_fraction is None
        or test_fraction is None
    ):
        raise ValueError(
            "Stored train and test fractions "
            "must be numeric."
        )

    if not (
        0 < train_fraction < 1
    ):
        raise ValueError(
            "Stored train fraction must "
            "be between 0 and 1."
        )

    if not (
        0 < test_fraction < 1
    ):
        raise ValueError(
            "Stored test fraction must "
            "be between 0 and 1."
        )

    if not np.isclose(
        train_fraction
        + test_fraction,
        1.0
    ):
        raise ValueError(
            "Stored train and test fractions "
            "must add up to 1."
        )

    if (
        validation_fraction is None
        or not (
            0
            < validation_fraction
            < 1
        )
    ):
        raise ValueError(
            "Stored validation fraction must "
            "be between 0 and 1."
        )

    try:
        sampled_rows = int(
            metadata[
                "sampled_rows"
            ]
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "sampled_rows in metadata "
            "must be an integer."
        )

    if sampled_rows <= 0:
        raise ValueError(
            "sampled_rows must be positive."
        )

    try:
        int(
            metadata[
                "random_state"
            ]
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "random_state in metadata "
            "must be an integer."
        )

    return True


# test split reconstruction

def split_from_training_metadata(
    dataframe,
    metadata
):
    if not isinstance(
        dataframe,
        pd.DataFrame
    ):
        raise TypeError(
            "Training data must be "
            "a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot reconstruct splits "
            "from an empty dataframe."
        )

    if "timestamp" not in dataframe.columns:
        raise ValueError(
            "Timestamp is required to "
            "reconstruct the training split."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "timestamp"
    ] = pd.to_numeric(
        dataframe[
            "timestamp"
        ],
        errors="coerce"
    )

    dataframe[
        "timestamp"
    ] = dataframe[
        "timestamp"
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    dataframe = dataframe.dropna(
        subset=[
            "timestamp"
        ]
    )

    dataframe = (
        dataframe
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    total_rows = len(
        dataframe
    )

    if total_rows < 20:
        raise ValueError(
            "Not enough rows are available "
            "to reconstruct the training "
            "and test splits."
        )

    train_fraction = float(
        metadata[
            "train_fraction"
        ]
    )

    validation_fraction = float(
        metadata[
            "validation_fraction_within_development"
        ]
    )

    development_end = int(
        total_rows
        * train_fraction
    )

    development_end = max(
        1,
        min(
            development_end,
            total_rows - 1
        )
    )

    development_data = (
        dataframe.iloc[
            :development_end
        ].copy()
    )

    test_data = (
        dataframe.iloc[
            development_end:
        ].copy()
    )

    validation_rows = max(
        1,
        int(
            len(
                development_data
            )
            * validation_fraction
        )
    )

    if (
        validation_rows
        >= len(
            development_data
        )
    ):
        raise ValueError(
            "Stored validation split "
            "cannot be reconstructed."
        )

    train_data = (
        development_data.iloc[
            :-validation_rows
        ].copy()
    )

    validation_data = (
        development_data.iloc[
            -validation_rows:
        ].copy()
    )

    if (
        train_data.empty
        or validation_data.empty
        or test_data.empty
    ):
        raise ValueError(
            "One or more reconstructed "
            "splits are empty."
        )

    return {
        "train":
            train_data,

        "validation":
            validation_data,

        "development":
            development_data,

        "test":
            test_data,
    }


def reconstruct_test_dataset(
    metadata,
    dataset_path=PROCESSED_DATASET_PATH
):
    validate_metadata(
        metadata
    )

    random_state = int(
        metadata[
            "random_state"
        ]
    )

    feature_set = metadata[
        "feature_set"
    ]

    requested_max_rows = (
        metadata.get(
            "requested_max_rows"
        )
    )

    if requested_max_rows is None:
        # Older metadata did not store
        # requested_max_rows.
        #
        # sampled_rows reproduces the
        # earlier behavior.

        sample_limit = int(
            metadata[
                "sampled_rows"
            ]
        )

    else:
        try:
            requested_max_rows = int(
                requested_max_rows
            )

        except (
            TypeError,
            ValueError
        ):
            raise ValueError(
                "requested_max_rows in "
                "metadata is invalid."
            )

        sample_limit = (
            None

            if requested_max_rows <= 0

            else requested_max_rows
        )

    print(
        "Reconstructing original "
        "training sample..."
    )

    raw_sample = (
        load_uniform_training_sample(
            dataset_path=dataset_path,
            max_rows=sample_limit,
            random_state=random_state
        )
    )

    actual_sampled_rows = len(
        raw_sample
    )

    expected_sampled_rows = int(
        metadata[
            "sampled_rows"
        ]
    )

    sample_row_match = (
        actual_sampled_rows
        == expected_sampled_rows
    )

    training_dataframe = (
        create_training_frame(
            raw_sample,
            feature_set=feature_set,
            include_context=True
        )
    )

    splits = (
        split_from_training_metadata(
            training_dataframe,
            metadata
        )
    )

    test_dataframe = (
        splits[
            "test"
        ].copy()
    )

    expected_usable_rows = (
        metadata.get(
            "usable_rows"
        )
    )

    expected_test_rows = (
        metadata.get(
            "test_rows"
        )
    )

    usable_row_match = True
    test_row_match = True

    if expected_usable_rows is not None:
        try:
            usable_row_match = (
                len(
                    training_dataframe
                )
                == int(
                    expected_usable_rows
                )
            )

        except (
            TypeError,
            ValueError
        ):
            usable_row_match = False

    if expected_test_rows is not None:
        try:
            test_row_match = (
                len(
                    test_dataframe
                )
                == int(
                    expected_test_rows
                )
            )

        except (
            TypeError,
            ValueError
        ):
            test_row_match = False

    return {
        "raw_sample":
            raw_sample,

        "training_dataframe":
            training_dataframe,

        "train_dataframe":
            splits[
                "train"
            ],

        "validation_dataframe":
            splits[
                "validation"
            ],

        "development_dataframe":
            splits[
                "development"
            ],

        "test_dataframe":
            test_dataframe,

        "sample_row_match":
            sample_row_match,

        "usable_row_match":
            usable_row_match,

        "test_row_match":
            test_row_match,

        "actual_sampled_rows":
            actual_sampled_rows,

        "expected_sampled_rows":
            expected_sampled_rows,
    }


# test model input

def prepare_test_data(
    test_dataframe,
    metadata
):
    validate_metadata(
        metadata
    )

    feature_set = metadata[
        "feature_set"
    ]

    x_test, y_test = (
        split_features_target(
            test_dataframe,
            feature_set=feature_set
        )
    )

    expected_features = (
        metadata[
            "feature_columns"
        ]
    )

    actual_features = list(
        x_test.columns
    )

    if actual_features != expected_features:
        raise ValueError(
            "Test feature order does not "
            "match the feature order used "
            "during model training."
        )

    y_test = pd.to_numeric(
        y_test,
        errors="coerce"
    )

    y_test = y_test.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    if y_test.isna().any():
        raise ValueError(
            "Final test targets contain "
            "invalid values."
        )

    if not y_test.between(
        0,
        1
    ).all():
        raise ValueError(
            "Final test targets must "
            "remain between 0 and 1."
        )

    return (
        x_test,
        y_test
    )


# scaling and feature preparation

def transform_test_data(
    x_test,
    scaler=None,
    metadata=None
):
    if metadata is None:
        metadata = load_metadata()

    return transform_model_input(
        x_test,
        scaler=scaler,
        metadata=metadata
    )


# prediction

def generate_predictions(
    model,
    x_test
):
    if not hasattr(
        model,
        "predict"
    ):
        raise TypeError(
            "Loaded model does not "
            "provide predict()."
        )

    raw_predictions = np.asarray(
        model.predict(
            x_test
        ),
        dtype=float
    ).reshape(
        -1
    )

    if raw_predictions.size == 0:
        raise ValueError(
            "The trained model produced "
            "no predictions."
        )

    invalid_count = int(
        np.sum(
            ~np.isfinite(
                raw_predictions
            )
        )
    )

    if invalid_count > 0:
        raise ValueError(
            "The trained model produced "
            f"{invalid_count} non-finite "
            "prediction(s)."
        )

    clipped_predictions, clipped_count = (
        normalize_predictions(
            raw_predictions
        )
    )

    return {
        "raw":
            raw_predictions,

        "clipped":
            clipped_predictions,

        "clipped_count":
            clipped_count,

        "clipped_fraction":
            (
                clipped_count
                / len(
                    clipped_predictions
                )
            ),
    }


# residual analysis

def calculate_residual_analysis(
    actual,
    predicted
):
    actual, predicted = (
        validate_numeric_pair_arrays(
            actual,
            predicted
        )
    )

    residuals = (
        predicted
        - actual
    )

    absolute_errors = np.abs(
        residuals
    )

    return {
        "mean_signed_error":
            float(
                np.mean(
                    residuals
                )
            ),

        "median_signed_error":
            float(
                np.median(
                    residuals
                )
            ),

        "mean_absolute_error":
            float(
                np.mean(
                    absolute_errors
                )
            ),

        "median_absolute_error":
            float(
                np.median(
                    absolute_errors
                )
            ),

        "maximum_absolute_error":
            float(
                np.max(
                    absolute_errors
                )
            ),

        "error_25th_percentile":
            float(
                np.percentile(
                    absolute_errors,
                    25
                )
            ),

        "error_50th_percentile":
            float(
                np.percentile(
                    absolute_errors,
                    50
                )
            ),

        "error_75th_percentile":
            float(
                np.percentile(
                    absolute_errors,
                    75
                )
            ),

        "error_90th_percentile":
            float(
                np.percentile(
                    absolute_errors,
                    90
                )
            ),

        "error_95th_percentile":
            float(
                np.percentile(
                    absolute_errors,
                    95
                )
            ),
    }


# prediction distribution

def calculate_prediction_distribution(
    actual,
    predicted
):
    actual, predicted = (
        validate_numeric_pair_arrays(
            actual,
            predicted
        )
    )

    return {
        "actual_mean":
            float(
                np.mean(
                    actual
                )
            ),

        "actual_std":
            float(
                np.std(
                    actual
                )
            ),

        "actual_min":
            float(
                np.min(
                    actual
                )
            ),

        "actual_max":
            float(
                np.max(
                    actual
                )
            ),

        "predicted_mean":
            float(
                np.mean(
                    predicted
                )
            ),

        "predicted_std":
            float(
                np.std(
                    predicted
                )
            ),

        "predicted_min":
            float(
                np.min(
                    predicted
                )
            ),

        "predicted_max":
            float(
                np.max(
                    predicted
                )
            ),

        "actual_prediction_correlation":
            calculate_correlation(
                actual,
                predicted
            ),

        "correlation_note":
            (
                "Correlation describes "
                "association between actual "
                "and predicted recall. It is "
                "not itself a measure of "
                "prediction accuracy."
            ),
    }


# calibration analysis

def calculate_calibration_bins(
    actual,
    predicted,
    bins=10
):
    actual, predicted = (
        validate_numeric_pair_arrays(
            actual,
            predicted
        )
    )

    try:
        bins = int(
            bins
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Calibration bin count must "
            "be an integer."
        )

    if bins < 2:
        raise ValueError(
            "At least two calibration "
            "bins are required."
        )

    if not (
        np.all(
            actual >= 0
        )
        and
        np.all(
            actual <= 1
        )
        and
        np.all(
            predicted >= 0
        )
        and
        np.all(
            predicted <= 1
        )
    ):
        raise ValueError(
            "Calibration values must "
            "remain between 0 and 1."
        )

    dataframe = pd.DataFrame(
        {
            "actual":
                actual,

            "predicted":
                predicted,
        }
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )

    dataframe[
        "prediction_bin"
    ] = pd.cut(
        dataframe[
            "predicted"
        ],
        bins=edges,
        include_lowest=True,
        duplicates="drop"
    )

    grouped = (
        dataframe
        .groupby(
            "prediction_bin",
            observed=True
        )
        .agg(
            observations=(
                "actual",
                "count"
            ),

            average_prediction=(
                "predicted",
                "mean"
            ),

            average_actual=(
                "actual",
                "mean"
            ),
        )
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():

        average_prediction = float(
            row[
                "average_prediction"
            ]
        )

        average_actual = float(
            row[
                "average_actual"
            ]
        )

        observations = int(
            row[
                "observations"
            ]
        )

        calibration_error = abs(
            average_prediction
            - average_actual
        )

        results.append(
            {
                "bin":
                    str(
                        row[
                            "prediction_bin"
                        ]
                    ),

                "observations":
                    observations,

                "average_prediction":
                    average_prediction,

                "average_actual":
                    average_actual,

                "calibration_error":
                    calibration_error,
            }
        )

    return results


def calculate_calibration_summary(
    calibration_bins
):
    if not calibration_bins:
        return {
            "expected_calibration_error":
                None,

            "maximum_calibration_error":
                None,

            "calibration_bins":
                0,
        }

    total = sum(
        row[
            "observations"
        ]

        for row
        in calibration_bins
    )

    if total <= 0:
        return {
            "expected_calibration_error":
                None,

            "maximum_calibration_error":
                None,

            "calibration_bins":
                len(
                    calibration_bins
                ),
        }

    expected_error = sum(
        (
            row[
                "observations"
            ]
            / total
        )
        * row[
            "calibration_error"
        ]

        for row
        in calibration_bins
    )

    maximum_error = max(
        row[
            "calibration_error"
        ]

        for row
        in calibration_bins
    )

    return {
        "expected_calibration_error":
            float(
                expected_error
            ),

        "expected_calibration_error_pp":
            float(
                expected_error
                * 100
            ),

        "maximum_calibration_error":
            float(
                maximum_error
            ),

        "maximum_calibration_error_pp":
            float(
                maximum_error
                * 100
            ),

        "calibration_bins":
            len(
                calibration_bins
            ),
    }


# delay analysis

def calculate_delay_groups(
    test_dataframe,
    predictions
):
    if (
        "delta_days"
        not in test_dataframe.columns
    ):
        return []

    if (
        TARGET_COLUMN
        not in test_dataframe.columns
    ):
        return []

    predictions = np.asarray(
        predictions,
        dtype=float
    ).reshape(
        -1
    )

    if len(
        predictions
    ) != len(
        test_dataframe
    ):
        raise ValueError(
            "Delay analysis prediction "
            "count does not match the "
            "test dataframe."
        )

    dataframe = pd.DataFrame(
        {
            "delta_days":
                pd.to_numeric(
                    test_dataframe[
                        "delta_days"
                    ],
                    errors="coerce"
                ).to_numpy(),

            "actual":
                pd.to_numeric(
                    test_dataframe[
                        TARGET_COLUMN
                    ],
                    errors="coerce"
                ).to_numpy(),

            "predicted":
                predictions,
        }
    )

    dataframe = dataframe.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    dataframe = dataframe.dropna()

    dataframe = dataframe[
        dataframe[
            "delta_days"
        ] >= 0
    ]

    groups = [
        (
            "Up to 1 day",
            0,
            1
        ),
        (
            "1 to 3 days",
            1,
            3
        ),
        (
            "3 to 7 days",
            3,
            7
        ),
        (
            "7 to 30 days",
            7,
            30
        ),
        (
            "More than 30 days",
            30,
            math.inf
        ),
    ]

    results = []

    for (
        label,
        minimum,
        maximum
    ) in groups:

        if math.isinf(
            maximum
        ):
            subset = dataframe[
                dataframe[
                    "delta_days"
                ] > minimum
            ]

        elif minimum == 0:
            subset = dataframe[
                (
                    dataframe[
                        "delta_days"
                    ] >= minimum
                )
                &
                (
                    dataframe[
                        "delta_days"
                    ] <= maximum
                )
            ]

        else:
            subset = dataframe[
                (
                    dataframe[
                        "delta_days"
                    ] > minimum
                )
                &
                (
                    dataframe[
                        "delta_days"
                    ] <= maximum
                )
            ]

        if subset.empty:
            continue

        metrics = (
            calculate_model_metrics(
                actual=subset[
                    "actual"
                ],
                predicted=subset[
                    "predicted"
                ],
                model_name=label
            )
        )

        results.append(
            {
                "delay_group":
                    label,

                "minimum_days_exclusive":
                    (
                        minimum
                        if minimum > 0
                        else None
                    ),

                "maximum_days_inclusive":
                    (
                        None
                        if math.isinf(
                            maximum
                        )
                        else maximum
                    ),

                "observations":
                    len(
                        subset
                    ),

                "mae":
                    metrics[
                        "mae"
                    ],

                "mae_pp":
                    metrics[
                        "mae_pp"
                    ],

                "rmse":
                    metrics[
                        "rmse"
                    ],

                "rmse_pp":
                    metrics[
                        "rmse_pp"
                    ],

                "bias":
                    metrics[
                        "bias"
                    ],

                "bias_pp":
                    metrics[
                        "bias_pp"
                    ],
            }
        )

    return results


# stored metric comparison

def compare_with_training_metadata(
    current_metrics,
    metadata,
    tolerance=1e-9
):
    try:
        tolerance = float(
            tolerance
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Reproducibility tolerance "
            "must be numeric."
        )

    if tolerance < 0:
        raise ValueError(
            "Reproducibility tolerance "
            "cannot be negative."
        )

    stored_metrics = metadata.get(
        "final_test_metrics",
        {}
    )

    if not stored_metrics:
        return {
            "available":
                False,

            "matches":
                False,

            "differences":
                {},

            "message":
                (
                    "No final test metrics "
                    "were stored during training."
                ),
        }

    fields = [
        "mae",
        "rmse",
        "r2",
        "bias",
        "within_5pp",
        "within_10pp",
        "within_15pp",
    ]

    differences = {}

    all_match = True

    for field in fields:

        stored = safe_float(
            stored_metrics.get(
                field
            )
        )

        current = safe_float(
            current_metrics.get(
                field
            )
        )

        if (
            stored is None
            and current is None
        ):
            differences[
                field
            ] = 0.0

            continue

        if (
            stored is None
            or current is None
        ):
            differences[
                field
            ] = None

            all_match = False

            continue

        difference = abs(
            stored
            - current
        )

        differences[
            field
        ] = difference

        if difference > tolerance:
            all_match = False

    return {
        "available":
            True,

        "matches":
            all_match,

        "tolerance":
            tolerance,

        "differences":
            differences,

        "message":
            (
                "Reloaded model metrics match "
                "the metrics recorded during "
                "training."

                if all_match

                else
                (
                    "Reloaded evaluation metrics "
                    "differ from the values "
                    "recorded during training."
                )
            ),
    }


# worst errors

def get_worst_prediction_errors(
    actual,
    predicted,
    limit=20
):
    actual, predicted = (
        validate_numeric_pair_arrays(
            actual,
            predicted
        )
    )

    try:
        limit = int(
            limit
        )

    except (
        TypeError,
        ValueError
    ):
        limit = 20

    if limit <= 0:
        return []

    errors = (
        predicted
        - actual
    )

    absolute_errors = np.abs(
        errors
    )

    order = np.argsort(
        absolute_errors
    )[
        ::-1
    ][
        :limit
    ]

    records = []

    for rank, index in enumerate(
        order,
        start=1
    ):
        signed_error = float(
            errors[
                index
            ]
        )

        if signed_error > 0:
            direction = (
                "Overprediction"
            )

        elif signed_error < 0:
            direction = (
                "Underprediction"
            )

        else:
            direction = (
                "Exact"
            )

        records.append(
            {
                "error_rank":
                    rank,

                "test_row":
                    int(
                        index
                    ),

                "actual":
                    float(
                        actual[
                            index
                        ]
                    ),

                "predicted":
                    float(
                        predicted[
                            index
                        ]
                    ),

                "signed_error":
                    signed_error,

                "absolute_error":
                    float(
                        absolute_errors[
                            index
                        ]
                    ),

                "absolute_error_pp":
                    float(
                        absolute_errors[
                            index
                        ]
                        * 100
                    ),

                "bias_direction":
                    direction,
            }
        )

    return records


# sanity checks

def perform_sanity_checks(
    metadata,
    model,
    scaler,
    x_test,
    y_test,
    prediction_data,
    reconstruction=None
):
    checks = []

    expected_model_name = (
        metadata.get(
            "model_name"
        )
    )

    actual_model_name = (
        model.__class__.__name__
    )

    checks.append(
        {
            "check":
                "Model class",

            "passed":
                (
                    expected_model_name
                    == actual_model_name
                ),

            "details":
                (
                    f"Expected "
                    f"{expected_model_name}, "
                    f"loaded "
                    f"{actual_model_name}."
                ),
        }
    )

    expected_features = (
        metadata.get(
            "feature_columns",
            []
        )
    )

    checks.append(
        {
            "check":
                "Feature order",

            "passed":
                (
                    list(
                        x_test.columns
                    )
                    == expected_features
                ),

            "details":
                (
                    f"{len(expected_features)} "
                    "expected feature(s)."
                ),
        }
    )

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None
    )

    feature_count_passed = True

    if model_feature_count is not None:
        try:
            feature_count_passed = (
                int(
                    model_feature_count
                )
                == len(
                    expected_features
                )
            )

        except (
            TypeError,
            ValueError
        ):
            feature_count_passed = False

    checks.append(
        {
            "check":
                "Model feature count",

            "passed":
                feature_count_passed,

            "details":
                (
                    "Saved model feature count "
                    "should match metadata."
                ),
        }
    )

    expected_scaler = metadata.get(
        "uses_scaler",
        False
    )

    checks.append(
        {
            "check":
                "Scaler availability",

            "passed":
                (
                    (
                        expected_scaler
                        and scaler
                        is not None
                    )
                    or
                    (
                        not expected_scaler
                        and scaler
                        is None
                    )
                ),

            "details":
                (
                    f"uses_scaler="
                    f"{expected_scaler}"
                ),
        }
    )

    checks.append(
        {
            "check":
                "Target range",

            "passed":
                bool(
                    (
                        y_test
                        .between(
                            0,
                            1
                        )
                    ).all()
                ),

            "details":
                (
                    "All final test target "
                    "values must remain "
                    "between 0 and 1."
                ),
        }
    )

    clipped_predictions = (
        prediction_data[
            "clipped"
        ]
    )

    checks.append(
        {
            "check":
                "Prediction range",

            "passed":
                bool(
                    np.all(
                        (
                            clipped_predictions
                            >= 0
                        )
                        &
                        (
                            clipped_predictions
                            <= 1
                        )
                    )
                ),

            "details":
                (
                    "All final predictions "
                    "must remain between "
                    "0 and 1."
                ),
        }
    )

    checks.append(
        {
            "check":
                "Finite predictions",

            "passed":
                bool(
                    np.isfinite(
                        clipped_predictions
                    ).all()
                ),

            "details":
                (
                    "Predictions must not "
                    "contain NaN or infinity."
                ),
        }
    )

    checks.append(
        {
            "check":
                "Non-empty test set",

            "passed":
                (
                    len(
                        y_test
                    )
                    > 0
                ),

            "details":
                (
                    f"{len(y_test):,} "
                    "test observations."
                ),
        }
    )

    expected_test_rows = (
        metadata.get(
            "test_rows"
        )
    )

    test_rows_passed = True

    if expected_test_rows is not None:
        try:
            test_rows_passed = (
                len(
                    y_test
                )
                == int(
                    expected_test_rows
                )
            )

        except (
            TypeError,
            ValueError
        ):
            test_rows_passed = False

    checks.append(
        {
            "check":
                "Reconstructed test size",

            "passed":
                test_rows_passed,

            "details":
                (
                    f"Expected "
                    f"{expected_test_rows}, "
                    f"reconstructed "
                    f"{len(y_test)}."
                ),
        }
    )

    if reconstruction is not None:

        checks.append(
            {
                "check":
                    "Sample reconstruction",

                "passed":
                    reconstruction.get(
                        "sample_row_match",
                        False
                    ),

                "details":
                    (
                        "Reconstructed sample "
                        "row count should match "
                        "training metadata."
                    ),
            }
        )

        checks.append(
            {
                "check":
                    "Usable-row reconstruction",

                "passed":
                    reconstruction.get(
                        "usable_row_match",
                        False
                    ),

                "details":
                    (
                        "Usable engineered rows "
                        "should match training "
                        "metadata."
                    ),
            }
        )

    return checks


# JSON conversion

def make_json_safe(
    value
):
    if isinstance(
        value,
        dict
    ):
        return {
            str(
                key
            ):
                make_json_safe(
                    item
                )

            for (
                key,
                item
            ) in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        )
    ):
        return [
            make_json_safe(
                item
            )

            for item
            in value
        ]

    if isinstance(
        value,
        np.ndarray
    ):
        return [
            make_json_safe(
                item
            )

            for item
            in value.tolist()
        ]

    if isinstance(
        value,
        np.integer
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating
    ):
        value = float(
            value
        )

        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        np.bool_
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        float
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        Path
    ):
        return str(
            value
        )

    if isinstance(
        value,
        datetime
    ):
        return value.isoformat()

    return value


# report saving

def save_evaluation_report(
    report,
    output_path=EVALUATION_REPORT_PATH
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = (
        output_path.parent
        / (
            output_path.name
            + ".tmp"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as report_file:

            json.dump(
                make_json_safe(
                    report
                ),
                report_file,
                indent=4,
                ensure_ascii=False
            )

        temporary_path.replace(
            output_path
        )

    except Exception:
        if temporary_path.exists():
            try:
                temporary_path.unlink()

            except OSError:
                pass

        raise

    return output_path


# evaluation pipeline

def evaluate_saved_model(
    dataset_path=PROCESSED_DATASET_PATH,
    save_report=True
):
    print(
        "MemoryMap Model Evaluation"
    )

    print(
        "=" * 40
    )

    validate_model_artifacts()

    metadata = load_metadata()

    validate_metadata(
        metadata
    )

    bundle = load_model_bundle()

    model = bundle[
        "model"
    ]

    scaler = bundle[
        "scaler"
    ]

    print(
        "Model:",
        metadata.get(
            "model_display_name",
            metadata.get(
                "model_name",
                "Unknown"
            )
        )
    )

    print(
        "Feature Set:",
        metadata[
            "feature_set"
        ]
    )

    print(
        "Features:",
        ", ".join(
            metadata[
                "feature_columns"
            ]
        )
    )

    print()

    reconstructed = (
        reconstruct_test_dataset(
            metadata,
            dataset_path
        )
    )

    test_dataframe = (
        reconstructed[
            "test_dataframe"
        ]
    )

    print()

    print(
        "Reconstructed final test rows:",
        f"{len(test_dataframe):,}"
    )

    x_test, y_test = (
        prepare_test_data(
            test_dataframe,
            metadata
        )
    )

    x_test_model = (
        transform_test_data(
            x_test,
            scaler=scaler,
            metadata=metadata
        )
    )

    prediction_data = (
        generate_predictions(
            model,
            x_test_model
        )
    )

    predictions = (
        prediction_data[
            "clipped"
        ]
    )

    if len(
        predictions
    ) != len(
        y_test
    ):
        raise RuntimeError(
            "Prediction count does not "
            "match the final test set."
        )

    metrics = (
        calculate_model_metrics(
            actual=y_test,
            predicted=predictions,
            model_name=metadata.get(
                "model_display_name",
                metadata.get(
                    "model_name",
                    "Retention Model"
                )
            )
        )
    )

    residual_analysis = (
        calculate_residual_analysis(
            y_test,
            predictions
        )
    )

    prediction_distribution = (
        calculate_prediction_distribution(
            y_test,
            predictions
        )
    )

    calibration = (
        calculate_calibration_bins(
            y_test,
            predictions
        )
    )

    calibration_summary = (
        calculate_calibration_summary(
            calibration
        )
    )

    delay_groups = (
        calculate_delay_groups(
            test_dataframe,
            predictions
        )
    )

    reproducibility = (
        compare_with_training_metadata(
            metrics,
            metadata
        )
    )

    sanity_checks = (
        perform_sanity_checks(
            metadata=metadata,
            model=model,
            scaler=scaler,
            x_test=x_test,
            y_test=y_test,
            prediction_data=prediction_data,
            reconstruction=reconstructed
        )
    )

    worst_errors = (
        get_worst_prediction_errors(
            actual=y_test,
            predicted=predictions,
            limit=20
        )
    )

    passed_checks = sum(
        1

        for check
        in sanity_checks

        if check[
            "passed"
        ]
    )

    all_sanity_checks_passed = (
        passed_checks
        == len(
            sanity_checks
        )
    )

    reconstruction_matches = (
        reconstructed[
            "sample_row_match"
        ]
        and reconstructed[
            "usable_row_match"
        ]
        and reconstructed[
            "test_row_match"
        ]
    )

    evaluation_reproduced = (
        reproducibility.get(
            "matches",
            False
        )
        and reconstruction_matches
        and all_sanity_checks_passed
    )

    report = {
        "evaluation_schema_version":
            1,

        "evaluated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "evaluation_role":
            (
                "Reproduction and diagnostic "
                "evaluation of the saved final "
                "test split."
            ),

        "independent_new_test_set":
            False,

        "model_name":
            metadata.get(
                "model_name"
            ),

        "model_display_name":
            metadata.get(
                "model_display_name"
            ),

        "candidate_key":
            metadata.get(
                "candidate_key"
            ),

        "feature_set":
            metadata[
                "feature_set"
            ],

        "feature_columns":
            metadata[
                "feature_columns"
            ],

        "target_column":
            metadata.get(
                "target_column"
            ),

        "target_scale":
            metadata.get(
                "target_scale"
            ),

        "training_source":
            metadata.get(
                "training_source"
            ),

        "model_role":
            metadata.get(
                "model_role"
            ),

        "memorymap_personal_model":
            metadata.get(
                "memorymap_personal_model",
                False
            ),

        "test_rows":
            len(
                y_test
            ),

        "reconstruction":
            {
                "sample_row_match":
                    reconstructed[
                        "sample_row_match"
                    ],

                "usable_row_match":
                    reconstructed[
                        "usable_row_match"
                    ],

                "test_row_match":
                    reconstructed[
                        "test_row_match"
                    ],

                "expected_sampled_rows":
                    reconstructed[
                        "expected_sampled_rows"
                    ],

                "actual_sampled_rows":
                    reconstructed[
                        "actual_sampled_rows"
                    ],

                "matches_training_structure":
                    reconstruction_matches,
            },

        "metrics":
            metrics,

        "prediction_clipping":
            {
                "clipped_count":
                    prediction_data[
                        "clipped_count"
                    ],

                "clipped_fraction":
                    prediction_data[
                        "clipped_fraction"
                    ],
            },

        "residual_analysis":
            residual_analysis,

        "prediction_distribution":
            prediction_distribution,

        "calibration_bins":
            calibration,

        "calibration_summary":
            calibration_summary,

        "delay_group_metrics":
            delay_groups,

        "reproducibility":
            reproducibility,

        "sanity_checks":
            sanity_checks,

        "sanity_checks_passed":
            passed_checks,

        "sanity_checks_total":
            len(
                sanity_checks
            ),

        "all_sanity_checks_passed":
            all_sanity_checks_passed,

        "evaluation_reproduced":
            evaluation_reproduced,

        "worst_prediction_errors":
            worst_errors,

        "methodological_note":
            (
                "This evaluation reconstructs "
                "the saved model's original "
                "sampled final test split. It "
                "does not constitute a second "
                "independent external test set."
            ),

        "research_limitations":
            metadata.get(
                "methodological_limitations",
                []
            ),
    }

    print()

    print(
        "Final Test Metrics"
    )

    print(
        "MAE:",
        f"{metrics['mae_pp']:.3f} pp"
    )

    print(
        "RMSE:",
        f"{metrics['rmse_pp']:.3f} pp"
    )

    if (
        metrics[
            "r2"
        ]
        is not None
    ):
        print(
            "R²:",
            f"{metrics['r2']:.4f}"
        )

    else:
        print(
            "R²: N/A"
        )

    print(
        "Bias:",
        f"{metrics['bias_pp']:.3f} pp",
        f"({metrics['bias_direction']})"
    )

    print()

    print(
        "Prediction Accuracy"
    )

    print(
        "Within 5pp:",
        (
            f"{metrics['within_5pp'] * 100:.2f}%"
        )
    )

    print(
        "Within 10pp:",
        (
            f"{metrics['within_10pp'] * 100:.2f}%"
        )
    )

    print(
        "Within 15pp:",
        (
            f"{metrics['within_15pp'] * 100:.2f}%"
        )
    )

    print()

    print(
        "Clipped Predictions:",
        prediction_data[
            "clipped_count"
        ]
    )

    correlation = (
        prediction_distribution[
            "actual_prediction_correlation"
        ]
    )

    print(
        "Prediction / Actual Correlation:",
        (
            f"{correlation:.4f}"

            if correlation
            is not None

            else "N/A"
        )
    )

    calibration_error = (
        calibration_summary.get(
            "expected_calibration_error_pp"
        )
    )

    print(
        "Expected Calibration Error:",
        (
            f"{calibration_error:.3f} pp"

            if calibration_error
            is not None

            else "N/A"
        )
    )

    print()

    print(
        "Reconstruction:"
    )

    print(
        "Sample rows:",
        (
            "MATCH"

            if reconstructed[
                "sample_row_match"
            ]

            else "MISMATCH"
        )
    )

    print(
        "Usable rows:",
        (
            "MATCH"

            if reconstructed[
                "usable_row_match"
            ]

            else "MISMATCH"
        )
    )

    print(
        "Test rows:",
        (
            "MATCH"

            if reconstructed[
                "test_row_match"
            ]

            else "MISMATCH"
        )
    )

    print()

    print(
        "Metric Reproducibility:"
    )

    print(
        reproducibility[
            "message"
        ]
    )

    print()

    print(
        "Sanity Checks:",
        f"{passed_checks}/"
        f"{len(sanity_checks)} passed"
    )

    for check in sanity_checks:

        status = (
            "PASS"

            if check[
                "passed"
            ]

            else "FAIL"
        )

        print(
            f"{status}: "
            f"{check['check']}"
        )

    print()

    print(
        "Evaluation Reproduced:",
        (
            "YES"
            if evaluation_reproduced
            else "NO"
        )
    )

    if save_report:
        report_path = (
            save_evaluation_report(
                report
            )
        )

        print()

        print(
            "Evaluation report saved:"
        )

        print(
            report_path
        )

    return report


# command line

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained "
            "MemoryMap retention model."
        )
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Evaluate without writing "
            "evaluation_report.json."
        )
    )

    return parser.parse_args()


# direct execution

if __name__ == "__main__":

    arguments = parse_arguments()

    evaluate_saved_model(
        save_report=(
            not arguments.no_save
        )
    )