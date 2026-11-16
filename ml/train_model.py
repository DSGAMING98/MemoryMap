import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from config import (
    PROCESSED_DATASET_PATH,
    MODELS_DIR,
    RETENTION_MODEL_PATH,
    SCALER_PATH,
    MODEL_METADATA_PATH,
    MODEL_CANDIDATES,
    MAX_TRAINING_ROWS,
    PREPROCESS_CHUNK_SIZE,
    RANDOM_STATE,
    TRAIN_FRACTION,
    TEST_FRACTION,
)

from core.metrics import (
    calculate_model_metrics,
    rank_models,
)

from ml.feature_engineering import (
    create_training_frame,
    split_features_target,
    get_feature_columns,
    create_feature_metadata,
    validate_feature_set,
)


VALIDATION_FRACTION = 0.10

DEFAULT_FEATURE_SET = "base"

PUBLIC_FEATURE_SETS = {
    "base",
    "extended",
}


# dataset validation

def validate_processed_dataset(
    dataset_path=PROCESSED_DATASET_PATH
):
    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Processed dataset was not found. "
            "Run ml/preprocess.py first."
        )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Processed dataset path is invalid."
        )

    if dataset_path.stat().st_size <= 0:
        raise ValueError(
            "Processed dataset is empty."
        )

    return dataset_path


def validate_public_feature_set(
    feature_set
):
    feature_set = validate_feature_set(
        feature_set
    )

    if feature_set not in PUBLIC_FEATURE_SETS:
        raise ValueError(
            "The public Duolingo training "
            "pipeline supports only the "
            "'base' and 'extended' feature "
            "sets. MemoryMap personal features "
            "must be trained later using real "
            "MemoryMap longitudinal data."
        )

    return feature_set


def validate_split_configuration():
    train_fraction = float(
        TRAIN_FRACTION
    )

    test_fraction = float(
        TEST_FRACTION
    )

    if not (
        0 < train_fraction < 1
    ):
        raise ValueError(
            "TRAIN_FRACTION must be "
            "between 0 and 1."
        )

    if not (
        0 < test_fraction < 1
    ):
        raise ValueError(
            "TEST_FRACTION must be "
            "between 0 and 1."
        )

    if not np.isclose(
        train_fraction
        + test_fraction,
        1.0
    ):
        raise ValueError(
            "TRAIN_FRACTION and "
            "TEST_FRACTION must add "
            "up to 1."
        )

    if not (
        0 < VALIDATION_FRACTION < 1
    ):
        raise ValueError(
            "VALIDATION_FRACTION must "
            "be between 0 and 1."
        )

    return True


# uniform streaming sample

def load_uniform_training_sample(
    dataset_path=PROCESSED_DATASET_PATH,
    max_rows=MAX_TRAINING_ROWS,
    chunk_size=PREPROCESS_CHUNK_SIZE,
    random_state=RANDOM_STATE
):
    dataset_path = (
        validate_processed_dataset(
            dataset_path
        )
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
            max_rows = None

    try:
        chunk_size = int(
            chunk_size
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Chunk size must be "
            "an integer."
        )

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be positive."
        )

    try:
        random_state = int(
            random_state
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Random state must be "
            "an integer."
        )

    if max_rows is None:
        print(
            "Loading full processed dataset..."
        )

        dataframe = pd.read_csv(
            dataset_path,
            low_memory=False
        )

        if dataframe.empty:
            raise RuntimeError(
                "Processed dataset contains "
                "no rows."
            )

        return dataframe

    rng = np.random.default_rng(
        random_state
    )

    reservoir = None

    total_rows_seen = 0

    print(
        "Creating uniform training sample..."
    )

    print(
        f"Maximum rows: {max_rows:,}"
    )

    print(
        f"Chunk size: {chunk_size:,}"
    )

    print()

    reader = pd.read_csv(
        dataset_path,
        chunksize=chunk_size,
        low_memory=False
    )

    for (
        chunk_number,
        chunk
    ) in enumerate(
        reader,
        start=1
    ):
        if chunk.empty:
            continue

        chunk = chunk.copy()

        total_rows_seen += len(
            chunk
        )

        # Every row receives an independent
        # random key. Keeping the smallest
        # keys produces a uniform sample
        # without loading the entire file.

        chunk[
            "_sample_key"
        ] = rng.random(
            len(
                chunk
            )
        )

        if reservoir is None:
            reservoir = chunk

        else:
            reservoir = pd.concat(
                [
                    reservoir,
                    chunk,
                ],
                ignore_index=True
            )

        if len(
            reservoir
        ) > max_rows:

            reservoir = (
                reservoir
                .nsmallest(
                    max_rows,
                    "_sample_key"
                )
                .copy()
            )

        print(
            f"Chunk {chunk_number}: "
            f"{total_rows_seen:,} rows scanned, "
            f"{len(reservoir):,} retained"
        )

    if (
        reservoir is None
        or reservoir.empty
    ):
        raise RuntimeError(
            "No rows were loaded from "
            "the processed dataset."
        )

    reservoir = reservoir.drop(
        columns=[
            "_sample_key"
        ]
    )

    reservoir = reservoir.reset_index(
        drop=True
    )

    print()

    print(
        f"Rows scanned: "
        f"{total_rows_seen:,}"
    )

    print(
        f"Final sample: "
        f"{len(reservoir):,} rows"
    )

    return reservoir


# chronological splitting

def chronological_split(
    dataframe,
    validation_fraction=VALIDATION_FRACTION
):
    validate_split_configuration()

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
            "Training dataframe is empty."
        )

    if "timestamp" not in dataframe.columns:
        raise ValueError(
            "Timestamp is required for "
            "chronological splitting."
        )

    try:
        validation_fraction = float(
            validation_fraction
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Validation fraction must "
            "be numeric."
        )

    if not (
        0 < validation_fraction < 1
    ):
        raise ValueError(
            "Validation fraction must "
            "be between 0 and 1."
        )

    dataframe = dataframe.copy()

    dataframe[
        "timestamp"
    ] = pd.to_numeric(
        dataframe[
            "timestamp"
        ],
        errors="coerce"
    )

    dataframe = dataframe.replace(
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
            "for training and evaluation."
        )

    development_end = int(
        total_rows
        * float(
            TRAIN_FRACTION
        )
    )

    development_end = max(
        1,
        min(
            development_end,
            total_rows - 1
        )
    )

    development_data = (
        dataframe
        .iloc[
            :development_end
        ]
        .copy()
    )

    test_data = (
        dataframe
        .iloc[
            development_end:
        ]
        .copy()
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
            "Validation split is too large."
        )

    train_data = (
        development_data
        .iloc[
            :-validation_rows
        ]
        .copy()
    )

    validation_data = (
        development_data
        .iloc[
            -validation_rows:
        ]
        .copy()
    )

    if (
        train_data.empty
        or validation_data.empty
        or test_data.empty
    ):
        raise ValueError(
            "One or more dataset splits "
            "are empty."
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


# candidate models

def create_candidate_model(
    model_name
):
    if model_name == (
        "linear_regression"
    ):
        return LinearRegression()

    if model_name == (
        "random_forest"
    ):
        return (
            RandomForestRegressor(
                n_estimators=40,
                max_depth=12,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )

    if model_name == (
        "gradient_boosting"
    ):
        return (
            HistGradientBoostingRegressor(
                learning_rate=0.08,
                max_iter=200,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=RANDOM_STATE
            )
        )

    raise ValueError(
        f"Unsupported model: "
        f"{model_name}"
    )


def model_requires_scaler(
    model_name
):
    return (
        model_name
        == "linear_regression"
    )


def get_model_display_name(
    model_name
):
    names = {
        "linear_regression":
            "Linear Regression",

        "random_forest":
            "Random Forest",

        "gradient_boosting":
            (
                "Histogram Gradient "
                "Boosting"
            ),
    }

    return names.get(
        model_name,
        model_name
    )


def get_model_hyperparameters(
    model_name
):
    model = create_candidate_model(
        model_name
    )

    parameters = model.get_params(
        deep=False
    )

    selected_parameters = {
        "linear_regression": [
            "fit_intercept",
        ],
        "random_forest": [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "random_state",
            "n_jobs",
        ],
        "gradient_boosting": [
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "l2_regularization",
            "random_state",
        ],
    }

    return {
        parameter:
            parameters.get(
                parameter
            )
        for parameter in selected_parameters[
            model_name
        ]
    }


# prediction normalization

def normalize_predictions(
    predictions
):
    predictions = np.asarray(
        predictions,
        dtype=float
    ).reshape(
        -1
    )

    if predictions.size == 0:
        raise ValueError(
            "Model returned no predictions."
        )

    if not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            "Model produced non-finite "
            "prediction values."
        )

    unclipped = (
        predictions.copy()
    )

    predictions = np.clip(
        predictions,
        0.0,
        1.0
    )

    clipped_count = int(
        np.sum(
            ~np.isclose(
                predictions,
                unclipped,
                rtol=1e-9,
                atol=1e-9
            )
        )
    )

    return (
        predictions,
        clipped_count
    )


# dataframe scaling

def scale_feature_dataframe(
    scaler,
    dataframe,
    fit=False
):
    if fit:
        transformed = (
            scaler.fit_transform(
                dataframe
            )
        )

    else:
        transformed = (
            scaler.transform(
                dataframe
            )
        )

    transformed = np.asarray(
        transformed
    )

    if transformed.ndim != 2:
        raise ValueError(
            "Scaler returned an invalid "
            "feature matrix."
        )

    if transformed.shape[
        1
    ] != len(
        dataframe.columns
    ):
        raise ValueError(
            "Scaler output feature count "
            "does not match the input."
        )

    if not np.isfinite(
        transformed
    ).all():
        raise ValueError(
            "Scaler produced non-finite "
            "values."
        )

    return pd.DataFrame(
        transformed,
        columns=dataframe.columns,
        index=dataframe.index
    )


# candidate training

def train_candidate(
    model_name,
    x_train,
    y_train,
    x_validation,
    y_validation
):
    model = create_candidate_model(
        model_name
    )

    scaler = None

    if model_requires_scaler(
        model_name
    ):
        scaler = StandardScaler()

        x_train_model = (
            scale_feature_dataframe(
                scaler,
                x_train,
                fit=True
            )
        )

        x_validation_model = (
            scale_feature_dataframe(
                scaler,
                x_validation,
                fit=False
            )
        )

    else:
        x_train_model = (
            x_train.copy()
        )

        x_validation_model = (
            x_validation.copy()
        )

    model.fit(
        x_train_model,
        y_train
    )

    raw_predictions = model.predict(
        x_validation_model
    )

    predictions, clipped_count = (
        normalize_predictions(
            raw_predictions
        )
    )

    if len(
        predictions
    ) != len(
        y_validation
    ):
        raise RuntimeError(
            "Candidate model returned an "
            "unexpected prediction count."
        )

    metrics = (
        calculate_model_metrics(
            actual=y_validation,
            predicted=predictions,
            model_name=(
                get_model_display_name(
                    model_name
                )
            )
        )
    )

    metrics[
        "candidate_key"
    ] = model_name

    metrics[
        "clipped_predictions"
    ] = clipped_count

    metrics[
        "uses_scaler"
    ] = (
        scaler is not None
    )

    return {
        "model":
            model,

        "scaler":
            scaler,

        "metrics":
            metrics,
    }


# candidate evaluation

def evaluate_candidates(
    x_train,
    y_train,
    x_validation,
    y_validation
):
    results = []

    for model_name in MODEL_CANDIDATES:
        print()

        print(
            f"Training "
            f"{get_model_display_name(model_name)}..."
        )

        try:
            result = train_candidate(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation
            )

        except Exception as error:
            print(
                "Candidate failed:"
            )

            print(
                error
            )

            continue

        metrics = result[
            "metrics"
        ]

        results.append(
            metrics
        )

        print(
            f"Validation MAE: "
            f"{metrics['mae_pp']:.3f} pp"
        )

        print(
            f"Validation RMSE: "
            f"{metrics['rmse_pp']:.3f} pp"
        )

        if (
            metrics[
                "r2"
            ]
            is not None
        ):
            print(
                f"Validation R²: "
                f"{metrics['r2']:.4f}"
            )

        print(
            "Clipped predictions:",
            metrics[
                "clipped_predictions"
            ]
        )

    ranked = rank_models(
        results,
        metric="mae"
    )

    if not ranked:
        raise RuntimeError(
            "No candidate model could "
            "be evaluated."
        )

    return (
        results,
        ranked
    )


# final model fitting

def fit_final_model(
    winner_name,
    development_dataframe,
    feature_set
):
    feature_set = (
        validate_public_feature_set(
            feature_set
        )
    )

    x_development, y_development = (
        split_features_target(
            development_dataframe,
            feature_set
        )
    )

    model = create_candidate_model(
        winner_name
    )

    scaler = None

    if model_requires_scaler(
        winner_name
    ):
        scaler = StandardScaler()

        x_model = (
            scale_feature_dataframe(
                scaler,
                x_development,
                fit=True
            )
        )

    else:
        x_model = (
            x_development.copy()
        )

    model.fit(
        x_model,
        y_development
    )

    return (
        model,
        scaler
    )


# final test evaluation

def evaluate_final_model(
    model,
    scaler,
    test_dataframe,
    feature_set,
    winner_name
):
    feature_set = (
        validate_public_feature_set(
            feature_set
        )
    )

    x_test, y_test = (
        split_features_target(
            test_dataframe,
            feature_set
        )
    )

    if scaler is not None:
        x_test_model = (
            scale_feature_dataframe(
                scaler,
                x_test,
                fit=False
            )
        )

    else:
        x_test_model = (
            x_test.copy()
        )

    raw_predictions = model.predict(
        x_test_model
    )

    predictions, clipped_count = (
        normalize_predictions(
            raw_predictions
        )
    )

    if len(
        predictions
    ) != len(
        y_test
    ):
        raise RuntimeError(
            "Final model returned an "
            "unexpected prediction count."
        )

    metrics = (
        calculate_model_metrics(
            actual=y_test,
            predicted=predictions,
            model_name=(
                get_model_display_name(
                    winner_name
                )
            )
        )
    )

    metrics[
        "candidate_key"
    ] = winner_name

    metrics[
        "clipped_predictions"
    ] = clipped_count

    metrics[
        "uses_scaler"
    ] = (
        scaler is not None
    )

    return metrics


# feature interpretation

def extract_feature_information(
    model,
    feature_columns
):
    if hasattr(
        model,
        "feature_importances_"
    ):
        values = np.asarray(
            model.feature_importances_,
            dtype=float
        ).reshape(
            -1
        )

        if len(
            values
        ) != len(
            feature_columns
        ):
            return {}

        return {
            feature:
                float(
                    value
                )

            for (
                feature,
                value
            ) in zip(
                feature_columns,
                values
            )

            if np.isfinite(
                value
            )
        }

    if hasattr(
        model,
        "coef_"
    ):
        coefficients = np.asarray(
            model.coef_,
            dtype=float
        ).reshape(
            -1
        )

        if len(
            coefficients
        ) != len(
            feature_columns
        ):
            return {}

        return {
            feature:
                float(
                    coefficient
                )

            for (
                feature,
                coefficient
            ) in zip(
                feature_columns,
                coefficients
            )

            if np.isfinite(
                coefficient
            )
        }

    return {}


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


# artifact saving

def save_training_artifacts(
    model,
    scaler,
    metadata
):
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not isinstance(
        metadata,
        dict
    ):
        raise ValueError(
            "Training metadata must be "
            "a dictionary."
        )

    temporary_model = (
        MODELS_DIR
        / (
            RETENTION_MODEL_PATH.name
            + ".tmp"
        )
    )

    temporary_scaler = (
        MODELS_DIR
        / (
            SCALER_PATH.name
            + ".tmp"
        )
    )

    temporary_metadata = (
        MODELS_DIR
        / (
            MODEL_METADATA_PATH.name
            + ".tmp"
        )
    )

    temporary_paths = [
        temporary_model,
        temporary_scaler,
        temporary_metadata,
    ]

    for path in temporary_paths:
        if path.exists():
            path.unlink()

    try:
        joblib.dump(
            model,
            temporary_model
        )

        if scaler is not None:
            joblib.dump(
                scaler,
                temporary_scaler
            )

        with open(
            temporary_metadata,
            "w",
            encoding="utf-8"
        ) as metadata_file:

            json.dump(
                make_json_safe(
                    metadata
                ),
                metadata_file,
                indent=4,
                ensure_ascii=False
            )

        temporary_model.replace(
            RETENTION_MODEL_PATH
        )

        if scaler is not None:
            temporary_scaler.replace(
                SCALER_PATH
            )

        elif SCALER_PATH.exists():
            SCALER_PATH.unlink()

        temporary_metadata.replace(
            MODEL_METADATA_PATH
        )

    except Exception:
        for path in temporary_paths:
            if path.exists():
                try:
                    path.unlink()

                except OSError:
                    pass

        raise


# training pipeline

def train_retention_model(
    dataset_path=PROCESSED_DATASET_PATH,
    feature_set=DEFAULT_FEATURE_SET,
    max_rows=MAX_TRAINING_ROWS
):
    feature_set = (
        validate_public_feature_set(
            feature_set
        )
    )

    validate_split_configuration()

    dataset_path = (
        validate_processed_dataset(
            dataset_path
        )
    )

    print(
        "MemoryMap Model Training"
    )

    print(
        "=" * 40
    )

    print(
        "Feature set:",
        feature_set
    )

    print(
        "Candidate models:",
        ", ".join(
            MODEL_CANDIDATES
        )
    )

    print()

    raw_sample = (
        load_uniform_training_sample(
            dataset_path=dataset_path,
            max_rows=max_rows
        )
    )

    sampled_rows = len(
        raw_sample
    )

    print()

    print(
        "Engineering features..."
    )

    training_dataframe = (
        create_training_frame(
            raw_sample,
            feature_set=feature_set,
            include_context=True
        )
    )

    feature_columns = (
        get_feature_columns(
            feature_set
        )
    )

    if training_dataframe.empty:
        raise RuntimeError(
            "No usable training records "
            "remain after feature validation."
        )

    print(
        f"Usable records: "
        f"{len(training_dataframe):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        "Feature columns:"
    )

    for feature in feature_columns:
        print(
            "-",
            feature
        )

    print()

    print(
        "Creating chronological "
        "train, validation and test splits..."
    )

    splits = chronological_split(
        training_dataframe
    )

    train_dataframe = (
        splits[
            "train"
        ]
    )

    validation_dataframe = (
        splits[
            "validation"
        ]
    )

    development_dataframe = (
        splits[
            "development"
        ]
    )

    test_dataframe = (
        splits[
            "test"
        ]
    )

    print(
        f"Train rows: "
        f"{len(train_dataframe):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_dataframe):,}"
    )

    print(
        f"Development rows: "
        f"{len(development_dataframe):,}"
    )

    print(
        f"Final test rows: "
        f"{len(test_dataframe):,}"
    )

    x_train, y_train = (
        split_features_target(
            train_dataframe,
            feature_set
        )
    )

    x_validation, y_validation = (
        split_features_target(
            validation_dataframe,
            feature_set
        )
    )

    validation_results, ranking = (
        evaluate_candidates(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation
        )
    )

    winner = ranking[
        0
    ]

    winner_name = winner[
        "candidate_key"
    ]

    winner_mae = winner[
        "mae"
    ]

    tied_candidates = [
        result[
            "candidate_key"
        ]

        for result in ranking

        if math.isclose(
            float(
                result[
                    "mae"
                ]
            ),
            float(
                winner_mae
            ),
            rel_tol=1e-9,
            abs_tol=1e-9
        )
    ]

    print()

    print(
        "Candidate Ranking"
    )

    for result in ranking:
        print(
            f"{result['rank']}. "
            f"{result['model']} "
            f"- MAE "
            f"{result['mae_pp']:.3f} pp"
        )

    print()

    print(
        "Selected Model:",
        get_model_display_name(
            winner_name
        )
    )

    if len(
        tied_candidates
    ) > 1:
        print(
            "Validation tie detected:",
            ", ".join(
                tied_candidates
            )
        )

        print(
            "The first tied model in the "
            "configured candidate order "
            "was selected."
        )

    print()

    print(
        "Refitting selected model on "
        "the complete development set..."
    )

    final_model, final_scaler = (
        fit_final_model(
            winner_name=winner_name,
            development_dataframe=(
                development_dataframe
            ),
            feature_set=feature_set
        )
    )

    print(
        "Evaluating untouched final "
        "test set..."
    )

    test_metrics = (
        evaluate_final_model(
            model=final_model,
            scaler=final_scaler,
            test_dataframe=(
                test_dataframe
            ),
            feature_set=feature_set,
            winner_name=winner_name
        )
    )

    print()

    print(
        f"Final Test MAE: "
        f"{test_metrics['mae_pp']:.3f} pp"
    )

    print(
        f"Final Test RMSE: "
        f"{test_metrics['rmse_pp']:.3f} pp"
    )

    if (
        test_metrics[
            "r2"
        ]
        is not None
    ):
        print(
            f"Final Test R²: "
            f"{test_metrics['r2']:.4f}"
        )

    print(
        "Clipped Test Predictions:",
        test_metrics[
            "clipped_predictions"
        ]
    )

    feature_information = (
        extract_feature_information(
            final_model,
            feature_columns
        )
    )

    feature_metadata = (
        create_feature_metadata(
            feature_set
        )
    )

    metadata = {
        "artifact_schema_version":
            1,

        "model_name":
            final_model.__class__.__name__,

        "model_display_name":
            get_model_display_name(
                winner_name
            ),

        "candidate_key":
            winner_name,

        "model_hyperparameters":
            get_model_hyperparameters(
                winner_name
            ),

        "feature_set":
            feature_set,

        "feature_columns":
            list(
                feature_columns
            ),

        "target_column":
            feature_metadata[
                "target_column"
            ],

        "target_scale":
            "0-1",

        "uses_scaler":
            (
                final_scaler
                is not None
            ),

        "feature_scales":
            {},

        "training_source":
            (
                "Duolingo Spaced Repetition "
                "public learning traces dataset"
            ),

        "model_role":
            (
                "Public bootstrap retention "
                "prediction model"
            ),

        "memorymap_personal_model":
            False,

        "memorymap_personal_features_trained":
            False,

        "sampling_strategy":
            (
                "Uniform random sample from the "
                "processed Duolingo dataset using "
                "streaming random keys."
            ),

        "split_strategy":
            (
                "The uniformly sampled rows were "
                "sorted by timestamp. The earliest "
                "TRAIN_FRACTION portion became the "
                "development set and the latest "
                "TEST_FRACTION portion became the "
                "untouched final test set. The final "
                "10% of the development set was used "
                "for candidate validation."
            ),

        "split_scope":
            (
                "Chronological within the sampled "
                "dataset, not the exact first 90% "
                "and final 10% of all original "
                "Duolingo traces."
            ),

        "train_fraction":
            TRAIN_FRACTION,

        "test_fraction":
            TEST_FRACTION,

        "validation_fraction_within_development":
            VALIDATION_FRACTION,

        "random_state":
            RANDOM_STATE,

        "requested_max_rows":
            max_rows,

        "sampled_rows":
            sampled_rows,

        "usable_rows":
            len(
                training_dataframe
            ),

        "train_rows":
            len(
                train_dataframe
            ),

        "validation_rows":
            len(
                validation_dataframe
            ),

        "development_rows":
            len(
                development_dataframe
            ),

        "test_rows":
            len(
                test_dataframe
            ),

        "candidate_validation_metrics":
            validation_results,

        "validation_ranking":
            ranking,

        "validation_tied_candidates":
            tied_candidates,

        "final_test_metrics":
            test_metrics,

        "feature_information":
            feature_information,

        "feature_descriptions":
            feature_metadata[
                "feature_descriptions"
            ],

        "blocked_leakage_columns":
            feature_metadata[
                "blocked_leakage_columns"
            ],

        "context_columns":
            feature_metadata.get(
                "context_columns",
                []
            ),

        "missing_value_policy":
            feature_metadata.get(
                "missing_value_policy"
            ),

        "methodological_limitations":
            [
                (
                    "The chronological split is "
                    "applied after uniform sampling, "
                    "so it is not identical to the "
                    "full-dataset chronological split "
                    "used in the original Duolingo "
                    "study."
                ),
                (
                    "The current processed training "
                    "frame does not preserve user and "
                    "lexeme identifiers for grouped "
                    "split analysis, so user or lexeme "
                    "overlap across sampled splits "
                    "cannot be excluded."
                ),
                (
                    "The public Duolingo model does "
                    "not learn MemoryMap-specific "
                    "features such as study duration, "
                    "topic difficulty, confidence, "
                    "or previous MemoryMap quiz "
                    "performance."
                ),
                (
                    "This model bootstraps retention "
                    "prediction. It must not be "
                    "presented as a personalized "
                    "MemoryMap model trained on the "
                    "project's future longitudinal "
                    "student dataset."
                ),
            ],

        "trained_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    save_training_artifacts(
        model=final_model,
        scaler=final_scaler,
        metadata=metadata
    )

    print()

    print(
        "Training artifacts saved."
    )

    print(
        "Model:",
        RETENTION_MODEL_PATH
    )

    if final_scaler is not None:
        print(
            "Scaler:",
            SCALER_PATH
        )

    else:
        print(
            "Scaler: Not required"
        )

    print(
        "Metadata:",
        MODEL_METADATA_PATH
    )

    return {
        "model":
            final_model,

        "scaler":
            final_scaler,

        "metadata":
            metadata,

        "validation_ranking":
            ranking,

        "test_metrics":
            test_metrics,
    }


# command line

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train the MemoryMap public "
            "bootstrap retention model."
        )
    )

    parser.add_argument(
        "--feature-set",
        choices=[
            "base",
            "extended",
        ],
        default=DEFAULT_FEATURE_SET,
        help=(
            "Safe Duolingo feature set "
            "used for public bootstrap "
            "model training."
        )
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_TRAINING_ROWS,
        help=(
            "Maximum number of uniformly "
            "sampled rows. Use 0 to load "
            "the full processed dataset."
        )
    )

    return parser.parse_args()


# direct execution

if __name__ == "__main__":

    arguments = parse_arguments()

    row_limit = (
        None

        if arguments.max_rows <= 0

        else arguments.max_rows
    )

    train_retention_model(
        feature_set=(
            arguments.feature_set
        ),
        max_rows=row_limit
    )
