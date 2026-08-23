from functools import lru_cache
import json
import math

import joblib
import numpy as np
import pandas as pd

from config import (
    RETENTION_MODEL_PATH,
    SCALER_PATH,
    MODEL_METADATA_PATH,
)

from ml.feature_engineering import (
    FEATURE_SETS,
    LEAKAGE_COLUMNS,
    check_for_target_leakage,
    get_feature_columns,
)


# artifact status

def get_artifact_status():
    return {
        "model_exists":
            RETENTION_MODEL_PATH.exists(),

        "scaler_exists":
            SCALER_PATH.exists(),

        "metadata_exists":
            MODEL_METADATA_PATH.exists(),

        "model_path":
            str(
                RETENTION_MODEL_PATH
            ),

        "scaler_path":
            str(
                SCALER_PATH
            ),

        "metadata_path":
            str(
                MODEL_METADATA_PATH
            ),
    }


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


# metadata validation

def validate_metadata(
    metadata
):
    if not isinstance(
        metadata,
        dict
    ):
        raise ValueError(
            "Model metadata must be "
            "stored as a JSON object."
        )

    required_fields = [
        "model_name",
        "feature_set",
        "feature_columns",
        "target_column",
        "target_scale",
        "uses_scaler",
    ]

    missing_fields = [
        field
        for field in required_fields

        if field
        not in metadata
    ]

    if missing_fields:
        raise ValueError(
            "Model metadata is missing "
            "required fields: "
            + ", ".join(
                missing_fields
            )
        )

    model_name = metadata.get(
        "model_name"
    )

    if not isinstance(
        model_name,
        str
    ) or not model_name.strip():
        raise ValueError(
            "model_name must be a "
            "non-empty string."
        )

    feature_set = metadata.get(
        "feature_set"
    )

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            "Unsupported feature set "
            f"in model metadata: {feature_set}"
        )

    feature_columns = metadata.get(
        "feature_columns"
    )

    if not isinstance(
        feature_columns,
        list
    ):
        raise ValueError(
            "feature_columns must be "
            "stored as a list."
        )

    if not feature_columns:
        raise ValueError(
            "The trained model has no "
            "recorded feature columns."
        )

    cleaned_features = []

    for feature in feature_columns:
        if not isinstance(
            feature,
            str
        ):
            raise ValueError(
                "Every feature column name "
                "must be a string."
            )

        feature = feature.strip()

        if not feature:
            raise ValueError(
                "Feature column names cannot "
                "be empty."
            )

        cleaned_features.append(
            feature
        )

    if len(
        cleaned_features
    ) != len(
        set(
            cleaned_features
        )
    ):
        raise ValueError(
            "Model metadata contains "
            "duplicate feature columns."
        )

    check_for_target_leakage(
        cleaned_features
    )

    expected_features = (
        get_feature_columns(
            feature_set
        )
    )

    if cleaned_features != expected_features:
        raise ValueError(
            "Model metadata feature columns "
            "do not match the recorded "
            f"feature set '{feature_set}'. "
            "Expected: "
            + ", ".join(
                expected_features
            )
            + ". Found: "
            + ", ".join(
                cleaned_features
            )
        )

    target_column = metadata.get(
        "target_column"
    )

    if not isinstance(
        target_column,
        str
    ) or not target_column.strip():
        raise ValueError(
            "target_column must be a "
            "non-empty string."
        )

    if target_column in cleaned_features:
        raise ValueError(
            "The prediction target cannot "
            "also appear as an input feature."
        )

    if target_column in LEAKAGE_COLUMNS:
        pass

    target_scale = metadata.get(
        "target_scale"
    )

    if target_scale not in {
        "0-1",
        "0-100",
    }:
        raise ValueError(
            "Unsupported target scale "
            "in model metadata."
        )

    uses_scaler = metadata.get(
        "uses_scaler"
    )

    if not isinstance(
        uses_scaler,
        bool
    ):
        raise ValueError(
            "uses_scaler must be stored "
            "as true or false."
        )

    return True


# metadata loading

@lru_cache(maxsize=1)
def load_metadata():
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "model_metadata.json was not found. "
            "Train the retention model first."
        )

    try:
        with open(
            MODEL_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as metadata_file:

            metadata = json.load(
                metadata_file
            )

    except json.JSONDecodeError as error:
        raise ValueError(
            "model_metadata.json contains "
            "invalid JSON."
        ) from error

    except OSError as error:
        raise RuntimeError(
            "model_metadata.json could "
            "not be read."
        ) from error

    validate_metadata(
        metadata
    )

    return metadata


# model loading

@lru_cache(maxsize=1)
def load_retention_model():
    if not RETENTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "retention_model.pkl was not found. "
            "Run the model training pipeline first."
        )

    try:
        model = joblib.load(
            RETENTION_MODEL_PATH
        )

    except Exception as error:
        raise RuntimeError(
            "The trained retention model "
            "could not be loaded."
        ) from error

    if not hasattr(
        model,
        "predict"
    ):
        raise TypeError(
            "The loaded model does not "
            "provide a predict() method."
        )

    return model


# scaler loading

@lru_cache(maxsize=1)
def load_scaler():
    metadata = load_metadata()

    uses_scaler = metadata.get(
        "uses_scaler",
        False
    )

    if not uses_scaler:
        return None

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "The model requires scaler.pkl, "
            "but the scaler file was not found."
        )

    try:
        scaler = joblib.load(
            SCALER_PATH
        )

    except Exception as error:
        raise RuntimeError(
            "The saved feature scaler "
            "could not be loaded."
        ) from error

    if not hasattr(
        scaler,
        "transform"
    ):
        raise TypeError(
            "The loaded scaler does not "
            "provide a transform() method."
        )

    return scaler


# artifact compatibility

def validate_feature_count(
    artifact,
    expected_count,
    artifact_name
):
    feature_count = getattr(
        artifact,
        "n_features_in_",
        None
    )

    if feature_count is None:
        return True

    try:
        feature_count = int(
            feature_count
        )

    except (
        TypeError,
        ValueError
    ):
        return True

    if feature_count != expected_count:
        raise ValueError(
            f"{artifact_name} expects "
            f"{feature_count} feature(s), "
            f"but metadata records "
            f"{expected_count}."
        )

    return True


def validate_feature_names(
    artifact,
    expected_features,
    artifact_name
):
    feature_names = getattr(
        artifact,
        "feature_names_in_",
        None
    )

    if feature_names is None:
        return True

    actual_features = [
        str(
            feature
        )
        for feature
        in list(
            feature_names
        )
    ]

    if actual_features != list(
        expected_features
    ):
        raise ValueError(
            f"{artifact_name} feature names "
            "do not match model metadata."
        )

    return True


# complete model bundle

@lru_cache(maxsize=1)
def load_model_bundle():
    metadata = load_metadata()

    model = load_retention_model()

    scaler = load_scaler()

    expected_model_name = metadata.get(
        "model_name"
    )

    actual_model_name = (
        model.__class__.__name__
    )

    if (
        expected_model_name
        != actual_model_name
    ):
        raise ValueError(
            "Loaded model class does not "
            "match model metadata. "
            f"Expected {expected_model_name}, "
            f"found {actual_model_name}."
        )

    feature_columns = metadata[
        "feature_columns"
    ]

    expected_count = len(
        feature_columns
    )

    validate_feature_count(
        model,
        expected_count,
        "Retention model"
    )

    validate_feature_names(
        model,
        feature_columns,
        "Retention model"
    )

    if metadata.get(
        "uses_scaler",
        False
    ):
        if scaler is None:
            raise ValueError(
                "Metadata indicates that a "
                "scaler is required."
            )

        validate_feature_count(
            scaler,
            expected_count,
            "Feature scaler"
        )

        validate_feature_names(
            scaler,
            feature_columns,
            "Feature scaler"
        )

    return {
        "model":
            model,

        "scaler":
            scaler,

        "metadata":
            metadata,
    }


# model information

def get_model_information():
    try:
        bundle = load_model_bundle()

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError,
    ) as error:

        return {
            "available":
                False,

            "message":
                str(
                    error
                ),

            "model_name":
                None,

            "model_class":
                None,

            "feature_set":
                None,

            "feature_columns":
                [],

            "target_column":
                None,

            "target_scale":
                None,

            "uses_scaler":
                False,

            "training_source":
                None,

            "trained_at":
                None,
        }

    metadata = bundle[
        "metadata"
    ]

    return {
        "available":
            True,

        "message":
            "Model loaded successfully.",

        "model_name":
            metadata.get(
                "model_display_name",
                metadata.get(
                    "model_name"
                )
            ),

        "model_class":
            metadata.get(
                "model_name"
            ),

        "feature_set":
            metadata.get(
                "feature_set"
            ),

        "feature_columns":
            list(
                metadata.get(
                    "feature_columns",
                    []
                )
            ),

        "target_column":
            metadata.get(
                "target_column"
            ),

        "target_scale":
            metadata.get(
                "target_scale"
            ),

        "uses_scaler":
            metadata.get(
                "uses_scaler",
                False
            ),

        "training_source":
            metadata.get(
                "training_source"
            ),

        "trained_at":
            metadata.get(
                "trained_at"
            ),
    }


# input validation

def validate_model_input(
    dataframe,
    metadata=None
):
    if metadata is None:
        metadata = load_metadata()

    validate_metadata(
        metadata
    )

    if not isinstance(
        dataframe,
        pd.DataFrame
    ):
        raise TypeError(
            "Model input must be "
            "a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Model input cannot be empty."
        )

    feature_columns = metadata[
        "feature_columns"
    ]

    missing_features = [
        feature
        for feature in feature_columns

        if feature
        not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Model input is missing "
            "required features: "
            + ", ".join(
                missing_features
            )
        )

    supplied_leakage = (
        set(
            dataframe.columns
        )
        & LEAKAGE_COLUMNS
    )

    if supplied_leakage:
        raise ValueError(
            "Model input contains outcome "
            "or leakage columns: "
            + ", ".join(
                sorted(
                    supplied_leakage
                )
            )
        )

    return True


# feature ordering

def prepare_model_input(
    dataframe,
    metadata=None
):
    if metadata is None:
        metadata = load_metadata()

    validate_model_input(
        dataframe,
        metadata
    )

    feature_columns = metadata[
        "feature_columns"
    ]

    prepared = dataframe[
        feature_columns
    ].copy()

    for feature in feature_columns:
        prepared[
            feature
        ] = pd.to_numeric(
            prepared[
                feature
            ],
            errors="coerce"
        )

    prepared = prepared.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    if prepared.isna().any().any():
        invalid = []

        for feature in feature_columns:
            if prepared[
                feature
            ].isna().any():

                invalid.append(
                    feature
                )

        raise ValueError(
            "Model input contains missing, "
            "non-numeric or non-finite "
            "feature values: "
            + ", ".join(
                invalid
            )
        )

    return prepared


# scaling

def transform_model_input(
    dataframe,
    scaler=None,
    metadata=None
):
    if metadata is None:
        metadata = load_metadata()

    prepared = prepare_model_input(
        dataframe,
        metadata
    )

    if not metadata.get(
        "uses_scaler",
        False
    ):
        return prepared

    if scaler is None:
        scaler = load_scaler()

    if scaler is None:
        raise ValueError(
            "Model metadata requires a "
            "scaler, but none is available."
        )

    try:
        transformed = scaler.transform(
            prepared
        )

    except Exception as error:
        raise ValueError(
            "Model features could not "
            "be transformed by the scaler."
        ) from error

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
        prepared.columns
    ):
        raise ValueError(
            "Scaler output feature count "
            "does not match model metadata."
        )

    if not np.isfinite(
        transformed
    ).all():
        raise ValueError(
            "Scaler produced non-finite "
            "feature values."
        )

    return pd.DataFrame(
        transformed,
        columns=prepared.columns,
        index=prepared.index
    )


# target normalization

def normalize_model_output(
    prediction,
    metadata=None
):
    if metadata is None:
        metadata = load_metadata()

    prediction = safe_float(
        prediction
    )

    if prediction is None:
        raise ValueError(
            "Model produced an invalid "
            "prediction value."
        )

    target_scale = metadata.get(
        "target_scale"
    )

    if target_scale == "0-100":
        prediction = (
            prediction
            / 100
        )

    elif target_scale != "0-1":
        raise ValueError(
            "Unsupported target scale "
            "in model metadata."
        )

    # Regression models can produce tiny
    # values outside the theoretical range.
    # Clamp only the final probability.

    return max(
        0.0,
        min(
            1.0,
            prediction
        )
    )


# cache control

def clear_model_cache():
    load_metadata.cache_clear()

    load_retention_model.cache_clear()

    load_scaler.cache_clear()

    load_model_bundle.cache_clear()


def reload_model_bundle():
    clear_model_cache()

    return load_model_bundle()


# readiness

def model_is_ready():
    try:
        load_model_bundle()

        return True

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError,
    ):
        return False


# direct execution

if __name__ == "__main__":

    print(
        "MemoryMap Model Loader"
    )

    print(
        "=" * 40
    )

    status = get_artifact_status()

    print(
        "Model File:",
        (
            "Found"
            if status[
                "model_exists"
            ]
            else "Missing"
        )
    )

    print(
        "Metadata File:",
        (
            "Found"
            if status[
                "metadata_exists"
            ]
            else "Missing"
        )
    )

    print(
        "Scaler File:",
        (
            "Found"
            if status[
                "scaler_exists"
            ]
            else "Optional / Missing"
        )
    )

    print()

    information = (
        get_model_information()
    )

    if not information[
        "available"
    ]:
        print(
            "Model Status: Not Ready"
        )

        print(
            information[
                "message"
            ]
        )

    else:
        print(
            "Model Status: Ready"
        )

        print(
            "Model:",
            information[
                "model_name"
            ]
        )

        print(
            "Model Class:",
            information[
                "model_class"
            ]
        )

        print(
            "Feature Set:",
            information[
                "feature_set"
            ]
        )

        print(
            "Uses Scaler:",
            information[
                "uses_scaler"
            ]
        )

        print()

        print(
            "Expected Features:"
        )

        for feature in information[
            "feature_columns"
        ]:
            print(
                "-",
                feature
            )

        print()

        print(
            "Target Scale:",
            information[
                "target_scale"
            ]
        )