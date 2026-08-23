from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RAW_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    DUOLINGO_COLUMNS,
    PREPROCESS_CHUNK_SIZE,
    SECONDS_PER_DAY,
)


# processed dataset columns

PROCESSED_COLUMNS = [
    "p_recall",
    "timestamp",
    "delta_days",
    "history_seen",
    "history_correct",
    "history_incorrect",
    "history_accuracy",
    "session_seen",
    "session_correct",
]


# dataset validation

def validate_dataset_file(
    dataset_path=RAW_DATASET_PATH
):
    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Duolingo dataset not found: "
            f"{dataset_path}"
        )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            "The dataset path does not "
            "point to a valid file."
        )

    return dataset_path


def get_dataset_columns(
    dataset_path=RAW_DATASET_PATH
):
    dataset_path = (
        validate_dataset_file(
            dataset_path
        )
    )

    header = pd.read_csv(
        dataset_path,
        compression="gzip",
        nrows=0
    )

    return list(
        header.columns
    )


def validate_dataset_columns(
    dataset_path=RAW_DATASET_PATH
):
    available_columns = (
        get_dataset_columns(
            dataset_path
        )
    )

    missing_columns = [
        column
        for column in DUOLINGO_COLUMNS
        if column
        not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            "The Duolingo dataset is missing "
            "required columns: "
            + ", ".join(
                missing_columns
            )
        )

    return True


# numeric conversion

def convert_numeric_columns(
    dataframe
):
    numeric_columns = [
        "p_recall",
        "timestamp",
        "delta",
        "history_seen",
        "history_correct",
        "session_seen",
        "session_correct",
    ]

    for column in numeric_columns:
        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce"
        )

    return dataframe


# invalid row filtering

def remove_missing_rows(
    dataframe
):
    required_columns = [
        "p_recall",
        "timestamp",
        "delta",
        "history_seen",
        "history_correct",
        "session_seen",
        "session_correct",
    ]

    return dataframe.dropna(
        subset=required_columns
    )


def remove_invalid_recall(
    dataframe
):
    return dataframe[
        dataframe[
            "p_recall"
        ].between(
            0.0,
            1.0
        )
    ]


def remove_invalid_time_values(
    dataframe
):
    return dataframe[
        (
            dataframe[
                "timestamp"
            ] >= 0
        )
        &
        (
            dataframe[
                "delta"
            ] >= 0
        )
    ]


def remove_invalid_counts(
    dataframe
):
    dataframe = dataframe[
        (
            dataframe[
                "history_seen"
            ] >= 0
        )
        &
        (
            dataframe[
                "history_correct"
            ] >= 0
        )
        &
        (
            dataframe[
                "session_seen"
            ] > 0
        )
        &
        (
            dataframe[
                "session_correct"
            ] >= 0
        )
    ]

    dataframe = dataframe[
        (
            dataframe[
                "history_correct"
            ]
            <= dataframe[
                "history_seen"
            ]
        )
        &
        (
            dataframe[
                "session_correct"
            ]
            <= dataframe[
                "session_seen"
            ]
        )
    ]

    return dataframe


# feature creation

def create_delta_days(
    dataframe
):
    dataframe[
        "delta_days"
    ] = (
        dataframe[
            "delta"
        ]
        / SECONDS_PER_DAY
    )

    return dataframe


def create_history_features(
    dataframe
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

    dataframe[
        "history_accuracy"
    ] = np.where(
        dataframe[
            "history_seen"
        ] > 0,
        (
            dataframe[
                "history_correct"
            ]
            / dataframe[
                "history_seen"
            ]
        ),
        0.0
    )

    return dataframe


# consistency check

def calculate_session_recall(
    dataframe
):
    return (
        dataframe[
            "session_correct"
        ]
        / dataframe[
            "session_seen"
        ]
    )


def count_recall_inconsistencies(
    dataframe,
    tolerance=0.0001
):
    session_recall = (
        calculate_session_recall(
            dataframe
        )
    )

    difference = abs(
        dataframe[
            "p_recall"
        ]
        - session_recall
    )

    return int(
        (
            difference
            > tolerance
        ).sum()
    )


# memory optimization

def optimize_dtypes(
    dataframe
):
    float_columns = [
        "p_recall",
        "delta_days",
        "history_accuracy",
    ]

    integer_columns = [
        "timestamp",
        "history_seen",
        "history_correct",
        "history_incorrect",
        "session_seen",
        "session_correct",
    ]

    for column in float_columns:
        dataframe[
            column
        ] = dataframe[
            column
        ].astype(
            "float32"
        )

    for column in integer_columns:
        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            downcast="integer"
        )

    return dataframe


# chunk preprocessing

def preprocess_chunk(
    dataframe
):
    original_rows = len(
        dataframe
    )

    dataframe = (
        dataframe.copy()
    )

    dataframe = (
        convert_numeric_columns(
            dataframe
        )
    )

    dataframe = (
        remove_missing_rows(
            dataframe
        )
    )

    dataframe = (
        remove_invalid_recall(
            dataframe
        )
    )

    dataframe = (
        remove_invalid_time_values(
            dataframe
        )
    )

    dataframe = (
        remove_invalid_counts(
            dataframe
        )
    )

    inconsistencies = (
        count_recall_inconsistencies(
            dataframe
        )
    )

    dataframe = (
        create_delta_days(
            dataframe
        )
    )

    dataframe = (
        create_history_features(
            dataframe
        )
    )

    dataframe = dataframe[
        PROCESSED_COLUMNS
    ]

    dataframe = (
        optimize_dtypes(
            dataframe
        )
    )

    cleaned_rows = len(
        dataframe
    )

    removed_rows = (
        original_rows
        - cleaned_rows
    )

    statistics = {
        "input_rows":
            original_rows,

        "output_rows":
            cleaned_rows,

        "removed_rows":
            removed_rows,

        "recall_inconsistencies":
            inconsistencies,
    }

    return (
        dataframe,
        statistics
    )


# processed dataset creation

def preprocess_dataset(
    input_path=RAW_DATASET_PATH,
    output_path=PROCESSED_DATASET_PATH,
    chunk_size=PREPROCESS_CHUNK_SIZE,
    max_rows=None
):
    input_path = (
        validate_dataset_file(
            input_path
        )
    )

    validate_dataset_columns(
        input_path
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():
        output_path.unlink()

    total_input_rows = 0

    total_output_rows = 0

    total_removed_rows = 0

    total_inconsistencies = 0

    chunks_processed = 0

    first_chunk = True

    print(
        "MemoryMap Dataset Preprocessing"
    )

    print(
        f"Input: {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Chunk size: {chunk_size:,}"
    )

    if max_rows is None:
        print(
            "Row limit: Full dataset"
        )

    else:
        print(
            f"Row limit: {max_rows:,}"
        )

    print()

    reader = pd.read_csv(
        input_path,
        compression="gzip",
        usecols=DUOLINGO_COLUMNS,
        chunksize=chunk_size,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):
        if (
            max_rows is not None
            and total_output_rows
            >= max_rows
        ):
            break

        cleaned_chunk, statistics = (
            preprocess_chunk(
                chunk
            )
        )

        if max_rows is not None:
            remaining_rows = (
                max_rows
                - total_output_rows
            )

            if len(
                cleaned_chunk
            ) > remaining_rows:
                cleaned_chunk = (
                    cleaned_chunk.iloc[
                        :remaining_rows
                    ]
                )

        cleaned_chunk.to_csv(
            output_path,
            mode=(
                "w"
                if first_chunk
                else "a"
            ),
            header=first_chunk,
            index=False
        )

        first_chunk = False

        chunks_processed += 1

        total_input_rows += (
            statistics[
                "input_rows"
            ]
        )

        total_output_rows += len(
            cleaned_chunk
        )

        total_removed_rows += (
            statistics[
                "removed_rows"
            ]
        )

        total_inconsistencies += (
            statistics[
                "recall_inconsistencies"
            ]
        )

        print(
            f"Chunk {chunk_number}: "
            f"{statistics['input_rows']:,} read, "
            f"{len(cleaned_chunk):,} saved"
        )

        print(
            f"Total saved: "
            f"{total_output_rows:,}"
        )

        print()

        if (
            max_rows is not None
            and total_output_rows
            >= max_rows
        ):
            break

    if first_chunk:
        raise RuntimeError(
            "No valid rows were written "
            "to the processed dataset."
        )

    result = {
        "input_path":
            str(
                input_path
            ),

        "output_path":
            str(
                output_path
            ),

        "chunks_processed":
            chunks_processed,

        "input_rows":
            total_input_rows,

        "output_rows":
            total_output_rows,

        "removed_rows":
            total_removed_rows,

        "recall_inconsistencies":
            total_inconsistencies,
    }

    return result


# dataset inspection

def inspect_processed_dataset(
    dataset_path=PROCESSED_DATASET_PATH,
    rows=5
):
    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Processed dataset does not exist."
        )

    dataframe = pd.read_csv(
        dataset_path,
        nrows=rows
    )

    return dataframe


# direct execution

if __name__ == "__main__":
    result = preprocess_dataset()

    print(
        "Preprocessing Complete"
    )

    print()

    print(
        "Chunks Processed:",
        result[
            "chunks_processed"
        ]
    )

    print(
        "Rows Read:",
        f"{result['input_rows']:,}"
    )

    print(
        "Rows Saved:",
        f"{result['output_rows']:,}"
    )

    print(
        "Rows Removed:",
        f"{result['removed_rows']:,}"
    )

    print(
        "Recall Inconsistencies:",
        f"{result['recall_inconsistencies']:,}"
    )

    print()

    print(
        "Processed Dataset:"
    )

    print(
        result[
            "output_path"
        ]
    )

    print()

    print(
        inspect_processed_dataset()
    )