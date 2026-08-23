from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    RECALL_CHECKPOINT_DAYS,
)

from database import (
    get_study_sessions_for_student,
    get_quiz_results_for_student,
    get_predictions_for_session,
)

from core.metrics import (
    calculate_model_metrics,
    calculate_correlation,
    classify_correlation,
    compare_with_baseline,
    rank_models,
    create_research_verdict,
)

from core.quiz_engine import (
    match_prediction_to_recall,
)

from services.student_service import (
    get_student_profile,
)

from services.study_service import (
    get_topic_profile,
)

from utils import (
    parse_datetime,
)


MODEL_COLUMNS = {
    "Fixed Forgetting Curve":
        "fixed_prediction",

    "Personalized Model":
        "personalized_prediction",

    "Adaptive Schedule":
        "adaptive_prediction",
}


SESSION_COLUMNS = [
    "session_id",
    "student_id",
    "topic_id",
    "topic_name",
    "subject_name",
    "difficulty",
    "study_duration",
    "confidence_level",
    "revision_count",
    "previous_quiz_score",
    "studied_at",
]


QUIZ_COLUMNS = [
    "quiz_id",
    "student_id",
    "topic_id",
    "topic_name",
    "session_id",
    "total_questions",
    "correct_answers",
    "actual_score",
    "recall_accuracy",
    "delay_days",
    "confidence_after_quiz",
    "quiz_date",
]


EVALUATION_COLUMNS = [
    "quiz_id",
    "student_id",
    "topic_id",
    "topic_name",
    "session_id",
    "quiz_date",
    "actual_recall",
    "actual_score",
    "delay_days",
    "prediction_for_days",
    "interval_difference",
    "strict_match",
    "prediction_created_at",
    "fixed_prediction",
    "personalized_prediction",
    "adaptive_prediction",
]


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
        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return default

    if not np.isfinite(
        value
    ):
        return default

    return value


def valid_retention(
    value
):
    value = safe_float(
        value
    )

    if value is None:
        return None

    if not (
        0 <= value <= 1
    ):
        return None

    return value


def empty_session_dataframe():
    return pd.DataFrame(
        columns=SESSION_COLUMNS
    )


def empty_quiz_dataframe():
    return pd.DataFrame(
        columns=QUIZ_COLUMNS
    )


def empty_evaluation_dataframe():
    return pd.DataFrame(
        columns=EVALUATION_COLUMNS
    )


# raw data retrieval

def get_student_sessions(
    student_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return []

    student_id = student[
        "student_id"
    ]

    try:
        sessions = (
            get_study_sessions_for_student(
                student_id
            )
        )

    except Exception:
        return []

    return normalize_records(
        sessions
    )


def get_student_quizzes(
    student_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return []

    student_id = student[
        "student_id"
    ]

    try:
        quizzes = (
            get_quiz_results_for_student(
                student_id
            )
        )

    except Exception:
        return []

    return normalize_records(
        quizzes
    )


def get_session_predictions(
    session_id
):
    session_id = normalize_id(
        session_id
    )

    if session_id is None:
        return []

    try:
        predictions = (
            get_predictions_for_session(
                session_id
            )
        )

    except Exception:
        return []

    return normalize_records(
        predictions
    )


# session dataframe

def create_session_dataframe(
    student_id
):
    sessions = get_student_sessions(
        student_id
    )

    if not sessions:
        return empty_session_dataframe()

    rows = []

    for session in sessions:
        topic = get_topic_profile(
            session.get(
                "topic_id"
            )
        )

        rows.append(
            {
                "session_id":
                    session.get(
                        "session_id"
                    ),

                "student_id":
                    session.get(
                        "student_id"
                    ),

                "topic_id":
                    session.get(
                        "topic_id"
                    ),

                "topic_name":
                    (
                        topic.get(
                            "topic_name"
                        )
                        if topic
                        else session.get(
                            "topic_name"
                        )
                    ),

                "subject_name":
                    (
                        topic.get(
                            "subject_name"
                        )
                        if topic
                        else session.get(
                            "subject_name"
                        )
                    ),

                "difficulty":
                    (
                        safe_float(
                            topic.get(
                                "difficulty"
                            )
                        )
                        if topic
                        else safe_float(
                            session.get(
                                "difficulty"
                            )
                        )
                    ),

                "study_duration":
                    safe_float(
                        session.get(
                            "study_duration"
                        )
                    ),

                "confidence_level":
                    safe_float(
                        session.get(
                            "confidence_level"
                        )
                    ),

                "revision_count":
                    safe_float(
                        session.get(
                            "revision_count"
                        )
                    ),

                "previous_quiz_score":
                    safe_float(
                        session.get(
                            "previous_quiz_score"
                        )
                    ),

                "studied_at":
                    session.get(
                        "studied_at"
                    ),
            }
        )

    dataframe = pd.DataFrame(
        rows,
        columns=SESSION_COLUMNS
    )

    if "studied_at" in dataframe.columns:
        dataframe[
            "studied_at"
        ] = pd.to_datetime(
            dataframe[
                "studied_at"
            ],
            errors="coerce"
        )

        dataframe = (
            dataframe
            .sort_values(
                "studied_at"
            )
            .reset_index(
                drop=True
            )
        )

    return dataframe


# quiz dataframe

def create_quiz_dataframe(
    student_id
):
    quizzes = get_student_quizzes(
        student_id
    )

    if not quizzes:
        return empty_quiz_dataframe()

    rows = []

    for quiz in quizzes:
        topic = get_topic_profile(
            quiz.get(
                "topic_id"
            )
        )

        rows.append(
            {
                "quiz_id":
                    quiz.get(
                        "quiz_id"
                    ),

                "student_id":
                    quiz.get(
                        "student_id"
                    ),

                "topic_id":
                    quiz.get(
                        "topic_id"
                    ),

                "topic_name":
                    (
                        topic.get(
                            "topic_name"
                        )
                        if topic
                        else quiz.get(
                            "topic_name"
                        )
                    ),

                "session_id":
                    quiz.get(
                        "session_id"
                    ),

                "total_questions":
                    safe_float(
                        quiz.get(
                            "total_questions"
                        )
                    ),

                "correct_answers":
                    safe_float(
                        quiz.get(
                            "correct_answers"
                        )
                    ),

                "actual_score":
                    safe_float(
                        quiz.get(
                            "actual_score"
                        )
                    ),

                "recall_accuracy":
                    valid_retention(
                        quiz.get(
                            "recall_accuracy"
                        )
                    ),

                "delay_days":
                    safe_float(
                        quiz.get(
                            "delay_days"
                        )
                    ),

                "confidence_after_quiz":
                    safe_float(
                        quiz.get(
                            "confidence_after_quiz"
                        )
                    ),

                "quiz_date":
                    quiz.get(
                        "quiz_date"
                    ),
            }
        )

    dataframe = pd.DataFrame(
        rows,
        columns=QUIZ_COLUMNS
    )

    dataframe[
        "quiz_date"
    ] = pd.to_datetime(
        dataframe[
            "quiz_date"
        ],
        errors="coerce"
    )

    dataframe = (
        dataframe
        .sort_values(
            "quiz_date"
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# combined learning dataframe

def create_learning_dataframe(
    student_id
):
    sessions = (
        create_session_dataframe(
            student_id
        )
    )

    quizzes = (
        create_quiz_dataframe(
            student_id
        )
    )

    if (
        sessions.empty
        or quizzes.empty
    ):
        return pd.DataFrame()

    session_columns = [
        "session_id",
        "study_duration",
        "confidence_level",
        "revision_count",
        "previous_quiz_score",
        "difficulty",
        "studied_at",
        "subject_name",
    ]

    available_columns = [
        column
        for column in session_columns
        if column in sessions.columns
    ]

    combined = quizzes.merge(
        sessions[
            available_columns
        ],
        on="session_id",
        how="left"
    )

    return combined


# prediction matching

def create_prediction_match(
    quiz,
    tolerance_days=0.5
):
    quiz = normalize_record(
        quiz
    )

    if not quiz:
        return None

    session_id = normalize_id(
        quiz.get(
            "session_id"
        )
    )

    if session_id is None:
        return None

    predictions = (
        get_session_predictions(
            session_id
        )
    )

    if not predictions:
        return None

    quiz_date = quiz.get(
        "quiz_date"
    )

    delay_days = safe_float(
        quiz.get(
            "delay_days"
        )
    )

    actual_recall = (
        valid_retention(
            quiz.get(
                "recall_accuracy"
            )
        )
    )

    if (
        quiz_date is None
        or delay_days is None
        or actual_recall is None
    ):
        return None

    try:
        match = (
            match_prediction_to_recall(
                predictions=predictions,
                quiz_at=quiz_date,
                delay_days=delay_days,
                tolerance_days=(
                    tolerance_days
                )
            )
        )

    except Exception:
        return None

    if not isinstance(
        match,
        dict
    ):
        return None

    prediction = match.get(
        "prediction"
    )

    if prediction is None:
        return None

    prediction = normalize_record(
        prediction
    )

    if prediction is None:
        return None

    result = {
        "quiz_id":
            quiz.get(
                "quiz_id"
            ),

        "student_id":
            quiz.get(
                "student_id"
            ),

        "topic_id":
            quiz.get(
                "topic_id"
            ),

        "session_id":
            session_id,

        "quiz_date":
            quiz_date,

        "actual_recall":
            actual_recall,

        "actual_score":
            actual_recall
            * 100,

        "delay_days":
            delay_days,

        "prediction_for_days":
            safe_float(
                prediction.get(
                    "prediction_for_days"
                )
            ),

        "interval_difference":
            safe_float(
                match.get(
                    "interval_difference"
                )
            ),

        "strict_match":
            bool(
                match.get(
                    "strict_match",
                    False
                )
            ),

        "prediction_created_at":
            prediction.get(
                "created_at"
            ),
    }

    for (
        model_name,
        column
    ) in MODEL_COLUMNS.items():

        result[
            column
        ] = valid_retention(
            prediction.get(
                column
            )
        )

    return result


# evaluation dataframe

def create_evaluation_dataframe(
    student_id,
    tolerance_days=0.5,
    strict_only=True
):
    quizzes = get_student_quizzes(
        student_id
    )

    if not quizzes:
        return empty_evaluation_dataframe()

    records = []

    for quiz in quizzes:
        record = (
            create_prediction_match(
                quiz,
                tolerance_days=(
                    tolerance_days
                )
            )
        )

        if record is None:
            continue

        if (
            strict_only
            and not record.get(
                "strict_match",
                False
            )
        ):
            continue

        topic = get_topic_profile(
            record.get(
                "topic_id"
            )
        )

        record[
            "topic_name"
        ] = (
            topic.get(
                "topic_name"
            )
            if topic
            else None
        )

        records.append(
            record
        )

    if not records:
        return empty_evaluation_dataframe()

    dataframe = pd.DataFrame(
        records
    )

    for column in EVALUATION_COLUMNS:
        if column not in dataframe.columns:
            dataframe[
                column
            ] = None

    dataframe[
        "quiz_date"
    ] = pd.to_datetime(
        dataframe[
            "quiz_date"
        ],
        errors="coerce"
    )

    dataframe[
        "prediction_created_at"
    ] = pd.to_datetime(
        dataframe[
            "prediction_created_at"
        ],
        errors="coerce"
    )

    dataframe = (
        dataframe
        .sort_values(
            "quiz_date"
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe[
        EVALUATION_COLUMNS
    ]


# individual metrics

def calculate_available_model_metrics(
    evaluation_dataframe
):
    if (
        evaluation_dataframe is None
        or evaluation_dataframe.empty
    ):
        return []

    results = []

    for (
        model_name,
        column
    ) in MODEL_COLUMNS.items():

        if (
            column
            not in evaluation_dataframe.columns
        ):
            continue

        subset = (
            evaluation_dataframe[
                [
                    "actual_recall",
                    column,
                ]
            ]
            .dropna()
        )

        if subset.empty:
            continue

        metrics = (
            calculate_model_metrics(
                actual=subset[
                    "actual_recall"
                ],
                predicted=subset[
                    column
                ],
                model_name=model_name
            )
        )

        results.append(
            metrics
        )

    return results


# common comparison subset

def create_common_model_subset(
    evaluation_dataframe,
    model_names=None
):
    if (
        evaluation_dataframe is None
        or evaluation_dataframe.empty
    ):
        return pd.DataFrame()

    if model_names is None:
        model_names = list(
            MODEL_COLUMNS.keys()
        )

    required_columns = [
        "actual_recall"
    ]

    valid_models = []

    for model_name in model_names:
        column = MODEL_COLUMNS.get(
            model_name
        )

        if (
            column is None
            or column
            not in evaluation_dataframe.columns
        ):
            continue

        if (
            evaluation_dataframe[
                column
            ]
            .notna()
            .any()
        ):
            required_columns.append(
                column
            )

            valid_models.append(
                model_name
            )

    if not valid_models:
        return pd.DataFrame()

    subset = (
        evaluation_dataframe
        .dropna(
            subset=required_columns
        )
        .copy()
    )

    subset.attrs[
        "models"
    ] = valid_models

    return subset


# fair common metrics

def calculate_common_subset_metrics(
    evaluation_dataframe,
    model_names=None
):
    common_subset = (
        create_common_model_subset(
            evaluation_dataframe,
            model_names
        )
    )

    if common_subset.empty:
        return {
            "observations":
                0,

            "models":
                [],

            "metrics":
                [],

            "ranking":
                [],

            "dataframe":
                common_subset,
        }

    if model_names is None:
        model_names = common_subset.attrs.get(
            "models",
            list(
                MODEL_COLUMNS.keys()
            )
        )

    metrics = []

    for model_name in model_names:
        column = MODEL_COLUMNS.get(
            model_name
        )

        if (
            column is None
            or column
            not in common_subset.columns
        ):
            continue

        if (
            common_subset[
                column
            ]
            .isna()
            .any()
        ):
            continue

        result = (
            calculate_model_metrics(
                actual=common_subset[
                    "actual_recall"
                ],
                predicted=common_subset[
                    column
                ],
                model_name=model_name
            )
        )

        metrics.append(
            result
        )

    ranking = (
        rank_models(
            metrics,
            metric="mae"
        )
        if metrics
        else []
    )

    return {
        "observations":
            len(
                common_subset
            ),

        "models":
            [
                result.get(
                    "model"
                )
                for result in metrics
            ],

        "metrics":
            metrics,

        "ranking":
            ranking,

        "dataframe":
            common_subset,
    }


# pairwise model comparison

def compare_models_pairwise(
    evaluation_dataframe,
    first_model,
    second_model
):
    if (
        evaluation_dataframe is None
        or evaluation_dataframe.empty
    ):
        return None

    first_column = MODEL_COLUMNS.get(
        first_model
    )

    second_column = MODEL_COLUMNS.get(
        second_model
    )

    if (
        first_column is None
        or second_column is None
    ):
        return None

    required_columns = [
        "actual_recall",
        first_column,
        second_column,
    ]

    for column in required_columns:
        if (
            column
            not in evaluation_dataframe.columns
        ):
            return None

    subset = (
        evaluation_dataframe
        .dropna(
            subset=required_columns
        )
        .copy()
    )

    if subset.empty:
        return None

    first_metrics = (
        calculate_model_metrics(
            actual=subset[
                "actual_recall"
            ],
            predicted=subset[
                first_column
            ],
            model_name=first_model
        )
    )

    second_metrics = (
        calculate_model_metrics(
            actual=subset[
                "actual_recall"
            ],
            predicted=subset[
                second_column
            ],
            model_name=second_model
        )
    )

    first_mae = safe_float(
        first_metrics.get(
            "mae"
        )
    )

    second_mae = safe_float(
        second_metrics.get(
            "mae"
        )
    )

    if (
        first_mae is None
        or second_mae is None
    ):
        winner = None
        difference = None

    elif first_mae < second_mae:
        winner = first_model

        difference = (
            second_mae
            - first_mae
        )

    elif second_mae < first_mae:
        winner = second_model

        difference = (
            first_mae
            - second_mae
        )

    else:
        winner = "Tie"
        difference = 0.0

    return {
        "observations":
            len(
                subset
            ),

        "first_model":
            first_metrics,

        "second_model":
            second_metrics,

        "winner":
            winner,

        "mae_difference":
            difference,

        "mae_difference_pp":
            (
                difference
                * 100
                if difference
                is not None
                else None
            ),
    }


# baseline comparison

def create_baseline_comparisons(
    evaluation_dataframe
):
    comparisons = {}

    for candidate in [
        "Personalized Model",
        "Adaptive Schedule",
    ]:

        result = (
            compare_models_pairwise(
                evaluation_dataframe,
                "Fixed Forgetting Curve",
                candidate
            )
        )

        if result is None:
            continue

        fixed_metrics = result[
            "first_model"
        ]

        candidate_metrics = result[
            "second_model"
        ]

        try:
            improvement = (
                compare_with_baseline(
                    fixed_metrics,
                    candidate_metrics
                )
            )

        except Exception:
            improvement = None

        comparisons[
            candidate
        ] = {
            "pairwise":
                result,

            "improvement":
                improvement,
        }

    return comparisons


# descriptive correlations

def calculate_learning_correlations(
    learning_dataframe
):
    if (
        learning_dataframe is None
        or learning_dataframe.empty
    ):
        return {}

    variables = {
        "Study Duration":
            "study_duration",

        "Topic Difficulty":
            "difficulty",

        "Confidence":
            "confidence_level",

        "Revision Count":
            "revision_count",

        "Previous Quiz Score":
            "previous_quiz_score",

        "Recall Delay":
            "delay_days",
    }

    results = {}

    for (
        label,
        column
    ) in variables.items():

        if (
            column
            not in learning_dataframe.columns
            or
            "recall_accuracy"
            not in learning_dataframe.columns
        ):
            continue

        subset = (
            learning_dataframe[
                [
                    column,
                    "recall_accuracy",
                ]
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .dropna()
        )

        if len(
            subset
        ) < 2:
            correlation = None
            classification = (
                "Insufficient Data"
            )

        else:
            try:
                correlation = (
                    calculate_correlation(
                        subset[
                            column
                        ],
                        subset[
                            "recall_accuracy"
                        ]
                    )
                )

                classification = (
                    classify_correlation(
                        correlation
                    )
                )

            except Exception:
                correlation = None
                classification = (
                    "Unavailable"
                )

        results[
            label
        ] = {
            "column":
                column,

            "correlation":
                correlation,

            "classification":
                classification,

            "observations":
                len(
                    subset
                ),

            "interpretation":
                (
                    "Descriptive association only. "
                    "This does not establish causation."
                ),
        }

    return results


# topic analytics

def calculate_topic_analytics(
    learning_dataframe
):
    if (
        learning_dataframe is None
        or learning_dataframe.empty
        or
        "topic_name"
        not in learning_dataframe.columns
    ):
        return pd.DataFrame()

    dataframe = (
        learning_dataframe
        .dropna(
            subset=[
                "topic_name"
            ]
        )
        .copy()
    )

    if dataframe.empty:
        return pd.DataFrame()

    rows = []

    for (
        topic_name,
        group
    ) in dataframe.groupby(
        "topic_name"
    ):

        recall = pd.to_numeric(
            group[
                "recall_accuracy"
            ],
            errors="coerce"
        ).dropna()

        delay = pd.to_numeric(
            group[
                "delay_days"
            ],
            errors="coerce"
        ).dropna()

        if (
            "study_duration"
            in group.columns
        ):
            study_duration = (
                pd.to_numeric(
                    group[
                        "study_duration"
                    ],
                    errors="coerce"
                )
                .dropna()
            )

        else:
            study_duration = pd.Series(
                dtype=float
            )

        rows.append(
            {
                "topic_name":
                    topic_name,

                "recall_tests":
                    len(
                        recall
                    ),

                "average_recall":
                    (
                        float(
                            recall.mean()
                        )
                        if not recall.empty
                        else None
                    ),

                "best_recall":
                    (
                        float(
                            recall.max()
                        )
                        if not recall.empty
                        else None
                    ),

                "lowest_recall":
                    (
                        float(
                            recall.min()
                        )
                        if not recall.empty
                        else None
                    ),

                "average_delay_days":
                    (
                        float(
                            delay.mean()
                        )
                        if not delay.empty
                        else None
                    ),

                "average_study_minutes":
                    (
                        float(
                            study_duration.mean()
                        )
                        if not study_duration.empty
                        else None
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if (
        not result.empty
        and
        "average_recall"
        in result.columns
    ):
        result = (
            result
            .sort_values(
                "average_recall",
                ascending=False,
                na_position="last"
            )
            .reset_index(
                drop=True
            )
        )

    return result


# checkpoints

def classify_delay_checkpoint(
    delay_days,
    tolerance=0.5
):
    delay_days = safe_float(
        delay_days
    )

    if delay_days is None:
        return "Unknown"

    checkpoints = list(
        RECALL_CHECKPOINT_DAYS
    )

    if not checkpoints:
        return "Custom"

    nearest = min(
        checkpoints,
        key=lambda checkpoint:
            abs(
                checkpoint
                - delay_days
            )
    )

    if (
        abs(
            nearest
            - delay_days
        )
        <= tolerance
    ):
        return (
            f"{nearest}-Day"
        )

    return "Custom"


def calculate_checkpoint_analytics(
    learning_dataframe
):
    if (
        learning_dataframe is None
        or learning_dataframe.empty
    ):
        return pd.DataFrame()

    if (
        "delay_days"
        not in learning_dataframe.columns
        or
        "recall_accuracy"
        not in learning_dataframe.columns
    ):
        return pd.DataFrame()

    dataframe = (
        learning_dataframe.copy()
    )

    dataframe[
        "checkpoint"
    ] = dataframe[
        "delay_days"
    ].apply(
        classify_delay_checkpoint
    )

    rows = []

    for (
        checkpoint,
        group
    ) in dataframe.groupby(
        "checkpoint"
    ):

        recall = pd.to_numeric(
            group[
                "recall_accuracy"
            ],
            errors="coerce"
        ).dropna()

        if recall.empty:
            continue

        rows.append(
            {
                "checkpoint":
                    checkpoint,

                "tests":
                    len(
                        recall
                    ),

                "average_recall":
                    float(
                        recall.mean()
                    ),

                "median_recall":
                    float(
                        recall.median()
                    ),

                "minimum_recall":
                    float(
                        recall.min()
                    ),

                "maximum_recall":
                    float(
                        recall.max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# research status

def calculate_research_status(
    student_id
):
    sessions = get_student_sessions(
        student_id
    )

    quizzes = get_student_quizzes(
        student_id
    )

    evaluation = (
        create_evaluation_dataframe(
            student_id,
            strict_only=True
        )
    )

    session_count = len(
        sessions
    )

    quiz_count = len(
        quizzes
    )

    strict_matches = len(
        evaluation
    )

    available_models = []

    if not evaluation.empty:
        for (
            model_name,
            column
        ) in MODEL_COLUMNS.items():

            if (
                column in evaluation.columns
                and
                evaluation[
                    column
                ].notna().any()
            ):
                available_models.append(
                    model_name
                )

    if strict_matches == 0:
        stage = (
            "Waiting for Matched Recall Data"
        )

    elif strict_matches < 5:
        stage = (
            "Insufficient Evaluation Data"
        )

    elif strict_matches < 20:
        stage = (
            "Preliminary Research Data"
        )

    elif strict_matches < 50:
        stage = (
            "Developing Evidence"
        )

    else:
        stage = (
            "Stronger Evaluation Dataset"
        )

    return {
        "study_sessions":
            session_count,

        "quiz_results":
            quiz_count,

        "strict_prediction_matches":
            strict_matches,

        "available_models":
            available_models,

        "research_stage":
            stage,

        "note":
            (
                "Prediction accuracy alone does not "
                "prove that one revision schedule is "
                "better. Schedule-level outcomes "
                "must also be evaluated."
            ),
    }


# research verdict

def get_research_verdict(
    student_id,
    tolerance_days=0.5
):
    evaluation = (
        create_evaluation_dataframe(
            student_id,
            tolerance_days=(
                tolerance_days
            ),
            strict_only=True
        )
    )

    if evaluation.empty:
        return {
            "winner":
                None,

            "status":
                "No Data",

            "message":
                (
                    "No strict prediction and recall "
                    "matches are available yet."
                ),
        }

    common = (
        calculate_common_subset_metrics(
            evaluation
        )
    )

    if (
        not common[
            "metrics"
        ]
        or
        len(
            common[
                "metrics"
            ]
        ) < 2
    ):
        return {
            "winner":
                None,

            "status":
                "Insufficient",

            "common_observations":
                common.get(
                    "observations",
                    0
                ),

            "message":
                (
                    "Matched recall data exists, but "
                    "at least two prediction methods "
                    "must be available on the same "
                    "observations for a fair comparison."
                ),
        }

    try:
        verdict = (
            create_research_verdict(
                common[
                    "metrics"
                ]
            )
        )

    except Exception as error:
        return {
            "winner":
                None,

            "status":
                "Unavailable",

            "common_observations":
                common[
                    "observations"
                ],

            "message":
                (
                    "Research verdict could not "
                    f"be calculated: {error}"
                ),
        }

    verdict[
        "common_observations"
    ] = common[
        "observations"
    ]

    verdict[
        "models_compared"
    ] = common[
        "models"
    ]

    verdict[
        "comparison_type"
    ] = (
        "Common matched observations"
    )

    return verdict


# complete analytics

def get_student_analytics(
    student_id,
    tolerance_days=0.5
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    sessions = (
        create_session_dataframe(
            student_id
        )
    )

    quizzes = (
        create_quiz_dataframe(
            student_id
        )
    )

    learning = (
        create_learning_dataframe(
            student_id
        )
    )

    evaluation = (
        create_evaluation_dataframe(
            student_id,
            tolerance_days=(
                tolerance_days
            ),
            strict_only=True
        )
    )

    individual_metrics = (
        calculate_available_model_metrics(
            evaluation
        )
    )

    common_metrics = (
        calculate_common_subset_metrics(
            evaluation
        )
    )

    baseline_comparisons = (
        create_baseline_comparisons(
            evaluation
        )
    )

    correlations = (
        calculate_learning_correlations(
            learning
        )
    )

    topic_analytics = (
        calculate_topic_analytics(
            learning
        )
    )

    checkpoint_analytics = (
        calculate_checkpoint_analytics(
            learning
        )
    )

    research_status = (
        calculate_research_status(
            student_id
        )
    )

    verdict = (
        get_research_verdict(
            student_id,
            tolerance_days=(
                tolerance_days
            )
        )
    )

    return {
        "success":
            True,

        "student":
            student,

        "sessions":
            sessions,

        "quizzes":
            quizzes,

        "learning_data":
            learning,

        "evaluation_data":
            evaluation,

        "individual_model_metrics":
            individual_metrics,

        "common_subset":
            common_metrics,

        "baseline_comparisons":
            baseline_comparisons,

        "correlations":
            correlations,

        "topic_analytics":
            topic_analytics,

        "checkpoint_analytics":
            checkpoint_analytics,

        "research_status":
            research_status,

        "research_verdict":
            verdict,

        "message":
            "Student analytics generated.",
    }


# direct execution

if __name__ == "__main__":

    from services.student_service import (
        list_students,
    )

    print(
        "MemoryMap Analytics Service"
    )

    print(
        "=" * 40
    )

    students = list_students()

    print(
        "Students:",
        len(
            students
        )
    )

    if students:
        student = students[
            0
        ]

        student_id = student[
            "student_id"
        ]

        analytics = (
            get_student_analytics(
                student_id
            )
        )

        print(
            "Student:",
            student.get(
                "name"
            )
        )

        print(
            "Student ID:",
            student_id
        )

        if analytics.get(
            "success",
            False
        ):
            status = analytics[
                "research_status"
            ]

            print(
                "Study Sessions:",
                status[
                    "study_sessions"
                ]
            )

            print(
                "Quiz Results:",
                status[
                    "quiz_results"
                ]
            )

            print(
                "Strict Matches:",
                status[
                    "strict_prediction_matches"
                ]
            )

            print(
                "Research Stage:",
                status[
                    "research_stage"
                ]
            )

            print()

            verdict = analytics[
                "research_verdict"
            ]

            print(
                "Research Verdict:"
            )

            print(
                verdict.get(
                    "message"
                )
            )

        else:
            print(
                analytics.get(
                    "message"
                )
            )