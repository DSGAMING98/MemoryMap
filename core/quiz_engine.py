from datetime import datetime

from config import (
    RECALL_CHECKPOINT_DAYS,
)

from utils import (
    calculate_prediction_error,
)


# datetime

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

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        return datetime.fromisoformat(
            text
        )

    except (
        TypeError,
        ValueError
    ):
        return None


# basic quiz validation

def validate_question_count(
    total_questions
):
    try:
        total_questions = int(
            total_questions
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Total questions must be "
            "a whole number."
        )

    if total_questions <= 0:
        raise ValueError(
            "A quiz must contain at least "
            "one question."
        )

    return total_questions


def validate_correct_answers(
    correct_answers,
    total_questions
):
    total_questions = (
        validate_question_count(
            total_questions
        )
    )

    try:
        correct_answers = int(
            correct_answers
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Correct answers must be "
            "a whole number."
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
            "Correct answers cannot exceed "
            "the total number of questions."
        )

    return correct_answers


# quiz scoring

def calculate_recall_accuracy(
    correct_answers,
    total_questions
):
    total_questions = (
        validate_question_count(
            total_questions
        )
    )

    correct_answers = (
        validate_correct_answers(
            correct_answers,
            total_questions
        )
    )

    return (
        correct_answers
        / total_questions
    )


def calculate_quiz_score(
    correct_answers,
    total_questions
):
    recall_accuracy = (
        calculate_recall_accuracy(
            correct_answers,
            total_questions
        )
    )

    return (
        recall_accuracy
        * 100
    )


def calculate_incorrect_answers(
    correct_answers,
    total_questions
):
    total_questions = (
        validate_question_count(
            total_questions
        )
    )

    correct_answers = (
        validate_correct_answers(
            correct_answers,
            total_questions
        )
    )

    return (
        total_questions
        - correct_answers
    )


# recall classification

def classify_recall(
    recall_accuracy
):
    try:
        recall_accuracy = float(
            recall_accuracy
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Recall accuracy must "
            "be numeric."
        )

    if not (
        0
        <= recall_accuracy
        <= 1
    ):
        raise ValueError(
            "Recall accuracy must be "
            "between 0 and 1."
        )

    if recall_accuracy >= 0.85:
        return "Strong Recall"

    if recall_accuracy >= 0.70:
        return "Good Recall"

    if recall_accuracy >= 0.50:
        return "Moderate Recall"

    return "Weak Recall"


# recall delay

def calculate_recall_delay(
    studied_at,
    quiz_at
):
    studied_datetime = (
        parse_datetime(
            studied_at
        )
    )

    quiz_datetime = (
        parse_datetime(
            quiz_at
        )
    )

    if studied_datetime is None:
        raise ValueError(
            "Invalid study datetime."
        )

    if quiz_datetime is None:
        raise ValueError(
            "Invalid quiz datetime."
        )

    try:
        if (
            quiz_datetime
            < studied_datetime
        ):
            raise ValueError(
                "Recall test cannot occur "
                "before the study session."
            )

        difference = (
            quiz_datetime
            - studied_datetime
        )

    except TypeError:
        raise ValueError(
            "Study and quiz datetimes use "
            "incompatible timezone formats."
        )

    delay_seconds = (
        difference.total_seconds()
    )

    return {
        "seconds":
            delay_seconds,

        "minutes":
            delay_seconds / 60,

        "hours":
            delay_seconds / 3600,

        "days":
            delay_seconds / 86400,
    }


# checkpoint matching

def find_nearest_checkpoint(
    delay_days,
    checkpoints=None
):
    try:
        delay_days = float(
            delay_days
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Recall delay must be numeric."
        )

    if delay_days < 0:
        raise ValueError(
            "Recall delay cannot be negative."
        )

    if checkpoints is None:
        checkpoints = (
            RECALL_CHECKPOINT_DAYS
        )

    if not checkpoints:
        return None

    usable_checkpoints = []

    for checkpoint in checkpoints:
        try:
            checkpoint = float(
                checkpoint
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        if checkpoint < 0:
            continue

        usable_checkpoints.append(
            checkpoint
        )

    if not usable_checkpoints:
        return None

    nearest = min(
        usable_checkpoints,
        key=lambda checkpoint:
            abs(
                checkpoint
                - delay_days
            )
    )

    difference = abs(
        nearest
        - delay_days
    )

    return {
        "checkpoint_days":
            nearest,

        "difference_days":
            difference,
    }


def classify_checkpoint(
    delay_days,
    tolerance_days=0.5
):
    try:
        tolerance_days = float(
            tolerance_days
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Checkpoint tolerance must "
            "be numeric."
        )

    if tolerance_days < 0:
        raise ValueError(
            "Checkpoint tolerance cannot "
            "be negative."
        )

    nearest = (
        find_nearest_checkpoint(
            delay_days
        )
    )

    if nearest is None:
        return {
            "matched":
                False,

            "checkpoint_days":
                None,

            "label":
                "Custom Recall",

            "difference_days":
                None,
        }

    if (
        nearest[
            "difference_days"
        ]
        <= tolerance_days
    ):
        checkpoint = (
            nearest[
                "checkpoint_days"
            ]
        )

        return {
            "matched":
                True,

            "checkpoint_days":
                checkpoint,

            "label":
                f"{checkpoint:g}-Day Recall",

            "difference_days":
                nearest[
                    "difference_days"
                ],
        }

    return {
        "matched":
            False,

        "checkpoint_days":
            nearest[
                "checkpoint_days"
            ],

        "label":
            "Custom Recall",

        "difference_days":
            nearest[
                "difference_days"
            ],
    }


# complete quiz result

def create_quiz_result(
    correct_answers,
    total_questions,
    studied_at,
    quiz_at,
    confidence_after_quiz=None
):
    total_questions = (
        validate_question_count(
            total_questions
        )
    )

    correct_answers = (
        validate_correct_answers(
            correct_answers,
            total_questions
        )
    )

    recall_accuracy = (
        calculate_recall_accuracy(
            correct_answers,
            total_questions
        )
    )

    actual_score = (
        recall_accuracy
        * 100
    )

    delay = calculate_recall_delay(
        studied_at,
        quiz_at
    )

    checkpoint = (
        classify_checkpoint(
            delay[
                "days"
            ]
        )
    )

    if (
        confidence_after_quiz
        is not None
    ):
        try:
            confidence_after_quiz = int(
                confidence_after_quiz
            )

        except (
            TypeError,
            ValueError
        ):
            raise ValueError(
                "Quiz confidence must "
                "be a whole number."
            )

        if not (
            1
            <= confidence_after_quiz
            <= 5
        ):
            raise ValueError(
                "Quiz confidence must be "
                "between 1 and 5."
            )

    quiz_datetime = parse_datetime(
        quiz_at
    )

    return {
        "total_questions":
            total_questions,

        "correct_answers":
            correct_answers,

        "incorrect_answers":
            calculate_incorrect_answers(
                correct_answers,
                total_questions
            ),

        "actual_score":
            actual_score,

        "recall_accuracy":
            recall_accuracy,

        "recall_status":
            classify_recall(
                recall_accuracy
            ),

        "delay_days":
            delay[
                "days"
            ],

        "delay_hours":
            delay[
                "hours"
            ],

        "checkpoint":
            checkpoint[
                "label"
            ],

        "checkpoint_days":
            checkpoint[
                "checkpoint_days"
            ],

        "checkpoint_matched":
            checkpoint[
                "matched"
            ],

        "checkpoint_difference_days":
            checkpoint[
                "difference_days"
            ],

        "confidence_after_quiz":
            confidence_after_quiz,

        "quiz_date":
            quiz_datetime.isoformat(),
    }


# question validation

def validate_question(
    question
):
    if not isinstance(
        question,
        dict
    ):
        raise ValueError(
            "Question must be a dictionary."
        )

    question_id = question.get(
        "question_id"
    )

    text = question.get(
        "question"
    )

    correct_answer = (
        question.get(
            "correct_answer"
        )
    )

    if question_id is None:
        raise ValueError(
            "Question ID is required."
        )

    if not str(
        text
    ).strip():
        raise ValueError(
            "Question text is required."
        )

    if correct_answer is None:
        raise ValueError(
            "Correct answer is required."
        )

    return True


def validate_question_bank(
    questions
):
    if not isinstance(
        questions,
        list
    ):
        raise ValueError(
            "Question bank must be a list."
        )

    if not questions:
        raise ValueError(
            "Question bank cannot be empty."
        )

    question_ids = set()

    for question in questions:
        validate_question(
            question
        )

        question_id = question[
            "question_id"
        ]

        if (
            question_id
            in question_ids
        ):
            raise ValueError(
                "Question IDs must be unique."
            )

        question_ids.add(
            question_id
        )

    return True


# question scoring

def score_question(
    question,
    student_answer
):
    validate_question(
        question
    )

    correct_answer = (
        question[
            "correct_answer"
        ]
    )

    is_correct = (
        str(
            student_answer
        ).strip()
        ==
        str(
            correct_answer
        ).strip()
    )

    return {
        "question_id":
            question[
                "question_id"
            ],

        "question":
            question[
                "question"
            ],

        "student_answer":
            student_answer,

        "correct_answer":
            correct_answer,

        "is_correct":
            is_correct,
    }


def score_question_bank(
    questions,
    answers
):
    validate_question_bank(
        questions
    )

    if not isinstance(
        answers,
        dict
    ):
        raise ValueError(
            "Answers must be provided "
            "as a dictionary."
        )

    results = []

    correct_answers = 0

    for question in questions:
        question_id = (
            question[
                "question_id"
            ]
        )

        student_answer = (
            answers.get(
                question_id
            )
        )

        result = score_question(
            question,
            student_answer
        )

        results.append(
            result
        )

        if result[
            "is_correct"
        ]:
            correct_answers += 1

    total_questions = len(
        questions
    )

    recall_accuracy = (
        correct_answers
        / total_questions
    )

    return {
        "total_questions":
            total_questions,

        "correct_answers":
            correct_answers,

        "incorrect_answers":
            (
                total_questions
                - correct_answers
            ),

        "recall_accuracy":
            recall_accuracy,

        "actual_score":
            recall_accuracy
            * 100,

        "recall_status":
            classify_recall(
                recall_accuracy
            ),

        "question_results":
            results,
    }


# prediction timing

def filter_predictions_before_quiz(
    predictions,
    quiz_at
):
    quiz_datetime = (
        parse_datetime(
            quiz_at
        )
    )

    if quiz_datetime is None:
        raise ValueError(
            "Invalid quiz datetime."
        )

    if not predictions:
        return []

    valid_predictions = []

    for prediction in predictions:
        if not isinstance(
            prediction,
            dict
        ):
            continue

        created_at = (
            prediction.get(
                "created_at"
            )
        )

        prediction_datetime = (
            parse_datetime(
                created_at
            )
        )

        if prediction_datetime is None:
            continue

        try:
            valid = (
                prediction_datetime
                <= quiz_datetime
            )

        except TypeError:
            continue

        if valid:
            valid_predictions.append(
                prediction
            )

    return valid_predictions


# prediction interval matching

def find_closest_prediction(
    predictions,
    delay_days
):
    try:
        delay_days = float(
            delay_days
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Recall delay must be numeric."
        )

    if delay_days < 0:
        raise ValueError(
            "Recall delay cannot be negative."
        )

    usable_predictions = []

    for prediction in predictions:
        if not isinstance(
            prediction,
            dict
        ):
            continue

        prediction_days = (
            prediction.get(
                "prediction_for_days"
            )
        )

        if prediction_days is None:
            continue

        try:
            prediction_days = float(
                prediction_days
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        if prediction_days < 0:
            continue

        prediction_copy = (
            prediction.copy()
        )

        prediction_copy[
            "prediction_for_days"
        ] = prediction_days

        prediction_copy[
            "_interval_difference"
        ] = abs(
            prediction_days
            - delay_days
        )

        prediction_copy[
            "_created_at"
        ] = parse_datetime(
            prediction.get(
                "created_at"
            )
        )

        usable_predictions.append(
            prediction_copy
        )

    if not usable_predictions:
        return None

    minimum_difference = min(
        prediction[
            "_interval_difference"
        ]
        for prediction
        in usable_predictions
    )

    closest_predictions = [
        prediction
        for prediction
        in usable_predictions
        if abs(
            prediction[
                "_interval_difference"
            ]
            - minimum_difference
        ) < 1e-12
    ]

    dated_predictions = [
        prediction
        for prediction
        in closest_predictions
        if prediction.get(
            "_created_at"
        ) is not None
    ]

    if dated_predictions:
        try:
            selected = max(
                dated_predictions,
                key=lambda prediction:
                    prediction[
                        "_created_at"
                    ]
            )

        except TypeError:
            selected = (
                closest_predictions[
                    0
                ]
            )

    else:
        selected = (
            closest_predictions[
                0
            ]
        )

    selected.pop(
        "_interval_difference",
        None
    )

    selected.pop(
        "_created_at",
        None
    )

    return selected


def match_prediction_to_recall(
    predictions,
    quiz_at,
    delay_days,
    tolerance_days=0.5
):
    try:
        tolerance_days = float(
            tolerance_days
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Prediction matching tolerance "
            "must be numeric."
        )

    if tolerance_days < 0:
        raise ValueError(
            "Prediction matching tolerance "
            "cannot be negative."
        )

    try:
        delay_days = float(
            delay_days
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Recall delay must be numeric."
        )

    if delay_days < 0:
        raise ValueError(
            "Recall delay cannot be negative."
        )

    valid_predictions = (
        filter_predictions_before_quiz(
            predictions,
            quiz_at
        )
    )

    prediction = (
        find_closest_prediction(
            valid_predictions,
            delay_days
        )
    )

    if prediction is None:
        return {
            "matched":
                False,

            "strict_match":
                False,

            "prediction":
                None,

            "interval_difference":
                None,

            "message":
                (
                    "No usable prediction was "
                    "created before the recall test."
                ),
        }

    interval_difference = abs(
        prediction[
            "prediction_for_days"
        ]
        - delay_days
    )

    strict_match = (
        interval_difference
        <= tolerance_days
    )

    return {
        "matched":
            True,

        "strict_match":
            strict_match,

        "prediction":
            prediction,

        "interval_difference":
            interval_difference,

        "message":
            (
                "Prediction matched successfully."
                if strict_match
                else
                (
                    "A prediction exists, but its "
                    "interval differs too much from "
                    "the actual recall delay."
                )
            ),
    }


# prediction comparison

def compare_single_prediction(
    predicted_recall,
    actual_recall
):
    if predicted_recall is None:
        return None

    try:
        predicted_recall = float(
            predicted_recall
        )

        actual_recall = float(
            actual_recall
        )

    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            "Predicted and actual recall "
            "must be numeric."
        )

    if not (
        0
        <= predicted_recall
        <= 1
    ):
        raise ValueError(
            "Predicted recall must be "
            "between 0 and 1."
        )

    if not (
        0
        <= actual_recall
        <= 1
    ):
        raise ValueError(
            "Actual recall must be "
            "between 0 and 1."
        )

    signed_error = (
        predicted_recall
        - actual_recall
    )

    absolute_error = (
        calculate_prediction_error(
            predicted_recall,
            actual_recall
        )
    )

    if signed_error > 0:
        bias_direction = (
            "Overprediction"
        )

    elif signed_error < 0:
        bias_direction = (
            "Underprediction"
        )

    else:
        bias_direction = (
            "Exact"
        )

    return {
        "predicted_recall":
            predicted_recall,

        "actual_recall":
            actual_recall,

        "signed_error":
            signed_error,

        "absolute_error":
            absolute_error,

        "absolute_error_pp":
            absolute_error
            * 100,

        "bias_direction":
            bias_direction,
    }


def compare_prediction_methods(
    prediction,
    actual_recall
):
    if prediction is None:
        return []

    methods = [
        (
            "Fixed Forgetting Curve",
            "fixed_prediction"
        ),
        (
            "Personalized Model",
            "personalized_prediction"
        ),
        (
            "Adaptive Schedule",
            "adaptive_prediction"
        ),
    ]

    results = []

    for (
        model_name,
        column
    ) in methods:

        predicted_value = (
            prediction.get(
                column
            )
        )

        if predicted_value is None:
            continue

        try:
            comparison = (
                compare_single_prediction(
                    predicted_value,
                    actual_recall
                )
            )

        except ValueError:
            continue

        if comparison is None:
            continue

        comparison[
            "model"
        ] = model_name

        results.append(
            comparison
        )

    results.sort(
        key=lambda result:
            result[
                "absolute_error"
            ]
    )

    return results


# complete experiment evaluation

def evaluate_recall_test(
    predictions,
    studied_at,
    quiz_at,
    correct_answers,
    total_questions,
    confidence_after_quiz=None,
    tolerance_days=0.5
):
    quiz_result = (
        create_quiz_result(
            correct_answers=(
                correct_answers
            ),
            total_questions=(
                total_questions
            ),
            studied_at=studied_at,
            quiz_at=quiz_at,
            confidence_after_quiz=(
                confidence_after_quiz
            )
        )
    )

    match = (
        match_prediction_to_recall(
            predictions=predictions,
            quiz_at=quiz_at,
            delay_days=(
                quiz_result[
                    "delay_days"
                ]
            ),
            tolerance_days=(
                tolerance_days
            )
        )
    )

    comparisons = []

    if (
        match.get(
            "prediction"
        )
        is not None
    ):
        comparisons = (
            compare_prediction_methods(
                match[
                    "prediction"
                ],
                quiz_result[
                    "recall_accuracy"
                ]
            )
        )

    exploratory_winner = None
    winner = None

    if comparisons:
        exploratory_winner = (
            comparisons[
                0
            ]
        )

    if (
        comparisons
        and match.get(
            "strict_match",
            False
        )
    ):
        winner = (
            comparisons[
                0
            ]
        )

    return {
        "quiz_result":
            quiz_result,

        "prediction_match":
            match,

        "comparisons":
            comparisons,

        "winner":
            winner,

        "exploratory_winner":
            exploratory_winner,

        "valid_for_strict_evaluation":
            match.get(
                "strict_match",
                False
            ),
    }


# database payload

def create_database_payload(
    student_id,
    topic_id,
    session_id,
    quiz_result
):
    if not isinstance(
        quiz_result,
        dict
    ):
        raise ValueError(
            "Quiz result must be "
            "a dictionary."
        )

    required_fields = [
        "total_questions",
        "correct_answers",
        "delay_days",
        "quiz_date",
    ]

    missing = [
        field
        for field in required_fields
        if quiz_result.get(
            field
        ) is None
    ]

    if missing:
        raise ValueError(
            "Quiz result is missing: "
            + ", ".join(
                missing
            )
        )

    return {
        "student_id":
            student_id,

        "topic_id":
            topic_id,

        "session_id":
            session_id,

        "total_questions":
            quiz_result[
                "total_questions"
            ],

        "correct_answers":
            quiz_result[
                "correct_answers"
            ],

        "delay_days":
            quiz_result[
                "delay_days"
            ],

        "confidence_after_quiz":
            quiz_result.get(
                "confidence_after_quiz"
            ),

        "quiz_date":
            quiz_result[
                "quiz_date"
            ],
    }


# recall summary

def summarize_recall_results(
    quiz_results
):
    if not quiz_results:
        return {
            "tests":
                0,

            "valid_recall_tests":
                0,

            "average_recall":
                None,

            "highest_recall":
                None,

            "lowest_recall":
                None,

            "average_delay":
                None,
        }

    recalls = []
    delays = []

    for quiz in quiz_results:
        if not isinstance(
            quiz,
            dict
        ):
            continue

        recall = quiz.get(
            "recall_accuracy"
        )

        delay = quiz.get(
            "delay_days"
        )

        if recall is not None:
            try:
                recall = float(
                    recall
                )

                if (
                    0
                    <= recall
                    <= 1
                ):
                    recalls.append(
                        recall
                    )

            except (
                TypeError,
                ValueError
            ):
                pass

        if delay is not None:
            try:
                delay = float(
                    delay
                )

                if delay >= 0:
                    delays.append(
                        delay
                    )

            except (
                TypeError,
                ValueError
            ):
                pass

    return {
        "tests":
            len(
                quiz_results
            ),

        "valid_recall_tests":
            len(
                recalls
            ),

        "average_recall":
            (
                sum(
                    recalls
                )
                / len(
                    recalls
                )
                if recalls
                else None
            ),

        "highest_recall":
            (
                max(
                    recalls
                )
                if recalls
                else None
            ),

        "lowest_recall":
            (
                min(
                    recalls
                )
                if recalls
                else None
            ),

        "average_delay":
            (
                sum(
                    delays
                )
                / len(
                    delays
                )
                if delays
                else None
            ),
    }


# direct execution

if __name__ == "__main__":

    print(
        "MemoryMap Quiz Engine"
    )

    print(
        "=" * 40
    )

    example_study_time = (
        "2026-08-20T10:00:00"
    )

    example_quiz_time = (
        "2026-08-23T10:30:00"
    )

    result = create_quiz_result(
        correct_answers=8,
        total_questions=10,
        studied_at=(
            example_study_time
        ),
        quiz_at=(
            example_quiz_time
        ),
        confidence_after_quiz=4
    )

    print(
        "Recall Score:",
        f"{result['actual_score']:.2f}%"
    )

    print(
        "Recall Accuracy:",
        f"{result['recall_accuracy']:.3f}"
    )

    print(
        "Delay:",
        f"{result['delay_days']:.2f} days"
    )

    print(
        "Checkpoint:",
        result[
            "checkpoint"
        ]
    )

    print(
        "Recall Status:",
        result[
            "recall_status"
        ]
    )

    print()

    example_questions = [
        {
            "question_id":
                1,

            "question":
                (
                    "Which data structure "
                    "uses FIFO order?"
                ),

            "correct_answer":
                "Queue",
        },
        {
            "question_id":
                2,

            "question":
                (
                    "Which traversal uses "
                    "a queue?"
                ),

            "correct_answer":
                "BFS",
        },
    ]

    example_answers = {
        1:
            "Queue",

        2:
            "BFS",
    }

    scored = score_question_bank(
        example_questions,
        example_answers
    )

    print(
        "Question Bank Score:",
        f"{scored['actual_score']:.2f}%"
    )

    print()

    example_predictions = [
        {
            "prediction_id":
                1,

            "prediction_for_days":
                3,

            "created_at":
                "2026-08-20T12:00:00",

            "fixed_prediction":
                0.70,

            "personalized_prediction":
                0.75,

            "adaptive_prediction":
                0.78,
        },
        {
            "prediction_id":
                2,

            "prediction_for_days":
                3,

            "created_at":
                "2026-08-22T12:00:00",

            "fixed_prediction":
                0.71,

            "personalized_prediction":
                0.79,

            "adaptive_prediction":
                0.81,
        },
    ]

    matched = (
        match_prediction_to_recall(
            predictions=(
                example_predictions
            ),
            quiz_at=(
                "2026-08-23T10:30:00"
            ),
            delay_days=3
        )
    )

    print(
        "Matched Prediction ID:",
        matched[
            "prediction"
        ][
            "prediction_id"
        ]
    )

    print(
        "Strict Match:",
        matched[
            "strict_match"
        ]
    )