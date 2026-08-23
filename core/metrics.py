import math

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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


def validate_numeric_sequence(
    values,
    name="values"
):
    if values is None:
        raise ValueError(
            f"{name} cannot be None."
        )

    try:
        values = list(
            values
        )

    except TypeError:
        raise ValueError(
            f"{name} must be an iterable "
            "of numeric values."
        )

    if not values:
        raise ValueError(
            f"{name} cannot be empty."
        )

    cleaned = []

    for value in values:
        number = safe_float(
            value
        )

        if number is None:
            continue

        cleaned.append(
            number
        )

    if not cleaned:
        raise ValueError(
            f"{name} contains no valid "
            "numeric values."
        )

    return cleaned


def validate_prediction_pairs(
    actual,
    predicted
):
    if actual is None:
        raise ValueError(
            "Actual values cannot be None."
        )

    if predicted is None:
        raise ValueError(
            "Predicted values cannot be None."
        )

    try:
        actual = list(
            actual
        )

        predicted = list(
            predicted
        )

    except TypeError:
        raise ValueError(
            "Actual and predicted values "
            "must be iterable."
        )

    if len(
        actual
    ) != len(
        predicted
    ):
        raise ValueError(
            "Actual and predicted values "
            "must have the same length."
        )

    valid_actual = []
    valid_predicted = []

    for (
        actual_value,
        predicted_value
    ) in zip(
        actual,
        predicted
    ):
        actual_number = safe_float(
            actual_value
        )

        predicted_number = safe_float(
            predicted_value
        )

        if (
            actual_number is None
            or predicted_number is None
        ):
            continue

        valid_actual.append(
            actual_number
        )

        valid_predicted.append(
            predicted_number
        )

    if not valid_actual:
        raise ValueError(
            "No valid prediction pairs "
            "are available."
        )

    return (
        valid_actual,
        valid_predicted
    )


# single prediction errors

def calculate_error(
    predicted,
    actual
):
    predicted = safe_float(
        predicted
    )

    actual = safe_float(
        actual
    )

    if (
        predicted is None
        or actual is None
    ):
        raise ValueError(
            "Predicted and actual values "
            "must be finite numbers."
        )

    return (
        predicted
        - actual
    )


def calculate_absolute_error(
    predicted,
    actual
):
    return abs(
        calculate_error(
            predicted,
            actual
        )
    )


def calculate_squared_error(
    predicted,
    actual
):
    error = calculate_error(
        predicted,
        actual
    )

    return (
        error ** 2
    )


def calculate_percentage_point_error(
    predicted,
    actual
):
    return (
        calculate_absolute_error(
            predicted,
            actual
        )
        * 100
    )


# mean absolute error

def calculate_mae(
    actual,
    predicted
):
    actual, predicted = (
        validate_prediction_pairs(
            actual,
            predicted
        )
    )

    return float(
        mean_absolute_error(
            actual,
            predicted
        )
    )


# root mean squared error

def calculate_rmse(
    actual,
    predicted
):
    actual, predicted = (
        validate_prediction_pairs(
            actual,
            predicted
        )
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    return float(
        math.sqrt(
            mse
        )
    )


# r squared

def calculate_r2(
    actual,
    predicted
):
    actual, predicted = (
        validate_prediction_pairs(
            actual,
            predicted
        )
    )

    if len(
        actual
    ) < 2:
        return None

    # R² is not meaningful when
    # actual observations have no variance.

    if math.isclose(
        float(
            np.std(
                actual
            )
        ),
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12
    ):
        return None

    try:
        value = r2_score(
            actual,
            predicted
        )

    except ValueError:
        return None

    value = safe_float(
        value
    )

    return value


# prediction bias

def calculate_bias(
    actual,
    predicted
):
    actual, predicted = (
        validate_prediction_pairs(
            actual,
            predicted
        )
    )

    errors = [
        predicted_value
        - actual_value

        for (
            actual_value,
            predicted_value
        ) in zip(
            actual,
            predicted
        )
    ]

    return float(
        sum(
            errors
        )
        / len(
            errors
        )
    )


def classify_bias(
    bias,
    tolerance=0.01
):
    bias = safe_float(
        bias
    )

    tolerance = safe_float(
        tolerance
    )

    if bias is None:
        return "Unavailable"

    if tolerance is None:
        raise ValueError(
            "Bias tolerance must be numeric."
        )

    tolerance = abs(
        tolerance
    )

    if abs(
        bias
    ) <= tolerance:
        return "Balanced"

    if bias > 0:
        return "Overprediction"

    return "Underprediction"


# correlation

def calculate_correlation(
    x_values,
    y_values
):
    if (
        x_values is None
        or y_values is None
    ):
        return None

    try:
        x_values = list(
            x_values
        )

        y_values = list(
            y_values
        )

    except TypeError:
        return None

    if len(
        x_values
    ) != len(
        y_values
    ):
        raise ValueError(
            "Both variables must contain "
            "the same number of values."
        )

    clean_x = []
    clean_y = []

    for (
        x_value,
        y_value
    ) in zip(
        x_values,
        y_values
    ):
        x_number = safe_float(
            x_value
        )

        y_number = safe_float(
            y_value
        )

        if (
            x_number is None
            or y_number is None
        ):
            continue

        clean_x.append(
            x_number
        )

        clean_y.append(
            y_number
        )

    if len(
        clean_x
    ) < 2:
        return None

    if (
        math.isclose(
            float(
                np.std(
                    clean_x
                )
            ),
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-12
        )
        or
        math.isclose(
            float(
                np.std(
                    clean_y
                )
            ),
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-12
        )
    ):
        return None

    correlation = np.corrcoef(
        clean_x,
        clean_y
    )[0, 1]

    correlation = safe_float(
        correlation
    )

    return correlation


def classify_correlation(
    correlation
):
    correlation = safe_float(
        correlation
    )

    if correlation is None:
        return "Unavailable"

    correlation = max(
        -1.0,
        min(
            1.0,
            correlation
        )
    )

    strength = abs(
        correlation
    )

    if strength < 0.20:
        level = "Very Weak"

    elif strength < 0.40:
        level = "Weak"

    elif strength < 0.60:
        level = "Moderate"

    elif strength < 0.80:
        level = "Strong"

    else:
        level = "Very Strong"

    if correlation > 0:
        direction = "Positive"

    elif correlation < 0:
        direction = "Negative"

    else:
        direction = "No Direction"

    return (
        f"{level} {direction}"
    )


# accuracy bands

def calculate_accuracy_band(
    actual,
    predicted,
    tolerance=0.10
):
    actual, predicted = (
        validate_prediction_pairs(
            actual,
            predicted
        )
    )

    tolerance = safe_float(
        tolerance
    )

    if tolerance is None:
        raise ValueError(
            "Accuracy tolerance must "
            "be numeric."
        )

    if tolerance < 0:
        raise ValueError(
            "Accuracy tolerance cannot "
            "be negative."
        )

    within_tolerance = 0

    for (
        actual_value,
        predicted_value
    ) in zip(
        actual,
        predicted
    ):
        error = abs(
            predicted_value
            - actual_value
        )

        if error <= tolerance:
            within_tolerance += 1

    return float(
        within_tolerance
        / len(
            actual
        )
    )


def calculate_accuracy_bands(
    actual,
    predicted
):
    return {
        "within_5pp":
            calculate_accuracy_band(
                actual,
                predicted,
                tolerance=0.05
            ),

        "within_10pp":
            calculate_accuracy_band(
                actual,
                predicted,
                tolerance=0.10
            ),

        "within_15pp":
            calculate_accuracy_band(
                actual,
                predicted,
                tolerance=0.15
            ),
    }


# complete model metrics

def calculate_model_metrics(
    actual,
    predicted,
    model_name="Model"
):
    actual, predicted = (
        validate_prediction_pairs(
            actual,
            predicted
        )
    )

    mae = calculate_mae(
        actual,
        predicted
    )

    rmse = calculate_rmse(
        actual,
        predicted
    )

    r2 = calculate_r2(
        actual,
        predicted
    )

    bias = calculate_bias(
        actual,
        predicted
    )

    accuracy_bands = (
        calculate_accuracy_bands(
            actual,
            predicted
        )
    )

    return {
        "model":
            str(
                model_name
            ),

        "observations":
            len(
                actual
            ),

        "mae":
            mae,

        "mae_pp":
            mae
            * 100,

        "rmse":
            rmse,

        "rmse_pp":
            rmse
            * 100,

        "r2":
            r2,

        "bias":
            bias,

        "bias_pp":
            bias
            * 100,

        "bias_direction":
            classify_bias(
                bias
            ),

        "within_5pp":
            accuracy_bands[
                "within_5pp"
            ],

        "within_10pp":
            accuracy_bands[
                "within_10pp"
            ],

        "within_15pp":
            accuracy_bands[
                "within_15pp"
            ],
    }


# record-level comparison

def create_error_records(
    actual,
    predicted
):
    actual, predicted = (
        validate_prediction_pairs(
            actual,
            predicted
        )
    )

    records = []

    for index, (
        actual_value,
        predicted_value
    ) in enumerate(
        zip(
            actual,
            predicted
        ),
        start=1
    ):
        signed_error = (
            predicted_value
            - actual_value
        )

        absolute_error = abs(
            signed_error
        )

        squared_error = (
            signed_error ** 2
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
                "record":
                    index,

                "actual":
                    actual_value,

                "predicted":
                    predicted_value,

                "signed_error":
                    signed_error,

                "absolute_error":
                    absolute_error,

                "absolute_error_pp":
                    absolute_error
                    * 100,

                "squared_error":
                    squared_error,

                "bias_direction":
                    direction,
            }
        )

    return records


# model improvement

def calculate_improvement(
    baseline_error,
    new_error
):
    baseline_error = safe_float(
        baseline_error
    )

    new_error = safe_float(
        new_error
    )

    if baseline_error is None:
        raise ValueError(
            "Baseline error must be "
            "a finite number."
        )

    if new_error is None:
        raise ValueError(
            "New error must be "
            "a finite number."
        )

    if baseline_error < 0:
        raise ValueError(
            "Baseline error cannot "
            "be negative."
        )

    if new_error < 0:
        raise ValueError(
            "New error cannot "
            "be negative."
        )

    equal = math.isclose(
        baseline_error,
        new_error,
        rel_tol=1e-9,
        abs_tol=1e-9
    )

    improvement = (
        baseline_error
        - new_error
    )

    if baseline_error == 0:
        return {
            "absolute_improvement":
                improvement,

            "percentage_improvement":
                None,

            "improved":
                False,

            "equal":
                equal,

            "baseline_zero":
                True,
        }

    percentage_improvement = (
        improvement
        / baseline_error
        * 100
    )

    return {
        "absolute_improvement":
            improvement,

        "percentage_improvement":
            percentage_improvement,

        "improved":
            improvement > 0,

        "equal":
            equal,

        "baseline_zero":
            False,
    }


# model ranking

def rank_models(
    model_metrics,
    metric="mae"
):
    if not model_metrics:
        return []

    supported_metrics = {
        "mae",
        "rmse",
        "r2",
        "bias",
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported ranking metric: "
            f"{metric}"
        )

    usable = []

    for metrics in model_metrics:
        if not isinstance(
            metrics,
            dict
        ):
            continue

        value = safe_float(
            metrics.get(
                metric
            )
        )

        if value is None:
            continue

        item = metrics.copy()

        item[
            "_ranking_value"
        ] = value

        usable.append(
            item
        )

    if metric in {
        "mae",
        "rmse",
    }:
        usable.sort(
            key=lambda item:
                item[
                    "_ranking_value"
                ]
        )

    elif metric == "bias":
        usable.sort(
            key=lambda item:
                abs(
                    item[
                        "_ranking_value"
                    ]
                )
        )

    else:
        usable.sort(
            key=lambda item:
                item[
                    "_ranking_value"
                ],
            reverse=True
        )

    ranked = []

    previous_value = None
    previous_rank = 0

    for index, item in enumerate(
        usable,
        start=1
    ):
        ranking_value = item[
            "_ranking_value"
        ]

        if (
            previous_value is not None
            and
            math.isclose(
                ranking_value,
                previous_value,
                rel_tol=1e-9,
                abs_tol=1e-9
            )
        ):
            rank = previous_rank

        else:
            rank = index

        item.pop(
            "_ranking_value",
            None
        )

        item[
            "rank"
        ] = rank

        ranked.append(
            item
        )

        previous_value = (
            ranking_value
        )

        previous_rank = rank

    return ranked


# best model

def get_best_model(
    model_metrics,
    metric="mae"
):
    ranked = rank_models(
        model_metrics,
        metric
    )

    if not ranked:
        return None

    return ranked[
        0
    ]


# multi-model evaluation

def evaluate_models(
    actual,
    predictions_by_model
):
    if not isinstance(
        predictions_by_model,
        dict
    ):
        raise ValueError(
            "Predictions must be provided "
            "as a dictionary."
        )

    results = []

    for (
        model_name,
        predictions
    ) in predictions_by_model.items():

        if predictions is None:
            continue

        try:
            metrics = (
                calculate_model_metrics(
                    actual,
                    predictions,
                    model_name
                )
            )

        except ValueError:
            continue

        results.append(
            metrics
        )

    return results


# baseline comparison

def compare_with_baseline(
    fixed_metrics,
    candidate_metrics
):
    if (
        fixed_metrics is None
        or candidate_metrics is None
    ):
        return None

    fixed_mae = safe_float(
        fixed_metrics.get(
            "mae"
        )
    )

    candidate_mae = safe_float(
        candidate_metrics.get(
            "mae"
        )
    )

    if (
        fixed_mae is None
        or candidate_mae is None
    ):
        return None

    improvement = (
        calculate_improvement(
            fixed_mae,
            candidate_mae
        )
    )

    baseline_observations = (
        fixed_metrics.get(
            "observations"
        )
    )

    candidate_observations = (
        candidate_metrics.get(
            "observations"
        )
    )

    fair_observation_count = (
        baseline_observations
        ==
        candidate_observations
    )

    return {
        "baseline_model":
            fixed_metrics.get(
                "model",
                "Fixed Baseline"
            ),

        "candidate_model":
            candidate_metrics.get(
                "model",
                "Candidate Model"
            ),

        "baseline_observations":
            baseline_observations,

        "candidate_observations":
            candidate_observations,

        "same_observation_count":
            fair_observation_count,

        "baseline_mae":
            fixed_mae,

        "candidate_mae":
            candidate_mae,

        "baseline_mae_pp":
            fixed_mae
            * 100,

        "candidate_mae_pp":
            candidate_mae
            * 100,

        "absolute_improvement":
            improvement.get(
                "absolute_improvement"
            ),

        "absolute_improvement_pp":
            (
                improvement.get(
                    "absolute_improvement"
                )
                * 100
            ),

        "percentage_improvement":
            improvement.get(
                "percentage_improvement"
            ),

        "improved":
            improvement.get(
                "improved",
                False
            ),

        "equal":
            improvement.get(
                "equal",
                False
            ),
    }


# sample size classification

def classify_sample_size(
    observations
):
    try:
        observations = int(
            observations
        )

    except (
        TypeError,
        ValueError
    ):
        return "No Data"

    if observations <= 0:
        return "No Data"

    if observations < 5:
        return "Insufficient"

    if observations < 20:
        return "Preliminary"

    if observations < 50:
        return "Developing"

    return "Stronger Evidence"


# research verdict

def create_research_verdict(
    model_metrics,
    baseline_name=(
        "Fixed Forgetting Curve"
    ),
    minimum_observations=5
):
    if not model_metrics:
        return {
            "winner":
                None,

            "status":
                "No Data",

            "tie":
                False,

            "tied_models":
                [],

            "comparison_to_baseline":
                None,

            "message":
                (
                    "No model comparison "
                    "is available yet."
                ),
        }

    try:
        minimum_observations = int(
            minimum_observations
        )

    except (
        TypeError,
        ValueError
    ):
        minimum_observations = 5

    minimum_observations = max(
        minimum_observations,
        1
    )

    ranked = rank_models(
        model_metrics,
        metric="mae"
    )

    if not ranked:
        return {
            "winner":
                None,

            "status":
                "No Data",

            "tie":
                False,

            "tied_models":
                [],

            "comparison_to_baseline":
                None,

            "message":
                (
                    "No usable prediction "
                    "metrics are available."
                ),
        }

    if len(
        ranked
    ) < 2:
        winner = ranked[
            0
        ]

        observations = (
            winner.get(
                "observations",
                0
            )
        )

        return {
            "winner":
                winner,

            "status":
                "Insufficient Comparison",

            "tie":
                False,

            "tied_models":
                [],

            "comparison_to_baseline":
                None,

            "message":
                (
                    f"{winner.get('model', 'The model')} "
                    "has usable metrics, but no second "
                    "model is available for comparison."
                ),

            "observations":
                observations,
        }

    winner = ranked[
        0
    ]

    observations = winner.get(
        "observations",
        0
    )

    try:
        observations = int(
            observations
        )

    except (
        TypeError,
        ValueError
    ):
        observations = 0

    sample_status = (
        classify_sample_size(
            observations
        )
    )

    winner_mae = safe_float(
        winner.get(
            "mae"
        )
    )

    tied_models = []

    if winner_mae is not None:
        for result in ranked:
            result_mae = safe_float(
                result.get(
                    "mae"
                )
            )

            if result_mae is None:
                continue

            if math.isclose(
                result_mae,
                winner_mae,
                rel_tol=1e-9,
                abs_tol=1e-9
            ):
                tied_models.append(
                    result.get(
                        "model"
                    )
                )

    tied_models = [
        name
        for name in tied_models
        if name is not None
    ]

    tie = (
        len(
            tied_models
        ) > 1
    )

    baseline = next(
        (
            metrics
            for metrics
            in model_metrics

            if isinstance(
                metrics,
                dict
            )

            and metrics.get(
                "model"
            )
            == baseline_name
        ),
        None
    )

    comparison = None

    if (
        baseline is not None
        and winner is not None
    ):
        comparison = (
            compare_with_baseline(
                baseline,
                winner
            )
        )

    observation_counts = set()

    for result in ranked:
        count = result.get(
            "observations"
        )

        try:
            count = int(
                count
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        observation_counts.add(
            count
        )

    same_observation_count = (
        len(
            observation_counts
        ) <= 1
    )

    if tie:
        names = ", ".join(
            tied_models
        )

        if observations < (
            minimum_observations
        ):
            message = (
                f"{names} currently have the same "
                f"lowest MAE, but only {observations} "
                "matched observation(s) are available."
            )

        else:
            message = (
                f"{names} currently have the same "
                f"lowest MAE at "
                f"{winner['mae_pp']:.2f} "
                "percentage points."
            )

    elif observations < (
        minimum_observations
    ):
        message = (
            f"{winner['model']} currently has "
            f"the lowest MAE, but only "
            f"{observations} matched "
            "observation(s) are available."
        )

    else:
        message = (
            f"{winner['model']} currently "
            f"has the lowest MAE at "
            f"{winner['mae_pp']:.2f} "
            "percentage points."
        )

    if not same_observation_count:
        message += (
            " The models were not evaluated on "
            "identical observation counts, so this "
            "ranking should not be treated as a fair "
            "head-to-head comparison."
        )

    return {
        "winner":
            winner,

        "status":
            sample_status,

        "tie":
            tie,

        "tied_models":
            tied_models,

        "observations":
            observations,

        "same_observation_count":
            same_observation_count,

        "comparison_to_baseline":
            comparison,

        "message":
            message,
    }


# metric formatting

def format_metric_value(
    value,
    decimals=3
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    return round(
        value,
        decimals
    )


def create_display_metrics(
    metrics
):
    if metrics is None:
        return None

    mae_pp = safe_float(
        metrics.get(
            "mae_pp"
        ),
        0
    )

    rmse_pp = safe_float(
        metrics.get(
            "rmse_pp"
        ),
        0
    )

    bias_pp = safe_float(
        metrics.get(
            "bias_pp"
        ),
        0
    )

    within_5 = safe_float(
        metrics.get(
            "within_5pp"
        ),
        0
    )

    within_10 = safe_float(
        metrics.get(
            "within_10pp"
        ),
        0
    )

    within_15 = safe_float(
        metrics.get(
            "within_15pp"
        ),
        0
    )

    return {
        "Model":
            metrics.get(
                "model"
            ),

        "Observations":
            metrics.get(
                "observations"
            ),

        "MAE (pp)":
            round(
                mae_pp,
                2
            ),

        "RMSE (pp)":
            round(
                rmse_pp,
                2
            ),

        "R²":
            format_metric_value(
                metrics.get(
                    "r2"
                ),
                3
            ),

        "Bias (pp)":
            round(
                bias_pp,
                2
            ),

        "Bias Direction":
            metrics.get(
                "bias_direction"
            ),

        "Within 5pp (%)":
            round(
                within_5
                * 100,
                1
            ),

        "Within 10pp (%)":
            round(
                within_10
                * 100,
                1
            ),

        "Within 15pp (%)":
            round(
                within_15
                * 100,
                1
            ),
    }


# direct execution

if __name__ == "__main__":

    print(
        "MemoryMap Metrics"
    )

    print(
        "=" * 40
    )

    actual = [
        0.80,
        0.62,
        0.91,
        0.73,
    ]

    fixed = [
        0.59,
        0.51,
        0.70,
        0.60,
    ]

    personalized = [
        0.76,
        0.65,
        0.87,
        0.70,
    ]

    adaptive = [
        0.78,
        0.64,
        0.89,
        0.72,
    ]

    fixed_metrics = (
        calculate_model_metrics(
            actual,
            fixed,
            "Fixed Forgetting Curve"
        )
    )

    personalized_metrics = (
        calculate_model_metrics(
            actual,
            personalized,
            "Personalized Model"
        )
    )

    adaptive_metrics = (
        calculate_model_metrics(
            actual,
            adaptive,
            "Adaptive Schedule"
        )
    )

    print(
        "Fixed MAE:",
        f"{fixed_metrics['mae_pp']:.2f} pp"
    )

    print(
        "Personalized MAE:",
        (
            f"{personalized_metrics['mae_pp']:.2f} pp"
        )
    )

    print(
        "Adaptive MAE:",
        (
            f"{adaptive_metrics['mae_pp']:.2f} pp"
        )
    )

    print()

    comparison = (
        compare_with_baseline(
            fixed_metrics,
            personalized_metrics
        )
    )

    if comparison is not None:
        percentage = (
            comparison.get(
                "percentage_improvement"
            )
        )

        if percentage is not None:
            print(
                "Personalized Improvement:",
                f"{percentage:.2f}%"
            )

    print()

    verdict = (
        create_research_verdict(
            [
                fixed_metrics,
                personalized_metrics,
                adaptive_metrics,
            ]
        )
    )

    winner = verdict.get(
        "winner"
    )

    if winner is not None:
        print(
            "Current Winner:",
            winner.get(
                "model"
            )
        )

    print(
        "Research Status:",
        verdict.get(
            "status"
        )
    )

    print(
        verdict.get(
            "message"
        )
    )