import math

import pandas as pd
import plotly.express as px
import streamlit as st

from database import (
    initialize_database,
)

from services.student_service import (
    list_students,
)

from services.analytics_service import (
    MODEL_COLUMNS,
    get_student_analytics,
)


st.set_page_config(
    page_title="Model Comparison | MemoryMap",
    page_icon="⚖️",
    layout="wide",
)


# initialization

initialize_database()

STRICT_TOLERANCE_DAYS = 0.5

if "student_id" not in st.session_state:
    st.session_state.student_id = None

if "student_name" not in st.session_state:
    st.session_state.student_name = None

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = None


# helpers

def safe_float(
    value,
    default=None
):
    if value is None:
        return default

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(value):
        return default

    return value


def safe_int(
    value,
    default=0
):
    if value is None:
        return default

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def ensure_dataframe(
    value
):
    if isinstance(
        value,
        pd.DataFrame
    ):
        return value.copy()

    if value is None:
        return pd.DataFrame()

    try:
        return pd.DataFrame(
            value
        )

    except Exception:
        return pd.DataFrame()


def get_record_id(
    record,
    canonical_name
):
    if not record:
        return None

    value = record.get(
        canonical_name
    )

    if value is None:
        value = record.get(
            "id"
        )

    return value


def format_pp(
    value,
    decimals=2
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    return f"{value:.{decimals}f} pp"


def format_percent(
    value,
    decimals=1
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    return f"{value * 100:.{decimals}f}%"


def format_number(
    value,
    decimals=3
):
    value = safe_float(
        value
    )

    if value is None:
        return "N/A"

    return f"{value:.{decimals}f}"


# student selection

def select_student():
    students = list_students()

    if not students:
        st.warning(
            "No student profiles exist yet."
        )

        st.info(
            "Create a student profile from "
            "the Dashboard first."
        )

        st.page_link(
            "pages/1_Dashboard.py",
            label="🧠 Open Dashboard"
        )

        return None

    student_map = {}

    for student in students:
        student_id = get_record_id(
            student,
            "student_id"
        )

        if student_id is None:
            continue

        name = (
            student.get(
                "name"
            )
            or "Unnamed Student"
        )

        email = student.get(
            "email"
        )

        label = (
            f"{name} • {email}"
            if email
            else str(name)
        )

        student_map[
            student_id
        ] = {
            "label":
                label,

            "name":
                str(name),
        }

    student_ids = list(
        student_map.keys()
    )

    if not student_ids:
        st.error(
            "No valid student profiles "
            "could be loaded."
        )

        return None

    current_student = (
        st.session_state.get(
            "student_id"
        )
    )

    if current_student in student_ids:
        selected_index = (
            student_ids.index(
                current_student
            )
        )

    else:
        selected_index = 0

    selected_student = (
        st.sidebar.selectbox(
            "Student Profile",
            options=student_ids,
            index=selected_index,
            format_func=lambda student_id:
                student_map[
                    student_id
                ][
                    "label"
                ],
            key="comparison_student_selector"
        )
    )

    previous_student = (
        st.session_state.get(
            "student_id"
        )
    )

    if (
        previous_student is not None
        and previous_student
        != selected_student
    ):
        st.session_state.selected_topic = None
        st.session_state.selected_session_id = None

    st.session_state.student_id = (
        selected_student
    )

    st.session_state.student_name = (
        student_map[
            selected_student
        ][
            "name"
        ]
    )

    return selected_student


# analytics loading

def load_comparison_data(
    student_id
):
    result = (
        get_student_analytics(
            student_id,
            tolerance_days=(
                STRICT_TOLERANCE_DAYS
            )
        )
    )

    if not result.get(
        "success",
        False
    ):
        st.error(
            result.get(
                "message",
                (
                    "Model comparison data "
                    "could not be loaded."
                )
            )
        )

        return None

    return result


# header

def show_header(
    analytics
):
    student = (
        analytics.get(
            "student"
        )
        or {}
    )

    student_name = (
        student.get(
            "name"
        )
        or st.session_state.get(
            "student_name"
        )
        or "Student"
    )

    st.session_state.student_name = (
        student_name
    )

    st.title(
        "⚖️ Model Comparison"
    )

    st.write(
        "Compare retention methods using "
        f"real delayed-recall data for "
        f"**{student_name}**."
    )

    st.caption(
        "Formal comparisons use only predictions "
        "created before the recall test and matched "
        f"within ±{STRICT_TOLERANCE_DAYS:g} day "
        "of the actual testing delay."
    )


# research question

def show_research_question():
    st.subheader(
        "Research Question"
    )

    st.info(
        "Can personalized learning data be used "
        "to predict a student's memory retention "
        "and recommend better revision timing than "
        "a fixed forgetting-curve schedule?"
    )


# research status

def show_research_status(
    analytics
):
    status = (
        analytics.get(
            "research_status"
        )
        or {}
    )

    evaluation = ensure_dataframe(
        analytics.get(
            "evaluation_data"
        )
    )

    common = (
        analytics.get(
            "common_subset"
        )
        or {}
    )

    total_quizzes = safe_int(
        status.get(
            "quiz_results"
        ),
        0
    )

    strict_matches = safe_int(
        status.get(
            "strict_prediction_matches"
        ),
        len(evaluation)
    )

    common_observations = safe_int(
        common.get(
            "observations"
        ),
        0
    )

    stage = status.get(
        "research_stage",
        "Unknown"
    )

    st.subheader(
        "Research Data Status"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:
        st.metric(
            "Actual Recall Tests",
            total_quizzes
        )

    with col2:
        st.metric(
            "Strict Prediction Matches",
            strict_matches
        )

    with col3:
        st.metric(
            "Common Comparison Rows",
            common_observations
        )

    with col4:
        st.metric(
            "Research Stage",
            stage
        )

    if strict_matches == 0:
        st.warning(
            "No prediction-recall pairs "
            "currently meet the strict "
            "matching requirement."
        )

    elif common_observations == 0:
        st.info(
            "Strict matches exist, but "
            "there is not yet a common "
            "comparison dataset."
        )

    elif common_observations < 5:
        st.info(
            "A numerical comparison is possible, "
            "but the dataset is still too small "
            "for a reliable conclusion."
        )

    elif common_observations < 20:
        st.info(
            "A preliminary common-observation "
            "comparison is available."
        )

    else:
        st.success(
            "A larger common-observation "
            "comparison dataset is available."
        )


# model availability

def show_model_availability(
    analytics
):
    st.subheader(
        "Model Availability"
    )

    evaluation = ensure_dataframe(
        analytics.get(
            "evaluation_data"
        )
    )

    rows = []

    for model_name, column in (
        MODEL_COLUMNS.items()
    ):
        observations = 0

        if (
            not evaluation.empty
            and column
            in evaluation.columns
        ):
            numeric = pd.to_numeric(
                evaluation[
                    column
                ],
                errors="coerce"
            )

            observations = int(
                numeric.notna().sum()
            )

        rows.append(
            {
                "Model":
                    model_name,

                "Strict Usable Predictions":
                    observations,

                "Status":
                    (
                        "Available"
                        if observations > 0
                        else "Not Yet Available"
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Missing personalized or adaptive "
        "predictions remain missing. MemoryMap "
        "does not replace them with fabricated "
        "values."
    )


# metrics table

def create_metrics_dataframe(
    metrics,
    include_rank=False
):
    rows = []

    for result in metrics:
        row = {}

        if include_rank:
            rank = result.get(
                "rank"
            )

            row[
                "Rank"
            ] = (
                safe_int(
                    rank,
                    None
                )
                if rank is not None
                else "N/A"
            )

        mae_pp = safe_float(
            result.get(
                "mae_pp"
            )
        )

        rmse_pp = safe_float(
            result.get(
                "rmse_pp"
            )
        )

        r2 = safe_float(
            result.get(
                "r2"
            )
        )

        bias_pp = safe_float(
            result.get(
                "bias_pp"
            )
        )

        within_5 = safe_float(
            result.get(
                "within_5pp"
            )
        )

        within_10 = safe_float(
            result.get(
                "within_10pp"
            )
        )

        within_15 = safe_float(
            result.get(
                "within_15pp"
            )
        )

        row.update(
            {
                "Model":
                    result.get(
                        "model",
                        "Unknown"
                    ),

                "Observations":
                    safe_int(
                        result.get(
                            "observations"
                        ),
                        0
                    ),

                "MAE":
                    (
                        f"{mae_pp:.2f} pp"
                        if mae_pp
                        is not None
                        else "N/A"
                    ),

                "RMSE":
                    (
                        f"{rmse_pp:.2f} pp"
                        if rmse_pp
                        is not None
                        else "N/A"
                    ),

                "R²":
                    (
                        f"{r2:.3f}"
                        if r2
                        is not None
                        else "N/A"
                    ),

                "Bias":
                    (
                        f"{bias_pp:.2f} pp"
                        if bias_pp
                        is not None
                        else "N/A"
                    ),

                "Bias Direction":
                    result.get(
                        "bias_direction",
                        "N/A"
                    ),

                "Within 5pp":
                    (
                        f"{within_5 * 100:.1f}%"
                        if within_5
                        is not None
                        else "N/A"
                    ),

                "Within 10pp":
                    (
                        f"{within_10 * 100:.1f}%"
                        if within_10
                        is not None
                        else "N/A"
                    ),

                "Within 15pp":
                    (
                        f"{within_15 * 100:.1f}%"
                        if within_15
                        is not None
                        else "N/A"
                    ),
            }
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# fair leaderboard

def show_fair_leaderboard(
    analytics
):
    st.subheader(
        "Fair Model Leaderboard"
    )

    common = (
        analytics.get(
            "common_subset"
        )
        or {}
    )

    observations = safe_int(
        common.get(
            "observations"
        ),
        0
    )

    models = (
        common.get(
            "models"
        )
        or []
    )

    metrics = (
        common.get(
            "metrics"
        )
        or []
    )

    ranking = (
        common.get(
            "ranking"
        )
        or []
    )

    if observations == 0:
        st.info(
            "No common strict evaluation "
            "subset is available yet."
        )

        return

    if not metrics:
        st.info(
            "No usable common-subset "
            "model metrics are available."
        )

        return

    display_metrics = (
        ranking
        if ranking
        else metrics
    )

    dataframe = (
        create_metrics_dataframe(
            display_metrics,
            include_rank=bool(
                ranking
            )
        )
    )

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True
    )

    if len(models) < 2:
        st.warning(
            "Only one prediction method "
            "currently has usable observations."
        )

        st.info(
            "A comparative winner is not "
            "reported until at least two "
            "methods are evaluated on the "
            "same recall outcomes."
        )

        return

    winner = (
        ranking[0]
        if ranking
        else metrics[0]
    )

    winner_name = winner.get(
        "model",
        "Unknown"
    )

    winner_mae = safe_float(
        winner.get(
            "mae_pp"
        )
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Current Numerical Leader",
            winner_name
        )

    with col2:
        st.metric(
            "Leader MAE",
            (
                f"{winner_mae:.2f} pp"
                if winner_mae
                is not None
                else "N/A"
            )
        )

    with col3:
        st.metric(
            "Common Observations",
            observations
        )

    if observations < 5:
        st.warning(
            "A numerical leader exists, "
            "but the comparison dataset is "
            "too small for a reliable conclusion."
        )

    st.caption(
        "All methods in this leaderboard "
        "are evaluated on the same "
        "recall observations."
    )


# error comparison

def get_common_metrics(
    analytics
):
    common = (
        analytics.get(
            "common_subset"
        )
        or {}
    )

    ranking = (
        common.get(
            "ranking"
        )
        or []
    )

    metrics = (
        common.get(
            "metrics"
        )
        or []
    )

    return (
        ranking
        if ranking
        else metrics
    )


def show_error_comparison(
    analytics
):
    st.subheader(
        "Prediction Error Comparison"
    )

    metrics = get_common_metrics(
        analytics
    )

    if not metrics:
        st.info(
            "No common-subset error "
            "metrics are available."
        )

        return

    rows = []

    for result in metrics:
        model = result.get(
            "model",
            "Unknown"
        )

        mae = safe_float(
            result.get(
                "mae_pp"
            )
        )

        rmse = safe_float(
            result.get(
                "rmse_pp"
            )
        )

        if mae is not None:
            rows.append(
                {
                    "Model":
                        model,

                    "Metric":
                        "MAE",

                    "Error":
                        mae,
                }
            )

        if rmse is not None:
            rows.append(
                {
                    "Model":
                        model,

                    "Metric":
                        "RMSE",

                    "Error":
                        rmse,
                }
            )

    if not rows:
        st.info(
            "No numeric error metrics "
            "are available."
        )

        return

    dataframe = pd.DataFrame(
        rows
    )

    figure = px.bar(
        dataframe,
        x="Model",
        y="Error",
        color="Metric",
        barmode="group",
        text_auto=".2f",
        title=(
            "MAE and RMSE on the "
            "Common Evaluation Subset"
        )
    )

    figure.update_yaxes(
        title=(
            "Prediction Error "
            "(percentage points)"
        )
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )

    st.caption(
        "Lower error is better. RMSE gives "
        "larger mistakes more influence."
    )


# accuracy bands

def show_accuracy_bands(
    analytics
):
    st.subheader(
        "Prediction Accuracy Bands"
    )

    metrics = get_common_metrics(
        analytics
    )

    if not metrics:
        st.info(
            "Accuracy-band comparison "
            "is not available yet."
        )

        return

    rows = []

    bands = [
        (
            "Within 5pp",
            "within_5pp"
        ),
        (
            "Within 10pp",
            "within_10pp"
        ),
        (
            "Within 15pp",
            "within_15pp"
        ),
    ]

    for result in metrics:
        for label, key in bands:
            value = safe_float(
                result.get(
                    key
                )
            )

            if value is None:
                continue

            rows.append(
                {
                    "Model":
                        result.get(
                            "model",
                            "Unknown"
                        ),

                    "Accuracy Band":
                        label,

                    "Percentage":
                        value * 100,
                }
            )

    if not rows:
        st.info(
            "No accuracy-band values "
            "are available."
        )

        return

    figure = px.bar(
        pd.DataFrame(
            rows
        ),
        x="Model",
        y="Percentage",
        color="Accuracy Band",
        barmode="group",
        text_auto=".1f",
        title=(
            "Predictions Close to "
            "Actual Recall"
        )
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Predictions (%)"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )


# bias comparison

def show_bias_analysis(
    analytics
):
    st.subheader(
        "Prediction Bias"
    )

    metrics = get_common_metrics(
        analytics
    )

    if not metrics:
        st.info(
            "Bias comparison is "
            "not available yet."
        )

        return

    rows = []

    for result in metrics:
        bias = safe_float(
            result.get(
                "bias_pp"
            )
        )

        if bias is None:
            continue

        rows.append(
            {
                "Model":
                    result.get(
                        "model",
                        "Unknown"
                    ),

                "Bias":
                    bias,

                "Direction":
                    result.get(
                        "bias_direction",
                        "N/A"
                    ),
            }
        )

    if not rows:
        st.info(
            "No numeric bias values "
            "are available."
        )

        return

    figure = px.bar(
        pd.DataFrame(
            rows
        ),
        x="Model",
        y="Bias",
        text_auto=".2f",
        hover_data=[
            "Direction"
        ],
        title=(
            "Average Prediction Bias"
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash"
    )

    figure.update_yaxes(
        title=(
            "Bias "
            "(percentage points)"
        )
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )

    st.caption(
        "Positive bias indicates average "
        "overprediction. Negative bias "
        "indicates average underprediction."
    )


# prediction dataframe

def create_prediction_dataframe(
    analytics
):
    evaluation = ensure_dataframe(
        analytics.get(
            "evaluation_data"
        )
    )

    if evaluation.empty:
        return pd.DataFrame()

    if (
        "actual_recall"
        not in evaluation.columns
    ):
        return pd.DataFrame()

    rows = []

    for _, record in (
        evaluation.iterrows()
    ):
        actual = safe_float(
            record.get(
                "actual_recall"
            )
        )

        if actual is None:
            continue

        if not (
            0 <= actual <= 1
        ):
            continue

        for model_name, column in (
            MODEL_COLUMNS.items()
        ):
            if column not in (
                evaluation.columns
            ):
                continue

            prediction = safe_float(
                record.get(
                    column
                )
            )

            if prediction is None:
                continue

            if not (
                0 <= prediction <= 1
            ):
                continue

            rows.append(
                {
                    "Model":
                        model_name,

                    "Quiz ID":
                        record.get(
                            "quiz_id"
                        ),

                    "Session ID":
                        record.get(
                            "session_id"
                        ),

                    "Topic":
                        record.get(
                            "topic_name",
                            "Unknown"
                        ),

                    "Actual Recall":
                        actual * 100,

                    "Predicted Recall":
                        prediction * 100,

                    "Absolute Error":
                        abs(
                            prediction
                            - actual
                        )
                        * 100,

                    "Delay Days":
                        safe_float(
                            record.get(
                                "delay_days"
                            )
                        ),

                    "Prediction Horizon":
                        safe_float(
                            record.get(
                                "prediction_for_days"
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# predicted vs actual

def show_predicted_vs_actual(
    analytics
):
    st.subheader(
        "Predicted vs Actual Recall"
    )

    dataframe = (
        create_prediction_dataframe(
            analytics
        )
    )

    if dataframe.empty:
        st.info(
            "A prediction-vs-actual graph "
            "appears after a saved prediction "
            "is followed by a strictly matched "
            "actual recall test."
        )

        return

    figure = px.scatter(
        dataframe,
        x="Predicted Recall",
        y="Actual Recall",
        color="Model",
        hover_data=[
            "Topic",
            "Delay Days",
            "Prediction Horizon",
            "Absolute Error",
        ],
        title=(
            "Strict Predictions Compared "
            "with Actual Recall"
        )
    )

    figure.update_traces(
        marker=dict(
            size=12
        )
    )

    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        line=dict(
            dash="dash"
        )
    )

    figure.update_xaxes(
        range=[
            0,
            100
        ],
        title="Predicted Recall (%)"
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ],
        title="Actual Recall (%)"
    )

    st.plotly_chart(
        figure,
        width="stretch"
    )

    st.caption(
        "Points closer to the diagonal "
        "represent smaller prediction errors."
    )


# test comparison

def show_test_by_test_comparison(
    analytics
):
    st.subheader(
        "Test-by-Test Comparison"
    )

    dataframe = (
        create_prediction_dataframe(
            analytics
        )
    )

    if dataframe.empty:
        st.info(
            "No usable test-by-test "
            "prediction comparison exists yet."
        )

        return

    display = dataframe.copy()

    for column in [
        "Actual Recall",
        "Predicted Recall",
        "Absolute Error",
        "Delay Days",
        "Prediction Horizon",
    ]:
        if column not in display.columns:
            continue

        numeric = pd.to_numeric(
            display[
                column
            ],
            errors="coerce"
        )

        display[
            column
        ] = numeric.map(
            lambda value:
                (
                    f"{value:.2f}"
                    if pd.notna(value)
                    else "N/A"
                )
        )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True
    )


# baseline comparisons

def show_baseline_comparison(
    analytics
):
    st.subheader(
        "Fixed Forgetting Curve vs Alternatives"
    )

    comparisons = (
        analytics.get(
            "baseline_comparisons"
        )
        or {}
    )

    if not comparisons:
        st.info(
            "No alternative model currently "
            "shares enough strict observations "
            "with the fixed baseline for a "
            "pairwise comparison."
        )

        return

    rows = []

    for candidate_name, result in (
        comparisons.items()
    ):
        pairwise = (
            result.get(
                "pairwise"
            )
            or {}
        )

        improvement = (
            result.get(
                "improvement"
            )
            or {}
        )

        fixed_metrics = (
            pairwise.get(
                "first_model"
            )
            or {}
        )

        candidate_metrics = (
            pairwise.get(
                "second_model"
            )
            or {}
        )

        improvement_percentage = (
            safe_float(
                improvement.get(
                    "percentage_improvement"
                )
            )
        )

        rows.append(
            {
                "Candidate":
                    candidate_name,

                "Shared Observations":
                    safe_int(
                        pairwise.get(
                            "observations"
                        ),
                        0
                    ),

                "Fixed MAE":
                    format_pp(
                        fixed_metrics.get(
                            "mae_pp"
                        )
                    ),

                "Candidate MAE":
                    format_pp(
                        candidate_metrics.get(
                            "mae_pp"
                        )
                    ),

                "Numerical Winner":
                    pairwise.get(
                        "winner",
                        "N/A"
                    ),

                "Relative MAE Improvement":
                    (
                        f"{improvement_percentage:.2f}%"
                        if improvement_percentage
                        is not None
                        else "N/A"
                    ),

                "Candidate Improved":
                    (
                        "Yes"
                        if improvement.get(
                            "improved",
                            False
                        )
                        else "No"
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Each candidate is compared with "
        "the fixed baseline using the same "
        "pairwise recall observations."
    )


# research verdict

def show_research_verdict(
    analytics
):
    st.subheader(
        "Current Research Finding"
    )

    verdict = (
        analytics.get(
            "research_verdict"
        )
        or {}
    )

    common = (
        analytics.get(
            "common_subset"
        )
        or {}
    )

    models = (
        common.get(
            "models"
        )
        or []
    )

    observations = safe_int(
        common.get(
            "observations"
        ),
        0
    )

    message = verdict.get(
        "message",
        (
            "No research finding "
            "is available yet."
        )
    )

    status = verdict.get(
        "status",
        "No Data"
    )

    if len(models) < 2:
        st.info(
            "The research question cannot yet "
            "be answered comparatively."
        )

        st.write(
            message
        )

        st.caption(
            "At least two approaches must "
            "have predictions on the same "
            "strict recall observations."
        )

        return

    if observations < 5:
        st.warning(
            message
        )

        st.caption(
            "The current ranking is numerical "
            "only and should not be treated as "
            "a reliable final conclusion."
        )

        return

    winner = verdict.get(
        "winner"
    )

    if winner:
        st.success(
            message
        )

    else:
        st.info(
            message
        )

    st.caption(
        f"Evidence classification: {status}"
    )

    if observations < 20:
        st.caption(
            "The finding remains preliminary "
            "while additional real delayed-recall "
            "observations are collected."
        )

    else:
        st.caption(
            "The result uses strictly matched "
            "delayed-recall observations. "
            "Additional participants would "
            "further strengthen the evidence."
        )


# raw data

def probability_display(
    dataframe
):
    dataframe = dataframe.copy()

    for column in [
        "actual_recall",
        "fixed_prediction",
        "personalized_prediction",
        "adaptive_prediction",
    ]:
        if column not in dataframe.columns:
            continue

        numeric = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce"
        )

        dataframe[
            column
        ] = numeric.map(
            lambda value:
                (
                    f"{value * 100:.2f}%"
                    if pd.notna(value)
                    else "N/A"
                )
        )

    return dataframe


def show_raw_comparison_data(
    analytics
):
    with st.expander(
        "View Comparison Data"
    ):
        evaluation = ensure_dataframe(
            analytics.get(
                "evaluation_data"
            )
        )

        common = (
            analytics.get(
                "common_subset"
            )
            or {}
        )

        common_dataframe = ensure_dataframe(
            common.get(
                "dataframe"
            )
        )

        individual_metrics = (
            analytics.get(
                "individual_model_metrics"
            )
            or []
        )

        common_metrics = (
            common.get(
                "metrics"
            )
            or []
        )

        tab1, tab2, tab3, tab4 = (
            st.tabs(
                [
                    "Strict Matches",
                    "Common Subset",
                    "Individual Metrics",
                    "Common Metrics",
                ]
            )
        )

        with tab1:
            if evaluation.empty:
                st.info(
                    "No strict matches."
                )

            else:
                st.dataframe(
                    probability_display(
                        evaluation
                    ),
                    width="stretch",
                    hide_index=True
                )

        with tab2:
            if common_dataframe.empty:
                st.info(
                    "No common comparison subset."
                )

            else:
                st.dataframe(
                    probability_display(
                        common_dataframe
                    ),
                    width="stretch",
                    hide_index=True
                )

        with tab3:
            if not individual_metrics:
                st.info(
                    "No individual model metrics."
                )

            else:
                st.dataframe(
                    create_metrics_dataframe(
                        individual_metrics
                    ),
                    width="stretch",
                    hide_index=True
                )

        with tab4:
            if not common_metrics:
                st.info(
                    "No common-subset metrics."
                )

            else:
                st.dataframe(
                    create_metrics_dataframe(
                        common_metrics
                    ),
                    width="stretch",
                    hide_index=True
                )


# explanation

def show_metric_explanation():
    with st.expander(
        "Understanding the comparison metrics"
    ):
        st.markdown(
            """
**MAE - Mean Absolute Error**

Average absolute distance between predicted recall and actual recall.

Lower is better.

**RMSE - Root Mean Squared Error**

Also measures prediction error, but larger mistakes receive more influence.

Lower is better.

**R²**

Measures how much variation in observed recall is explained by the prediction method.

With only one observation, R² is not meaningful. MemoryMap displays `N/A` instead of inventing a value.

**Bias**

Average signed prediction error.

Positive values indicate overprediction. Negative values indicate underprediction.

**Accuracy bands**

Within 5, 10 and 15 percentage points shows how often predictions fall close to actual recall.

**Common-observation comparison**

When models compete directly, MemoryMap compares them on the same actual recall records.
            """
        )


def show_comparison_rules():
    with st.expander(
        "Research safeguards used by MemoryMap"
    ):
        st.markdown(
            f"""
MemoryMap applies the following rules:

- Only actual recorded recall tests are treated as observed outcomes.
- Predictions must exist before the recall test.
- Prediction horizons must be close to the actual recall delay.
- Strict evaluation currently uses a tolerance of **±{STRICT_TOLERANCE_DAYS:g} day**.
- Missing personalized or adaptive predictions remain missing.
- Direct model rankings use common observations.
- Pairwise baseline comparisons use the same observation subset.
- One available model is not automatically declared the research winner.
- Tiny samples are treated as preliminary.
- R² is shown as `N/A` when it cannot be meaningfully calculated.
- Prediction error does not by itself prove that a revision schedule is better.
- The public bootstrap model does not by itself validate MemoryMap personalization.
            """
        )


# page

def show_page():
    student_id = select_student()

    if student_id is None:
        return

    analytics = load_comparison_data(
        student_id
    )

    if analytics is None:
        return

    show_header(
        analytics
    )

    st.divider()

    show_research_question()

    st.divider()

    show_research_status(
        analytics
    )

    st.divider()

    show_model_availability(
        analytics
    )

    st.divider()

    show_fair_leaderboard(
        analytics
    )

    st.divider()

    show_error_comparison(
        analytics
    )

    st.divider()

    show_accuracy_bands(
        analytics
    )

    st.divider()

    show_bias_analysis(
        analytics
    )

    st.divider()

    show_predicted_vs_actual(
        analytics
    )

    st.divider()

    show_test_by_test_comparison(
        analytics
    )

    st.divider()

    show_baseline_comparison(
        analytics
    )

    st.divider()

    show_research_verdict(
        analytics
    )

    st.divider()

    st.page_link(
        "pages/6_Analytics.py",
        label="📊 Back to Analytics"
    )

    show_raw_comparison_data(
        analytics
    )

    show_metric_explanation()

    show_comparison_rules()


show_page()
