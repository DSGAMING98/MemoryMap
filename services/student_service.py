import sqlite3

from database import (
    add_student,
    get_student,
    get_student_by_email,
    get_all_students,
    get_student_statistics,
    get_study_sessions_for_student,
    get_quiz_results_for_student,
    get_upcoming_revisions,
    get_due_revisions,
    delete_student,
)

from utils import (
    clean_name,
    clean_email,
    validate_name,
    validate_email,
)


# normalization

def normalize_student(
    student
):
    if not student:
        return None

    if isinstance(
        student,
        dict
    ):
        return student.copy()

    try:
        return dict(
            student
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
        item = normalize_student(
            record
        )

        if item is not None:
            normalized.append(
                item
            )

    return normalized


# validation

def validate_student_name(
    name
):
    name = clean_name(
        name
    )

    if not name:
        raise ValueError(
            "Student name is required."
        )

    if not validate_name(
        name
    ):
        raise ValueError(
            "Student name must contain "
            "at least two valid characters."
        )

    if len(
        name
    ) > 100:
        raise ValueError(
            "Student name is too long."
        )

    return name


def validate_student_email(
    email
):
    email = clean_email(
        email
    )

    if not email:
        return None

    if len(
        email
    ) > 254:
        raise ValueError(
            "Email address is too long."
        )

    if not validate_email(
        email
    ):
        raise ValueError(
            "Enter a valid email address."
        )

    return email


def normalize_student_id(
    student_id
):
    if student_id is None:
        return None

    try:
        student_id = int(
            student_id
        )

    except (
        TypeError,
        ValueError
    ):
        return None

    if student_id <= 0:
        return None

    return student_id


# creation

def create_student(
    name,
    email=None
):
    try:
        name = validate_student_name(
            name
        )

        email = validate_student_email(
            email
        )

    except ValueError as error:
        return {
            "success":
                False,

            "student":
                None,

            "message":
                str(
                    error
                ),
        }

    if email:
        existing_student = (
            get_student_by_email(
                email
            )
        )

        existing_student = (
            normalize_student(
                existing_student
            )
        )

        if existing_student:
            return {
                "success":
                    False,

                "student":
                    existing_student,

                "message":
                    (
                        "A student profile already "
                        "exists with this email address."
                    ),
            }

    try:
        student_id = add_student(
            name=name,
            email=email
        )

    except sqlite3.IntegrityError:
        return {
            "success":
                False,

            "student":
                None,

            "message":
                (
                    "A student profile with this "
                    "email address already exists."
                ),
        }

    except Exception as error:
        return {
            "success":
                False,

            "student":
                None,

            "message":
                (
                    "Student profile could not "
                    f"be created: {error}"
                ),
        }

    student = get_student(
        student_id
    )

    student = normalize_student(
        student
    )

    if student is None:
        return {
            "success":
                False,

            "student":
                None,

            "message":
                (
                    "Student was created but the "
                    "profile could not be reloaded."
                ),
        }

    return {
        "success":
            True,

        "student":
            student,

        "message":
            "Student profile created successfully.",
    }


# retrieval

def get_student_profile(
    student_id
):
    student_id = normalize_student_id(
        student_id
    )

    if student_id is None:
        return None

    try:
        student = get_student(
            student_id
        )

    except Exception:
        return None

    return normalize_student(
        student
    )


def get_student_by_email_address(
    email
):
    try:
        email = validate_student_email(
            email
        )

    except ValueError:
        return None

    if email is None:
        return None

    try:
        student = get_student_by_email(
            email
        )

    except Exception:
        return None

    return normalize_student(
        student
    )


def list_students():
    try:
        students = get_all_students()

    except Exception:
        return []

    return normalize_records(
        students
    )


def student_exists(
    student_id
):
    return (
        get_student_profile(
            student_id
        )
        is not None
    )


# display

def get_student_display_name(
    student
):
    student = normalize_student(
        student
    )

    if not student:
        return "Unknown Student"

    name = clean_name(
        student.get(
            "name"
        )
    )

    if not name:
        return "Unnamed Student"

    return name


def get_student_selector_options():
    students = list_students()

    options = {}

    for student in students:
        student_id = student.get(
            "student_id"
        )

        if student_id is None:
            continue

        name = get_student_display_name(
            student
        )

        email = student.get(
            "email"
        )

        if email:
            label = (
                f"{name} ({email})"
            )

        else:
            label = name

        if label in options:
            label = (
                f"{label} "
                f"[ID {student_id}]"
            )

        options[
            label
        ] = student_id

    return options


# statistics

def get_student_summary(
    student_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "student":
                None,

            "statistics":
                {},

            "message":
                "Student profile not found.",
        }

    try:
        statistics = (
            get_student_statistics(
                student[
                    "student_id"
                ]
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "student":
                student,

            "statistics":
                {},

            "message":
                (
                    "Student statistics could not "
                    f"be loaded: {error}"
                ),
        }

    if not isinstance(
        statistics,
        dict
    ):
        statistics = {}

    return {
        "success":
            True,

        "student":
            student,

        "statistics":
            statistics,

        "message":
            "Student summary loaded.",
    }


# activity

def get_student_activity(
    student_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "student":
                None,

            "study_sessions":
                [],

            "quiz_results":
                [],

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    try:
        study_sessions = (
            get_study_sessions_for_student(
                student_id
            )
        )

        quiz_results = (
            get_quiz_results_for_student(
                student_id
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "student":
                student,

            "study_sessions":
                [],

            "quiz_results":
                [],

            "message":
                (
                    "Student activity could not "
                    f"be loaded: {error}"
                ),
        }

    return {
        "success":
            True,

        "student":
            student,

        "study_sessions":
            normalize_records(
                study_sessions
            ),

        "quiz_results":
            normalize_records(
                quiz_results
            ),

        "message":
            "Student activity loaded.",
    }


# revision overview

def get_student_revision_overview(
    student_id
):
    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "success":
                False,

            "due":
                [],

            "upcoming":
                [],

            "due_count":
                0,

            "upcoming_count":
                0,

            "message":
                "Student profile not found.",
        }

    student_id = student[
        "student_id"
    ]

    try:
        due_revisions = (
            get_due_revisions(
                student_id
            )
        )

        upcoming_revisions = (
            get_upcoming_revisions(
                student_id
            )
        )

    except Exception as error:
        return {
            "success":
                False,

            "due":
                [],

            "upcoming":
                [],

            "due_count":
                0,

            "upcoming_count":
                0,

            "message":
                (
                    "Revision data could not "
                    f"be loaded: {error}"
                ),
        }

    due_revisions = (
        normalize_records(
            due_revisions
        )
    )

    upcoming_revisions = (
        normalize_records(
            upcoming_revisions
        )
    )

    return {
        "success":
            True,

        "due":
            due_revisions,

        "upcoming":
            upcoming_revisions,

        "due_count":
            len(
                due_revisions
            ),

        "upcoming_count":
            len(
                upcoming_revisions
            ),

        "message":
            "Revision overview loaded.",
    }


# dashboard

def get_student_dashboard_data(
    student_id
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

    summary = get_student_summary(
        student_id
    )

    activity = get_student_activity(
        student_id
    )

    revisions = (
        get_student_revision_overview(
            student_id
        )
    )

    study_sessions = activity.get(
        "study_sessions",
        []
    )

    quiz_results = activity.get(
        "quiz_results",
        []
    )

    latest_session = (
        study_sessions[
            0
        ]
        if study_sessions
        else None
    )

    latest_quiz = (
        quiz_results[
            0
        ]
        if quiz_results
        else None
    )

    return {
        "success":
            True,

        "student":
            student,

        "statistics":
            summary.get(
                "statistics",
                {}
            ),

        "study_sessions":
            study_sessions,

        "quiz_results":
            quiz_results,

        "due_revisions":
            revisions.get(
                "due",
                []
            ),

        "upcoming_revisions":
            revisions.get(
                "upcoming",
                []
            ),

        "latest_session":
            latest_session,

        "latest_quiz":
            latest_quiz,

        "message":
            "Dashboard data loaded.",
    }


# progress

def calculate_student_progress(
    student_id
):
    activity = get_student_activity(
        student_id
    )

    if not activity.get(
        "success",
        False
    ):
        return {
            "study_sessions":
                0,

            "quiz_tests":
                0,

            "valid_recall_tests":
                0,

            "average_recall":
                None,

            "progress_level":
                "No Data",
        }

    study_sessions = activity[
        "study_sessions"
    ]

    quiz_results = activity[
        "quiz_results"
    ]

    recall_values = []

    for quiz in quiz_results:
        recall = quiz.get(
            "recall_accuracy"
        )

        if recall is None:
            continue

        try:
            recall = float(
                recall
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        if (
            0
            <= recall
            <= 1
        ):
            recall_values.append(
                recall
            )

    if recall_values:
        average_recall = (
            sum(
                recall_values
            )
            / len(
                recall_values
            )
        )

    else:
        average_recall = None

    valid_recall_count = len(
        recall_values
    )

    if valid_recall_count == 0:
        progress_level = (
            "Waiting for Recall Data"
        )

    elif valid_recall_count < 5:
        progress_level = (
            "Early Data Collection"
        )

    elif valid_recall_count < 20:
        progress_level = (
            "Personalization Developing"
        )

    else:
        progress_level = (
            "Personalization Ready"
        )

    return {
        "study_sessions":
            len(
                study_sessions
            ),

        "quiz_tests":
            len(
                quiz_results
            ),

        "valid_recall_tests":
            valid_recall_count,

        "average_recall":
            average_recall,

        "progress_level":
            progress_level,
    }


# deletion

def remove_student(
    student_id
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

    try:
        deleted = delete_student(
            student[
                "student_id"
            ]
        )

    except Exception as error:
        return {
            "success":
                False,

            "message":
                (
                    "Student profile could not "
                    f"be deleted: {error}"
                ),
        }

    if not deleted:
        return {
            "success":
                False,

            "message":
                (
                    "Student profile was not deleted."
                ),
        }

    return {
        "success":
            True,

        "message":
            (
                "Student profile and related "
                "learning records were deleted."
            ),
    }


# active student

def validate_active_student(
    student_id
):
    if student_id is None:
        return {
            "valid":
                False,

            "student":
                None,

            "message":
                (
                    "Select or create a "
                    "student profile first."
                ),
        }

    student = get_student_profile(
        student_id
    )

    if student is None:
        return {
            "valid":
                False,

            "student":
                None,

            "message":
                (
                    "The selected student profile "
                    "no longer exists."
                ),
        }

    return {
        "valid":
            True,

        "student":
            student,

        "message":
            "Student profile is active.",
    }


# direct execution

if __name__ == "__main__":

    print(
        "MemoryMap Student Service"
    )

    print(
        "=" * 40
    )

    students = list_students()

    print(
        "Student Profiles:",
        len(
            students
        )
    )

    for student in students:
        print(
            "-",
            get_student_display_name(
                student
            )
        )

    if students:
        first_student = students[
            0
        ]

        student_id = first_student[
            "student_id"
        ]

        progress = (
            calculate_student_progress(
                student_id
            )
        )

        print()

        print(
            "Example Student:",
            first_student.get(
                "name"
            )
        )

        print(
            "Student ID:",
            student_id
        )

        print(
            "Study Sessions:",
            progress[
                "study_sessions"
            ]
        )

        print(
            "Quiz Tests:",
            progress[
                "quiz_tests"
            ]
        )

        print(
            "Progress Level:",
            progress[
                "progress_level"
            ]
        )