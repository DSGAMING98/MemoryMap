import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import (
    DATABASE_PATH,
    STUDENTS_TABLE,
    TOPICS_TABLE,
    STUDY_SESSIONS_TABLE,
    PREDICTIONS_TABLE,
    REVISION_SCHEDULE_TABLE,
    QUIZ_RESULTS_TABLE,
)


# connection

@contextmanager
def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# row conversion

def row_to_dict(
    row,
    id_field=None
):
    if row is None:
        return None

    record = dict(
        row
    )

    if (
        id_field
        and id_field in record
    ):
        record.setdefault(
            "id",
            record[
                id_field
            ]
        )

    if "subject_name" in record:
        record.setdefault(
            "subject",
            record[
                "subject_name"
            ]
        )

    return record


def rows_to_dicts(
    rows,
    id_field=None
):
    return [
        row_to_dict(
            row,
            id_field
        )
        for row in rows
    ]


# datetime helpers

def normalize_datetime(
    value=None
):
    if value is None:
        return datetime.now().isoformat()

    if isinstance(
        value,
        datetime
    ):
        return value.isoformat()

    return str(
        value
    )


# validation helpers

def validate_probability(
    value,
    field_name
):
    if value is None:
        return

    value = float(
        value
    )

    if not 0 <= value <= 1:
        raise ValueError(
            f"{field_name} must be "
            "between 0 and 1."
        )


def validate_positive(
    value,
    field_name
):
    if value is None:
        return

    if float(value) <= 0:
        raise ValueError(
            f"{field_name} must be "
            "greater than zero."
        )


def validate_non_negative(
    value,
    field_name
):
    if value is None:
        return

    if float(value) < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )


# schema helpers

def get_table_columns(
    connection,
    table_name
):
    cursor = connection.cursor()

    cursor.execute(
        f"""
        PRAGMA table_info({table_name})
        """
    )

    return {
        row[
            "name"
        ]
        for row in cursor.fetchall()
    }


def add_column_if_missing(
    connection,
    table_name,
    column_name,
    column_definition
):
    columns = get_table_columns(
        connection,
        table_name
    )

    if column_name in columns:
        return

    connection.execute(
        f"""
        ALTER TABLE {table_name}
        ADD COLUMN {column_name}
        {column_definition}
        """
    )


def migrate_database(
    connection
):
    add_column_if_missing(
        connection,
        REVISION_SCHEDULE_TABLE,
        "schedule_mode",
        "TEXT"
    )

    add_column_if_missing(
        connection,
        REVISION_SCHEDULE_TABLE,
        "source",
        "TEXT"
    )

    add_column_if_missing(
        connection,
        REVISION_SCHEDULE_TABLE,
        "estimated_half_life",
        "REAL"
    )

    add_column_if_missing(
        connection,
        REVISION_SCHEDULE_TABLE,
        "generated_at",
        "TEXT"
    )

    connection.execute(
        f"""
        UPDATE {REVISION_SCHEDULE_TABLE}

        SET generated_at = created_at

        WHERE generated_at IS NULL
        """
    )


# initialization

def initialize_database():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with get_connection() as connection:
        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        cursor = connection.cursor()

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {STUDENTS_TABLE}
            (
                student_id
                    INTEGER
                    PRIMARY KEY
                    AUTOINCREMENT,

                name
                    TEXT
                    NOT NULL,

                email
                    TEXT
                    UNIQUE,

                created_at
                    TEXT
                    NOT NULL
            )
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {TOPICS_TABLE}
            (
                topic_id
                    INTEGER
                    PRIMARY KEY
                    AUTOINCREMENT,

                student_id
                    INTEGER
                    NOT NULL,

                topic_name
                    TEXT
                    NOT NULL,

                subject_name
                    TEXT,

                difficulty
                    INTEGER
                    NOT NULL,

                created_at
                    TEXT
                    NOT NULL,

                FOREIGN KEY (
                    student_id
                )
                REFERENCES
                    {STUDENTS_TABLE}
                    (student_id)
                ON DELETE CASCADE,

                CHECK (
                    difficulty BETWEEN 1 AND 5
                )
            )
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {STUDY_SESSIONS_TABLE}
            (
                session_id
                    INTEGER
                    PRIMARY KEY
                    AUTOINCREMENT,

                student_id
                    INTEGER
                    NOT NULL,

                topic_id
                    INTEGER
                    NOT NULL,

                study_duration
                    REAL
                    NOT NULL,

                confidence_level
                    INTEGER
                    NOT NULL,

                revision_count
                    INTEGER
                    NOT NULL
                    DEFAULT 0,

                previous_quiz_score
                    REAL,

                studied_at
                    TEXT
                    NOT NULL,

                notes
                    TEXT,

                FOREIGN KEY (
                    student_id
                )
                REFERENCES
                    {STUDENTS_TABLE}
                    (student_id)
                ON DELETE CASCADE,

                FOREIGN KEY (
                    topic_id
                )
                REFERENCES
                    {TOPICS_TABLE}
                    (topic_id)
                ON DELETE CASCADE,

                CHECK (
                    study_duration > 0
                ),

                CHECK (
                    confidence_level
                    BETWEEN 1 AND 5
                ),

                CHECK (
                    revision_count >= 0
                ),

                CHECK (
                    previous_quiz_score
                    IS NULL
                    OR
                    previous_quiz_score
                    BETWEEN 0 AND 100
                )
            )
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {PREDICTIONS_TABLE}
            (
                prediction_id
                    INTEGER
                    PRIMARY KEY
                    AUTOINCREMENT,

                session_id
                    INTEGER
                    NOT NULL,

                fixed_prediction
                    REAL,

                personalized_prediction
                    REAL,

                adaptive_prediction
                    REAL,

                predicted_half_life
                    REAL,

                prediction_for_days
                    REAL,

                prediction_mode
                    TEXT,

                created_at
                    TEXT
                    NOT NULL,

                FOREIGN KEY (
                    session_id
                )
                REFERENCES
                    {STUDY_SESSIONS_TABLE}
                    (session_id)
                ON DELETE CASCADE,

                CHECK (
                    fixed_prediction
                    IS NULL
                    OR
                    fixed_prediction
                    BETWEEN 0 AND 1
                ),

                CHECK (
                    personalized_prediction
                    IS NULL
                    OR
                    personalized_prediction
                    BETWEEN 0 AND 1
                ),

                CHECK (
                    adaptive_prediction
                    IS NULL
                    OR
                    adaptive_prediction
                    BETWEEN 0 AND 1
                ),

                CHECK (
                    predicted_half_life
                    IS NULL
                    OR
                    predicted_half_life > 0
                ),

                CHECK (
                    prediction_for_days
                    IS NULL
                    OR
                    prediction_for_days >= 0
                )
            )
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {REVISION_SCHEDULE_TABLE}
            (
                revision_id
                    INTEGER
                    PRIMARY KEY
                    AUTOINCREMENT,

                student_id
                    INTEGER
                    NOT NULL,

                topic_id
                    INTEGER
                    NOT NULL,

                session_id
                    INTEGER,

                scheduled_for
                    TEXT
                    NOT NULL,

                predicted_retention
                    REAL,

                threshold
                    REAL,

                priority
                    TEXT,

                schedule_mode
                    TEXT,

                source
                    TEXT,

                estimated_half_life
                    REAL,

                generated_at
                    TEXT,

                completed
                    INTEGER
                    NOT NULL
                    DEFAULT 0,

                completed_at
                    TEXT,

                created_at
                    TEXT
                    NOT NULL,

                FOREIGN KEY (
                    student_id
                )
                REFERENCES
                    {STUDENTS_TABLE}
                    (student_id)
                ON DELETE CASCADE,

                FOREIGN KEY (
                    topic_id
                )
                REFERENCES
                    {TOPICS_TABLE}
                    (topic_id)
                ON DELETE CASCADE,

                FOREIGN KEY (
                    session_id
                )
                REFERENCES
                    {STUDY_SESSIONS_TABLE}
                    (session_id)
                ON DELETE SET NULL,

                CHECK (
                    predicted_retention
                    IS NULL
                    OR
                    predicted_retention
                    BETWEEN 0 AND 1
                ),

                CHECK (
                    threshold
                    IS NULL
                    OR
                    threshold
                    BETWEEN 0 AND 1
                ),

                CHECK (
                    estimated_half_life
                    IS NULL
                    OR
                    estimated_half_life > 0
                ),

                CHECK (
                    completed IN (0, 1)
                )
            )
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {QUIZ_RESULTS_TABLE}
            (
                quiz_id
                    INTEGER
                    PRIMARY KEY
                    AUTOINCREMENT,

                student_id
                    INTEGER
                    NOT NULL,

                topic_id
                    INTEGER
                    NOT NULL,

                session_id
                    INTEGER,

                total_questions
                    INTEGER
                    NOT NULL,

                correct_answers
                    INTEGER
                    NOT NULL,

                actual_score
                    REAL
                    NOT NULL,

                recall_accuracy
                    REAL
                    NOT NULL,

                delay_days
                    REAL,

                confidence_after_quiz
                    INTEGER,

                quiz_date
                    TEXT
                    NOT NULL,

                FOREIGN KEY (
                    student_id
                )
                REFERENCES
                    {STUDENTS_TABLE}
                    (student_id)
                ON DELETE CASCADE,

                FOREIGN KEY (
                    topic_id
                )
                REFERENCES
                    {TOPICS_TABLE}
                    (topic_id)
                ON DELETE CASCADE,

                FOREIGN KEY (
                    session_id
                )
                REFERENCES
                    {STUDY_SESSIONS_TABLE}
                    (session_id)
                ON DELETE SET NULL,

                CHECK (
                    total_questions > 0
                ),

                CHECK (
                    correct_answers >= 0
                ),

                CHECK (
                    correct_answers
                    <= total_questions
                ),

                CHECK (
                    actual_score
                    BETWEEN 0 AND 100
                ),

                CHECK (
                    recall_accuracy
                    BETWEEN 0 AND 1
                ),

                CHECK (
                    delay_days
                    IS NULL
                    OR
                    delay_days >= 0
                ),

                CHECK (
                    confidence_after_quiz
                    IS NULL
                    OR
                    confidence_after_quiz
                    BETWEEN 1 AND 5
                )
            )
            """
        )

        migrate_database(
            connection
        )

        create_indexes(
            connection
        )


# indexes

def create_indexes(
    connection
):
    cursor = connection.cursor()

    indexes = [
        (
            "idx_topics_student",
            TOPICS_TABLE,
            "student_id"
        ),
        (
            "idx_sessions_student",
            STUDY_SESSIONS_TABLE,
            "student_id"
        ),
        (
            "idx_sessions_topic",
            STUDY_SESSIONS_TABLE,
            "topic_id"
        ),
        (
            "idx_sessions_studied_at",
            STUDY_SESSIONS_TABLE,
            "studied_at"
        ),
        (
            "idx_predictions_session",
            PREDICTIONS_TABLE,
            "session_id"
        ),
        (
            "idx_predictions_created_at",
            PREDICTIONS_TABLE,
            "created_at"
        ),
        (
            "idx_revision_student",
            REVISION_SCHEDULE_TABLE,
            "student_id"
        ),
        (
            "idx_revision_topic",
            REVISION_SCHEDULE_TABLE,
            "topic_id"
        ),
        (
            "idx_revision_scheduled_for",
            REVISION_SCHEDULE_TABLE,
            "scheduled_for"
        ),
        (
            "idx_quiz_student",
            QUIZ_RESULTS_TABLE,
            "student_id"
        ),
        (
            "idx_quiz_topic",
            QUIZ_RESULTS_TABLE,
            "topic_id"
        ),
        (
            "idx_quiz_session",
            QUIZ_RESULTS_TABLE,
            "session_id"
        ),
        (
            "idx_quiz_date",
            QUIZ_RESULTS_TABLE,
            "quiz_date"
        ),
    ]

    for (
        index_name,
        table_name,
        column_name
    ) in indexes:

        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
            {index_name}

            ON {table_name}
            ({column_name})
            """
        )


# ownership checks

def validate_topic_owner(
    connection,
    student_id,
    topic_id
):
    cursor = connection.cursor()

    cursor.execute(
        f"""
        SELECT student_id
        FROM {TOPICS_TABLE}
        WHERE topic_id = ?
        """,
        (
            topic_id,
        ),
    )

    row = cursor.fetchone()

    if row is None:
        raise ValueError(
            "Topic does not exist."
        )

    if (
        row[
            "student_id"
        ]
        != student_id
    ):
        raise ValueError(
            "Topic does not belong "
            "to this student."
        )


def validate_session_context(
    connection,
    session_id,
    student_id=None,
    topic_id=None
):
    if session_id is None:
        return

    cursor = connection.cursor()

    cursor.execute(
        f"""
        SELECT
            student_id,
            topic_id

        FROM {STUDY_SESSIONS_TABLE}

        WHERE session_id = ?
        """,
        (
            session_id,
        ),
    )

    row = cursor.fetchone()

    if row is None:
        raise ValueError(
            "Study session does not exist."
        )

    if (
        student_id is not None
        and
        row[
            "student_id"
        ]
        != student_id
    ):
        raise ValueError(
            "Study session does not "
            "belong to this student."
        )

    if (
        topic_id is not None
        and
        row[
            "topic_id"
        ]
        != topic_id
    ):
        raise ValueError(
            "Study session does not "
            "belong to this topic."
        )


# students

def add_student(
    name,
    email=None
):
    name = str(
        name
    ).strip()

    if not name:
        raise ValueError(
            "Student name cannot be empty."
        )

    if email is not None:
        email = str(
            email
        ).strip()

        if not email:
            email = None

    created_at = (
        datetime.now().isoformat()
    )

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            INSERT INTO {STUDENTS_TABLE}
            (
                name,
                email,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                created_at,
            ),
        )

        return cursor.lastrowid


def get_student(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT *
            FROM {STUDENTS_TABLE}
            WHERE student_id = ?
            """,
            (
                student_id,
            ),
        )

        return row_to_dict(
            cursor.fetchone(),
            "student_id"
        )


def get_student_by_email(
    email
):
    if email is None:
        return None

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT *
            FROM {STUDENTS_TABLE}
            WHERE email = ?
            """,
            (
                str(
                    email
                ).strip(),
            ),
        )

        return row_to_dict(
            cursor.fetchone(),
            "student_id"
        )


def get_all_students():
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT *
            FROM {STUDENTS_TABLE}
            ORDER BY created_at DESC
            """
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "student_id"
        )


# topics

def add_topic(
    student_id,
    topic_name,
    difficulty,
    subject_name=None,
    subject=None
):
    topic_name = str(
        topic_name
    ).strip()

    if not topic_name:
        raise ValueError(
            "Topic name cannot be empty."
        )

    difficulty = int(
        difficulty
    )

    if not 1 <= difficulty <= 5:
        raise ValueError(
            "Difficulty must be between 1 and 5."
        )

    if (
        subject_name is None
        and
        subject is not None
    ):
        subject_name = subject

    if subject_name is not None:
        subject_name = str(
            subject_name
        ).strip()

        if not subject_name:
            subject_name = None

    created_at = (
        datetime.now().isoformat()
    )

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT 1
            FROM {STUDENTS_TABLE}
            WHERE student_id = ?
            """,
            (
                student_id,
            ),
        )

        if cursor.fetchone() is None:
            raise ValueError(
                "Student does not exist."
            )

        cursor.execute(
            f"""
            INSERT INTO {TOPICS_TABLE}
            (
                student_id,
                topic_name,
                subject_name,
                difficulty,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                topic_name,
                subject_name,
                difficulty,
                created_at,
            ),
        )

        return cursor.lastrowid


def get_topic(
    topic_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT *
            FROM {TOPICS_TABLE}
            WHERE topic_id = ?
            """,
            (
                topic_id,
            ),
        )

        return row_to_dict(
            cursor.fetchone(),
            "topic_id"
        )


def get_topics_for_student(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT *
            FROM {TOPICS_TABLE}

            WHERE student_id = ?

            ORDER BY created_at DESC
            """,
            (
                student_id,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "topic_id"
        )


# study sessions

def add_study_session(
    student_id,
    topic_id,
    study_duration,
    confidence_level,
    revision_count=0,
    previous_quiz_score=None,
    studied_at=None,
    notes=None
):
    study_duration = float(
        study_duration
    )

    confidence_level = int(
        confidence_level
    )

    revision_count = int(
        revision_count
    )

    validate_positive(
        study_duration,
        "Study duration"
    )

    if not 1 <= confidence_level <= 5:
        raise ValueError(
            "Confidence level must "
            "be between 1 and 5."
        )

    validate_non_negative(
        revision_count,
        "Revision count"
    )

    if previous_quiz_score is not None:
        previous_quiz_score = float(
            previous_quiz_score
        )

        if not (
            0
            <= previous_quiz_score
            <= 100
        ):
            raise ValueError(
                "Previous quiz score must "
                "be between 0 and 100."
            )

    studied_at = normalize_datetime(
        studied_at
    )

    with get_connection() as connection:
        validate_topic_owner(
            connection,
            student_id,
            topic_id
        )

        cursor = connection.cursor()

        cursor.execute(
            f"""
            INSERT INTO {STUDY_SESSIONS_TABLE}
            (
                student_id,
                topic_id,
                study_duration,
                confidence_level,
                revision_count,
                previous_quiz_score,
                studied_at,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                topic_id,
                study_duration,
                confidence_level,
                revision_count,
                previous_quiz_score,
                studied_at,
                notes,
            ),
        )

        return cursor.lastrowid


def get_study_session(
    session_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                s.*,
                t.topic_name,
                t.subject_name,
                t.difficulty

            FROM {STUDY_SESSIONS_TABLE} AS s

            JOIN {TOPICS_TABLE} AS t
            ON s.topic_id = t.topic_id

            WHERE s.session_id = ?
            """,
            (
                session_id,
            ),
        )

        return row_to_dict(
            cursor.fetchone(),
            "session_id"
        )


def get_study_sessions_for_student(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                s.*,
                t.topic_name,
                t.subject_name,
                t.difficulty

            FROM {STUDY_SESSIONS_TABLE} AS s

            JOIN {TOPICS_TABLE} AS t
            ON s.topic_id = t.topic_id

            WHERE s.student_id = ?

            ORDER BY s.studied_at DESC
            """,
            (
                student_id,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "session_id"
        )


def get_latest_study_session(
    student_id,
    topic_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                s.*,
                t.topic_name,
                t.subject_name,
                t.difficulty

            FROM {STUDY_SESSIONS_TABLE} AS s

            JOIN {TOPICS_TABLE} AS t
            ON s.topic_id = t.topic_id

            WHERE s.student_id = ?
            AND s.topic_id = ?

            ORDER BY s.studied_at DESC

            LIMIT 1
            """,
            (
                student_id,
                topic_id,
            ),
        )

        return row_to_dict(
            cursor.fetchone(),
            "session_id"
        )


# predictions

def add_prediction(
    session_id,
    fixed_prediction=None,
    personalized_prediction=None,
    adaptive_prediction=None,
    predicted_half_life=None,
    prediction_for_days=None,
    prediction_mode=None
):
    validate_probability(
        fixed_prediction,
        "Fixed prediction"
    )

    validate_probability(
        personalized_prediction,
        "Personalized prediction"
    )

    validate_probability(
        adaptive_prediction,
        "Adaptive prediction"
    )

    validate_positive(
        predicted_half_life,
        "Predicted half-life"
    )

    validate_non_negative(
        prediction_for_days,
        "Prediction interval"
    )

    created_at = (
        datetime.now().isoformat()
    )

    with get_connection() as connection:
        validate_session_context(
            connection,
            session_id
        )

        cursor = connection.cursor()

        cursor.execute(
            f"""
            INSERT INTO {PREDICTIONS_TABLE}
            (
                session_id,
                fixed_prediction,
                personalized_prediction,
                adaptive_prediction,
                predicted_half_life,
                prediction_for_days,
                prediction_mode,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                fixed_prediction,
                personalized_prediction,
                adaptive_prediction,
                predicted_half_life,
                prediction_for_days,
                prediction_mode,
                created_at,
            ),
        )

        return cursor.lastrowid


def get_predictions_for_session(
    session_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT *
            FROM {PREDICTIONS_TABLE}

            WHERE session_id = ?

            ORDER BY created_at DESC
            """,
            (
                session_id,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "prediction_id"
        )


def get_predictions_for_student(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                p.*,
                s.student_id,
                s.topic_id,
                s.studied_at,
                t.topic_name

            FROM {PREDICTIONS_TABLE} AS p

            JOIN {STUDY_SESSIONS_TABLE} AS s
            ON p.session_id = s.session_id

            JOIN {TOPICS_TABLE} AS t
            ON s.topic_id = t.topic_id

            WHERE s.student_id = ?

            ORDER BY p.created_at DESC
            """,
            (
                student_id,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "prediction_id"
        )


# revisions

def add_revision(
    student_id,
    topic_id,
    scheduled_for,
    session_id=None,
    predicted_retention=None,
    threshold=None,
    priority="Normal",
    schedule_mode=None,
    source=None,
    estimated_half_life=None,
    generated_at=None
):
    validate_probability(
        predicted_retention,
        "Predicted retention"
    )

    validate_probability(
        threshold,
        "Retention threshold"
    )

    validate_positive(
        estimated_half_life,
        "Estimated half-life"
    )

    scheduled_for = normalize_datetime(
        scheduled_for
    )

    created_at = (
        datetime.now().isoformat()
    )

    if generated_at is None:
        generated_at = created_at

    else:
        generated_at = (
            normalize_datetime(
                generated_at
            )
        )

    with get_connection() as connection:
        validate_topic_owner(
            connection,
            student_id,
            topic_id
        )

        validate_session_context(
            connection,
            session_id,
            student_id=student_id,
            topic_id=topic_id
        )

        cursor = connection.cursor()

        cursor.execute(
            f"""
            INSERT INTO {REVISION_SCHEDULE_TABLE}
            (
                student_id,
                topic_id,
                session_id,
                scheduled_for,
                predicted_retention,
                threshold,
                priority,
                schedule_mode,
                source,
                estimated_half_life,
                generated_at,
                completed,
                created_at
            )
            VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                student_id,
                topic_id,
                session_id,
                scheduled_for,
                predicted_retention,
                threshold,
                priority,
                schedule_mode,
                source,
                estimated_half_life,
                generated_at,
                created_at,
            ),
        )

        return cursor.lastrowid


def get_revision(
    revision_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                r.*,
                t.topic_name,
                t.subject_name

            FROM {REVISION_SCHEDULE_TABLE} AS r

            JOIN {TOPICS_TABLE} AS t
            ON r.topic_id = t.topic_id

            WHERE r.revision_id = ?
            """,
            (
                revision_id,
            ),
        )

        return row_to_dict(
            cursor.fetchone(),
            "revision_id"
        )


def get_revisions_for_student(
    student_id,
    include_completed=True
):
    with get_connection() as connection:
        cursor = connection.cursor()

        sql = f"""
            SELECT
                r.*,
                t.topic_name,
                t.subject_name

            FROM {REVISION_SCHEDULE_TABLE} AS r

            JOIN {TOPICS_TABLE} AS t
            ON r.topic_id = t.topic_id

            WHERE r.student_id = ?
        """

        parameters = [
            student_id
        ]

        if not include_completed:
            sql += """
                AND r.completed = 0
            """

        sql += """
            ORDER BY r.scheduled_for ASC
        """

        cursor.execute(
            sql,
            parameters
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "revision_id"
        )


def get_upcoming_revisions(
    student_id
):
    current_time = (
        datetime.now().isoformat()
    )

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                r.*,
                t.topic_name,
                t.subject_name

            FROM {REVISION_SCHEDULE_TABLE} AS r

            JOIN {TOPICS_TABLE} AS t
            ON r.topic_id = t.topic_id

            WHERE r.student_id = ?
            AND r.completed = 0
            AND r.scheduled_for >= ?

            ORDER BY r.scheduled_for ASC
            """,
            (
                student_id,
                current_time,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "revision_id"
        )


def get_due_revisions(
    student_id
):
    current_time = (
        datetime.now().isoformat()
    )

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                r.*,
                t.topic_name,
                t.subject_name

            FROM {REVISION_SCHEDULE_TABLE} AS r

            JOIN {TOPICS_TABLE} AS t
            ON r.topic_id = t.topic_id

            WHERE r.student_id = ?
            AND r.completed = 0
            AND r.scheduled_for <= ?

            ORDER BY r.scheduled_for ASC
            """,
            (
                student_id,
                current_time,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "revision_id"
        )


def mark_revision_completed(
    revision_id
):
    completed_at = (
        datetime.now().isoformat()
    )

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            UPDATE {REVISION_SCHEDULE_TABLE}

            SET
                completed = 1,
                completed_at = ?

            WHERE revision_id = ?
            """,
            (
                completed_at,
                revision_id,
            ),
        )

        return (
            cursor.rowcount > 0
        )


# quizzes

def add_quiz_result(
    student_id,
    topic_id,
    total_questions,
    correct_answers,
    session_id=None,
    delay_days=None,
    confidence_after_quiz=None,
    quiz_date=None
):
    total_questions = int(
        total_questions
    )

    correct_answers = int(
        correct_answers
    )

    if total_questions <= 0:
        raise ValueError(
            "Total questions must be "
            "greater than zero."
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
            "Correct answers cannot "
            "exceed total questions."
        )

    if delay_days is not None:
        delay_days = float(
            delay_days
        )

        validate_non_negative(
            delay_days,
            "Recall delay"
        )

    if confidence_after_quiz is not None:
        confidence_after_quiz = int(
            confidence_after_quiz
        )

        if not (
            1
            <= confidence_after_quiz
            <= 5
        ):
            raise ValueError(
                "Confidence after quiz must "
                "be between 1 and 5."
            )

    quiz_date = normalize_datetime(
        quiz_date
    )

    recall_accuracy = (
        correct_answers
        / total_questions
    )

    actual_score = (
        recall_accuracy
        * 100
    )

    with get_connection() as connection:
        validate_topic_owner(
            connection,
            student_id,
            topic_id
        )

        validate_session_context(
            connection,
            session_id,
            student_id=student_id,
            topic_id=topic_id
        )

        cursor = connection.cursor()

        cursor.execute(
            f"""
            INSERT INTO {QUIZ_RESULTS_TABLE}
            (
                student_id,
                topic_id,
                session_id,
                total_questions,
                correct_answers,
                actual_score,
                recall_accuracy,
                delay_days,
                confidence_after_quiz,
                quiz_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                topic_id,
                session_id,
                total_questions,
                correct_answers,
                actual_score,
                recall_accuracy,
                delay_days,
                confidence_after_quiz,
                quiz_date,
            ),
        )

        return cursor.lastrowid


def get_quiz_result(
    quiz_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                q.*,
                t.topic_name,
                t.subject_name

            FROM {QUIZ_RESULTS_TABLE} AS q

            JOIN {TOPICS_TABLE} AS t
            ON q.topic_id = t.topic_id

            WHERE q.quiz_id = ?
            """,
            (
                quiz_id,
            ),
        )

        return row_to_dict(
            cursor.fetchone(),
            "quiz_id"
        )


def get_quiz_results_for_student(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                q.*,
                t.topic_name,
                t.subject_name

            FROM {QUIZ_RESULTS_TABLE} AS q

            JOIN {TOPICS_TABLE} AS t
            ON q.topic_id = t.topic_id

            WHERE q.student_id = ?

            ORDER BY q.quiz_date DESC
            """,
            (
                student_id,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall(),
            "quiz_id"
        )


def get_quiz_results_for_topic(
    student_id=None,
    topic_id=None
):
    # Compatibility:
    # get_quiz_results_for_topic(topic_id)
    # and
    # get_quiz_results_for_topic(student_id, topic_id)

    if topic_id is None:
        topic_id = student_id
        student_id = None

    with get_connection() as connection:
        cursor = connection.cursor()

        if student_id is None:
            cursor.execute(
                f"""
                SELECT
                    q.*,
                    t.topic_name,
                    t.subject_name

                FROM {QUIZ_RESULTS_TABLE} AS q

                JOIN {TOPICS_TABLE} AS t
                ON q.topic_id = t.topic_id

                WHERE q.topic_id = ?

                ORDER BY q.quiz_date ASC
                """,
                (
                    topic_id,
                ),
            )

        else:
            cursor.execute(
                f"""
                SELECT
                    q.*,
                    t.topic_name,
                    t.subject_name

                FROM {QUIZ_RESULTS_TABLE} AS q

                JOIN {TOPICS_TABLE} AS t
                ON q.topic_id = t.topic_id

                WHERE q.student_id = ?
                AND q.topic_id = ?

                ORDER BY q.quiz_date ASC
                """,
                (
                    student_id,
                    topic_id,
                ),
            )

        return rows_to_dicts(
            cursor.fetchall(),
            "quiz_id"
        )


# research comparison

def get_prediction_recall_pairs(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT
                s.session_id,
                s.student_id,
                s.topic_id,
                s.studied_at,

                t.topic_name,
                t.subject_name,
                t.difficulty,

                p.prediction_id,
                p.fixed_prediction,
                p.personalized_prediction,
                p.adaptive_prediction,
                p.predicted_half_life,
                p.prediction_for_days,
                p.prediction_mode,
                p.created_at
                    AS prediction_created_at,

                q.quiz_id,
                q.actual_score,
                q.recall_accuracy,
                q.delay_days,
                q.quiz_date

            FROM {STUDY_SESSIONS_TABLE} AS s

            JOIN {TOPICS_TABLE} AS t
            ON s.topic_id = t.topic_id

            JOIN {PREDICTIONS_TABLE} AS p
            ON s.session_id = p.session_id

            JOIN {QUIZ_RESULTS_TABLE} AS q
            ON s.session_id = q.session_id

            WHERE s.student_id = ?

            AND p.created_at <= q.quiz_date

            ORDER BY q.quiz_date ASC,
                     p.created_at ASC
            """,
            (
                student_id,
            ),
        )

        return rows_to_dicts(
            cursor.fetchall()
        )


# statistics

def get_student_statistics(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {TOPICS_TABLE}
            WHERE student_id = ?
            """,
            (
                student_id,
            ),
        )

        topic_count = (
            cursor.fetchone()[0]
        )

        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {STUDY_SESSIONS_TABLE}
            WHERE student_id = ?
            """,
            (
                student_id,
            ),
        )

        session_count = (
            cursor.fetchone()[0]
        )

        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {QUIZ_RESULTS_TABLE}
            WHERE student_id = ?
            """,
            (
                student_id,
            ),
        )

        quiz_count = (
            cursor.fetchone()[0]
        )

        cursor.execute(
            f"""
            SELECT AVG(actual_score)
            FROM {QUIZ_RESULTS_TABLE}
            WHERE student_id = ?
            """,
            (
                student_id,
            ),
        )

        average_score = (
            cursor.fetchone()[0]
        )

        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {REVISION_SCHEDULE_TABLE}

            WHERE student_id = ?
            AND completed = 0
            """,
            (
                student_id,
            ),
        )

        pending_revisions = (
            cursor.fetchone()[0]
        )

        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {REVISION_SCHEDULE_TABLE}

            WHERE student_id = ?
            AND completed = 1
            """,
            (
                student_id,
            ),
        )

        completed_revisions = (
            cursor.fetchone()[0]
        )

    if average_score is None:
        average_score = 0.0

    return {
        "topics":
            topic_count,

        "study_sessions":
            session_count,

        "quizzes":
            quiz_count,

        "average_quiz_score":
            round(
                average_score,
                2
            ),

        "pending_revisions":
            pending_revisions,

        "completed_revisions":
            completed_revisions,
    }


# deletion

def delete_topic(
    topic_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            DELETE FROM {TOPICS_TABLE}
            WHERE topic_id = ?
            """,
            (
                topic_id,
            ),
        )

        return (
            cursor.rowcount > 0
        )


def delete_student(
    student_id
):
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            DELETE FROM {STUDENTS_TABLE}
            WHERE student_id = ?
            """,
            (
                student_id,
            ),
        )

        return (
            cursor.rowcount > 0
        )


# database status

def get_database_status():
    if not DATABASE_PATH.exists():
        return {
            "database_exists":
                False,

            "students":
                0,

            "topics":
                0,

            "study_sessions":
                0,

            "predictions":
                0,

            "revision_schedule":
                0,

            "quiz_results":
                0,
        }

    tables = {
        "students":
            STUDENTS_TABLE,

        "topics":
            TOPICS_TABLE,

        "study_sessions":
            STUDY_SESSIONS_TABLE,

        "predictions":
            PREDICTIONS_TABLE,

        "revision_schedule":
            REVISION_SCHEDULE_TABLE,

        "quiz_results":
            QUIZ_RESULTS_TABLE,
    }

    status = {
        "database_exists":
            True
    }

    with get_connection() as connection:
        cursor = connection.cursor()

        for (
            key,
            table_name
        ) in tables.items():

            try:
                cursor.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table_name}
                    """
                )

                status[
                    key
                ] = cursor.fetchone()[0]

            except sqlite3.OperationalError:
                status[
                    key
                ] = 0

    return status


# direct execution

if __name__ == "__main__":
    initialize_database()

    print()
    print(
        "MemoryMap Database"
    )
    print(
        "=" * 40
    )

    print(
        f"Database path: {DATABASE_PATH}"
    )

    print(
        "Database initialized successfully."
    )

    print()

    status = (
        get_database_status()
    )

    for (
        key,
        value
    ) in status.items():

        print(
            f"{key}: {value}"
        )