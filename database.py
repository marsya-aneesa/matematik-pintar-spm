# ============================================================
# database.py
# SQLite database layer for Adaptive Mathematics Learning System
# Includes users, profiles, quiz sessions, answers, chatbot logs,
# and FSRS-style adaptation records.
# ============================================================

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("adaptive_math.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        study_hours REAL,
        attendance REAL,
        resources INTEGER,
        extracurricular INTEGER,
        motivation INTEGER,
        internet INTEGER,
        gender INTEGER,
        age INTEGER,
        learning_style INTEGER,
        online_courses INTEGER,
        discussions INTEGER,
        assignment_completion REAL,
        exam_score REAL,
        edutech INTEGER,
        stress_level INTEGER,
        predicted_level INTEGER,
        predicted_level_text TEXT,
        confidence REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        initial_level INTEGER,
        final_level INTEGER,
        score REAL DEFAULT 0,
        correct_count INTEGER DEFAULT 0,
        total_questions INTEGER DEFAULT 0,
        hints_used INTEGER DEFAULT 0,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        question_id TEXT,
        topic TEXT,
        difficulty TEXT,
        selected_answer TEXT,
        correct_answer TEXT,
        is_correct INTEGER,
        time_taken REAL,
        max_time REAL,
        used_hint INTEGER,
        level_before INTEGER,
        level_after INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES quiz_sessions(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        question_id TEXT,
        user_message TEXT,
        bot_reply TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES quiz_sessions(id)
    )
    """)

    # FSRS-style review log: every answered question becomes a review event.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fsrs_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER NOT NULL,
        question_id TEXT,
        topic TEXT,
        grade INTEGER,
        response_ratio REAL,
        fsrs_difficulty REAL,
        fsrs_stability REAL,
        retrievability REAL,
        next_level INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(session_id) REFERENCES quiz_sessions(id)
    )
    """)

    # Latest FSRS state per user-topic.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fsrs_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        fsrs_difficulty REAL DEFAULT 5.0,
        fsrs_stability REAL DEFAULT 1.0,
        latest_grade INTEGER,
        current_level INTEGER DEFAULT 0,
        review_count INTEGER DEFAULT 0,
        last_reviewed TEXT,
        UNIQUE(user_id, topic),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()


def get_or_create_user(username: str) -> int:
    username = username.strip()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if row:
        conn.close()
        return int(row["id"])
    cur.execute("INSERT INTO users (username, created_at) VALUES (?, ?)", (username, datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return int(user_id)


def save_profile(user_id: int, profile: dict, predicted_level: int, predicted_level_text: str, confidence: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO profiles (
        user_id, study_hours, attendance, resources, extracurricular, motivation, internet, gender, age,
        learning_style, online_courses, discussions, assignment_completion, exam_score, edutech, stress_level,
        predicted_level, predicted_level_text, confidence, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        profile.get("StudyHours"), profile.get("Attendance"), profile.get("Resources"), profile.get("Extracurricular"),
        profile.get("Motivation"), profile.get("Internet"), profile.get("Gender"), profile.get("Age"),
        profile.get("LearningStyle"), profile.get("OnlineCourses"), profile.get("Discussions"),
        profile.get("AssignmentCompletion"), profile.get("ExamScore"), profile.get("EduTech"), profile.get("StressLevel"),
        predicted_level, predicted_level_text, confidence, datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()


def create_quiz_session(user_id: int, initial_level: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO quiz_sessions (user_id, initial_level, final_level, started_at)
    VALUES (?, ?, ?, ?)
    """, (user_id, initial_level, initial_level, datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return int(session_id)


def save_answer(session_id: int, answer: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO quiz_answers (
        session_id, question_id, topic, difficulty, selected_answer, correct_answer, is_correct,
        time_taken, max_time, used_hint, level_before, level_after, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        str(answer.get("question_id")), answer.get("topic"), answer.get("difficulty"), answer.get("selected_answer"),
        answer.get("correct_answer"), int(answer.get("is_correct", 0)), float(answer.get("time_taken", 0)),
        float(answer.get("max_time", 0)), int(answer.get("used_hint", 0)), int(answer.get("level_before", 0)),
        int(answer.get("level_after", 0)), datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()


def save_chat(session_id: int, question_id, user_message: str, bot_reply: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO chat_logs (session_id, question_id, user_message, bot_reply, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (session_id, str(question_id), user_message, bot_reply, datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    conn.close()


def complete_session(session_id: int, final_level: int, score: float, correct_count: int, total_questions: int, hints_used: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    UPDATE quiz_sessions
    SET final_level = ?, score = ?, correct_count = ?, total_questions = ?, hints_used = ?, completed_at = ?
    WHERE id = ?
    """, (final_level, score, correct_count, total_questions, hints_used, datetime.now().isoformat(timespec="seconds"), session_id))
    conn.commit()
    conn.close()


def fetch_sessions(user_id: int):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM quiz_sessions
        WHERE user_id = ? AND completed_at IS NOT NULL
        ORDER BY completed_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_answers(session_id: int):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM quiz_answers
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_latest_profile(user_id: int):
    conn = get_conn()
    row = conn.execute("""
        SELECT * FROM profiles
        WHERE user_id = ?
        ORDER BY id DESC LIMIT 1
    """, (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_fsrs_progress(user_id: int):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM fsrs_progress
        WHERE user_id = ?
        ORDER BY topic ASC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_fsrs_topic_state(user_id: int, topic: str):
    conn = get_conn()
    row = conn.execute("""
        SELECT * FROM fsrs_progress
        WHERE user_id = ? AND topic = ?
    """, (user_id, topic)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "fsrs_difficulty": 5.0,
        "fsrs_stability": 1.0,
        "current_level": 0,
        "review_count": 0,
    }


def save_fsrs_review(user_id: int, session_id: int, question_id: str, topic: str, fsrs_result: dict):
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO fsrs_reviews (
        user_id, session_id, question_id, topic, grade, response_ratio,
        fsrs_difficulty, fsrs_stability, retrievability, next_level, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, session_id, str(question_id), topic,
        int(fsrs_result["grade"]), float(fsrs_result["response_ratio"]),
        float(fsrs_result["difficulty"]), float(fsrs_result["stability"]),
        float(fsrs_result["retrievability"]), int(fsrs_result["new_level"]), now
    ))

    cur.execute("""
    INSERT INTO fsrs_progress (
        user_id, topic, fsrs_difficulty, fsrs_stability, latest_grade,
        current_level, review_count, last_reviewed
    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
    ON CONFLICT(user_id, topic) DO UPDATE SET
        fsrs_difficulty = excluded.fsrs_difficulty,
        fsrs_stability = excluded.fsrs_stability,
        latest_grade = excluded.latest_grade,
        current_level = excluded.current_level,
        review_count = fsrs_progress.review_count + 1,
        last_reviewed = excluded.last_reviewed
    """, (
        user_id, topic, float(fsrs_result["difficulty"]), float(fsrs_result["stability"]),
        int(fsrs_result["grade"]), int(fsrs_result["new_level"]), now
    ))

    conn.commit()
    conn.close()
