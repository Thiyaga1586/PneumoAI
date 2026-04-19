from typing import Optional

from pneumoai.storage.sqlite import get_connection, init_db


def add_task(request_id: str, payload: dict) -> None:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tasks
            (request_id, status, submitted_at, image_uri, true_label)
            VALUES (?, ?, ?, ?, ?)
        """, (
            request_id,
            payload["status"],
            payload["submitted_at"],
            payload["image_uri"],
            payload.get("true_label"),
        ))
        conn.commit()
    finally:
        conn.close()


def get_task(request_id: str) -> Optional[dict]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT request_id, status, submitted_at, image_uri, true_label
            FROM tasks
            WHERE request_id = ?
        """, (request_id,))
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "request_id": row["request_id"],
        "status": row["status"],
        "submitted_at": row["submitted_at"],
        "image_uri": row["image_uri"],
        "true_label": row["true_label"],
    }


def pop_next_queued_task() -> Optional[dict]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT request_id, status, submitted_at, image_uri, true_label
            FROM tasks
            WHERE status = 'queued'
            ORDER BY submitted_at ASC
            LIMIT 1
        """)
        row = cursor.fetchone()

        if row is None:
            return None

        request_id = row["request_id"]

        cursor.execute("""
            UPDATE tasks
            SET status = 'processing'
            WHERE request_id = ?
        """, (request_id,))
        conn.commit()

        return {
            "request_id": row["request_id"],
            "status": "processing",
            "submitted_at": row["submitted_at"],
            "image_uri": row["image_uri"],
            "true_label": row["true_label"],
        }
    finally:
        conn.close()


def mark_task_completed(request_id: str) -> None:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE tasks
            SET status = 'completed'
            WHERE request_id = ?
        """, (request_id,))
        conn.commit()
    finally:
        conn.close()


def clear_tasks() -> None:
    init_db(force=True)
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks")
        conn.commit()
    finally:
        conn.close()