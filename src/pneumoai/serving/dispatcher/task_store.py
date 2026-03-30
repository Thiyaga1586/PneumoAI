from typing import Optional

from pneumoai.storage.sqlite import get_connection


def add_task(request_id: str, payload: dict) -> None:
    conn = get_connection()
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
    conn.close()


def get_task(request_id: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT request_id, status, submitted_at, image_uri, true_label
        FROM tasks
        WHERE request_id = ?
    """, (request_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "request_id": row[0],
        "status": row[1],
        "submitted_at": row[2],
        "image_uri": row[3],
        "true_label": row[4],
    }


def pop_next_queued_task() -> Optional[dict]:
    conn = get_connection()
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
        conn.close()
        return None

    request_id = row[0]

    cursor.execute("""
        UPDATE tasks
        SET status = 'processing'
        WHERE request_id = ?
    """, (request_id,))
    conn.commit()
    conn.close()

    return {
        "request_id": row[0],
        "status": "processing",
        "submitted_at": row[2],
        "image_uri": row[3],
        "true_label": row[4],
    }


def mark_task_completed(request_id: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE tasks
        SET status = 'completed'
        WHERE request_id = ?
    """, (request_id,))
    conn.commit()
    conn.close()


def clear_tasks() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks")
    conn.commit()
    conn.close()