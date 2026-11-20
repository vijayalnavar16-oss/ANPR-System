"""
SQLite Database Handler for AI-Based Indian Vehicle ANPR System.

Manages persistent logging of vehicle detections, audit trails,
safe PRAGMA schema migrations, and real-time dashboard analytics.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from config import DATABASE_PATH


def get_db_connection():
    """Create and return a SQLite connection with row dict access."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize database tables with safe column migrations and indexes.
    Preserves all existing data without resetting or deleting tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Base table creation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT
        )
    """)
    conn.commit()

    # 2. Check existing columns for safe migration
    cursor.execute("PRAGMA table_info(vehicle_logs)")
    columns = [row["name"] for row in cursor.fetchall()]

    if "raw_text" not in columns:
        cursor.execute("ALTER TABLE vehicle_logs ADD COLUMN raw_text TEXT DEFAULT ''")

    if "verification_status" not in columns:
        cursor.execute("ALTER TABLE vehicle_logs ADD COLUMN verification_status TEXT DEFAULT 'MANUAL_VERIFICATION'")

    if "verified_by" not in columns:
        cursor.execute("ALTER TABLE vehicle_logs ADD COLUMN verified_by TEXT DEFAULT 'SYSTEM'")

    if "manual_notes" not in columns:
        cursor.execute("ALTER TABLE vehicle_logs ADD COLUMN manual_notes TEXT DEFAULT ''")

    if "detection_confidence" not in columns:
        cursor.execute("ALTER TABLE vehicle_logs ADD COLUMN detection_confidence REAL DEFAULT 0.0")

    if "registration_type" not in columns:
        cursor.execute("ALTER TABLE vehicle_logs ADD COLUMN registration_type TEXT DEFAULT 'STANDARD'")

    if "frame_count" not in columns:
        cursor.execute("ALTER TABLE vehicle_logs ADD COLUMN frame_count INTEGER DEFAULT 1")

    # 3. Create performance indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate_number ON vehicle_logs(plate_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON vehicle_logs(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON vehicle_logs(verification_status)")

    conn.commit()
    conn.close()


def insert_record(
    plate_number: str,
    raw_text: str = "",
    confidence: float = 0.0,
    image_path: str = "",
    verification_status: str = "MANUAL_VERIFICATION",
    verified_by: str = "SYSTEM",
    manual_notes: str = "",
    detection_confidence: float = 0.0,
    registration_type: str = "STANDARD",
    frame_count: int = 1
) -> int:
    """
    Insert a vehicle plate entry into the database.
    Stores recognized plate, raw OCR text, confidence, timestamp, crop path,
    validation status, detection confidence, registration type, and frame observations.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO vehicle_logs (
            plate_number, raw_text, confidence, timestamp, image_path,
            verification_status, verified_by, manual_notes,
            detection_confidence, registration_type, frame_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        plate_number,
        raw_text,
        round(confidence, 4),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_path,
        verification_status,
        verified_by,
        manual_notes,
        round(detection_confidence, 4),
        registration_type,
        int(frame_count)
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_all_records(limit: int = 50):
    """Fetch recent vehicle logs ordered by newest first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plate_number, raw_text, confidence, timestamp, image_path,
               verification_status, verified_by, manual_notes,
               detection_confidence, registration_type, frame_count
        FROM vehicle_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_record_by_id(record_id: int):
    """Retrieve a single vehicle log by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plate_number, raw_text, confidence, timestamp, image_path,
               verification_status, verified_by, manual_notes,
               detection_confidence, registration_type, frame_count
        FROM vehicle_logs
        WHERE id = ?
    """, (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def search_and_filter_records(
    search_query: str = "",
    date_filter: str = "",
    status_filter: str = "",
    limit: int = 100
):
    """
    Filter records by plate number, specific date (YYYY-MM-DD), and verification status.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT id, plate_number, raw_text, confidence, timestamp, image_path,
               verification_status, verified_by, manual_notes,
               detection_confidence, registration_type, frame_count
        FROM vehicle_logs
        WHERE 1=1
    """
    params = []

    if search_query:
        query += " AND (plate_number LIKE ? OR raw_text LIKE ?)"
        term = f"%{search_query.strip().upper()}%"
        params.extend([term, term])

    if date_filter:
        query += " AND date(timestamp) = date(?)"
        params.append(date_filter.strip())

    if status_filter and status_filter.upper() != "ALL":
        query += " AND verification_status = ?"
        params.append(status_filter.strip().upper())

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_summary_stats():
    """
    Aggregate metrics for the dashboard header:
    - Total unique vehicle events
    - Today's vehicle entries
    - Automatically accepted valid Indian plates
    - Pending rechecks
    - Rejected false detections / watermarks
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM vehicle_logs")
    total_detected = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE date(timestamp) = date('now', 'localtime')")
    today_entries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE verification_status IN ('VALID_INDIAN_PLATE', 'CONFIRMED')")
    valid_plates = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE verification_status IN ('PENDING_RECHECK', 'MANUAL_VERIFICATION')")
    pending_verification = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE verification_status IN ('REJECTED', 'REJECTED_INVALID_FORMAT', 'REJECTED_WATERMARK', 'UNREADABLE')")
    rejected_detections = cursor.fetchone()[0]

    conn.close()
    return {
        "total_detected": total_detected,
        "today_entries": today_entries,
        "valid_plates": valid_plates,
        "pending_verification": pending_verification,
        "rejected_detections": rejected_detections
    }


def update_record_manual_verification(record_id: int, corrected_plate: str, new_status: str, notes: str = ""):
    """
    Update a plate entry following human review.
    Preserves original raw_text for audit integrity.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE vehicle_logs
        SET plate_number = ?,
            verification_status = ?,
            verified_by = 'MANUAL_AUDIT',
            manual_notes = ?
        WHERE id = ?
    """, (corrected_plate.strip().upper(), new_status.strip().upper(), notes.strip(), record_id))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# Run safe migrations on module initialization
init_db()
