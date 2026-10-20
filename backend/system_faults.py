"""Durable, bounded system-fault records shared by the API and host collector."""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def database_path() -> Path:
    root = Path(os.getenv("DEEPVISION_DATA_DIR", "/app/data"))
    return root / "system_faults" / "faults.sqlite3"


def _redact(value: str, limit: int = 6000) -> str:
    text = str(value or "")[:limit]
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", text)
    text = re.sub(r'''(?i)(\b[\w-]*(?:password|passwd|token|secret|api[_-]?key)[\w-]*["']?\s*[:=]\s*["']?)[^"'\s,;&]+''',
                  r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(rtsps?://[^:/\s]+:)[^@/\s]+@", r"\1[REDACTED]@", text)
    return text


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        path.parent.chmod(0o700)
    conn = sqlite3.connect(path, timeout=5)
    if os.name != "nt":
        path.chmod(0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""CREATE TABLE IF NOT EXISTS faults (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_at TEXT NOT NULL,
        service TEXT NOT NULL,
        severity TEXT NOT NULL,
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL,
        container_id TEXT NOT NULL DEFAULT ''
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_faults_time ON faults(occurred_at DESC, id DESC)")
    return conn


def record_fault(*, service: str, severity: str, category: str, message: str,
                 details: str = "", source: str = "application", container_id: str = "",
                 occurred_at: str | None = None) -> None:
    if occurred_at:
        parsed = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        timestamp = parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
    else:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn:
        with conn:
            conn.execute("""INSERT INTO faults
                (occurred_at, service, severity, category, message, details, source, container_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, _redact(service, 80), _redact(severity, 20), _redact(category, 80),
                 _redact(message, 1200), _redact(details), _redact(source, 40), _redact(container_id, 80)))


def list_faults(*, page: int = 1, page_size: int = 20, service: str = "",
                severity: str = "", start: str = "", end: str = "") -> dict:
    filters, values = [], []
    for column, value in (("service", service), ("severity", severity)):
        if value:
            filters.append(f"{column} = ?")
            values.append(value)
    if start:
        start = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds")
        filters.append("occurred_at >= ?")
        values.append(start)
    if end:
        end = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds")
        filters.append("occurred_at <= ?")
        values.append(end)
    where = " WHERE " + " AND ".join(filters) if filters else ""
    with closing(_connect()) as conn:
        total = conn.execute("SELECT count(*) FROM faults" + where, values).fetchone()[0]
        rows = conn.execute("SELECT * FROM faults" + where + " ORDER BY occurred_at DESC, id DESC LIMIT ? OFFSET ?",
                            (*values, page_size, (page - 1) * page_size)).fetchall()
    return {"total": total, "rows": [dict(row) for row in rows]}
