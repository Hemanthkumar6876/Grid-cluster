"""Minimal local account system (SQLite + salted PBKDF2-HMAC-SHA256).

Accounts live in ``data/users.db`` (override with the MICROGRID_DB environment
variable). Passwords are never stored in plain text. This is a local, single-machine
login for the demo application - it is not an OAuth / single-sign-on service.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _db_path() -> Path:
    p = Path(os.environ.get("MICROGRID_DB", Path(__file__).parent / "data" / "users.db"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_db_path())
    c.row_factory = sqlite3.Row
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               email TEXT NOT NULL UNIQUE,
               salt BLOB NOT NULL,
               pwd_hash BLOB NOT NULL,
               created TEXT NOT NULL)"""
    )
    return c


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def _public(row: sqlite3.Row) -> dict:
    return {"name": row["name"], "email": row["email"], "created": row["created"], "guest": False}


def validate_password(password: str) -> str | None:
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if password.isalpha() or password.isdigit():
        return "Use a mix of letters and numbers."
    return None


def create_user(name: str, email: str, password: str) -> tuple[bool, str]:
    name, email = name.strip(), email.strip().lower()
    if len(name) < 2:
        return False, "Please enter your full name."
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    err = validate_password(password)
    if err:
        return False, err
    salt = secrets.token_bytes(16)
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (name, email, salt, pwd_hash, created) VALUES (?,?,?,?,?)",
                (name, email, salt, _hash(password, salt), datetime.now(timezone.utc).strftime("%Y-%m-%d")),
            )
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    return True, "Account created."


def verify(email: str, password: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if row is None:
        _hash(password, b"\x00" * 16)  # keep timing similar for unknown emails
        return None
    if hmac.compare_digest(_hash(password, row["salt"]), row["pwd_hash"]):
        return _public(row)
    return None


def update_name(email: str, name: str) -> bool:
    name = name.strip()
    if len(name) < 2:
        return False
    with _conn() as c:
        c.execute("UPDATE users SET name = ? WHERE email = ?", (name, email.lower()))
    return True


def change_password(email: str, old: str, new: str) -> tuple[bool, str]:
    if verify(email, old) is None:
        return False, "Current password is incorrect."
    err = validate_password(new)
    if err:
        return False, err
    salt = secrets.token_bytes(16)
    with _conn() as c:
        c.execute("UPDATE users SET salt = ?, pwd_hash = ? WHERE email = ?", (salt, _hash(new, salt), email.lower()))
    return True, "Password updated."


def guest() -> dict:
    return {"name": "Guest", "email": "guest@local", "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "guest": True}
