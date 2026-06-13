"""
auth/auth.py
Authentication: password hashing, session tokens, role-based access.
21 CFR Part 11 compliant audit trail on every action.
"""

import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g

logger = logging.getLogger(__name__)

# Roles hierarchy
ROLES = {
    "admin":     4,
    "manager":   3,
    "analyst":   2,
    "viewer":    1,
}

SESSION_HOURS = 8


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    )
    return f"{salt}:{h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":")
        new_h = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100_000
        )
        return secrets.compare_digest(h, new_h.hex())
    except Exception:
        return False


def create_session(user_id: int, ip: str = None, ua: str = None) -> str:
    from auth.database import db
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(hours=SESSION_HOURS)
    with db() as conn:
        conn.execute("""
            INSERT INTO sessions (user_id, token, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, token, expires, ip, ua))
    return token


def get_user_from_token(token: str):
    from auth.database import db
    if not token:
        return None
    with db() as conn:
        row = conn.execute("""
            SELECT u.*, s.id as session_id, s.expires_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ? AND s.is_active = 1
        """, (token,)).fetchone()
        if not row:
            return None
        exp = row["expires_at"]

        if isinstance(exp, str):
            exp = datetime.fromisoformat(exp)

        if exp < datetime.utcnow():
            conn.execute("UPDATE sessions SET is_active=0 WHERE token=?", (token,))
            return None
        # Update last login
        conn.execute("UPDATE users SET last_login=? WHERE id=?",
                     (datetime.utcnow(), row["id"]))
    return dict(row)


def audit_log(user_id, username, action, resource=None,
              resource_id=None, details=None, status="SUCCESS"):
    from auth.database import db
    try:
        ip = request.remote_addr if request else None
        sid = getattr(g, "session_token", None) if request else None
        with db() as conn:
            conn.execute("""
                INSERT INTO audit_logs
                (user_id, username, action, resource, resource_id,
                 details, ip_address, status, session_id)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (user_id, username, action, resource, resource_id,
                  details, ip, status, sid))
    except Exception as e:
        logger.error(f"Audit log failed: {e}")


def require_auth(min_role: str = "viewer"):
    """Decorator: require authenticated user with minimum role."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
            else:
                token = (
                    request.headers.get("X-Auth-Token")
                    or request.cookies.get("qmolsim_token")
                )
            user = get_user_from_token(token)
            if not user:
                audit_log(None, "anonymous", "AUTH_FAILED",
                          details=f"Attempted {request.path}",
                          status="FAILED")
                return jsonify({"error": "Authentication required",
                                "code": "AUTH_REQUIRED"}), 401
            if ROLES.get(user["role"], 0) < ROLES.get(min_role, 0):
                audit_log(user["id"], user["username"], "PERMISSION_DENIED",
                          details=f"Required {min_role}, has {user['role']}",
                          status="FAILED")
                return jsonify({"error": "Insufficient permissions",
                                "code": "PERMISSION_DENIED"}), 403
            g.user = user
            g.session_token = token
            return f(*args, **kwargs)
        return wrapped
    return decorator


def require_admin(f):
    return require_auth("admin")(f)


def current_user():
    return getattr(g, "user", None)
