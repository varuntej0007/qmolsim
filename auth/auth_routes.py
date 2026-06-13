"""
auth/auth_routes.py
Auth API endpoints:
  POST /auth/login
  POST /auth/logout
  GET  /auth/me
  POST /auth/users          (admin)
  GET  /auth/users          (admin)
  PUT  /auth/users/<id>     (admin)
  GET  /auth/audit          (admin/manager)
  POST /auth/sign/<analysis_id>  (electronic signature)
"""

import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, g

from auth.auth import (
    verify_password, create_session, get_user_from_token,
    audit_log, require_auth, require_admin, current_user, hash_password,
)
from auth.database import db

logger   = logging.getLogger(__name__)
auth_bp  = Blueprint("auth", __name__)


@auth_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username","").strip()
    password = data.get("password","")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    with db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1",
            (username,)
        ).fetchone()

    if not user or not verify_password(password, user["password_hash"]):
        audit_log(None, username, "LOGIN_FAILED",
                  details="Invalid credentials", status="FAILED")
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_session(
        user["id"],
        ip=request.remote_addr,
        ua=request.headers.get("User-Agent","")
    )
    audit_log(user["id"], user["username"], "LOGIN",
              details=f"Login from {request.remote_addr}")

    resp = jsonify({
        "token":    token,
        "user": {
            "id":         user["id"],
            "username":   user["username"],
            "email":      user["email"],
            "role":       user["role"],
            "department": user["department"],
            "full_name":  user["full_name"],
        },
        "expires_in_hours": 8,
    })
    resp.set_cookie("qmolsim_token", token, httponly=True,
                    samesite="Lax", max_age=8*3600)
    return resp


@auth_bp.route("/auth/logout", methods=["POST"])
def logout():
    token = request.headers.get("X-Auth-Token") or \
            request.cookies.get("qmolsim_token")
    if token:
        user = get_user_from_token(token)
        with db() as conn:
            conn.execute("UPDATE sessions SET is_active=0 WHERE token=?",
                         (token,))
        if user:
            audit_log(user["id"], user["username"], "LOGOUT")
    resp = jsonify({"message": "Logged out"})
    resp.delete_cookie("qmolsim_token")
    return resp


@auth_bp.route("/auth/me", methods=["GET"])
@require_auth("viewer")
def me():
    u = current_user()
    return jsonify({
        "id":         u["id"],
        "username":   u["username"],
        "email":      u["email"],
        "role":       u["role"],
        "department": u["department"],
        "full_name":  u["full_name"],
        "last_login": u["last_login"],
    })


@auth_bp.route("/auth/users", methods=["POST"])
@require_admin
def create_user():
    data = request.get_json(silent=True) or {}
    required = ["username", "email", "password", "role"]
    for f in required:
        if not data.get(f):
            return jsonify({"error": f"Missing: {f}"}), 400

    if data["role"] not in ["admin","manager","analyst","viewer"]:
        return jsonify({"error": "Invalid role"}), 400

    try:
        with db() as conn:
            conn.execute("""
                INSERT INTO users
                (username, email, password_hash, role, department,
                 full_name, created_by)
                VALUES (?,?,?,?,?,?,?)
            """, (
                data["username"], data["email"],
                hash_password(data["password"]),
                data["role"],
                data.get("department",""),
                data.get("full_name",""),
                g.user["id"],
            ))
        u = g.user
        audit_log(u["id"], u["username"], "USER_CREATED",
                  resource="users", resource_id=data["username"],
                  details=f"Role: {data['role']}")
        return jsonify({"message": f"User {data['username']} created"}), 201
    except Exception as e:
        if "UNIQUE" in str(e):
            return jsonify({"error": "Username or email already exists"}), 409
        raise


@auth_bp.route("/auth/users", methods=["GET"])
@require_auth("manager")
def list_users():
    with db() as conn:
        rows = conn.execute("""
            SELECT id, username, email, role, department,
                   full_name, is_active, created_at, last_login
            FROM users ORDER BY created_at DESC
        """).fetchall()
    return jsonify([dict(r) for r in rows])


@auth_bp.route("/auth/users/<int:uid>", methods=["PUT"])
@require_admin
def update_user(uid):
    data = request.get_json(silent=True) or {}
    allowed = ["role","department","full_name","is_active"]
    updates = {k: data[k] for k in allowed if k in data}
    if "password" in data:
        updates["password_hash"] = hash_password(data["password"])
    if not updates:
        return jsonify({"error": "Nothing to update"}), 400
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values     = list(updates.values()) + [uid]
    with db() as conn:
        conn.execute(f"UPDATE users SET {set_clause} WHERE id=?", values)
    u = g.user
    audit_log(u["id"], u["username"], "USER_UPDATED",
              resource="users", resource_id=str(uid),
              details=str(updates))
    return jsonify({"message": "User updated"})


@auth_bp.route("/auth/audit", methods=["GET"])
@require_auth("manager")
def get_audit_log():
    limit  = min(int(request.args.get("limit", 100)), 1000)
    offset = int(request.args.get("offset", 0))
    action = request.args.get("action")
    user   = request.args.get("user")

    query  = "SELECT * FROM audit_logs WHERE 1=1"
    params = []
    if action:
        query += " AND action LIKE ?"
        params.append(f"%{action}%")
    if user:
        query += " AND username LIKE ?"
        params.append(f"%{user}%")
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    with db() as conn:
        rows  = conn.execute(query, params).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]

    return jsonify({
        "total":  total,
        "limit":  limit,
        "offset": offset,
        "logs":   [dict(r) for r in rows],
    })


@auth_bp.route("/auth/sign/<int:analysis_id>", methods=["POST"])
@require_auth("analyst")
def electronic_sign(analysis_id):
    """
    21 CFR Part 11 electronic signature.
    Requires password re-verification.
    """
    import hashlib, secrets as sc
    data     = request.get_json(silent=True) or {}
    password = data.get("password","")
    meaning  = data.get("meaning", "I approve this analysis result")

    u = g.user
    with db() as conn:
        user_row = conn.execute(
            "SELECT * FROM users WHERE id=?", (u["id"],)
        ).fetchone()

    if not verify_password(password, user_row["password_hash"]):
        audit_log(u["id"], u["username"], "SIGNATURE_FAILED",
                  resource="analyses", resource_id=str(analysis_id),
                  status="FAILED")
        return jsonify({"error": "Password verification failed"}), 401

    sig_data  = f"{u['id']}:{analysis_id}:{meaning}:{datetime.utcnow().isoformat()}"
    signature = hashlib.sha256(sig_data.encode()).hexdigest()

    with db() as conn:
        conn.execute("""
            INSERT INTO signatures
            (user_id, analysis_id, meaning, signature, ip_address)
            VALUES (?,?,?,?,?)
        """, (u["id"], analysis_id, meaning, signature, request.remote_addr))

    audit_log(u["id"], u["username"], "ELECTRONIC_SIGNATURE",
              resource="analyses", resource_id=str(analysis_id),
              details=f"Meaning: {meaning}")

    return jsonify({
        "signed":      True,
        "analysis_id": analysis_id,
        "signed_by":   u["username"],
        "meaning":     meaning,
        "signature":   signature,
        "timestamp":   datetime.utcnow().isoformat(),
    })
