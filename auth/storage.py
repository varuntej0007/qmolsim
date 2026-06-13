"""
auth/storage.py
Persistent storage for analyses, molecules, projects.
Every save is audit-logged automatically.
"""

import json
import logging
from datetime import datetime
from auth.database import db
from auth.auth import audit_log

logger = logging.getLogger(__name__)


def save_analysis(user_id: int, smiles: str, molecule_name: str,
                  analysis_type: str, result: dict,
                  project_id: int = None,
                  notes: str = None) -> int:
    """Save analysis result to DB. Returns analysis ID."""
    api_score = result.get("api_score") or result.get("overall_score")
    with db() as conn:
        cur = conn.execute("""
            INSERT INTO analyses
            (user_id, smiles, molecule_name, analysis_type,
             result_json, api_score, project_id, notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            user_id, smiles, molecule_name, analysis_type,
            json.dumps(result), api_score, project_id, notes,
        ))
        analysis_id = cur.lastrowid

    with db() as conn:
        user = conn.execute("SELECT username FROM users WHERE id=?",
                            (user_id,)).fetchone()
    username = user["username"] if user else str(user_id)

    audit_log(user_id, username, "ANALYSIS_SAVED",
              resource="analyses", resource_id=str(analysis_id),
              details=f"{analysis_type} | {molecule_name} | score={api_score}")
    return analysis_id


def get_analyses(user_id: int = None, project_id: int = None,
                 analysis_type: str = None, limit: int = 50) -> list:
    query  = "SELECT * FROM analyses WHERE 1=1"
    params = []
    if user_id:
        query += " AND user_id=?"
        params.append(user_id)
    if project_id:
        query += " AND project_id=?"
        params.append(project_id)
    if analysis_type:
        query += " AND analysis_type=?"
        params.append(analysis_type)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db() as conn:
        rows = conn.execute(query, params).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["result"] = json.loads(d["result_json"])
        except:
            d["result"] = {}
        results.append(d)
    return results


def save_molecule(smiles: str, name: str, category: str,
                  user_id: int, notes: str = None) -> int:
    with db() as conn:
        existing = conn.execute(
            "SELECT id FROM molecules WHERE smiles=?", (smiles,)
        ).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute("""
            INSERT INTO molecules (smiles, name, category, created_by, notes)
            VALUES (?,?,?,?,?)
        """, (smiles, name, category, user_id, notes))
        return cur.lastrowid


def create_project(name: str, description: str, owner_id: int) -> int:
    with db() as conn:
        cur = conn.execute("""
            INSERT INTO projects (name, description, owner_id)
            VALUES (?,?,?)
        """, (name, description, owner_id))
        return cur.lastrowid


def get_projects(user_id: int) -> list:
    with db() as conn:
        rows = conn.execute("""
            SELECT p.*, u.username as owner_name,
                   COUNT(a.id) as analysis_count
            FROM projects p
            JOIN users u ON u.id = p.owner_id
            LEFT JOIN analyses a ON a.project_id = p.id
            WHERE p.owner_id=? OR p.id IN (
                SELECT DISTINCT project_id FROM analyses WHERE user_id=?
            )
            GROUP BY p.id
            ORDER BY p.updated_at DESC
        """, (user_id, user_id)).fetchall()
    return [dict(r) for r in rows]
