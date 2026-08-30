"""
Database access. One place that knows how to reach Postgres, so scripts,
services and the UI all connect the same way.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Read .env into the environment if present.

    Deliberately minimal — no dependency on python-dotenv, and existing
    environment variables win so a shell export can override the file.
    """
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()


def _from_streamlit(name: str) -> str | None:
    """Look the setting up in Streamlit's secrets store.

    Hosted deployments have no .env file — secrets are entered through the
    platform instead. Importing streamlit is deliberately deferred and
    wrapped: the CLI scripts run without it installed, and outside a
    Streamlit session the secrets store raises rather than returning empty.
    """
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value is not None else None


def setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name)
    if value is None:
        value = _from_streamlit(name)
    if value is None:
        value = default
    if value is None:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            "or add it to the Streamlit secrets for a hosted deployment."
        )
    return value


def dsn() -> str:
    return setting("PG_URL")


def corpus_path() -> Path:
    """The test corpus in use.

    Hardcoding this in each script meant a version bump had to be chased
    through five files, and it was missed three times. One setting, one
    place, read the same way as everything else.
    """
    return _ROOT / setting("CORPUS_FILE", "corpus/corpus_v2.json")


def policy_path() -> Path:
    """The policy the system assesses against."""
    return _ROOT / setting("POLICY_FILE", "corpus/expense_policy_v2_1.md")


@contextmanager
def connect():
    """A connection that commits on success and rolls back on error.

    Rows come back as dictionaries — more readable at the call site than
    tuple indexing, and it keeps column changes from silently shifting
    meaning.
    """
    conn = psycopg.connect(dsn(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_connection() -> str:
    """Confirm the database is reachable. Returns the server version."""
    with connect() as conn:
        row = conn.execute("SELECT version() AS v").fetchone()
    return row["v"]
