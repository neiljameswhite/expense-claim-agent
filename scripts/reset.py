#!/usr/bin/env python3
"""
Reset — return the database to a known empty state.

    python scripts/reset.py            clears everything, with a prompt
    python scripts/reset.py --yes      no prompt
    python scripts/reset.py --status   show what is there without touching it

Traces in Langfuse are left alone. They are grouped by run label and are the
record of what happened; wiping the database does not un-happen it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.db import connect  # noqa: E402

# Order matters: children before parents. run_events and payment_requests
# both reference runs; runs references claims.
TABLES = ["run_events", "payment_requests", "test_results", "runs", "claims"]


def counts(conn) -> dict[str, int]:
    out = {}
    for table in TABLES:
        row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
        out[table] = row["n"]
    return out


def show_status() -> None:
    with connect() as conn:
        rows = counts(conn)
        pending = conn.execute(
            "SELECT status, count(*) AS n FROM claims GROUP BY status ORDER BY status"
        ).fetchall()
        verdicts = conn.execute(
            """
            SELECT claim_status, count(*) AS n
            FROM runs GROUP BY claim_status ORDER BY claim_status
            """
        ).fetchall()

    for table, n in rows.items():
        print(f"  {table:<18} {n:>5}")
    if pending:
        print("\n  claims by status:")
        for r in pending:
            print(f"    {r['status']:<12} {r['n']:>5}")
    if verdicts:
        print("\n  runs by outcome:")
        for r in verdicts:
            print(f"    {r['claim_status']:<18} {r['n']:>5}")


def reset(confirm: bool = True) -> None:
    with connect() as conn:
        before = counts(conn)
        total = sum(before.values())

        if total == 0:
            print("Already empty.")
            return

        if confirm:
            print("This will delete:")
            for table, n in before.items():
                if n:
                    print(f"  {table:<18} {n:>5}")
            answer = input("\nProceed? [y/N] ").strip().lower()
            if answer != "y":
                print("Cancelled.")
                return

        # TRUNCATE rather than DELETE: faster, and resets the sequences so
        # event ids start from 1 again on a fresh corpus run.
        conn.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")

    print(f"Reset. {total} rows cleared across {len(TABLES)} tables.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear the database.")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    parser.add_argument("--status", action="store_true", help="show counts only")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    reset(confirm=not args.yes)


if __name__ == "__main__":
    main()
