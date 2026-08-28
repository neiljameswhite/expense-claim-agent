#!/usr/bin/env python3
"""
Seed — load the test corpus into the claims table.

Claims land with status 'new'. Nothing processes them until n8n polls, or
until they are released from the submit tab.

    python scripts/seed.py                     load every record
    python scripts/seed.py --only H1 H2 J6     load specific records
    python scripts/seed.py --list              show what is in the corpus

Re-seeding after a reset produces identical claim ids, so a run is
reproducible. Seeding without a reset appends a fresh generation with
suffixed ids, which is what you want when comparing prompt changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.db import connect, setting  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "corpus_v1.json"

INSERT = """
INSERT INTO claims (
    claim_id, record_id, status, submitter,
    claim_amount, claim_currency, claim_category, claim_date,
    business_purpose, tax_amount,
    cost_exception_rationale, other_category_rationale,
    receipt_ref, extraction
) VALUES (
    %(claim_id)s, %(record_id)s, 'new', %(submitter)s,
    %(claim_amount)s, %(claim_currency)s, %(claim_category)s, %(claim_date)s,
    %(business_purpose)s, %(tax_amount)s,
    %(cost_exception_rationale)s, %(other_category_rationale)s,
    %(receipt_ref)s, %(extraction)s
)
"""


def load_corpus() -> dict:
    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found at {CORPUS}")
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def next_generation(conn, record_ids: list[str]) -> str:
    """Suffix for this batch, so re-seeding without a reset does not collide.

    First load of a record gives claim_id 'EXP-H1'; a second gives
    'EXP-H1-2'. The plain id is the common case and stays readable.
    """
    row = conn.execute(
        "SELECT count(*) AS n FROM claims WHERE record_id = ANY(%s)", (record_ids,)
    ).fetchone()
    if row["n"] == 0:
        return ""
    row = conn.execute(
        """
        SELECT count(DISTINCT claim_id) AS n
        FROM claims WHERE record_id = ANY(%s)
        """,
        (record_ids,),
    ).fetchone()
    return f"-{(row['n'] // max(len(record_ids), 1)) + 2}"


def seed(only: list[str] | None = None) -> int:
    corpus = load_corpus()
    records = corpus["records"]

    if only:
        wanted = set(only)
        unknown = wanted - {r["record_id"] for r in records}
        if unknown:
            raise SystemExit(f"Unknown record id(s): {', '.join(sorted(unknown))}")
        records = [r for r in records if r["record_id"] in wanted]

    policy_version = corpus.get("policy_version", "unknown")
    expected = setting("POLICY_VERSION", policy_version)
    if expected != policy_version:
        print(
            f"  ! corpus targets policy {policy_version} but POLICY_VERSION is {expected}.\n"
            f"    Expected outcomes are only valid against the policy they were written for.",
            file=sys.stderr,
        )

    with connect() as conn:
        suffix = next_generation(conn, [r["record_id"] for r in records])
        for rec in records:
            claim = rec["claim"]
            conn.execute(
                INSERT,
                {
                    "claim_id": f"EXP-{rec['record_id']}{suffix}",
                    "record_id": rec["record_id"],
                    "submitter": claim["submitter"],
                    "claim_amount": claim["claim_amount"],
                    "claim_currency": claim["claim_currency"],
                    "claim_category": claim["claim_category"],
                    "claim_date": claim["claim_date"],
                    "business_purpose": claim["business_purpose"],
                    "tax_amount": claim.get("tax_amount"),
                    "cost_exception_rationale": claim.get("cost_exception_rationale"),
                    "other_category_rationale": claim.get("other_category_rationale"),
                    "receipt_ref": f"stub://{rec['record_id']}",
                    "extraction": json.dumps(rec["extraction"]),
                },
            )

    if suffix:
        print(f"Seeded {len(records)} claims (generation {suffix.lstrip('-')}).")
    else:
        print(f"Seeded {len(records)} claims.")
    return len(records)


def show_list() -> None:
    corpus = load_corpus()
    print(f"Corpus v{corpus['corpus_version']}, policy v{corpus['policy_version']}\n")
    for rec in corpus["records"]:
        c = rec["claim"]
        print(
            f"  {rec['record_id']:<4} {c['claim_category']:<20} "
            f"£{c['claim_amount']:>7.2f}  → {rec['expected_verdict']:<8} {rec['purpose']}"
        )
    print(f"\n  {len(corpus['records'])} records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the test corpus into claims.")
    parser.add_argument("--only", nargs="+", metavar="ID", help="seed specific record ids")
    parser.add_argument("--list", action="store_true", help="show the corpus without loading")
    args = parser.parse_args()

    if args.list:
        show_list()
        return

    seed(only=args.only)


if __name__ == "__main__":
    main()
