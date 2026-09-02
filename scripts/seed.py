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
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.db import connect, corpus_path, setting  # noqa: E402

CORPUS = corpus_path()

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


def claim_id_for(conn, record_id: str) -> str:
    """A unique claim id for this submission of a record.

    First submission uses the record id unchanged — EXP-3. A second gives
    EXP-3-2, a third EXP-3-3, and so on, so a record can be resubmitted to
    compare runs without colliding.

    Derived per record rather than per batch. The previous version counted
    rows across the whole batch and divided, which produced the wrong
    generation whenever records had been submitted different numbers of
    times.
    """
    rows = conn.execute(
        "SELECT claim_id FROM claims WHERE record_id = %s", (record_id,)
    ).fetchall()
    existing = {r["claim_id"] for r in rows}

    if record_id not in existing:
        return record_id

    n = 2
    while f"{record_id}-{n}" in existing:
        n += 1
    return f"{record_id}-{n}"


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

    today = date.today()

    with connect() as conn:
        generations = []
        for rec in records:
            claim = rec["claim"]
            extraction = dict(rec["extraction"])
            conn.execute(
                INSERT,
                {
                    "claim_id": claim_id,
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
                    "extraction": json.dumps(extraction),
                },
            )

    if generations:
        print(f"Seeded {len(records)} claims ({len(generations)} as repeat submissions).")
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
