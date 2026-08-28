#!/usr/bin/env python3
"""
Process claims end to end: all eight checks, assessment, and a run written
to the database.

    python scripts/process.py                    every claim with status 'new'
    python scripts/process.py --only H1 J6       specific record ids
    python scripts/process.py --dry-run          read the corpus, write nothing

The UI does the same work through the same pipeline, so the two cannot drift
apart. This exists for running the whole corpus without clicking, and for
comparing results against the corpus expectations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import process_pending  # noqa: E402
from agent.policy import load  # noqa: E402
from agent.runner import format_outcome, run_claim  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "corpus_v1.json"
POLICY = ROOT / "corpus" / "expense_policy_v1.md"

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def expectations() -> dict[str, dict]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    return {r["record_id"]: r for r in corpus["records"]}


def claims_from_corpus(only: list[str] | None) -> list[dict]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    rows = []
    for rec in corpus["records"]:
        if only and rec["record_id"] not in only:
            continue
        row = dict(rec["claim"])
        row["claim_id"] = f"EXP-{rec['record_id']}"
        row["record_id"] = rec["record_id"]
        row["extraction"] = rec["extraction"]
        rows.append(row)
    return rows


def score(record_id: str, outcome, expected: dict) -> tuple[int, int, str]:
    rec = expected.get(record_id, {})
    exp_checks = rec.get("expected_checks", {})

    matched = 0
    for cid, exp in exp_checks.items():
        actual = outcome.by_id.get(int(cid))
        if actual is not None and actual.result.value == exp:
            matched += 1

    exp_verdict = rec.get("expected_verdict")
    flag = ""
    if exp_verdict:
        flag = (
            f"{GREEN}ok{RESET}"
            if outcome.verdict == exp_verdict
            else f"{RED}MISMATCH expected {exp_verdict}{RESET}"
        )
    return matched, len(exp_checks), flag


def main() -> None:
    parser = argparse.ArgumentParser(description="Process claims through all eight checks.")
    parser.add_argument("--only", nargs="+", metavar="ID", help="specific record ids")
    parser.add_argument("--dry-run", action="store_true", help="read the corpus, write nothing")
    args = parser.parse_args()

    policy = load(POLICY)
    expected = expectations()

    matched = total = tokens = 0
    verdict_ok = verdict_total = 0

    if args.dry_run:
        rows = claims_from_corpus(args.only)
        if not rows:
            print("Nothing to process.")
            return
        print(f"Policy v{policy.version}, {len(rows)} claim(s), reading corpus, no writes\n")

        for row in rows:
            record_id = row["record_id"]
            outcome = run_claim(row, policy)
            tokens += outcome.tokens()
            m, t, flag = score(record_id, outcome, expected)
            matched += m
            total += t
            if expected.get(record_id, {}).get("expected_verdict"):
                verdict_total += 1
                if "ok" in flag:
                    verdict_ok += 1
            print(f"{BOLD}{record_id}{RESET}  {outcome.verdict:<8} {flag}")
            print(f"{DIM}{expected.get(record_id, {}).get('purpose', '')}{RESET}")
            print(format_outcome(outcome, expected=expected.get(record_id, {}).get("expected_checks", {})))
            print()
    else:
        results = process_pending(policy, only=args.only)
        if not results:
            print("No claims with status 'new'. Run scripts/seed.py first.")
            return
        print(f"Policy v{policy.version}, {len(results)} claim(s)\n")

        for r in results:
            if not r.ok:
                print(f"{BOLD}{r.record_id or r.claim_id}{RESET}  {RED}failed{RESET}: {r.error}\n")
                continue

            record_id = r.record_id or r.claim_id.replace("EXP-", "")
            outcome = r.outcome
            tokens += outcome.tokens()
            m, t, flag = score(record_id, outcome, expected)
            matched += m
            total += t
            if expected.get(record_id, {}).get("expected_verdict"):
                verdict_total += 1
                if "ok" in flag:
                    verdict_ok += 1
            print(f"{BOLD}{record_id}{RESET}  {outcome.verdict:<8} {flag}")
            print(f"{DIM}{expected.get(record_id, {}).get('purpose', '')}{RESET}")
            print(format_outcome(outcome, expected=expected.get(record_id, {}).get("expected_checks", {})))
            print(f"  {DIM}run {r.run_id}{RESET}\n")

    if verdict_total:
        colour = GREEN if verdict_ok == verdict_total else RED
        print(f"{colour}{verdict_ok}/{verdict_total} verdicts matched{RESET}")
    if total:
        print(f"{matched}/{total} check results matched")
    print(f"{tokens} tokens used")


if __name__ == "__main__":
    main()
