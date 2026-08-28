#!/usr/bin/env python3
"""
Run checks 6 and 7 in dependency order against corpus records.

    python scripts/try_checks.py                  every record
    python scripts/try_checks.py H1 G3 J3         specific records
    python scripts/try_checks.py --trials 3 H6    repeat, to see variance

Check 6 establishes whether the limit was exceeded; check 7 only runs where
it was. This replaces the circular arrangement in try_check7.py, where
over_limit was read from the answer sheet.

A development harness, not the eval suite. It prints for inspection and
reports agreement; it does not assert.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.assessment import (  # noqa: E402
    CHECK_COST_RATIONALE,
    CHECK_WITHIN_LIMIT,
    Result,
)
from agent.checks import cost_rationale, within_limit  # noqa: E402
from agent.policy import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "corpus_v1.json"
POLICY = ROOT / "corpus" / "expense_policy_v1.md"

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def mark(actual: str, expected: str | None) -> tuple[str, str]:
    if expected is None:
        return "?", DIM
    if actual == expected:
        return "ok", GREEN
    return "MISMATCH", RED


def run_record(rec: dict, policy, trials: int) -> dict:
    claim_data = dict(rec["claim"])
    claim_data["claim_id"] = f"EXP-{rec['record_id']}"

    expected = rec.get("expected_checks", {})
    exp6 = expected.get(str(CHECK_WITHIN_LIMIT))
    exp7 = expected.get(str(CHECK_COST_RATIONALE))

    outcomes6: list[str] = []
    outcomes7: list[str] = []
    tokens = 0
    last6 = last7 = None

    for _ in range(trials):
        claim6 = within_limit.Claim.from_row(claim_data)
        r6 = within_limit.run(claim6, policy)
        outcomes6.append(r6.result.value)
        tokens += r6.inputs.get("input_tokens", 0) + r6.inputs.get("output_tokens", 0)
        last6 = r6

        over = r6.result is Result.FAIL
        claim7 = cost_rationale.Claim.from_row(claim_data)
        r7 = cost_rationale.run(claim7, policy, over_limit=over)
        outcomes7.append(r7.result.value)
        tokens += r7.inputs.get("input_tokens", 0) + r7.inputs.get("output_tokens", 0)
        last7 = r7

    return {
        "record": rec,
        "exp6": exp6,
        "exp7": exp7,
        "outcomes6": outcomes6,
        "outcomes7": outcomes7,
        "last6": last6,
        "last7": last7,
        "tokens": tokens,
    }


def report(res: dict, trials: int) -> tuple[int, int]:
    rec = res["record"]
    print(f"{BOLD}{rec['record_id']}{RESET}  {DIM}{rec['purpose']}{RESET}")

    agreed = scored = 0

    for label, key_out, key_exp, key_last in (
        ("6 within limit    ", "outcomes6", "exp6", "last6"),
        ("7 cost rationale  ", "outcomes7", "exp7", "last7"),
    ):
        outcomes = res[key_out]
        expected = res[key_exp]
        counts = Counter(outcomes)
        actual = counts.most_common(1)[0][0]
        m, colour = mark(actual, expected)

        if expected is not None:
            scored += 1
            if actual == expected:
                agreed += 1

        spread = ""
        if len(counts) > 1:
            spread = f"  {YELLOW}unstable: {dict(counts)}{RESET}"
        elif trials > 1:
            spread = f"  {DIM}{trials}/{trials} consistent{RESET}"

        print(f"  {label} {colour}{m:<9}{RESET} expected {expected or '-':<15} got {actual}{spread}")

        last = res[key_last]
        if last is not None and last.detail:
            print(f"     {DIM}{last.detail}{RESET}")

    print()
    return agreed, scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checks 6 and 7 over the corpus.")
    parser.add_argument("records", nargs="*", metavar="ID")
    parser.add_argument("--trials", type=int, default=1, help="repeat each record")
    args = parser.parse_args()

    policy = load(POLICY)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    records = corpus["records"]

    if args.records:
        wanted = set(args.records)
        unknown = wanted - {r["record_id"] for r in records}
        if unknown:
            raise SystemExit(f"Unknown record id(s): {', '.join(sorted(unknown))}")
        records = [r for r in records if r["record_id"] in wanted]

    print(f"Policy v{policy.version}, {len(records)} record(s), {args.trials} trial(s) each\n")

    agreed = scored = tokens = 0
    unstable: list[str] = []

    for rec in records:
        res = run_record(rec, policy, args.trials)
        a, s = report(res, args.trials)
        agreed += a
        scored += s
        tokens += res["tokens"]
        if args.trials > 1 and (
            len(set(res["outcomes6"])) > 1 or len(set(res["outcomes7"])) > 1
        ):
            unstable.append(rec["record_id"])

    print(f"{agreed}/{scored} check results matched expectation")
    if unstable:
        print(f"{YELLOW}unstable across trials: {', '.join(unstable)}{RESET}")
    print(f"{tokens} tokens used")


if __name__ == "__main__":
    main()
