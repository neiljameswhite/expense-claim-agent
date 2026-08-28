#!/usr/bin/env python3
"""
Run check 7 against corpus records and compare with what was expected.

    python scripts/try_check7.py              every record where check 7 applies
    python scripts/try_check7.py H1 H2 K4     specific records

Reads the corpus directly rather than the database, so it can be run before
any claims have been seeded. This is a development harness, not the eval
suite — it prints results for inspection rather than asserting them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.assessment import CHECK_COST_RATIONALE, Result  # noqa: E402
from agent.checks.cost_rationale import Claim, run  # noqa: E402
from agent.policy import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "corpus_v1.json"
POLICY = ROOT / "corpus" / "expense_policy_v1.md"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def expected_for(record: dict) -> str | None:
    return record.get("expected_checks", {}).get(str(CHECK_COST_RATIONALE))


def main() -> None:
    wanted = set(sys.argv[1:])

    policy = load(POLICY)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))

    records = corpus["records"]
    if wanted:
        records = [r for r in records if r["record_id"] in wanted]
    else:
        records = [r for r in records if expected_for(r) is not None]

    if not records:
        print("Nothing to run.")
        return

    print(f"Policy v{policy.version}, {len(records)} record(s)\n")

    agreed = 0
    tokens = 0

    for rec in records:
        claim_data = dict(rec["claim"])
        claim_data["claim_id"] = f"EXP-{rec['record_id']}"
        claim = Claim.from_row(claim_data)

        expected = expected_for(rec)
        # Check 6 has not been built yet. Infer whether the limit was
        # exceeded from what the record expects of check 6.
        over_limit = rec.get("expected_checks", {}).get("6") == "fail"

        result = run(claim, policy, over_limit=over_limit)
        tokens += result.inputs.get("input_tokens", 0) + result.inputs.get("output_tokens", 0)

        actual = result.result.value
        if expected is None:
            mark, colour = "?", DIM
        elif actual == expected:
            mark, colour = "ok", GREEN
            agreed += 1
        else:
            mark, colour = "MISMATCH", RED

        print(f"{colour}{rec['record_id']:<4} {mark:<9}{RESET} "
              f"expected {expected or '-':<14} got {actual}")
        print(f"     {DIM}{rec['purpose']}{RESET}")
        if result.clause_refs:
            print(f"     ground: {', '.join(result.clause_refs)}")
        if result.detail:
            print(f"     {result.detail}")
        conf = result.inputs.get("confidence")
        if conf is not None:
            print(f"     confidence: {conf}")
        print()

    scored = [r for r in records if expected_for(r) is not None]
    if scored:
        print(f"{agreed}/{len(scored)} matched expectation")
    if tokens:
        print(f"{tokens} tokens used")


if __name__ == "__main__":
    main()
