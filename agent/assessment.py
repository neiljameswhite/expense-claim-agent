"""
Assessment — converts check findings into a verdict.

This module is deliberately deterministic. Checks may involve model judgement;
the decision about what those findings *mean* does not. The same set of check
results always produces the same verdict.

See solution design v0.4 section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Result(str, Enum):
    """The result vocabulary. See design v0.4 section 5.1."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"
    INCONCLUSIVE = "inconclusive"


class Verdict(str, Enum):
    APPROVE = "approve"
    DECLINE = "decline"


# Check identifiers. Numbers match the design document.
CHECK_IS_RECEIPT = 1
CHECK_LEGIBLE = 2
CHECK_AMOUNT_MATCHES = 3
CHECK_CATEGORY_CONSISTENT = 4
CHECK_VAT = 5
CHECK_WITHIN_LIMIT = 6
CHECK_COST_RATIONALE = 7
CHECK_OTHER_RATIONALE = 8

CHECK_NAMES = {
    CHECK_IS_RECEIPT: "Document is a receipt",
    CHECK_LEGIBLE: "Receipt legible",
    CHECK_AMOUNT_MATCHES: "Amount matches receipt",
    CHECK_CATEGORY_CONSISTENT: "Category consistent with receipt",
    CHECK_VAT: "VAT declaration",
    CHECK_WITHIN_LIMIT: "Within category limit",
    CHECK_COST_RATIONALE: "Cost exception rationale",
    CHECK_OTHER_RATIONALE: "Other category rationale",
}


@dataclass
class CheckResult:
    """One check's finding, with whatever it was derived from."""

    check_id: int
    result: Result
    inputs: dict = field(default_factory=dict)
    clause_refs: list[str] = field(default_factory=list)
    detail: str = ""

    def __post_init__(self):
        if self.check_id not in CHECK_NAMES:
            raise ValueError(f"Unknown check id: {self.check_id}")
        if not isinstance(self.result, Result):
            self.result = Result(self.result)

    @property
    def name(self) -> str:
        return CHECK_NAMES[self.check_id]


@dataclass
class Assessment:
    verdict: Verdict
    grounds: list[CheckResult]  # every check that contributed to a decline

    @property
    def is_approved(self) -> bool:
        return self.verdict is Verdict.APPROVE


class MissingCheckError(ValueError):
    """Raised when a required check has no recorded result.

    Invariant 12: a check that did not run records not_applicable or
    not_evaluated, never nothing at all. Silence is not a pass.
    """


REQUIRED_CHECKS = set(CHECK_NAMES)


def assess(check_results: list[CheckResult]) -> Assessment:
    """Produce a verdict from a complete set of check results.

    Rules, in order of application:

      1. Every check must have a recorded result. A missing result is an
         error, not an implicit pass.
      2. Check 6 (over limit) is excused where check 7 (cost rationale)
         passes — a valid exception supports the excess.
      3. Any other fail declines.
      4. Any inconclusive declines — the reviewer must look.
      5. Otherwise approve.

    Grounds accumulate: a claim may decline on several bases at once, and
    all of them are cited.
    """
    by_id = {}
    for cr in check_results:
        if cr.check_id in by_id:
            raise ValueError(f"Duplicate result for check {cr.check_id}")
        by_id[cr.check_id] = cr

    missing = REQUIRED_CHECKS - set(by_id)
    if missing:
        raise MissingCheckError(
            "No result recorded for check(s): " + ", ".join(str(m) for m in sorted(missing))
        )

    cost_rationale_upholds = by_id[CHECK_COST_RATIONALE].result is Result.PASS

    grounds: list[CheckResult] = []
    for check_id in sorted(by_id):
        cr = by_id[check_id]

        if cr.result in (Result.PASS, Result.NOT_APPLICABLE, Result.NOT_EVALUATED):
            continue

        # Rule 2: a valid cost exception excuses the limit breach itself.
        if check_id == CHECK_WITHIN_LIMIT and cr.result is Result.FAIL and cost_rationale_upholds:
            continue

        grounds.append(cr)

    verdict = Verdict.DECLINE if grounds else Verdict.APPROVE
    return Assessment(verdict=verdict, grounds=grounds)


def summarise(assessment: Assessment) -> str:
    """Plain-language summary of the grounds. Not the reviewer-facing narrative,
    which is generated separately with policy citations."""
    if assessment.is_approved:
        return "All checks satisfied."

    parts = []
    for cr in assessment.grounds:
        verb = "could not be determined" if cr.result is Result.INCONCLUSIVE else "failed"
        parts.append(f"{cr.name} {verb}")
    return "; ".join(parts) + "."
