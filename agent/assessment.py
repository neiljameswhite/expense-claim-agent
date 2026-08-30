"""
Assessment — converts check findings into a verdict.

This module is deliberately deterministic. Checks may involve judgement; the
decision about what those findings mean does not. The same set of check
results always produces the same verdict.

Three verdicts, not two. `review` means the system could not reach a
position on the evidence supplied — distinct from `decline`, which is a
position. Collapsing the two would make a claim the model could not judge
indistinguishable from one it judged and rejected, and would count correct
restraint as a wrong answer.

Six checks, all reasoning over what the claimant declared against what the
policy says. Nothing here reads the receipt: extraction is stubbed, so a
check comparing a claimed amount against an "extracted" total would be
comparing one invented value against another and demonstrating nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Result(str, Enum):
    """The result vocabulary."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"
    INCONCLUSIVE = "inconclusive"


class Verdict(str, Enum):
    APPROVE = "approve"
    DECLINE = "decline"
    REVIEW = "review"


# Check identifiers.
CHECK_CATEGORY_CORRECT = 1
CHECK_SUBSISTENCE_ELIGIBLE = 2
CHECK_MILEAGE_JOURNEY = 3
CHECK_WITHIN_LIMIT = 4
CHECK_COST_EXPLANATION = 5
CHECK_OTHER_DESCRIPTION = 6

CHECK_NAMES = {
    CHECK_CATEGORY_CORRECT: "Category correctly selected",
    CHECK_SUBSISTENCE_ELIGIBLE: "Subsistence eligibility",
    CHECK_MILEAGE_JOURNEY: "Mileage journey description",
    CHECK_WITHIN_LIMIT: "Within category limit",
    CHECK_COST_EXPLANATION: "Cost exception explanation",
    CHECK_OTHER_DESCRIPTION: "Other category description",
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
    grounds: list[CheckResult]       # checks that failed
    undetermined: list[CheckResult]  # checks that could not be resolved

    @property
    def is_approved(self) -> bool:
        return self.verdict is Verdict.APPROVE


class MissingCheckError(ValueError):
    """Raised when a required check has no recorded result.

    A check that did not run records not_applicable. Silence is not a pass.
    """


REQUIRED_CHECKS = set(CHECK_NAMES)

SETTLED = (Result.PASS, Result.NOT_APPLICABLE, Result.NOT_EVALUATED)


def assess(check_results: list[CheckResult]) -> Assessment:
    """Produce a verdict from a complete set of check results.

    Rules, in order:

      1. Every check must have a recorded result. A missing result is an
         error, not an implicit pass.
      2. Check 4 (over limit) is excused where check 5 (cost explanation)
         passes — a valid exception supports the excess.
      3. Any other fail declines. A definite ground to decline is
         dispositive even where another check was inconclusive: the claim
         can be decided, so it is.
      4. Otherwise, any inconclusive result means the claim cannot be
         determined on what was supplied. Verdict is review.
      5. Otherwise approve.

    Grounds accumulate: a claim may decline on several bases at once, and
    all of them are cited.
    """
    by_id: dict[int, CheckResult] = {}
    for cr in check_results:
        if cr.check_id in by_id:
            raise ValueError(f"Duplicate result for check {cr.check_id}")
        by_id[cr.check_id] = cr

    missing = REQUIRED_CHECKS - set(by_id)
    if missing:
        raise MissingCheckError(
            "No result recorded for check(s): " + ", ".join(str(m) for m in sorted(missing))
        )

    explanation_upholds = by_id[CHECK_COST_EXPLANATION].result is Result.PASS

    grounds: list[CheckResult] = []
    undetermined: list[CheckResult] = []

    for check_id in sorted(by_id):
        cr = by_id[check_id]

        if cr.result in SETTLED:
            continue

        # Rule 2: a valid cost explanation excuses the limit breach itself,
        # and nothing else.
        if (
            check_id == CHECK_WITHIN_LIMIT
            and cr.result is Result.FAIL
            and explanation_upholds
        ):
            continue

        if cr.result is Result.FAIL:
            grounds.append(cr)
        else:
            undetermined.append(cr)

    if grounds:
        verdict = Verdict.DECLINE
    elif undetermined:
        verdict = Verdict.REVIEW
    else:
        verdict = Verdict.APPROVE

    return Assessment(verdict=verdict, grounds=grounds, undetermined=undetermined)


def summarise(assessment: Assessment) -> str:
    """Plain-language summary. Not the reviewer-facing narrative, which is
    assembled from the check details with their clause references."""
    if assessment.verdict is Verdict.APPROVE:
        return "All applicable checks satisfied."

    parts = [f"{cr.name} failed" for cr in assessment.grounds]
    parts += [f"{cr.name} could not be determined" for cr in assessment.undetermined]
    return "; ".join(parts) + "."
