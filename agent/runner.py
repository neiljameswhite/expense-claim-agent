"""
Runner — sequences the six checks and produces a verdict.

Every check reasons over what the claimant declared against what the policy
says. Nothing reads the receipt: extraction is stubbed, so a check comparing
a claimed amount against an "extracted" total would compare one invented
value against another and demonstrate nothing.

Check 2 runs before check 4 because it establishes which of the two
subsistence limits applies. Passing that forward rather than letting check 4
re-derive it means the two cannot reach different conclusions about the same
claim.

The runner produces findings and hands them to assess(). It decides nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from agent.assessment import (
    CHECK_CATEGORY_CORRECT,
    CHECK_COST_EXPLANATION,
    CHECK_MILEAGE_JOURNEY,
    CHECK_NAMES,
    CHECK_OTHER_DESCRIPTION,
    CHECK_SUBSISTENCE_ELIGIBLE,
    CHECK_WITHIN_LIMIT,
    Assessment,
    CheckResult,
    Result,
    Verdict,
    assess,
    summarise,
)
from agent.checks import (
    category,
    cost_rationale,
    mileage,
    other_rationale,
    subsistence,
    within_limit,
)
from agent.policy import Policy

_MARKUP = re.compile(r":(?:red|orange|gray)\[(.*?)\]")
_MARKER = re.compile(r"^[✗?·]\s*")

# Every check reasons over the policy. The deterministic work — the limit
# comparison, the routing, the verdict — happens inside the checks and in
# assess(), not as checks of its own.
MODEL_CHECKS = set(CHECK_NAMES)


@dataclass
class RunOutcome:
    """Everything one pass over a claim produced."""

    claim_id: str
    check_results: list[CheckResult]
    assessment: Assessment
    events: list[dict] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return self.assessment.verdict.value

    @property
    def by_id(self) -> dict[int, CheckResult]:
        return {cr.check_id: cr for cr in self.check_results}

    def confidence(self) -> float | None:
        """Lowest confidence across the checks that reported one.

        The weakest link, not the average: a verdict resting on one shaky
        finding is a shaky verdict, and averaging would hide it.
        """
        scores = []
        for cr in self.check_results:
            raw = cr.inputs.get("confidence")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                scores.append(float(raw))
        return min(scores) if scores else None

    def tokens(self) -> int:
        return sum(
            cr.inputs.get("input_tokens", 0) + cr.inputs.get("output_tokens", 0)
            for cr in self.check_results
        )

    def as_json(self) -> list[dict]:
        return [
            {
                "check_id": cr.check_id,
                "name": cr.name,
                "result": cr.result.value,
                "inputs": cr.inputs,
                "clause_refs": cr.clause_refs,
                "detail": cr.detail,
            }
            for cr in self.check_results
        ]

    def plain_narrative(self) -> str:
        """The narrative as the submitter receives it.

        No markers and no markup. The reviewer's copy carries coloured
        symbols so a ground for decline reads as a status rather than
        punctuation, but those are display affordances: in an email the
        markup appears literally, and a cross beside every line says nothing
        when every line is a ground.

        This is also what the reviewer edits before sending, in a plain text
        box that renders nothing.
        """
        lines = []
        for raw in self.narrative().splitlines():
            line = raw.strip()
            if not line:
                continue
            line = _MARKUP.sub(r"\1", line).replace("**", "")
            line = _MARKER.sub("", line).strip()
            if line:
                lines.append(line)
        return "\n\n".join(lines)

    def narrative(self) -> str:
        """The reason detail shown under the AI Review heading.

        Assembled from the findings rather than generated. Each line is a
        check's own recorded detail with the clauses it cited, so the summary
        cannot say anything the checks did not establish.

        What it lists depends on the verdict, because the reviewer's question
        differs. On a decline they need the grounds, and five lines of things
        that passed bury the one that matters. On a review they need what
        could not be settled. On an approval there are no grounds, so the
        useful thing is what was actually established — which limit applied,
        what was verified — since an approval is where a claim might
        otherwise be released unexamined.

        The full check table sits below in every case, so nothing is hidden.
        """
        if self.assessment.verdict is Verdict.DECLINE:
            return "\n\n".join(self._line(cr, "✗") for cr in self.assessment.grounds)

        if self.assessment.verdict is Verdict.REVIEW:
            return "\n\n".join(self._line(cr, "?") for cr in self.assessment.undetermined)

        lines: list[str] = []
        skipped: list[str] = []
        for cr in sorted(self.check_results, key=lambda c: c.check_id):
            if cr.result in (Result.NOT_APPLICABLE, Result.NOT_EVALUATED):
                skipped.append(cr.name)
                continue
            lines.append(self._line(cr, "·"))
        if skipped:
            lines.append(f":gray[· Not applicable: {', '.join(skipped)}.]")
        return "\n\n".join(lines)

    @staticmethod
    def _line(cr: CheckResult, marker: str) -> str:
        """One finding, with a coloured marker.

        Streamlit renders :red[...] and :orange[...] in markdown. Without a
        colour the marker reads as a stray character rather than a status —
        a bare cross next to a sentence looks like part of the sentence.
        """
        refs = f" ({', '.join(cr.clause_refs)})" if cr.clause_refs else ""
        coloured = {
            "✗": ":red[**✗**]",
            "?": ":orange[**?**]",
            "·": ":gray[·]",
        }.get(marker, marker)
        return f"{coloured} {cr.detail or cr.result.value}{refs}"


def run_claim(
    claim_row: dict,
    policy: Policy,
    *,
    run_id: str = "",
    on_event: Callable[[str, dict], None] | None = None,
) -> RunOutcome:
    """Run every check over one claim and assess the findings."""
    claim_category = str(claim_row.get("claim_category", ""))
    results: list[CheckResult] = []
    events: list[dict] = []

    def emit(event_type: str, detail: dict) -> None:
        if on_event is not None:
            on_event(event_type, detail)

    def starting(check_id: int) -> None:
        emit(
            "check_started",
            {
                "check_id": check_id,
                "name": CHECK_NAMES[check_id],
                "uses_model": True,
                "policy_sections": [s.number for s in policy.resolve_sections(check_id)],
            },
        )

    def record(cr: CheckResult) -> CheckResult:
        results.append(cr)
        detail = {
            "check_id": cr.check_id,
            "name": cr.name,
            "result": cr.result.value,
            "clause_refs": cr.clause_refs,
            "detail": cr.detail,
            "input_tokens": cr.inputs.get("input_tokens", 0),
            "output_tokens": cr.inputs.get("output_tokens", 0),
            "model": cr.inputs.get("model"),
        }
        events.append(
            {"event_type": "check_completed", "stage": "checked", "detail_json": detail}
        )
        emit("check_completed", detail)
        return cr

    # --- 1: is the claim in the right category at all -------------------
    starting(CHECK_CATEGORY_CORRECT)
    record(category.run(category.Claim.from_row(claim_row), policy, run_id=run_id))

    # --- 2: subsistence eligibility, and which limit applies ------------
    starting(CHECK_SUBSISTENCE_ELIGIBLE)
    c2 = record(subsistence.run(subsistence.Claim.from_row(claim_row), policy, run_id=run_id))
    subsistence_limit = c2.inputs.get("applicable_limit")

    # --- 3: mileage journey ---------------------------------------------
    starting(CHECK_MILEAGE_JOURNEY)
    record(mileage.journey(mileage.Claim.from_row(claim_row), policy, run_id=run_id))

    # --- 4, 5, 6: the limit and the two explanations ---------------------
    starting(CHECK_WITHIN_LIMIT)
    c4 = record(
        within_limit.run(
            within_limit.Claim.from_row(claim_row),
            policy,
            subsistence_limit=subsistence_limit,
            run_id=run_id,
        )
    )

    starting(CHECK_COST_EXPLANATION)
    record(
        cost_rationale.run(
            cost_rationale.Claim.from_row(claim_row),
            policy,
            over_limit=c4.result is Result.FAIL,
            run_id=run_id,
        )
    )

    starting(CHECK_OTHER_DESCRIPTION)
    record(
        other_rationale.run(
            other_rationale.Claim.from_row(claim_row), policy, run_id=run_id
        )
    )

    outcome = RunOutcome(
        claim_id=claim_row["claim_id"],
        check_results=results,
        assessment=assess(results),
        events=events,
    )

    verdict_detail = {
        "verdict": outcome.verdict,
        "grounds": [cr.check_id for cr in outcome.assessment.grounds],
        "undetermined": [cr.check_id for cr in outcome.assessment.undetermined],
        "confidence": outcome.confidence(),
        "tokens": outcome.tokens(),
    }
    events.append(
        {"event_type": "ai_verdict_written", "stage": "assessed", "detail_json": verdict_detail}
    )
    emit("ai_verdict_written", verdict_detail)

    return outcome


def format_outcome(outcome: RunOutcome, *, expected: dict | None = None) -> str:
    """Human-readable summary for the development harness."""
    lines = []
    by_id = outcome.by_id
    for check_id in sorted(CHECK_NAMES):
        cr = by_id.get(check_id)
        if cr is None:
            lines.append(f"  {check_id:>2}  {CHECK_NAMES[check_id]:<36} (no result)")
            continue
        exp = (expected or {}).get(str(check_id))
        flag = ""
        if exp is not None:
            flag = "  ok" if cr.result.value == exp else f"  MISMATCH (expected {exp})"
        lines.append(f"  {check_id:>2}  {cr.name:<36} {cr.result.value:<16}{flag}")
    lines.append(f"\n  verdict: {outcome.verdict}")
    lines.append(f"  {summarise(outcome.assessment)}")
    return "\n".join(lines)
