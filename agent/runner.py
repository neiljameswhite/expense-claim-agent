"""
Runner — sequences the eight checks and produces a verdict.

The sequence is not arbitrary. Checks 1 and 2 establish whether the evidence
is usable; where either fails, checks 3, 4 and 5 are not_applicable because
they compare the claim against evidence that cannot be read. Checks 6, 7 and
8 assess the claimant's declared values and run regardless.

That is the skip path, and its purpose is to avoid sequential rejection.
Returning a claim for an unreadable receipt, only to decline it on policy at
the second attempt, costs two round trips to reach a decision available at
the first. Both findings are surfaced together.

The runner produces findings and hands them to assess(). It does not decide
anything itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agent.assessment import (
    CHECK_AMOUNT_MATCHES,
    CHECK_CATEGORY_CONSISTENT,
    CHECK_COST_RATIONALE,
    CHECK_IS_RECEIPT,
    CHECK_LEGIBLE,
    CHECK_NAMES,
    CHECK_OTHER_RATIONALE,
    CHECK_VAT,
    CHECK_WITHIN_LIMIT,
    Assessment,
    CheckResult,
    Result,
    assess,
    summarise,
)
from agent.checks import category, cost_rationale, evidence, other_rationale, vat, within_limit
from agent.policy import Policy

NO_USABLE_EVIDENCE = "The receipt could not be used, so this comparison does not arise."


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

        Only numeric confidences count. A check reporting something else
        under that key is ignored rather than allowed to break the run.
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
        """check_results in the shape the database column expects."""
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

    def narrative(self) -> str:
        """The reason detail shown under the AI Review heading, and on a
        decline communicated to the submitter.

        Assembled from the findings rather than generated. Each line is a
        check's own recorded detail with the clauses it cited, so the summary
        cannot say anything the checks did not establish. A model asked to
        summarise its own reasoning would produce something plausible that
        may not correspond to what actually happened.

        Approvals get the same treatment as declines. "All checks satisfied"
        tells a reviewer nothing about which limits applied or what was
        verified, and an approval is precisely where a reviewer might
        otherwise wave a claim through unexamined.
        """
        grounds = {cr.check_id for cr in self.assessment.grounds}
        lines: list[str] = []
        skipped: list[str] = []

        for cr in sorted(self.check_results, key=lambda c: c.check_id):
            if cr.result in (Result.NOT_APPLICABLE, Result.NOT_EVALUATED):
                skipped.append(cr.name)
                continue

            refs = f" ({', '.join(cr.clause_refs)})" if cr.clause_refs else ""
            detail = cr.detail or cr.result.value
            marker = "✗" if cr.check_id in grounds else "·"
            lines.append(f"{marker} {detail}{refs}")

        if skipped:
            lines.append(f"· Not applicable: {', '.join(skipped)}.")

        return "\n\n".join(lines)


def run_claim(
    claim_row: dict,
    policy: Policy,
    *,
    run_id: str = "",
    on_event: Callable[[str, dict], None] | None = None,
) -> RunOutcome:
    """Run every check over one claim and assess the findings.

    claim_row carries the submitted fields plus an "extraction" dict — what
    the extractor returned. Extraction is currently stubbed, but sits behind
    the interface a real one would satisfy.

    on_event fires as work happens: check_started before each check, then
    check_completed with its result, then ai_verdict_written. A caller can
    use it to show progress; nothing depends on it.
    """
    extraction = claim_row.get("extraction") or {}
    results: list[CheckResult] = []
    events: list[dict] = []

    def emit(event_type: str, detail: dict) -> None:
        if on_event is not None:
            on_event(event_type, detail)

    def starting(check_id: int, uses_model: bool, sections: list[str] | None = None) -> None:
        emit(
            "check_started",
            {
                "check_id": check_id,
                "name": CHECK_NAMES[check_id],
                "uses_model": uses_model,
                "policy_sections": sections or [],
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

    def sections_for(check_id: int) -> list[str]:
        return [s.number for s in policy.resolve_sections(check_id)]

    # --- 1 and 2: is the evidence usable at all -------------------------
    starting(CHECK_IS_RECEIPT, uses_model=False)
    c1 = record(evidence.is_receipt(extraction))

    starting(CHECK_LEGIBLE, uses_model=False)
    c2 = record(evidence.legible(extraction))

    evidence_usable = c1.result is Result.PASS and c2.result is Result.PASS

    # --- 3, 4, 5: compare the claim against that evidence ---------------
    if evidence_usable:
        starting(CHECK_AMOUNT_MATCHES, uses_model=False)
        record(evidence.amount_matches(float(claim_row["claim_amount"]), extraction))

        starting(CHECK_CATEGORY_CONSISTENT, uses_model=True,
                 sections=sections_for(CHECK_CATEGORY_CONSISTENT))
        record(
            category.run(
                category.Claim.from_row(claim_row), extraction, policy, run_id=run_id
            )
        )

        starting(CHECK_VAT, uses_model=True, sections=sections_for(CHECK_VAT))
        record(vat.run(vat.Claim.from_row(claim_row), extraction, policy, run_id=run_id))
    else:
        for check_id in (CHECK_AMOUNT_MATCHES, CHECK_CATEGORY_CONSISTENT, CHECK_VAT):
            starting(check_id, uses_model=False)
            record(evidence.not_applicable(check_id, NO_USABLE_EVIDENCE))

    # --- 6, 7, 8: assess the declared claim against policy --------------
    starting(CHECK_WITHIN_LIMIT, uses_model=True, sections=sections_for(CHECK_WITHIN_LIMIT))
    c6 = record(within_limit.run(within_limit.Claim.from_row(claim_row), policy, run_id=run_id))

    over_limit = c6.result is Result.FAIL
    starting(CHECK_COST_RATIONALE, uses_model=over_limit,
             sections=sections_for(CHECK_COST_RATIONALE) if over_limit else [])
    record(
        cost_rationale.run(
            cost_rationale.Claim.from_row(claim_row),
            policy,
            over_limit=over_limit,
            run_id=run_id,
        )
    )

    is_other = str(claim_row.get("claim_category", "")).strip().lower() == "other"
    starting(CHECK_OTHER_RATIONALE, uses_model=is_other,
             sections=sections_for(CHECK_OTHER_RATIONALE) if is_other else [])
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
            lines.append(f"  {check_id}  {CHECK_NAMES[check_id]:<34} (no result)")
            continue
        exp = (expected or {}).get(str(check_id))
        flag = ""
        if exp is not None:
            flag = "  ok" if cr.result.value == exp else f"  MISMATCH (expected {exp})"
        lines.append(
            f"  {check_id}  {cr.name:<34} {cr.result.value:<16}{flag}"
        )
    lines.append(f"\n  verdict: {outcome.verdict}")
    lines.append(f"  {summarise(outcome.assessment)}")
    return "\n".join(lines)
