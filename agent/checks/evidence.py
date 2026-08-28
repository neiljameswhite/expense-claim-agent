"""
Checks 1, 2 and 3 — evidence.

These three read the extraction result and apply thresholds and comparisons.
No model is involved. Extraction is currently stubbed, so what they read is
authored corpus data — but they sit behind the same interface a real
extractor would satisfy, so nothing here changes when extraction becomes
real.

  1  Document is a receipt      is the attachment a receipt at all
  2  Receipt legible            can every required field be read
  3  Amount matches receipt     does the claimed total equal the receipt's

Checks 1 and 2 gate everything downstream that compares the claim against
its evidence. Where either fails, checks 3, 4 and 5 are not_applicable —
they could never have run, rather than having been skipped.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from agent.assessment import (
    CHECK_AMOUNT_MATCHES,
    CHECK_IS_RECEIPT,
    CHECK_LEGIBLE,
    CheckResult,
    Result,
)
from agent.db import setting

# Clause 1.3: a valid receipt shows the retailer, the date, a description of
# what was purchased, the individual item costs and the total paid.
REQUIRED_FIELDS = ["retailer", "date", "total", "line_items"]


class ConfigurationError(ValueError):
    """A configured value is outside the range that makes sense.

    Raised rather than tolerated. A legibility threshold of 0 does not make
    the check lenient — it disables it, and every claim passes regardless of
    how unreadable its receipt was. The run continues, the verdicts still
    look plausible, and an entire control has quietly stopped existing.

    Failing loudly here is the same principle as validating the policy
    structure at load: a configuration that changes what the system does
    must not do so silently.
    """


# Below 0.5 the check stops discriminating; at or above 1.0 nothing can pass.
THRESHOLD_MIN = 0.5
THRESHOLD_MAX = 0.99


def _threshold() -> float:
    raw = setting("LEGIBILITY_THRESHOLD", "0.80")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"LEGIBILITY_THRESHOLD is {raw!r}, which is not a number."
        ) from None

    if not (THRESHOLD_MIN <= value <= THRESHOLD_MAX):
        raise ConfigurationError(
            f"LEGIBILITY_THRESHOLD is {value}, outside the usable range "
            f"{THRESHOLD_MIN}–{THRESHOLD_MAX}. A value below {THRESHOLD_MIN} "
            "effectively disables check 2 rather than relaxing it."
        )
    return value


def is_receipt(extraction: dict) -> CheckResult:
    """Check 1 — is the attachment a receipt.

    The extractor reports this. A null answer means it could not tell, which
    is inconclusive rather than a failure.
    """
    value = extraction.get("is_receipt")

    if value is None:
        return CheckResult(
            check_id=CHECK_IS_RECEIPT,
            result=Result.INCONCLUSIVE,
            inputs={"is_receipt": None},
            clause_refs=["1.3"],
            detail="Extraction could not determine whether the attachment is a receipt.",
        )

    if value is True:
        return CheckResult(
            check_id=CHECK_IS_RECEIPT,
            result=Result.PASS,
            inputs={"is_receipt": True},
            clause_refs=["1.3"],
            detail="Attachment is a receipt.",
        )

    return CheckResult(
        check_id=CHECK_IS_RECEIPT,
        result=Result.FAIL,
        inputs={"is_receipt": False},
        clause_refs=["1.3", "6.1"],
        detail="Attachment is not a receipt. Clause 1.3 requires a valid receipt.",
    )


def legible(extraction: dict) -> CheckResult:
    """Check 2 — can every required field be read.

    All or nothing, deliberately. Per-field legibility is more realistic but
    makes every downstream check's applicability conditional on which fields
    happened to be readable, which multiplies the state space without adding
    to what the system demonstrates. See design v0.4 section 5.3.
    """
    threshold = _threshold()
    confidence = extraction.get("confidence") or {}

    scores: dict[str, float | None] = {}
    for field in REQUIRED_FIELDS:
        raw = confidence.get(field)
        scores[field] = float(raw) if raw is not None else None

    missing = [f for f, s in scores.items() if s is None]
    below = [f for f, s in scores.items() if s is not None and s < threshold]

    # Near the threshold on either side: the extractor cannot confidently
    # call it either way.
    margin = 0.05
    borderline = [
        f for f, s in scores.items()
        if s is not None and abs(s - threshold) < margin
    ]

    inputs = {"threshold": threshold, "field_confidence": scores}

    if missing:
        return CheckResult(
            check_id=CHECK_LEGIBLE,
            result=Result.FAIL,
            inputs={**inputs, "missing": missing},
            clause_refs=["1.3", "6.1"],
            detail="Required field(s) could not be read: " + ", ".join(missing) + ".",
        )

    if below:
        if all(f in borderline for f in below):
            return CheckResult(
                check_id=CHECK_LEGIBLE,
                result=Result.INCONCLUSIVE,
                inputs={**inputs, "borderline": borderline},
                clause_refs=["1.3"],
                detail="Field(s) read close to the confidence threshold: "
                + ", ".join(borderline) + ".",
            )
        return CheckResult(
            check_id=CHECK_LEGIBLE,
            result=Result.FAIL,
            inputs={**inputs, "below_threshold": below},
            clause_refs=["1.3", "6.1"],
            detail="Field(s) below the legibility threshold: " + ", ".join(below) + ".",
        )

    return CheckResult(
        check_id=CHECK_LEGIBLE,
        result=Result.PASS,
        inputs=inputs,
        clause_refs=["1.3"],
        detail="All required fields legible.",
    )


def amount_matches(claim_amount: float, extraction: dict) -> CheckResult:
    """Check 3 — does the claimed amount equal the receipt total.

    Clause 1.4: the amount claimed must equal the total shown on the receipt,
    and partial claims against a larger receipt are not permitted.
    """
    raw_total = extraction.get("total")

    if raw_total is None:
        return CheckResult(
            check_id=CHECK_AMOUNT_MATCHES,
            result=Result.INCONCLUSIVE,
            inputs={"claim_amount": claim_amount, "receipt_total": None},
            clause_refs=["1.4"],
            detail="No total could be read from the receipt.",
        )

    try:
        claimed = Decimal(str(claim_amount))
        total = Decimal(str(raw_total))
    except (InvalidOperation, ValueError):
        return CheckResult(
            check_id=CHECK_AMOUNT_MATCHES,
            result=Result.INCONCLUSIVE,
            inputs={"claim_amount": claim_amount, "receipt_total": raw_total},
            clause_refs=["1.4"],
            detail="Claimed amount or receipt total was not a usable number.",
        )

    inputs = {
        "claim_amount": float(claimed),
        "receipt_total": float(total),
        "difference": float(claimed - total),
    }

    if claimed == total:
        return CheckResult(
            check_id=CHECK_AMOUNT_MATCHES,
            result=Result.PASS,
            inputs=inputs,
            clause_refs=["1.4"],
            detail=f"Claimed {claimed} matches receipt total {total}.",
        )

    direction = "less than" if claimed < total else "more than"
    return CheckResult(
        check_id=CHECK_AMOUNT_MATCHES,
        result=Result.FAIL,
        inputs=inputs,
        clause_refs=["1.4"],
        detail=f"Claimed {claimed} is {direction} the receipt total {total}. "
        "Clause 1.4 requires them to be equal.",
    )


def not_applicable(check_id: int, reason: str) -> CheckResult:
    """A check that could never have run for this claim.

    Used where checks 1 or 2 failed: there is no usable evidence, so the
    comparisons against it do not arise.
    """
    return CheckResult(
        check_id=check_id,
        result=Result.NOT_APPLICABLE,
        inputs={"reason": reason},
        detail=reason,
    )
