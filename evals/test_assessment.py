"""
Tests for the deterministic assessment logic.

These need no database, no containers, no API key and no model. They are the
J-group cases from the coverage matrix, plus exhaustive coverage of the
verdict rules.
"""

import pytest

from agent.assessment import (
    CHECK_AMOUNT_MATCHES,
    CHECK_CATEGORY_CONSISTENT,
    CHECK_COST_RATIONALE,
    CHECK_IS_RECEIPT,
    CHECK_LEGIBLE,
    CHECK_OTHER_RATIONALE,
    CHECK_VAT,
    CHECK_WITHIN_LIMIT,
    Assessment,
    CheckResult,
    MissingCheckError,
    Result,
    Verdict,
    assess,
    summarise,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# A clean claim: receipt fine, within limit, standard category, no VAT needed.
BASELINE = {
    CHECK_IS_RECEIPT: Result.PASS,
    CHECK_LEGIBLE: Result.PASS,
    CHECK_AMOUNT_MATCHES: Result.PASS,
    CHECK_CATEGORY_CONSISTENT: Result.PASS,
    CHECK_VAT: Result.NOT_APPLICABLE,
    CHECK_WITHIN_LIMIT: Result.PASS,
    CHECK_COST_RATIONALE: Result.NOT_APPLICABLE,
    CHECK_OTHER_RATIONALE: Result.NOT_APPLICABLE,
}


def results(**overrides) -> list[CheckResult]:
    """Baseline check results with named overrides.

    results(within_limit=Result.FAIL, cost_rationale=Result.PASS)
    """
    name_to_id = {
        "is_receipt": CHECK_IS_RECEIPT,
        "legible": CHECK_LEGIBLE,
        "amount_matches": CHECK_AMOUNT_MATCHES,
        "category_consistent": CHECK_CATEGORY_CONSISTENT,
        "vat": CHECK_VAT,
        "within_limit": CHECK_WITHIN_LIMIT,
        "cost_rationale": CHECK_COST_RATIONALE,
        "other_rationale": CHECK_OTHER_RATIONALE,
    }
    merged = dict(BASELINE)
    for name, value in overrides.items():
        merged[name_to_id[name]] = value
    return [CheckResult(check_id=cid, result=res) for cid, res in merged.items()]


def ground_ids(assessment: Assessment) -> set[int]:
    return {cr.check_id for cr in assessment.grounds}


# --------------------------------------------------------------------------
# A — baseline
# --------------------------------------------------------------------------


def test_all_pass_approves():
    a = assess(results())
    assert a.verdict is Verdict.APPROVE
    assert a.grounds == []


def test_not_applicable_does_not_decline():
    a = assess(
        results(
            vat=Result.NOT_APPLICABLE,
            cost_rationale=Result.NOT_APPLICABLE,
            other_rationale=Result.NOT_APPLICABLE,
        )
    )
    assert a.verdict is Verdict.APPROVE


# --------------------------------------------------------------------------
# Single-check failures each decline
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check_name,check_id",
    [
        ("is_receipt", CHECK_IS_RECEIPT),
        ("legible", CHECK_LEGIBLE),
        ("amount_matches", CHECK_AMOUNT_MATCHES),
        ("category_consistent", CHECK_CATEGORY_CONSISTENT),
        ("vat", CHECK_VAT),
        ("other_rationale", CHECK_OTHER_RATIONALE),
    ],
)
def test_any_fail_declines(check_name, check_id):
    a = assess(results(**{check_name: Result.FAIL}))
    assert a.verdict is Verdict.DECLINE
    assert check_id in ground_ids(a)


@pytest.mark.parametrize(
    "check_name,check_id",
    [
        ("is_receipt", CHECK_IS_RECEIPT),
        ("legible", CHECK_LEGIBLE),
        ("amount_matches", CHECK_AMOUNT_MATCHES),
        ("category_consistent", CHECK_CATEGORY_CONSISTENT),
        ("vat", CHECK_VAT),
        ("cost_rationale", CHECK_COST_RATIONALE),
        ("other_rationale", CHECK_OTHER_RATIONALE),
    ],
)
def test_any_inconclusive_declines(check_name, check_id):
    a = assess(results(**{check_name: Result.INCONCLUSIVE}))
    assert a.verdict is Verdict.DECLINE
    assert check_id in ground_ids(a)


# --------------------------------------------------------------------------
# H — the cost exception. The only route to approval through a failed check.
# --------------------------------------------------------------------------


def test_h1_over_limit_with_supported_rationale_approves():
    """The key case: check 6 fails, check 7 upholds it, claim approves."""
    a = assess(results(within_limit=Result.FAIL, cost_rationale=Result.PASS))
    assert a.verdict is Verdict.APPROVE
    assert a.grounds == []


def test_h2_over_limit_with_unsupported_rationale_declines():
    a = assess(results(within_limit=Result.FAIL, cost_rationale=Result.FAIL))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_WITHIN_LIMIT, CHECK_COST_RATIONALE}


def test_over_limit_with_inconclusive_rationale_declines():
    a = assess(results(within_limit=Result.FAIL, cost_rationale=Result.INCONCLUSIVE))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_WITHIN_LIMIT, CHECK_COST_RATIONALE}


def test_over_limit_with_no_rationale_applicable_declines():
    """Over limit but the rationale check never applied — nothing excuses it."""
    a = assess(results(within_limit=Result.FAIL, cost_rationale=Result.NOT_APPLICABLE))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_WITHIN_LIMIT}


def test_cost_rationale_does_not_excuse_other_failures():
    """A valid cost exception rescues the limit breach and nothing else."""
    a = assess(
        results(
            within_limit=Result.FAIL,
            cost_rationale=Result.PASS,
            category_consistent=Result.FAIL,
        )
    )
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_CATEGORY_CONSISTENT}


# --------------------------------------------------------------------------
# J — accumulation and combinations
# --------------------------------------------------------------------------


def test_j1_failures_accumulate_across_independent_grounds():
    """Not a receipt AND over limit with a valid rationale: still declines,
    on the evidence ground only."""
    a = assess(
        results(
            is_receipt=Result.FAIL,
            amount_matches=Result.NOT_APPLICABLE,
            category_consistent=Result.NOT_APPLICABLE,
            vat=Result.NOT_APPLICABLE,
            within_limit=Result.FAIL,
            cost_rationale=Result.PASS,
        )
    )
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_IS_RECEIPT}


def test_j2_two_independent_failures_both_cited():
    a = assess(results(category_consistent=Result.FAIL, other_rationale=Result.FAIL))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_CATEGORY_CONSISTENT, CHECK_OTHER_RATIONALE}


def test_j3_other_category_over_the_other_limit():
    """Category is Other and the amount exceeds the Other limit: both
    rationales are in play."""
    a = assess(
        results(
            within_limit=Result.FAIL,
            cost_rationale=Result.PASS,
            other_rationale=Result.PASS,
        )
    )
    assert a.verdict is Verdict.APPROVE


def test_j3_other_over_limit_with_bad_other_rationale_declines():
    a = assess(
        results(
            within_limit=Result.FAIL,
            cost_rationale=Result.PASS,
            other_rationale=Result.FAIL,
        )
    )
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_OTHER_RATIONALE}


def test_j5_skip_path_declines_on_evidence_alone():
    """Illegible receipt, in limit, standard category. Checks 3-5 could never
    have run; 6-8 still assessed the declared values."""
    a = assess(
        results(
            legible=Result.FAIL,
            amount_matches=Result.NOT_APPLICABLE,
            category_consistent=Result.NOT_APPLICABLE,
            vat=Result.NOT_APPLICABLE,
        )
    )
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_LEGIBLE}


def test_multiple_inconclusive_all_cited():
    a = assess(results(vat=Result.INCONCLUSIVE, category_consistent=Result.INCONCLUSIVE))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_VAT, CHECK_CATEGORY_CONSISTENT}


# --------------------------------------------------------------------------
# Invariant 12 — silence is not a pass
# --------------------------------------------------------------------------


def test_missing_check_raises():
    partial = [cr for cr in results() if cr.check_id != CHECK_VAT]
    with pytest.raises(MissingCheckError) as exc:
        assess(partial)
    assert "5" in str(exc.value)


def test_several_missing_checks_all_named():
    partial = [
        cr for cr in results() if cr.check_id not in (CHECK_VAT, CHECK_WITHIN_LIMIT)
    ]
    with pytest.raises(MissingCheckError) as exc:
        assess(partial)
    assert "5" in str(exc.value) and "6" in str(exc.value)


def test_duplicate_check_result_raises():
    doubled = results() + [CheckResult(check_id=CHECK_VAT, result=Result.PASS)]
    with pytest.raises(ValueError, match="Duplicate"):
        assess(doubled)


def test_unknown_check_id_raises():
    with pytest.raises(ValueError, match="Unknown check id"):
        CheckResult(check_id=99, result=Result.PASS)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_input_same_verdict():
    """Stated as a property because it is the reason this logic is not a model."""
    payload = results(within_limit=Result.FAIL, cost_rationale=Result.PASS)
    verdicts = {assess(payload).verdict for _ in range(50)}
    assert verdicts == {Verdict.APPROVE}


def test_result_order_does_not_matter():
    forward = results(vat=Result.FAIL)
    backward = list(reversed(forward))
    assert assess(forward).verdict is assess(backward).verdict
    assert ground_ids(assess(forward)) == ground_ids(assess(backward))


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------


def test_summarise_approved():
    assert summarise(assess(results())) == "All checks satisfied."


def test_summarise_distinguishes_fail_from_inconclusive():
    a = assess(results(vat=Result.FAIL, category_consistent=Result.INCONCLUSIVE))
    text = summarise(a)
    assert "failed" in text
    assert "could not be determined" in text
