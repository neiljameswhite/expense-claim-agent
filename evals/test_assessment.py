"""
Tests for the deterministic assessment logic.

No database, no containers, no API key, no model. These cover the verdict
rules exhaustively, including the three-way distinction between approve,
decline and review.
"""

import pytest

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
    MissingCheckError,
    Result,
    Verdict,
    assess,
    summarise,
)

NAME_TO_ID = {
    "category": CHECK_CATEGORY_CORRECT,
    "subsistence": CHECK_SUBSISTENCE_ELIGIBLE,
    "mileage": CHECK_MILEAGE_JOURNEY,
    "within_limit": CHECK_WITHIN_LIMIT,
    "cost_explanation": CHECK_COST_EXPLANATION,
    "other_description": CHECK_OTHER_DESCRIPTION,
}

# A clean travel claim: right category, within limit, no exceptions in play.
BASELINE = {
    CHECK_CATEGORY_CORRECT: Result.PASS,
    CHECK_SUBSISTENCE_ELIGIBLE: Result.NOT_APPLICABLE,
    CHECK_MILEAGE_JOURNEY: Result.NOT_APPLICABLE,
    CHECK_WITHIN_LIMIT: Result.PASS,
    CHECK_COST_EXPLANATION: Result.NOT_APPLICABLE,
    CHECK_OTHER_DESCRIPTION: Result.NOT_APPLICABLE,
}


def results(**overrides) -> list[CheckResult]:
    merged = dict(BASELINE)
    for name, value in overrides.items():
        merged[NAME_TO_ID[name]] = value
    return [CheckResult(check_id=cid, result=res) for cid, res in merged.items()]


def ground_ids(a: Assessment) -> set[int]:
    return {cr.check_id for cr in a.grounds}


def undetermined_ids(a: Assessment) -> set[int]:
    return {cr.check_id for cr in a.undetermined}


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_all_pass_approves():
    a = assess(results())
    assert a.verdict is Verdict.APPROVE
    assert a.grounds == [] and a.undetermined == []


def test_not_applicable_does_not_decline():
    a = assess(results(within_limit=Result.NOT_APPLICABLE))
    assert a.verdict is Verdict.APPROVE


# --------------------------------------------------------------------------
# Fails decline, inconclusive reviews
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "category", "subsistence", "mileage", "other_description",
])
def test_any_fail_declines(name):
    a = assess(results(**{name: Result.FAIL}))
    assert a.verdict is Verdict.DECLINE
    assert NAME_TO_ID[name] in ground_ids(a)


@pytest.mark.parametrize("name", list(NAME_TO_ID))
def test_any_inconclusive_reviews(name):
    """An unresolved check is not a decision. The verdict is review."""
    a = assess(results(**{name: Result.INCONCLUSIVE}))
    assert a.verdict is Verdict.REVIEW
    assert NAME_TO_ID[name] in undetermined_ids(a)
    assert a.grounds == []


def test_several_inconclusive_all_recorded():
    a = assess(results(category=Result.INCONCLUSIVE, subsistence=Result.INCONCLUSIVE))
    assert a.verdict is Verdict.REVIEW
    assert undetermined_ids(a) == {CHECK_CATEGORY_CORRECT, CHECK_SUBSISTENCE_ELIGIBLE}


def test_a_definite_fail_beats_an_inconclusive():
    """A claim that can be decided is decided, even where something else was
    unresolved."""
    a = assess(results(category=Result.FAIL, subsistence=Result.INCONCLUSIVE))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_CATEGORY_CORRECT}
    assert undetermined_ids(a) == {CHECK_SUBSISTENCE_ELIGIBLE}


# --------------------------------------------------------------------------
# The cost exception: the only route to approval through a failed check
# --------------------------------------------------------------------------


def test_over_limit_with_supported_explanation_approves():
    a = assess(results(within_limit=Result.FAIL, cost_explanation=Result.PASS))
    assert a.verdict is Verdict.APPROVE
    assert a.grounds == []


def test_over_limit_with_unsupported_explanation_declines():
    a = assess(results(within_limit=Result.FAIL, cost_explanation=Result.FAIL))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_WITHIN_LIMIT, CHECK_COST_EXPLANATION}


def test_over_limit_with_inconclusive_explanation_declines_on_the_limit():
    a = assess(results(within_limit=Result.FAIL, cost_explanation=Result.INCONCLUSIVE))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_WITHIN_LIMIT}
    assert undetermined_ids(a) == {CHECK_COST_EXPLANATION}


def test_over_limit_with_no_explanation_applicable_declines():
    a = assess(results(within_limit=Result.FAIL, cost_explanation=Result.NOT_APPLICABLE))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_WITHIN_LIMIT}


def test_explanation_excuses_the_limit_and_nothing_else():
    a = assess(results(
        within_limit=Result.FAIL,
        cost_explanation=Result.PASS,
        category=Result.FAIL,
    ))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_CATEGORY_CORRECT}


# --------------------------------------------------------------------------
# Category-shaped combinations
# --------------------------------------------------------------------------


def test_subsistence_ineligible_declines():
    a = assess(results(subsistence=Result.FAIL, within_limit=Result.NOT_APPLICABLE))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_SUBSISTENCE_ELIGIBLE}


def test_mileage_shape_approves():
    """Mileage: no limit, journey description instead."""
    a = assess(results(within_limit=Result.NOT_APPLICABLE, mileage=Result.PASS))
    assert a.verdict is Verdict.APPROVE


def test_mileage_incomplete_journey_declines():
    a = assess(results(within_limit=Result.NOT_APPLICABLE, mileage=Result.FAIL))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_MILEAGE_JOURNEY}


def test_wrong_category_declines_even_within_limit():
    """Fuel claimed as travel: the amount is fine, the category is not."""
    a = assess(results(category=Result.FAIL, within_limit=Result.PASS))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_CATEGORY_CORRECT}


def test_failures_accumulate():
    a = assess(results(category=Result.FAIL, other_description=Result.FAIL))
    assert a.verdict is Verdict.DECLINE
    assert ground_ids(a) == {CHECK_CATEGORY_CORRECT, CHECK_OTHER_DESCRIPTION}


# --------------------------------------------------------------------------
# Silence is not a pass
# --------------------------------------------------------------------------


def test_missing_check_raises():
    partial = [cr for cr in results() if cr.check_id != CHECK_WITHIN_LIMIT]
    with pytest.raises(MissingCheckError) as exc:
        assess(partial)
    assert "4" in str(exc.value)


def test_several_missing_checks_all_named():
    partial = [cr for cr in results()
               if cr.check_id not in (CHECK_WITHIN_LIMIT, CHECK_MILEAGE_JOURNEY)]
    with pytest.raises(MissingCheckError) as exc:
        assess(partial)
    assert "3" in str(exc.value) and "4" in str(exc.value)


def test_duplicate_check_result_raises():
    doubled = results() + [CheckResult(check_id=CHECK_WITHIN_LIMIT, result=Result.PASS)]
    with pytest.raises(ValueError, match="Duplicate"):
        assess(doubled)


def test_unknown_check_id_raises():
    with pytest.raises(ValueError, match="Unknown check id"):
        CheckResult(check_id=99, result=Result.PASS)


def test_six_checks_numbered_one_to_six():
    assert set(CHECK_NAMES) == set(range(1, 7))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_input_same_verdict():
    payload = results(within_limit=Result.FAIL, cost_explanation=Result.PASS)
    assert {assess(payload).verdict for _ in range(50)} == {Verdict.APPROVE}


def test_result_order_does_not_matter():
    forward = results(category=Result.FAIL)
    backward = list(reversed(forward))
    assert assess(forward).verdict is assess(backward).verdict
    assert ground_ids(assess(forward)) == ground_ids(assess(backward))


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------


def test_summarise_approved():
    assert summarise(assess(results())) == "All applicable checks satisfied."


def test_summarise_distinguishes_fail_from_inconclusive():
    a = assess(results(category=Result.FAIL, subsistence=Result.INCONCLUSIVE))
    text = summarise(a)
    assert "failed" in text and "could not be determined" in text
