"""
Tests for the policy loader.

Two things are being proved. First, that the document parses into clauses
that can be cited exactly. Second — and more importantly — that a policy
whose structure has drifted from the check mapping fails at load rather than
quietly retrieving the wrong text.
"""

import pytest

from agent.assessment import (
    CHECK_COST_RATIONALE,
    CHECK_OTHER_RATIONALE,
    CHECK_VAT,
    CHECK_WITHIN_LIMIT,
)
from agent.policy import (
    CHECK_SECTIONS,
    PolicyStructureError,
    parse,
)

# --------------------------------------------------------------------------
# A miniature policy with the same shape as the real one.
# --------------------------------------------------------------------------

GOOD = """
# Expense Policy
**Version 1.0**

## 1. Scope and principles

**1.1** This policy applies to all employees.

**1.3** All claims must be supported by a valid receipt. A valid receipt shows
the retailer, the date, a description, the item costs and the total.

## 2. Categories and limits

**2.1** Every claim must be assigned to a category.

| Category | Limit |
|---|---|
| Subsistence | £40 |
| Other | £50 |

**2.3** Other must not be used to avoid a category limit.

## 3. The "Other" category

**3.1** Other is reserved for expenses outside the listed categories.

## 4. Exceeding a category limit

**4.1** A claim may exceed its limit only with a supported rationale.

**4.2** A rationale is supported where the excess arose from one of:
- (a) Prior written approval.
- (b) Travel disruption.

## 5. VAT

**5.2** VAT must be recorded for: Subsistence, Accommodation, Office supplies.

## 6. Claims that cannot be assessed

**6.1** Where the document is not a receipt the claim cannot be verified.

## 7. Categories in detail

**7.1 Subsistence.** Meals purchased while working away.
"""


@pytest.fixture
def policy():
    p = parse(GOOD, version="1.0", source="test")
    p.validate()
    return p


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_sections_are_found(policy):
    assert set(policy.sections) == {"1", "2", "3", "4", "5", "6", "7"}


def test_section_headings(policy):
    assert policy.sections["4"].heading == "Exceeding a category limit"


def test_clause_retrievable_by_reference(policy):
    c = policy.clause("4.2")
    assert c.section == "4"
    assert "supported where the excess" in c.text


def test_clause_carries_its_continuation_lines(policy):
    """4.2's lettered grounds are part of the clause, not orphaned."""
    text = policy.clause("4.2").text
    assert "Prior written approval" in text
    assert "Travel disruption" in text


def test_table_is_captured(policy):
    table = policy.sections["2"].tables[0]
    assert "Subsistence" in table
    assert "£40" in table


def test_unknown_clause_raises(policy):
    with pytest.raises(KeyError):
        policy.clause("9.9")


def test_version_parsed_from_document():
    from agent.policy import _VERSION_RE

    assert _VERSION_RE.search(GOOD).group(1) == "1.0"


# --------------------------------------------------------------------------
# Retrieval — each check sees its sections and nothing else
# --------------------------------------------------------------------------


def test_cost_rationale_check_sees_section_four(policy):
    ctx = policy.context_for(CHECK_COST_RATIONALE)
    assert "Prior written approval" in ctx


def test_cost_rationale_check_does_not_see_vat_rules(policy):
    """The point of per-check retrieval: narrow context, less to sift."""
    ctx = policy.context_for(CHECK_COST_RATIONALE)
    assert "VAT" not in ctx


def test_limit_check_sees_the_limits_table(policy):
    ctx = policy.context_for(CHECK_WITHIN_LIMIT)
    assert "£40" in ctx
    assert "Subsistence" in ctx


def test_vat_check_sees_only_vat(policy):
    ctx = policy.context_for(CHECK_VAT)
    assert "Office supplies" in ctx
    assert "Travel disruption" not in ctx


def test_other_rationale_check_sees_the_anti_avoidance_clause(policy):
    """2.3 matters here — Other must not be used to dodge a limit."""
    ctx = policy.context_for(CHECK_OTHER_RATIONALE)
    assert "avoid a category limit" in ctx


def test_every_check_has_a_mapping():
    from agent.assessment import CHECK_NAMES

    assert set(CHECK_SECTIONS) == set(CHECK_NAMES)


def test_unmapped_check_raises(policy):
    with pytest.raises(KeyError):
        policy.resolve_sections(99)


# --------------------------------------------------------------------------
# Validation — the maintainability guard
# --------------------------------------------------------------------------


def test_missing_section_fails_at_load():
    broken = GOOD.replace('## 4. Exceeding a category limit', '## 44. Exceeding a category limit')
    with pytest.raises(PolicyStructureError) as exc:
        parse(broken).validate()
    assert "section 4" in str(exc.value)


def test_renumbered_anchor_clause_fails_at_load():
    """The dangerous case: section 4 still exists but 4.2 has become 4.3.
    Retrieval would return plausible text about the wrong thing."""
    broken = GOOD.replace("**4.2**", "**4.3**")
    with pytest.raises(PolicyStructureError) as exc:
        parse(broken).validate()
    assert "4.2" in str(exc.value)


def test_missing_vat_anchor_fails_at_load():
    broken = GOOD.replace("**5.2**", "**5.7**")
    with pytest.raises(PolicyStructureError) as exc:
        parse(broken).validate()
    assert "5.2" in str(exc.value)


def test_emptied_section_fails_at_load():
    broken = GOOD.replace('**3.1** Other is reserved for expenses outside the listed categories.', '')
    with pytest.raises(PolicyStructureError) as exc:
        parse(broken).validate()
    assert "section 3" in str(exc.value)


def test_all_problems_reported_together():
    """A stale mapping usually breaks in several places at once. Report them
    all rather than one per run."""
    broken = GOOD.replace("**4.2**", "**4.3**").replace("**5.2**", "**5.7**")
    with pytest.raises(PolicyStructureError) as exc:
        parse(broken).validate()
    message = str(exc.value)
    assert "4.2" in message and "5.2" in message


def test_valid_policy_passes_validation(policy):
    policy.validate()  # no exception


def test_load_can_skip_validation():
    """Useful when deliberately inspecting a broken policy."""
    broken = GOOD.replace("**4.2**", "**4.3**")
    p = parse(broken)
    assert "4" in p.sections  # parsed fine; simply not validated


# --------------------------------------------------------------------------
# Citation
# --------------------------------------------------------------------------


def test_cited_clause_includes_its_reference(policy):
    assert policy.clause("2.3").cited().startswith("2.3 ")
