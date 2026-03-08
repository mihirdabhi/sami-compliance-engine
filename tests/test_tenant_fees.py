"""
Tests for Tenant Fees Act 2019 validation rule.
Ensures prohibited fees are correctly detected and
permitted payments are not flagged.
"""

from app.rules.tenant_fees import TenantFeesRule


def test_detects_admin_fee():
    """Should flag administration fee as prohibited."""
    rule = TenantFeesRule()
    result = rule.validate("The tenant must pay an administration fee of £150.")
    assert result.status == "FAIL"
    assert result.total_violations == 1
    assert result.violations[0].severity == "HIGH"


def test_detects_renewal_fee():
    """Should flag renewal fee as prohibited."""
    rule = TenantFeesRule()
    result = rule.validate("A renewal fee of £75 applies annually.")
    assert result.status == "FAIL"
    assert result.total_violations == 1


def test_detects_semantic_variant():
    """Should flag processing charge as likely prohibited."""
    rule = TenantFeesRule()
    result = rule.validate("A processing charge of £50 is required.")
    assert result.status == "FAIL"
    assert result.violations[0].match_type == "semantic_match"
    assert result.violations[0].severity == "MEDIUM"


def test_passes_clean_document():
    """Should pass a document with no prohibited fees."""
    rule = TenantFeesRule()
    result = rule.validate(
        "Monthly rent is £1,200. A tenancy deposit of £1,200 is required. "
        "The tenant is responsible for council tax."
    )
    assert result.status == "PASS"
    assert result.total_violations == 0


def test_detects_multiple_violations():
    """Should detect multiple prohibited fees in one document."""
    rule = TenantFeesRule()
    text = (
        "The tenant shall pay an administration fee of £150, "
        "a check-in fee of £100, and a renewal fee of £75."
    )
    result = rule.validate(text)
    assert result.status == "FAIL"
    assert result.total_violations == 3


def test_ignores_permitted_payments():
    """Should not flag permitted payments like rent and deposit."""
    rule = TenantFeesRule()
    text = (
        "Rent: £1,500 per month. "
        "Tenancy deposit: £1,500. "
        "Council tax: payable by tenant. "
        "Holding deposit: £150."
    )
    result = rule.validate(text)
    assert result.status == "PASS"


def test_case_insensitive():
    """Should detect prohibited fees regardless of case."""
    rule = TenantFeesRule()
    result = rule.validate("An ADMINISTRATION FEE of £200 is payable.")
    assert result.status == "FAIL"
    assert result.total_violations == 1


def test_provides_legal_citation():
    """Should include specific legal citation for each violation."""
    rule = TenantFeesRule()
    result = rule.validate("A referencing fee of £30 per person applies.")
    assert result.status == "FAIL"
    assert "Tenant Fees Act 2019" in result.violations[0].citation


if __name__ == "__main__":
    tests = [
        test_detects_admin_fee,
        test_detects_renewal_fee,
        test_detects_semantic_variant,
        test_passes_clean_document,
        test_detects_multiple_violations,
        test_ignores_permitted_payments,
        test_case_insensitive,
        test_provides_legal_citation,
    ]

    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} — {e}")

    print(f"\n{len(tests)} tests completed.")
