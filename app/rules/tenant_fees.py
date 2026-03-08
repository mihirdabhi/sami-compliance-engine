"""
Tenant Fees Act 2019 — Compliance Validation Rule

Detects prohibited fees in tenancy agreements and related documents.
The Act prohibits landlords and letting agents from charging tenants
certain fees in connection with a tenancy.

Uses two detection methods:
1. Exact matching against known prohibited terms
2. Semantic matching for variant phrasings using regex patterns

Reference: Tenant Fees Act 2019, Schedule 1
https://www.legislation.gov.uk/ukpga/2019/4/schedule/1
"""

import re
from typing import List
from .base import ComplianceRule, ValidationResult, Violation


# Known prohibited fee terms per Schedule 1 of the Act
PROHIBITED_TERMS = [
    "administration fee",
    "admin fee",
    "check-in fee",
    "check-out fee",
    "checkout fee",
    "reference fee",
    "referencing fee",
    "inventory fee",
    "inventory check fee",
    "renewal fee",
    "tenancy renewal fee",
    "credit check fee",
    "guarantor fee",
    "cleaning fee",
    "professional cleaning fee",
    "contract fee",
    "setup fee",
    "registration fee",
    "viewing fee",
    "sign-up fee",
]

# Semantic variants that likely indicate prohibited fees
# Each tuple: (regex pattern, description of why it is likely prohibited)
SEMANTIC_VARIANTS = [
    (r"processing\s+charge", "Likely variant of administration fee"),
    (r"documentation\s+fee", "Likely variant of admin fee"),
    (r"documentation\s+charge", "Likely variant of admin fee"),
    (r"move[\s-]?in\s+fee", "Likely variant of check-in fee"),
    (r"move[\s-]?in\s+charge", "Likely variant of check-in fee"),
    (r"move[\s-]?out\s+fee", "Likely variant of check-out fee"),
    (r"move[\s-]?out\s+charge", "Likely variant of check-out fee"),
    (r"sign[\s-]?up\s+fee", "Likely variant of setup fee"),
    (r"sign[\s-]?up\s+charge", "Likely variant of setup fee"),
    (r"onboarding\s+fee", "Likely variant of administration fee"),
    (r"onboarding\s+charge", "Likely variant of administration fee"),
    (r"tenant\s+verification\s+fee", "Likely variant of referencing fee"),
    (r"vetting\s+fee", "Likely variant of referencing fee"),
    (r"property\s+inspection\s+fee", "Likely variant of inventory fee"),
]

# Permitted payments under the Act (for context — these should NOT be flagged)
PERMITTED_PAYMENTS = [
    "rent",
    "tenancy deposit",
    "holding deposit",
    "payment in the event of a default",
    "payment on termination at tenant request",
    "council tax",
    "utilities",
    "communication services",
    "tv licence",
]


class TenantFeesRule(ComplianceRule):
    """Validates documents against Tenant Fees Act 2019."""

    @property
    def name(self) -> str:
        return "Tenant Fees Act 2019"

    @property
    def legislation(self) -> str:
        return "Tenant Fees Act 2019, Schedule 1 — Prohibited Payments"

    def validate(self, document_text: str) -> ValidationResult:
        """
        Check document for prohibited fees under the Tenant Fees Act 2019.

        Runs two passes:
        1. Exact term matching for known prohibited fees
        2. Semantic pattern matching for variant phrasings
        """
        violations = []

        # Pass 1: Exact prohibited term matching
        violations.extend(self._check_exact_terms(document_text))

        # Pass 2: Semantic variant matching
        violations.extend(self._check_semantic_variants(document_text))

        # Remove duplicates (same position flagged by both passes)
        violations = self._deduplicate(violations)

        status = "FAIL" if violations else "PASS"
        summary = (
            f"Found {len(violations)} potential violation(s) of {self.name}"
            if violations
            else f"No prohibited fees detected under {self.name}"
        )

        return ValidationResult(
            rule_name=self.name,
            status=status,
            total_violations=len(violations),
            violations=violations,
            summary=summary,
        )

    def _check_exact_terms(self, text: str) -> List[Violation]:
        """Check for exact matches of prohibited fee terms."""
        violations = []
        text_lower = text.lower()

        for term in PROHIBITED_TERMS:
            # Use word boundary matching to avoid partial matches
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                violations.append(
                    Violation(
                        term=term,
                        position=match.start(),
                        context=self._extract_context(text, match.start()),
                        severity="HIGH",
                        citation=self.legislation,
                        match_type="exact_match",
                        description=f"Prohibited payment: '{term}'",
                    )
                )

        return violations

    def _check_semantic_variants(self, text: str) -> List[Violation]:
        """Check for semantic variants of prohibited fees."""
        violations = []

        for pattern_str, description in SEMANTIC_VARIANTS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                violations.append(
                    Violation(
                        term=match.group(),
                        position=match.start(),
                        context=self._extract_context(text, match.start()),
                        severity="MEDIUM",
                        citation="Tenant Fees Act 2019, Section 1(1)",
                        match_type="semantic_match",
                        description=description,
                    )
                )

        return violations

    def _deduplicate(self, violations: List[Violation]) -> List[Violation]:
        """Remove duplicate violations at the same position."""
        seen_positions = set()
        unique = []
        for v in violations:
            if v.position not in seen_positions:
                seen_positions.add(v.position)
                unique.append(v)
        return unique


if __name__ == "__main__":
    # Quick test with sample document
    sample = """
    TENANCY AGREEMENT — 42 Oak Street, London

    1. The tenant shall pay an administration fee of £150 upon signing
       this agreement.
    2. A renewal fee of £75 will be charged at each contract renewal.
    3. The processing charge for references is £50 per applicant.
    4. Monthly rent: £1,200 payable on the 1st of each month.
    5. A tenancy deposit of £1,200 (5 weeks rent) is required.
    6. The tenant is responsible for council tax payments.
    """

    rule = TenantFeesRule()
    result = rule.validate(sample)

    print(f"\nRule: {result.rule_name}")
    print(f"Status: {result.status}")
    print(f"Violations: {result.total_violations}")
    print(f"Summary: {result.summary}\n")

    for v in result.violations:
        print(f"  [{v.severity}] '{v.term}'")
        print(f"    Citation: {v.citation}")
        print(f"    Type: {v.match_type}")
        print(f"    Context: ...{v.context}...")
        print()
