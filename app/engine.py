"""
SAMI Compliance Engine — Unified Validation Pipeline

Combines deterministic rule validation with ML classification
to provide comprehensive compliance checking. Rules provide
guaranteed detection of known violations, while the ML layer
catches potential violations that rules might miss.
"""

from typing import Dict, List
from .rules.tenant_fees import TenantFeesRule
from .rules.gdpr import GDPRRule
from .rules.deposit import DepositProtectionRule
from .ml.classifier import ComplianceClassifier


class ComplianceEngine:
    """
    Unified compliance validation engine.

    Runs documents through two layers:
    1. Deterministic rules — guaranteed detection of known violations
    2. ML classifier — probabilistic detection of potential violations

    Results are merged and deduplicated, with deterministic
    findings always taking priority over ML predictions.
    """

    def __init__(self):
        self.rules = [
            TenantFeesRule(),
            GDPRRule(),
            DepositProtectionRule(),
        ]
        self.classifier = ComplianceClassifier()
        self.classifier.train()

    def validate(self, document_text: str) -> Dict:
        """
        Run full compliance validation on a document.

        Returns combined results from both deterministic
        rules and ML classification.
        """
        # Layer 1: Deterministic rules
        rule_results = []
        total_rule_violations = 0

        for rule in self.rules:
            result = rule.validate(document_text)
            rule_results.append(result.to_dict())
            total_rule_violations += result.total_violations

        # Layer 2: ML classification
        ml_results = self.classifier.analyse_document(document_text)

        # Determine overall status
        has_rule_violations = total_rule_violations > 0
        has_ml_warnings = ml_results["high_risk_clauses"] > 0

        if has_rule_violations:
            overall_status = "FAIL"
            confidence = "HIGH"
            reason = "Deterministic rules detected confirmed violations"
        elif has_ml_warnings:
            overall_status = "REVIEW"
            confidence = "MEDIUM"
            reason = "ML classifier flagged potential issues requiring human review"
        else:
            overall_status = "PASS"
            confidence = "HIGH"
            reason = "No violations detected by rules or ML classifier"

        return {
            "overall_status": overall_status,
            "confidence": confidence,
            "reason": reason,
            "deterministic_layer": {
                "rules_checked": len(self.rules),
                "total_violations": total_rule_violations,
                "results": rule_results,
            },
            "ml_layer": {
                "clauses_analysed": ml_results["total_clauses"],
                "high_risk_clauses": ml_results["high_risk_clauses"],
                "medium_risk_clauses": ml_results["medium_risk_clauses"],
                "overall_risk": ml_results["overall_risk"],
                "clause_details": ml_results["clause_analysis"],
            },
            "recommendation": self._generate_recommendation(
                has_rule_violations, has_ml_warnings, total_rule_violations, ml_results
            ),
        }

    def _generate_recommendation(
        self,
        has_rule_violations: bool,
        has_ml_warnings: bool,
        violation_count: int,
        ml_results: Dict,
    ) -> str:
        """Generate a human-readable recommendation based on results."""
        if has_rule_violations and has_ml_warnings:
            return (
                f"URGENT: {violation_count} confirmed regulatory violation(s) detected "
                f"plus {ml_results['high_risk_clauses']} additional clause(s) flagged by "
                f"ML analysis. This document requires immediate review before any action."
            )
        elif has_rule_violations:
            return (
                f"ACTION REQUIRED: {violation_count} confirmed regulatory violation(s) "
                f"detected. These must be resolved before the document can be approved."
            )
        elif has_ml_warnings:
            return (
                f"REVIEW RECOMMENDED: ML analysis flagged "
                f"{ml_results['high_risk_clauses']} clause(s) as potentially "
                f"non-compliant. Manual review recommended before approval."
            )
        else:
            return (
                "CLEAR: No violations detected by deterministic rules and no "
                "significant concerns flagged by ML analysis. Document appears compliant."
            )

    def get_status(self) -> Dict:
        """Return current engine configuration."""
        return {
            "engine": "SAMI Compliance Engine",
            "version": "0.2.0",
            "deterministic_rules": len(self.rules),
            "ml_classifier": "TF-IDF + Logistic Regression",
            "regulations_covered": [rule.name for rule in self.rules],
            "architecture": "Dual-layer: Deterministic + ML",
        }


if __name__ == "__main__":
    engine = ComplianceEngine()

    sample = """
    ASSURED SHORTHOLD TENANCY AGREEMENT

    Property: 42 Oak Street, London, E1 6AN

    The monthly rent is £1,400 payable on the 1st of each month.
    The tenant shall pay an administration fee of £200 upon signing.
    A referencing fee of £50 per person is required before move-in.
    The deposit of £2,000 will be held by the landlord.
    The landlord will indefinitely retain all personal data collected.
    The tenant provides irrevocable consent to data processing.
    Council tax is the responsibility of the tenant.
    """

    print("\n" + "=" * 60)
    print("  SAMI COMPLIANCE ENGINE — Full Validation Report")
    print("=" * 60)

    result = engine.validate(sample)

    print(f"\n  Status: {result['overall_status']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reason: {result['reason']}")

    print(f"\n  --- Deterministic Layer ---")
    print(f"  Rules checked: {result['deterministic_layer']['rules_checked']}")
    print(f"  Violations found: {result['deterministic_layer']['total_violations']}")

    for rule_result in result["deterministic_layer"]["results"]:
        status = rule_result["status"]
        print(f"\n  [{status}] {rule_result['rule']}")
        for v in rule_result["violations"]:
            print(f"    - [{v['severity']}] '{v['term']}'")
            print(f"      {v['citation']}")

    print(f"\n  --- ML Layer ---")
    print(f"  Clauses analysed: {result['ml_layer']['clauses_analysed']}")
    print(f"  High risk: {result['ml_layer']['high_risk_clauses']}")
    print(f"  Overall ML risk: {result['ml_layer']['overall_risk']}")

    print(f"\n  RECOMMENDATION: {result['recommendation']}")
    print("=" * 60)
