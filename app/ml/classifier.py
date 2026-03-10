"""
SAMI Compliance Engine — ML Text Classifier

Uses scikit-learn to classify document clauses as compliant
or potentially non-compliant. This adds a machine learning
layer on top of the deterministic rule matching.

The classifier is trained on labelled examples of compliant
and non-compliant clauses, then used to flag suspicious
clauses that the regex rules might miss.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import List, Dict, Tuple


# Training data: labelled examples of compliant and non-compliant clauses
# Label 1 = non-compliant (violation), Label 0 = compliant
TRAINING_DATA = [
    # Non-compliant examples (label 1)
    ("The tenant shall pay an administration fee of £150", 1),
    ("A referencing fee of £50 per applicant is required", 1),
    ("Renewal fee of £75 applies at each contract extension", 1),
    ("Check-out fee of £100 is payable upon vacating", 1),
    ("A processing charge will be applied for all applications", 1),
    ("The tenant must pay a setup fee before moving in", 1),
    ("An inventory check fee of £80 is mandatory", 1),
    ("Credit check fee of £30 per person applies", 1),
    ("A cleaning fee of £200 is deducted from the deposit", 1),
    ("Documentation fee of £60 is required upon signing", 1),
    ("The tenant agrees to pay a viewing fee of £25", 1),
    ("A move-in charge of £120 applies to all new tenants", 1),
    ("Tenancy signup fee is non-refundable", 1),
    ("A vetting fee is required for each guarantor", 1),
    ("Property inspection fee payable annually by tenant", 1),
    ("The landlord retains all data indefinitely", 1),
    ("Tenant waives their data protection rights", 1),
    ("Consent to data sharing is irrevocable", 1),
    ("Personal data may be shared with any third parties", 1),
    ("The tenant cannot request deletion of their records", 1),

    # Compliant examples (label 0)
    ("Monthly rent is £1200 payable on the first of each month", 0),
    ("A tenancy deposit of five weeks rent is required", 0),
    ("The holding deposit shall not exceed one weeks rent", 0),
    ("The tenant is responsible for council tax payments", 0),
    ("Utility bills are the responsibility of the tenant", 0),
    ("The landlord will protect the deposit in an approved scheme", 0),
    ("Notice period of two months is required from either party", 0),
    ("The property is let on an assured shorthold tenancy basis", 0),
    ("Rent is reviewed annually in line with market rates", 0),
    ("The tenant shall maintain the property in good condition", 0),
    ("The landlord will provide an EPC before the tenancy begins", 0),
    ("Gas safety certificate will be provided annually", 0),
    ("The deposit will be registered with the DPS within 30 days", 0),
    ("Prescribed information will be provided to the tenant", 0),
    ("The tenant has the right to request their personal data", 0),
    ("Data will be retained for six years after tenancy ends", 0),
    ("The privacy notice is available on our website", 0),
    ("Tenants may withdraw consent for marketing at any time", 0),
    ("We process data under legitimate interest for tenancy management", 0),
    ("The tenant may lodge a complaint with the ICO", 0),
]


class ComplianceClassifier:
    """
    ML classifier for detecting potentially non-compliant clauses.

    Uses TF-IDF vectorization and Logistic Regression to classify
    text as compliant or potentially non-compliant. This supplements
    the deterministic rule matching by catching clauses that may
    not match exact patterns but have similar characteristics to
    known violations.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                stop_words="english",
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
            )),
        ])
        self.is_trained = False

    def train(self, data: List[Tuple[str, int]] = None):
        """
        Train the classifier on labelled examples.

        Args:
            data: List of (text, label) tuples. Uses built-in
                  training data if not provided.
        """
        if data is None:
            data = TRAINING_DATA

        texts = [item[0] for item in data]
        labels = [item[1] for item in data]

        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def predict(self, text: str) -> Dict:
        """
        Classify a single clause as compliant or non-compliant.

        Returns:
            Dict with prediction, confidence score, and risk level.
        """
        if not self.is_trained:
            self.train()

        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]

        confidence = max(probabilities)
        risk_score = probabilities[1]  # probability of non-compliant

        if risk_score >= 0.8:
            risk_level = "HIGH"
        elif risk_score >= 0.5:
            risk_level = "MEDIUM"
        elif risk_score >= 0.3:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        return {
            "text": text,
            "prediction": "NON_COMPLIANT" if prediction == 1 else "COMPLIANT",
            "confidence": round(confidence, 3),
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Classify multiple clauses at once."""
        return [self.predict(text) for text in texts]

    def analyse_document(self, document_text: str) -> Dict:
        """
        Split a document into clauses and classify each one.

        Splits on sentence boundaries and analyses each clause
        independently, returning an overall risk assessment.
        """
        if not self.is_trained:
            self.train()

        # Split into clauses (simple sentence splitting)
        clauses = [
            clause.strip()
            for clause in document_text.replace("\n", " ").split(".")
            if clause.strip() and len(clause.strip()) > 20
        ]

        results = []
        high_risk_count = 0
        medium_risk_count = 0

        for clause in clauses:
            result = self.predict(clause)
            results.append(result)

            if result["risk_level"] == "HIGH":
                high_risk_count += 1
            elif result["risk_level"] == "MEDIUM":
                medium_risk_count += 1

        # Overall document risk
        if high_risk_count >= 2:
            overall_risk = "HIGH"
        elif high_risk_count >= 1 or medium_risk_count >= 2:
            overall_risk = "MEDIUM"
        elif medium_risk_count >= 1:
            overall_risk = "LOW"
        else:
            overall_risk = "MINIMAL"

        return {
            "total_clauses": len(clauses),
            "high_risk_clauses": high_risk_count,
            "medium_risk_clauses": medium_risk_count,
            "overall_risk": overall_risk,
            "clause_analysis": results,
        }


if __name__ == "__main__":
    # Demo: classify sample clauses
    classifier = ComplianceClassifier()
    classifier.train()

    test_clauses = [
        "The tenant must pay an administration fee of £150",
        "Monthly rent is £1200 payable on the first",
        "A processing charge of £50 applies to all applicants",
        "The deposit will be protected in the DPS scheme",
        "The landlord will retain all data indefinitely",
        "Tenant may request deletion of personal data at any time",
        "An onboarding charge of £80 is required",
        "Council tax is the responsibility of the tenant",
    ]

    print("\n" + "=" * 70)
    print("  SAMI COMPLIANCE ENGINE — ML Clause Classification")
    print("=" * 70)

    for clause in test_clauses:
        result = classifier.predict(clause)
        status = result["prediction"]
        risk = result["risk_level"]
        score = result["risk_score"]

        icon = "WARNING" if status == "NON_COMPLIANT" else "OK"
        print(f"\n  [{icon}] {clause[:60]}...")
        print(f"       Prediction: {status} | Risk: {risk} | Score: {score}")

    # Demo: full document analysis
    print("\n" + "=" * 70)
    print("  FULL DOCUMENT ANALYSIS")
    print("=" * 70)

    sample_doc = """
    The monthly rent is £1400 payable on the first of each month.
    The tenant shall pay an administration fee of £200 upon signing.
    A referencing fee of £50 per person is required before move-in.
    The deposit of £1400 will be protected in an approved scheme.
    The landlord will retain all personal data indefinitely.
    Council tax and utilities are the responsibility of the tenant.
    """

    doc_result = classifier.analyse_document(sample_doc)
    print(f"\n  Total Clauses Analysed: {doc_result['total_clauses']}")
    print(f"  High Risk: {doc_result['high_risk_clauses']}")
    print(f"  Medium Risk: {doc_result['medium_risk_clauses']}")
    print(f"  Overall Risk: {doc_result['overall_risk']}")
