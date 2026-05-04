"""
src/auditor.py — PII Auditor using Microsoft Presidio NER.
"""
from presidio_analyzer import AnalyzerEngine


class PIIAuditor:
    def __init__(self):
        print("[Auditor] Initializing Presidio AnalyzerEngine...")
        self.engine = AnalyzerEngine()
        self.entities = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "US_SSN", "DATE_TIME", "LOCATION", "ORGANIZATION",
            "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "NRP"
        ]

    def audit(self, text: str) -> list:
        """
        Returns a list of detected PII entity dicts.
        Each dict: {entity_type, start, end, score, text_snippet}
        """
        results = self.engine.analyze(
            text=text,
            entities=self.entities,
            language="en"
        )
        entities = []
        for r in results:
            entities.append({
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
                "text_snippet": text[r.start:r.end]
            })
        return entities