class VerdictExplainer:
    def explain(self, reasoning: str, evidence_count: int) -> str:
        return f"{reasoning} Evidence count: {evidence_count}."
