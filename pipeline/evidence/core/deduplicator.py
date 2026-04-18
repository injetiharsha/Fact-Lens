from collections import defaultdict
from pipeline.evidence.core.aggregator import EvidenceAggregator


class EvidenceDeduplicator(EvidenceAggregator):
    """Adds domain diversity hard-cap behavior on top of current dedup."""

    def apply_domain_diversity(self, evidence_list, max_per_domain=3):
        kept = []
        seen = defaultdict(int)
        for ev in evidence_list:
            domain = (ev.get("url", "") or ev.get("source", "unknown")).split("/")[2:3]
            key = domain[0] if domain else "unknown"
            if seen[key] >= max_per_domain:
                continue
            seen[key] += 1
            kept.append(ev)
        return kept
