"""Stance detection using NLI models."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class StanceDetector:
    """Detect stance (support/refute/neutral) of evidence towards claim using NLI."""

    def __init__(self, model_path: str = None):
        """Initialize with NLI model."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._idx_to_stance = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load NLI model for stance detection."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            except Exception as tok_err:
                logger.warning(
                    "Slow tokenizer load failed for %s: %s. Retrying with fast tokenizer.",
                    self.model_path,
                    tok_err,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=3  # entailment/support, neutral, contradiction/refute
            ).to(self.device)
            self.model.eval()
            self._build_index_mapping()
            logger.info(f"Stance detection model loaded from {self.model_path} on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load stance model: {e}. Using keyword fallback.")
            self.model = None

    def _build_index_mapping(self):
        """Build model-logit index -> canonical stance label mapping from checkpoint config."""
        self._idx_to_stance = {0: "support", 1: "neutral", 2: "refute"}  # safe fallback
        try:
            cfg = getattr(self.model, "config", None)
            id2label = getattr(cfg, "id2label", None) if cfg is not None else None
            if not id2label:
                return

            normalized = {}
            for raw_k, raw_v in id2label.items():
                try:
                    k = int(raw_k)
                except Exception:
                    continue
                v = str(raw_v).strip().lower()
                if v in {"support", "supported", "entailment", "entails", "true"}:
                    normalized[k] = "support"
                elif v in {"refute", "refuted", "contradiction", "contradicts", "false"}:
                    normalized[k] = "refute"
                elif v in {"neutral", "nei", "not enough info"}:
                    normalized[k] = "neutral"

            # Keep fallback if mapping incomplete.
            if set(normalized.values()) >= {"support", "refute", "neutral"}:
                self._idx_to_stance = normalized
                logger.info("Stance label mapping resolved from checkpoint: %s", self._idx_to_stance)
        except Exception:
            pass

    def detect(self, claim: str, evidence: str) -> Dict[str, float]:
        """
        Detect stance of evidence towards claim.
        
        Returns:
            Dict with {support: prob, refute: prob, neutral: prob}
        """
        if self.model:
            return self._detect_with_model(claim, evidence)
        return self._detect_with_keywords(claim, evidence)
    
    def _detect_with_model(self, claim: str, evidence: str) -> Dict[str, float]:
        """Use NLI model for stance detection."""
        import torch
        
        try:
            # Format: [CLS] claim [SEP] evidence [SEP]
            inputs = self.tokenizer(
                claim,
                evidence,
                return_tensors="pt",
                truncation="only_second",
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            stance_probs = {"support": 0.0, "refute": 0.0, "neutral": 0.0}
            idx_map = self._idx_to_stance or {0: "support", 1: "neutral", 2: "refute"}
            for i in range(probs.shape[-1]):
                stance = idx_map.get(i)
                if stance in stance_probs:
                    stance_probs[stance] = probs[0][i].item()

            # If mapping produced empty/partial values, fallback to old convention.
            if sum(stance_probs.values()) <= 0:
                stance_probs = {
                    "support": probs[0][0].item(),
                    "neutral": probs[0][1].item(),
                    "refute": probs[0][2].item(),
                }
            return stance_probs
        
        except Exception as e:
            logger.error(f"Model stance detection failed: {e}")
            return self._detect_with_keywords(claim, evidence)
    
    def _detect_with_keywords(self, claim: str, evidence: str) -> Dict[str, float]:
        """Keyword-based stance detection (fallback)."""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Supporting indicators
        support_words = ["confirmed", "verified", "true", "correct", "accurate", "supports",
                         "agrees", "consistent", "validates", "proves", "evidence shows"]
        
        # Refuting indicators
        refute_words = ["false", "incorrect", "debunked", "misleading", "wrong", "disproves",
                        "contradicts", "refutes", "denies", "not true", "hoax", "fake"]
        
        # Neutral indicators
        neutral_words = ["unclear", "uncertain", "mixed", "partial", "some evidence",
                         "conflicting", "inconclusive", "needs more"]
        
        support_score = sum(1 for word in support_words if word in evidence_lower)
        refute_score = sum(1 for word in refute_words if word in evidence_lower)
        neutral_score = sum(1 for word in neutral_words if word in evidence_lower)
        
        # Check claim-evidence word overlap
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        overlap = len(claim_words & evidence_words)
        
        if overlap > 3:
            support_score += overlap * 0.3
        
        # Normalize to probabilities
        total = support_score + refute_score + neutral_score + 0.1  # Small epsilon
        if total == 0:
            return {"support": 0.33, "refute": 0.33, "neutral": 0.34}
        
        return {
            "support": support_score / total,
            "refute": refute_score / total,
            "neutral": neutral_score / total
        }
