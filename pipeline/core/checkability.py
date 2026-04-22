"""Checkability classification model wrapper."""

import logging
from typing import Tuple
from pathlib import Path

logger = logging.getLogger(__name__)
MIN_CLAIM_WORDS = 6


class CheckabilityClassifier:
    """Determine if a claim is checkable."""

    UNCHECKABLE_REASONS = {
        "too_short": f"Claim shorter than {MIN_CLAIM_WORDS} words",
        "opinion": "Subjective opinion, not factual claim",
        "question": "Question format, not a statement",
        "irrelevant": "Irrelevant or meaningless context"
    }

    def __init__(self, model_path: str = None):
        """Initialize with model checkpoint."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.id2label = {}
        self.checkable_ids = set()
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        ckpt_path = Path(str(self.model_path))
        required_any = ["config.json", "tokenizer.json", "tokenizer_config.json", "spm.model", "vocab.txt"]
        has_config = (ckpt_path / "config.json").exists()
        has_tokenizer = any((ckpt_path / n).exists() for n in required_any[1:])
        has_weights = any((ckpt_path / n).exists() for n in ["model.safetensors", "pytorch_model.bin"])
        if not (has_config and has_tokenizer and has_weights):
            logger.warning(
                "Checkability checkpoint incomplete at %s; falling back to heuristic.",
                ckpt_path,
            )
            self.model = None
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

            cfg = AutoConfig.from_pretrained(self.model_path)
            raw_map = getattr(cfg, "id2label", {}) or {}
            self.id2label = {int(k): str(v) for k, v in raw_map.items()}
            # Default/fallback mapping:
            # - binary heads often use label 1 as checkable
            # - v5 multilingual checkpoint uses FACTUAL_CLAIM as checkable (label 0)
            self.checkable_ids = {1}
            if self.id2label:
                normalized = {i: lbl.strip().lower() for i, lbl in self.id2label.items()}
                factual_like = {
                    i for i, lbl in normalized.items()
                    if lbl in {"factual_claim", "checkable", "factual", "claim_checkable"}
                }
                if factual_like:
                    self.checkable_ids = factual_like
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            logger.info(f"Checkability model loaded from {self.model_path} on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load checkability model: {e}. Using heuristic fallback.")
            self.model = None

    def classify(self, claim: str) -> Tuple[bool, str]:
        """
        Classify claim checkability.
        Returns: (is_checkable, reason_if_not)
        """
        # Heuristic checks (fast, no model needed)
        # Keep hard short-claim rejection only when model is unavailable.
        # For meme/image text, model-based checkability should still run.
        if (self.model is None) and len(claim.split()) < MIN_CLAIM_WORDS:
            return False, self.UNCHECKABLE_REASONS["too_short"]
        
        if claim.strip().endswith('?'):
            return False, self.UNCHECKABLE_REASONS["question"]
        
        # Opinion indicators
        opinion_words = ['i think', 'i believe', 'in my opinion', 'i feel', 
                         'should', 'ought to', 'favorite']
        if any(opt in claim.lower() for opt in opinion_words):
            return False, self.UNCHECKABLE_REASONS["opinion"]
        
        # If model available, use it for classification
        if self.model:
            return self._classify_with_model(claim)
        
        # Fallback: assume checkable if passed heuristics
        return True, ""
    
    def _classify_with_model(self, claim: str) -> Tuple[bool, str]:
        """Use trained model for checkability classification."""
        import torch
        
        inputs = self.tokenizer(claim, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted].item()
        
        label = str(self.id2label.get(predicted, f"LABEL_{predicted}"))
        if predicted not in self.checkable_ids:
            return False, f"Model label={label} (conf={confidence:.2f})"
        
        return True, ""
