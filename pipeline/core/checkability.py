"""Checkability classification model wrapper."""

import logging
import os
import re
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)
MIN_CLAIM_WORDS = 6


class CheckabilityClassifier:
    """Determine if a claim is checkable."""

    UNCHECKABLE_REASONS = {
        "too_short": f"Claim shorter than {MIN_CLAIM_WORDS} words",
        "opinion": "Subjective opinion, not factual claim",
        "question": "Question format, not a statement",
        "irrelevant": "Irrelevant or meaningless context",
    }

    def __init__(self, model_path: str = None):
        """Initialize with model checkpoint."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.id2label = {}
        self.checkable_ids = set()

        # Multi-language relaxation toggles to reduce over-blocking.
        self.multi_relax_enable = os.getenv(
            "CHECKABILITY_MULTI_RELAX_ENABLE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_factual_override = os.getenv(
            "CHECKABILITY_MULTI_FACTUAL_OVERRIDE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_min_conf_block = float(
            os.getenv("CHECKABILITY_MULTI_MIN_CONF_BLOCK", "0.90")
        )
        relax_labels_raw = os.getenv(
            "CHECKABILITY_MULTI_RELAX_LABELS", "QUESTION_OR_REWRITE,OTHER_UNCHECKABLE"
        )
        self.multi_relax_labels = {
            str(x).strip().upper()
            for x in str(relax_labels_raw).split(",")
            if str(x).strip()
        }
        bypass_langs_raw = os.getenv("CHECKABILITY_BYPASS_LANGS", "")
        self.checkability_bypass_langs = {
            str(x).strip().lower()
            for x in str(bypass_langs_raw).split(",")
            if str(x).strip()
        }

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
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

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
            logger.info("Checkability model loaded from %s on %s", self.model_path, self.device)
        except Exception as e:
            logger.warning("Failed to load checkability model: %s. Using heuristic fallback.", e)
            self.model = None

    def classify(self, claim: str, language: str = "en") -> Tuple[bool, str]:
        """
        Classify claim checkability.
        Returns: (is_checkable, reason_if_not)
        """
        lang = str(language or "en").strip().lower()
        if lang in self.checkability_bypass_langs:
            return True, ""

        # Heuristic checks (fast, no model needed)
        # Keep hard short-claim rejection only when model is unavailable.
        if (self.model is None) and len(claim.split()) < MIN_CLAIM_WORDS:
            return False, self.UNCHECKABLE_REASONS["too_short"]

        if claim.strip().endswith("?"):
            return False, self.UNCHECKABLE_REASONS["question"]

        # Opinion indicators
        opinion_words = ["i think", "i believe", "in my opinion", "i feel", "should", "ought to", "favorite"]
        if any(opt in claim.lower() for opt in opinion_words):
            return False, self.UNCHECKABLE_REASONS["opinion"]

        # If model available, use it for classification
        if self.model:
            return self._classify_with_model(claim, language=language)

        # Fallback: assume checkable if passed heuristics
        return True, ""

    def _classify_with_model(self, claim: str, language: str = "en") -> Tuple[bool, str]:
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
            if self._should_relax_to_checkable(
                claim=claim,
                language=language,
                label=label,
                confidence=confidence,
            ):
                return True, ""
            return False, f"Model label={label} (conf={confidence:.2f})"

        return True, ""

    def _should_relax_to_checkable(
        self,
        claim: str,
        language: str,
        label: str,
        confidence: float,
    ) -> bool:
        """Optional multi-language relaxation to reduce over-blocking."""
        lang = str(language or "en").strip().lower()
        if not self.multi_relax_enable or lang.startswith("en"):
            return False
        lbl = str(label or "").strip().upper()
        if lbl not in self.multi_relax_labels:
            return False
        # Keep strong uncheckable predictions blocked.
        if float(confidence or 0.0) >= self.multi_min_conf_block:
            return False
        if self.multi_factual_override and not self._looks_factual_shape(claim):
            return False
        return True

    def _looks_factual_shape(self, claim: str) -> bool:
        text = str(claim or "").strip()
        if not text:
            return False
        # Numeric/date anchor is a strong factual cue across languages.
        has_digit = bool(re.search(r"\d", text))
        has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", text))
        # Entity-ish cue for Latin-script claims.
        has_capital_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text))
        token_count = len(re.findall(r"\S+", text))
        return bool(has_digit or has_year or has_capital_entity or token_count >= 10)
