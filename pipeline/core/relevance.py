"""Relevance scoring with optional two-stage retrieval/reranking."""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """Score relevance of evidence to claim with optional 2-stage ranker."""

    def __init__(
        self,
        model_path: str = None,
        bi_model_name: Optional[str] = None,
        enable_two_stage: bool = True,
        shortlist_k: int = 20,
        top_k: Optional[int] = None,
    ):
        """Initialize relevance models.

        Stage 1 (optional): bi-encoder shortlist.
        Stage 2: cross-encoder / HF seq-classifier rerank.
        """
        self.model_path = model_path
        self.model = None  # stage-2 model
        self.model_kind = None  # "cross_encoder" | "hf_seqcls" | None
        self.tokenizer = None
        self.bi_model = None
        self.device = "cpu"

        self.enable_two_stage = bool(enable_two_stage)
        self.shortlist_k = max(1, int(shortlist_k))
        self.top_k = int(top_k) if top_k is not None and int(top_k) > 0 else None
        self.bi_model_name = (
            bi_model_name
            or os.getenv("RELEVANCE_BI_ENCODER_MODEL", "intfloat/multilingual-e5-small")
        )

        if model_path:
            self._load_model()
        if self.enable_two_stage:
            self._load_bi_encoder()

    def _load_model(self):
        """Load stage-2 reranker model."""
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1) Prefer standard HF sequence-classification checkpoints.
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            self.model_kind = "hf_seqcls"
            logger.info(f"Relevance HF seq-classifier loaded from {self.model_path} on {self.device}")
            return
        except Exception as e:
            logger.info(f"HF seq-classifier load failed for {self.model_path}: {e}")

        # 2) Fallback: try sentence-transformers CrossEncoder checkpoints.
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_path, device=self.device)
            self.model_kind = "cross_encoder"
            logger.info(f"Relevance cross-encoder loaded from {self.model_path} on {self.device}")
            return
        except Exception as e:
            logger.warning(f"Failed to load relevance model: {e}. Using keyword fallback.")
            self.model = None
            self.model_kind = None

    def _load_bi_encoder(self):
        """Load stage-1 bi-encoder model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.bi_model = SentenceTransformer(self.bi_model_name, device=self.device)
            logger.info(f"Relevance bi-encoder loaded: {self.bi_model_name} on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load bi-encoder {self.bi_model_name}: {e}")
            self.bi_model = None

    def score(self, claim: str, evidence: str) -> float:
        """
        Score relevance of evidence to claim.

        Returns:
            Relevance score 0.0-1.0
        """
        if self.model:
            return self._score_with_model(claim, evidence)
        return self._score_with_keywords(claim, evidence)
    
    def _score_with_model(self, claim: str, evidence: str) -> float:
        """Use cross-encoder for relevance scoring."""
        try:
            if self.model_kind == "cross_encoder":
                import numpy as np

                raw = self.model.predict([[claim, evidence]])[0]
                arr = np.asarray(raw)

                # Case A: scalar logit/score
                if arr.ndim == 0:
                    return float(1 / (1 + np.exp(-arr.item())))

                # Case B: class logits/probs vector (binary/multi-class)
                if arr.size >= 2:
                    # Prefer class-1 (RELEVANT) as relevance.
                    ex = np.exp(arr - np.max(arr))
                    probs = ex / np.sum(ex)
                    return float(probs[1])

                # Case C: size-1 vector
                return float(1 / (1 + np.exp(-arr.reshape(-1)[0])))

            if self.model_kind == "hf_seqcls":
                import torch

                inputs = self.tokenizer(
                    claim,
                    evidence,
                    return_tensors="pt",
                    truncation="only_second",
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0]
                    probs = torch.softmax(logits, dim=-1)

                # Binary classifier with RELEVANT as class 1 (config in v9_run1).
                if probs.shape[0] >= 2:
                    return float(probs[1].item())
                return float(probs[0].item())

            return 0.5
        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            return 0.5

    def rank_evidence(
        self,
        claim: str,
        evidence_list: List[Dict],
        language: str = "en",
        shortlist_k: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Two-stage relevance ranking over an evidence list.

        Stage-1: bi-encoder fast filter (N -> shortlist_k)
        Stage-2: reranker model scoring on shortlist
        """
        if not evidence_list:
            return []

        sk = max(1, int(shortlist_k if shortlist_k is not None else self.shortlist_k))
        tk = top_k if top_k is not None else self.top_k
        if tk is not None and tk <= 0:
            tk = None

        # Stage-1 scores for all candidates.
        stage1_scores = self._score_many_stage1(claim, evidence_list)
        ranked_stage1 = sorted(
            enumerate(stage1_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        shortlist_ids = {idx for idx, _ in ranked_stage1[: min(sk, len(ranked_stage1))]}

        ranked: List[Dict] = []
        for idx, ev in enumerate(evidence_list):
            text = str(ev.get("text", ""))
            s1 = float(stage1_scores[idx])

            # Stage-2 only on shortlist. Otherwise keep stage-1 score.
            if idx in shortlist_ids and self.model is not None and text:
                s2 = float(self._score_with_model(claim, text))
                final = s2
                source = f"two_stage:{self.model_kind}"
            else:
                s2 = None
                final = s1 if self.enable_two_stage else float(self.score(claim, text))
                source = "bi_encoder" if self.enable_two_stage else "single_stage"

            ev2 = dict(ev)
            ev2["relevance_stage1"] = s1
            if s2 is not None:
                ev2["relevance_stage2"] = s2
            ev2["relevance"] = float(final)
            ev2["relevance_source"] = source
            ranked.append(ev2)

        ranked.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
        if tk is not None:
            ranked = ranked[: min(tk, len(ranked))]
        return ranked

    def _score_many_stage1(self, claim: str, evidence_list: List[Dict]) -> List[float]:
        """Stage-1 bi-encoder scores for all evidence rows."""
        if not evidence_list:
            return []

        if self.enable_two_stage and self.bi_model is not None:
            try:
                import numpy as np

                from sentence_transformers import util

                query = f"query: {claim}"
                passages = [f"passage: {str(ev.get('text', ''))}" for ev in evidence_list]

                q_emb = self.bi_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
                p_emb = self.bi_model.encode(passages, convert_to_tensor=True, normalize_embeddings=True)
                sims = util.cos_sim(q_emb, p_emb)[0].detach().cpu().numpy()
                # Cosine [-1, 1] -> [0, 1]
                return [float((x + 1.0) * 0.5) for x in np.asarray(sims).reshape(-1)]
            except Exception as e:
                logger.warning(f"Bi-encoder scoring failed; falling back to keyword stage1: {e}")

        # Fallback stage-1 if bi-encoder unavailable.
        return [
            float(self._score_with_keywords(claim, str(ev.get("text", ""))))
            for ev in evidence_list
        ]

    def _score_with_keywords(self, claim: str, evidence: str) -> float:
        """Keyword-based relevance scoring (fallback)."""
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                     'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                     'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                     'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
                     'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its'}
        
        claim_words -= stopwords
        evidence_words -= stopwords
        
        if not claim_words:
            return 0.3
        
        # Calculate overlap
        overlap = claim_words & evidence_words
        overlap_score = len(overlap) / len(claim_words)
        
        # Normalize to 0-1 range
        return min(overlap_score * 1.5, 1.0)  # Scale up slightly
