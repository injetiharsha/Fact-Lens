"""Main pipeline orchestrator - coordinates all pipeline stages."""

import logging
import os
import re
import time
from typing import Callable
from typing import Dict, List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from pipeline.core.normalizer import ClaimNormalizer
from pipeline.core.checkability import CheckabilityClassifier
from pipeline.core.context_classifier import ContextClassifier
from pipeline.core.domain_router import DomainRouter
from pipeline.evidence.core.gatherer import EvidenceGatherer
from pipeline.core.relevance import RelevanceScorer
from pipeline.core.stance import StanceDetector
from pipeline.scoring import EvidenceScorer
from pipeline.verdict import VerdictEngine
from pipeline.verdict.llm_verifier import LLMVerifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for pipeline output."""
    verdict: str
    confidence: float
    evidence: List[Dict]
    reasoning: str
    details: Dict


class FactCheckingPipeline:
    """Main orchestrator for fact-checking pipeline."""
    
    def __init__(self, config: Dict):
        """Initialize all pipeline components."""
        logger.info("Initializing fact-checking pipeline...")
        self.config = config
        
        # Initialize components
        self.normalizer = ClaimNormalizer()
        self.checkability = CheckabilityClassifier(
            model_path=(config.get("claim_checkability_checkpoint") or "checkpoints/claim_checkability/latest")
        )
        self.context_classifier = ContextClassifier(
            model_path=(config.get("context_checkpoint") or "checkpoints/context/latest")
        )
        self.domain_router = DomainRouter()
        self.evidence_gatherer = EvidenceGatherer(config)
        self.relevance_scorer = RelevanceScorer(
            model_path=(config.get("relevance_checkpoint") or "checkpoints/relevance/latest"),
            bi_model_name=config.get("relevance_bi_encoder_model"),
            enable_two_stage=bool(config.get("enable_two_stage_relevance", True)),
            shortlist_k=int(config.get("relevance_shortlist_k", 20)),
            top_k=config.get("relevance_top_k"),
        )
        self.stance_detector = StanceDetector(
            model_path=(config.get("stance_checkpoint") or "checkpoints/stance/latest")
        )
        self.evidence_scorer = EvidenceScorer()
        self.verdict_engine = VerdictEngine()
        self.enable_llm_verifier = bool(config.get("enable_llm_verifier", True))
        self.llm_neutral_only = bool(config.get("llm_neutral_only", True))
        self.relevance_drop_threshold = float(config.get("relevance_drop_threshold", 0.30))
        self.relevance_min_keep = max(1, int(os.getenv("RELEVANCE_MIN_KEEP", "3")))
        self.multi_tavily_boost_enable = os.getenv(
            "MULTI_NEUTRAL_TAVILY_BOOST_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.image_enable_tavily_boost = os.getenv(
            "IMAGE_ENABLE_TAVILY_BOOST", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.image_enable_scraper_enrichment = os.getenv(
            "IMAGE_ENABLE_SCRAPER_ENRICHMENT", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_tavily_boost_threshold = float(
            os.getenv("MULTI_NEUTRAL_TAVILY_BOOST_THRESHOLD", "0.6")
        )
        self.multi_tavily_boost_max_queries = max(
            1, int(os.getenv("MULTI_NEUTRAL_TAVILY_BOOST_MAX_QUERIES", "2"))
        )
        self.multi_tavily_boost_max_results = max(
            1, int(os.getenv("MULTI_NEUTRAL_TAVILY_BOOST_MAX_RESULTS", "5"))
        )
        self.image_multi_en_fallback_enable = os.getenv(
            "IMAGE_MULTI_EN_FALLBACK_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.image_multi_en_fallback_disabled_langs = {
            tok.strip().lower()
            for tok in os.getenv("IMAGE_MULTI_EN_FALLBACK_DISABLED_LANGS", "te,kn").split(",")
            if tok.strip()
        }
        self.image_multi_en_fallback_max_queries = max(
            1, int(os.getenv("IMAGE_MULTI_EN_FALLBACK_MAX_QUERIES", "2"))
        )
        self.image_multi_en_fallback_max_results = max(
            1, int(os.getenv("IMAGE_MULTI_EN_FALLBACK_MAX_RESULTS", "6"))
        )
        self.image_multi_en_fallback_min_non_en_kept = max(
            0, int(os.getenv("IMAGE_MULTI_EN_FALLBACK_MIN_NON_EN_KEPT", "2"))
        )
        self.image_multi_en_fallback_max_keep = max(
            0, int(os.getenv("IMAGE_MULTI_EN_FALLBACK_MAX_KEEP", "2"))
        )
        try:
            self.max_evidence = max(1, int(config.get("max_evidence", 5)))
        except Exception:
            self.max_evidence = 5
        self.llm_verifier = LLMVerifier(
            provider=str(config.get("llm_provider", "openai")),
            model=str(config.get("llm_model", "gpt-4o-mini")),
        )
        self.live_progress = os.getenv("PIPELINE_LIVE_PROGRESS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.enable_polarity_adjust = os.getenv(
            "ENABLE_POLARITY_CONFLICT_ADJUST", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.polarity_min_relevance = float(os.getenv("POLARITY_MIN_RELEVANCE", "0.90"))
        self.polarity_neutral_floor = float(os.getenv("POLARITY_NEUTRAL_FLOOR", "0.20"))
        self.polarity_shift = float(os.getenv("POLARITY_SHIFT", "0.12"))
        self.llm_only_high_quality_neutral = os.getenv(
            "LLM_ONLY_HIGH_QUALITY_NEUTRAL", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.llm_hq_neutral_min_relevance = float(
            os.getenv("LLM_HQ_NEUTRAL_MIN_RELEVANCE", "0.90")
        )
        self.llm_hq_neutral_min_credibility = float(
            os.getenv("LLM_HQ_NEUTRAL_MIN_CREDIBILITY", "0.70")
        )
        
        logger.info("Pipeline initialized.")
    
    def analyze(
        self,
        claim: str,
        language: str = "en",
        image_path: Optional[str] = None,
        recency_mode: str = "general",
        recency_start: Optional[str] = None,
        recency_end: Optional[str] = None,
        progress_hook: Optional[Callable[[str], None]] = None,
    ) -> PipelineResult:
        """Run complete fact-checking pipeline."""
        logger.info(f"Starting analysis: claim='{claim[:50]}...', lang={language}")
        t_all = time.perf_counter()
        timings: Dict[str, float] = {}
        claim_head = (claim or "")[:70].replace("\n", " ")

        def _emit(msg: str) -> None:
            if progress_hook is not None:
                try:
                    progress_hook(msg)
                except Exception:
                    pass
            if self.live_progress:
                print(f"[pipeline] {msg}", flush=True)
        
        # Stage 1: Normalize claim
        _emit(f"stage1_normalize:start lang={language} claim='{claim_head}'")
        is_image_mode = bool(image_path)
        t = time.perf_counter()
        normalized_claim = self.normalizer.normalize(claim)
        search_queries = self.normalizer.rephrase_for_search(normalized_claim, language=language)
        timings["stage1_normalize"] = round(time.perf_counter() - t, 4)
        _emit(f"stage1_normalize:done sec={timings['stage1_normalize']}")
        logger.info(f"Normalized claim: {normalized_claim}")
        
        # Stage 2: Checkability
        _emit("stage2_checkability:start")
        t = time.perf_counter()
        is_checkable, checkability_reason = self.checkability.classify(normalized_claim)
        timings["stage2_checkability"] = round(time.perf_counter() - t, 4)
        _emit(
            f"stage2_checkability:done sec={timings['stage2_checkability']} checkable={is_checkable}"
        )
        if not is_checkable:
            timings["total_pipeline"] = round(time.perf_counter() - t_all, 4)
            _emit(f"pipeline:done early_uncheckable total_sec={timings['total_pipeline']}")
            return PipelineResult(
                verdict="neutral",
                confidence=0.0,
                evidence=[],
                reasoning=f"Claim uncheckable: {checkability_reason}",
                details={
                    "checkability": f"Uncheckable ({checkability_reason})",
                    "context": "N/A",
                    "sources_checked": 0,
                    "evidence_count": 0,
                    "timings": timings,
                }
            )
        
        # Stage 3: Context classification (hierarchical)
        _emit("stage3_context:start")
        t = time.perf_counter()
        l1_label, l2_label, l1_conf, l2_conf = self.context_classifier.classify(normalized_claim)
        l1_label, l2_label = self._apply_context_overrides(
            normalized_claim,
            l1_label,
            l2_label,
            l1_conf,
        )
        timings["stage3_context"] = round(time.perf_counter() - t, 4)
        _emit(
            f"stage3_context:done sec={timings['stage3_context']} context={l1_label}/{l2_label}"
        )
        logger.info(f"Context: {l1_label}/{l2_label} (l1_conf={l1_conf:.2f}, l2_conf={l2_conf:.2f})")
        
        # Stage 4: Domain routing (using level1 + level2)
        _emit("stage4_routing:start")
        t = time.perf_counter()
        evidence_sources = self._route_with_fallbacks(
            claim=normalized_claim,
            level1=l1_label,
            level2=l2_label,
            l1_conf=l1_conf,
            l2_conf=l2_conf,
        )
        timings["stage4_routing"] = round(time.perf_counter() - t, 4)
        _emit(
            f"stage4_routing:done sec={timings['stage4_routing']} sources={len(evidence_sources)}"
        )
        logger.info(f"Routing to sources: {evidence_sources}")
        
        # Stage 5: Evidence gathering
        _emit("stage5_evidence_gather:start")
        t = time.perf_counter()
        raw_evidence = self.evidence_gatherer.gather(
            claim=normalized_claim,
            queries=search_queries,
            sources=evidence_sources,
            language=language,
            max_evidence=self.max_evidence,
            scraper_enrichment_override=(
                self.image_enable_scraper_enrichment if is_image_mode else None
            ),
        )
        timings["stage5_evidence_gather"] = round(time.perf_counter() - t, 4)
        _emit(
            f"stage5_evidence_gather:done sec={timings['stage5_evidence_gather']} raw={len(raw_evidence)}"
        )
        logger.info(f"Gathered {len(raw_evidence)} raw evidence items")
        
        if not raw_evidence:
            timings["total_pipeline"] = round(time.perf_counter() - t_all, 4)
            _emit(f"pipeline:done no_evidence total_sec={timings['total_pipeline']}")
            return PipelineResult(
                verdict="neutral",
                confidence=0.3,
                evidence=[],
                reasoning="No evidence found from available sources.",
                details={
                    "checkability": "Checkable",
                    "context": f"{l1_label}/{l2_label}",
                    "sources_checked": len(evidence_sources),
                    "evidence_count": 0,
                    "timings": timings,
                }
            )
        
        # Stage 6: Relevance scoring
        _emit("stage6_relevance:start")
        t = time.perf_counter()
        scored_evidence = self.relevance_scorer.rank_evidence(
            claim=normalized_claim,
            evidence_list=raw_evidence,
            language=language,
        )
        scored_evidence = self._dedupe_ranked_evidence(scored_evidence)
        
        # Filter low-relevance evidence
        filtered_evidence = [
            ev for ev in scored_evidence if float(ev.get("relevance", 0.0)) >= self.relevance_drop_threshold
        ]
        filtered_evidence = self._apply_relevance_noise_guard(filtered_evidence, normalized_claim)
        if filtered_evidence:
            # Keep floor of evidence rows to avoid over-pruning in multilingual/OCR cases.
            if len(filtered_evidence) < self.relevance_min_keep:
                merged = list(filtered_evidence)
                seen = set(
                    self._canonicalize_url(str(ev.get("url") or "").strip()) or
                    " ".join(str(ev.get("text") or "").strip().lower().split())[:220]
                    for ev in merged
                )
                for ev in scored_evidence:
                    if len(merged) >= self.relevance_min_keep:
                        break
                    key = self._canonicalize_url(str(ev.get("url") or "").strip()) or \
                        " ".join(str(ev.get("text") or "").strip().lower().split())[:220]
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    merged.append(ev)
                scored_evidence = merged
            else:
                scored_evidence = filtered_evidence
        else:
            # Avoid hard-failing the pipeline if calibration is off.
            scored_evidence = scored_evidence[: min(self.relevance_min_keep, len(scored_evidence))]
        timings["stage6_relevance"] = round(time.perf_counter() - t, 4)
        _emit(
            f"stage6_relevance:done sec={timings['stage6_relevance']} kept={len(scored_evidence)}"
        )
        logger.info(f"After relevance filter: {len(scored_evidence)} items")
        
        # Recency is internal-only scoring signal from UI preference (no hard filter).
        recency_policy = {
            "mode": str(recency_mode or "general").strip().lower(),
            "start": str(recency_start or "").strip(),
            "end": str(recency_end or "").strip(),
        }

        # Stage 7: Stance detection
        _emit("stage7_stance:start")
        t = time.perf_counter()
        for ev in scored_evidence:
            stance_probs = self.stance_detector.detect(normalized_claim, ev["text"])
            if self.enable_polarity_adjust:
                stance_probs = self._apply_polarity_adjustment(normalized_claim, ev, stance_probs)
            ev["stance_probs"] = stance_probs  # {support: x, refute: y, neutral: z}
            ev["stance"] = max(stance_probs, key=stance_probs.get)
        timings["stage7_stance"] = round(time.perf_counter() - t, 4)
        _emit(f"stage7_stance:done sec={timings['stage7_stance']}")
        
        # Stage 8: Evidence scoring (relevance + credibility + temporal + stance)
        _emit("stage8_evidence_scoring:start")
        t = time.perf_counter()
        for ev in scored_evidence:
            ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                ev, recency_policy=recency_policy
            )
            ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
        scored_evidence = self._dedupe_scored_evidence(scored_evidence)
        timings["stage8_evidence_scoring"] = round(time.perf_counter() - t, 4)
        _emit(f"stage8_evidence_scoring:done sec={timings['stage8_evidence_scoring']}")
        
        # Stage 9: Aggregation and verdict
        _emit("stage9_aggregate_verdict:start")
        t = time.perf_counter()
        verdict_result = self.verdict_engine.compute(scored_evidence, normalized_claim)
        timings["stage9_aggregate_verdict"] = round(time.perf_counter() - t, 4)
        _emit(
            f"stage9_aggregate_verdict:done sec={timings['stage9_aggregate_verdict']} pre={verdict_result.get('verdict')}"
        )
        pre_llm_verdict = str(verdict_result.get("verdict", "neutral"))
        pre_llm_confidence = float(verdict_result.get("confidence", 0.0))
        llm_status = {
            "enabled": bool(self.enable_llm_verifier),
            "triggered": False,
            "used": False,
            "provider": self.llm_verifier.provider,
            "model": self.llm_verifier.model,
            "neutral_only": bool(self.llm_neutral_only),
            "conf_threshold": None,
            "pre_llm_verdict": pre_llm_verdict,
            "pre_llm_confidence": pre_llm_confidence,
            "reason": "not_triggered",
            "multi_tavily_boost_attempted": False,
            "multi_tavily_boost_added": 0,
            "image_multi_en_fallback_attempted": False,
            "image_multi_en_fallback_added": 0,
            "image_multi_en_fallback_kept": 0,
        }
        if self.enable_llm_verifier:
            _emit("stage10_llm_verify:start")
            t = time.perf_counter()
            # Old-style policy: run verifier only when pre-LLM verdict is neutral.
            should_verify = str(verdict_result.get("verdict", "neutral")).lower() == "neutral"
            if should_verify:
                llm_status["triggered"] = True
                llm_limit = max(0, int(os.getenv("LLM_VERIFIER_EVIDENCE_LIMIT", "0")))
                llm_min_neutral = max(0, int(os.getenv("LLM_VERIFIER_MIN_NEUTRAL_EVIDENCE", "3")))
                indexed_rows = list(enumerate(scored_evidence))
                neutral_idx = [i for i, ev in indexed_rows if str(ev.get("stance", "neutral")).lower() == "neutral"]
                if self.llm_only_high_quality_neutral:
                    neutral_idx = [
                        i
                        for i in neutral_idx
                        if float(scored_evidence[i].get("relevance", 0.0) or 0.0) >= self.llm_hq_neutral_min_relevance
                        and float(scored_evidence[i].get("credibility", 0.0) or 0.0) >= self.llm_hq_neutral_min_credibility
                    ]
                chosen_idx: List[int] = []
                if llm_limit > 0:
                    chosen_idx.extend(neutral_idx[: min(len(neutral_idx), llm_min_neutral, llm_limit)])
                    # Keep LLM input restricted to high-quality neutral only.
                else:
                    chosen_idx = list(neutral_idx)
                llm_input = [scored_evidence[i] for i in chosen_idx]
                llm_status["evidence_sent"] = len(llm_input)
                llm_status["neutral_sent"] = sum(
                    1 for ev in llm_input if str(ev.get("stance", "neutral")).lower() == "neutral"
                )
                llm_status["hq_neutral_only"] = bool(self.llm_only_high_quality_neutral)
                llm_status["hq_neutral_thresholds"] = {
                    "relevance": self.llm_hq_neutral_min_relevance,
                    "credibility": self.llm_hq_neutral_min_credibility,
                }
                # Multi-only branch: if neutral confidence is already >= threshold,
                # do one Tavily boost pass, recompute verdict once, then call LLM (no loopback).
                is_multi = str(language or "en").lower() != "en"
                pre_conf = float(verdict_result.get("confidence", 0.0))
                if (
                    is_multi
                    and self.multi_tavily_boost_enable
                    and ((not is_image_mode) or self.image_enable_tavily_boost)
                    and pre_conf >= self.multi_tavily_boost_threshold
                ):
                    llm_status["multi_tavily_boost_attempted"] = True
                    _emit("stage10a_multi_tavily_boost:start")
                    t_boost = time.perf_counter()
                    boosted_rows = self._multi_tavily_boost(
                        queries=search_queries,
                        language=language,
                    )
                    llm_status["multi_tavily_boost_added"] = len(boosted_rows)
                    if boosted_rows:
                        combined_raw = list(raw_evidence) + boosted_rows
                        rescored = self.relevance_scorer.rank_evidence(
                            claim=normalized_claim,
                            evidence_list=combined_raw,
                            language=language,
                        )
                        rescored = self._dedupe_ranked_evidence(rescored)
                        filtered = [
                            ev
                            for ev in rescored
                            if float(ev.get("relevance", 0.0)) >= self.relevance_drop_threshold
                        ]
                        rescored = filtered if filtered else rescored[: min(3, len(rescored))]
                        for ev in rescored:
                            stance_probs = self.stance_detector.detect(normalized_claim, ev["text"])
                            if self.enable_polarity_adjust:
                                stance_probs = self._apply_polarity_adjustment(normalized_claim, ev, stance_probs)
                            ev["stance_probs"] = stance_probs
                            ev["stance"] = max(stance_probs, key=stance_probs.get)
                            ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                                ev, recency_policy=recency_policy
                            )
                            ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                        rescored = self._dedupe_scored_evidence(rescored)
                        verdict_result = self.verdict_engine.compute(rescored, normalized_claim)
                        scored_evidence = rescored
                        llm_status["pre_llm_verdict"] = str(verdict_result.get("verdict", "neutral"))
                        llm_status["pre_llm_confidence"] = float(verdict_result.get("confidence", 0.0))
                    timings["stage10a_multi_tavily_boost"] = round(time.perf_counter() - t_boost, 4)
                    _emit(
                        f"stage10a_multi_tavily_boost:done sec={timings['stage10a_multi_tavily_boost']} added={llm_status['multi_tavily_boost_added']}"
                    )
                if not llm_input:
                    llm_status["used"] = False
                    llm_status["reason"] = "no high-quality neutral evidence for LLM re-stance"
                    llm_status["verdict"] = str(verdict_result.get("verdict", "neutral"))
                    llm_status["confidence"] = float(verdict_result.get("confidence", 0.0))
                    llm_updates = []
                else:
                    llm_result = self.llm_verifier.verify(normalized_claim, llm_input)
                    llm_status["reason"] = str(llm_result.get("reason", "no reason provided"))
                    llm_status["verdict"] = str(llm_result.get("verdict", "neutral"))
                    llm_status["confidence"] = float(llm_result.get("confidence", 0.5))
                    llm_updates = llm_result.get("evidence_updates") or []
                llm_status["evidence_updates"] = llm_updates
                llm_status["evidence_updates_applied"] = 0
                reason_l = llm_status["reason"].lower()
                llm_failed = ("verify call failed" in reason_l) or ("429" in reason_l)
                if llm_failed:
                    llm_status["used"] = False
                    llm_status["reason"] = f"{llm_status['reason']} (ignored; kept pre-LLM verdict)"
                else:
                    # Apply LLM evidence updates to the same evidence rows, then recompute verdict.
                    applied = 0
                    for upd in llm_updates:
                        try:
                            idx = int(upd.get("index", 0))
                        except Exception:
                            continue
                        if idx < 1 or idx > len(llm_input):
                            continue
                        src_pos = chosen_idx[idx - 1]
                        ev = scored_evidence[src_pos]
                        stance = str(upd.get("stance", "neutral")).lower()
                        if stance not in {"support", "refute", "neutral"}:
                            stance = "neutral"
                        rel_val = upd.get("relevance")
                        if rel_val is not None:
                            try:
                                ev["relevance"] = max(0.0, min(1.0, float(rel_val)))
                            except Exception:
                                pass
                        ev["stance"] = stance
                        # Keep probabilities coherent with LLM-updated stance.
                        if stance == "support":
                            ev["stance_probs"] = {"support": 0.98, "refute": 0.01, "neutral": 0.01}
                        elif stance == "refute":
                            ev["stance_probs"] = {"support": 0.01, "refute": 0.98, "neutral": 0.01}
                        else:
                            ev["stance_probs"] = {"support": 0.01, "refute": 0.01, "neutral": 0.98}
                        ev["llm_adjusted"] = True
                        ev["llm_adjust_reason"] = str(llm_result.get("reason", ""))
                        ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                            ev, recency_policy=recency_policy
                        )
                        ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                        applied += 1

                    llm_status["evidence_updates_applied"] = applied
                    if applied > 0:
                        verdict_result = self.verdict_engine.compute(scored_evidence, normalized_claim)
                        verdict_result["reasoning"] = (
                            f"{verdict_result['reasoning']} LLM verifier ({self.llm_verifier.provider}) "
                            f"updated {applied} evidence items; reason: {llm_result.get('reason', 'no reason provided')}."
                        )
                        llm_status["used"] = True
                    else:
                        llm_status["used"] = False
                        llm_status["reason"] = (
                            f"{llm_status['reason']} (ignored; no valid evidence updates)"
                        )
            # Image + multi fallback: if still neutral after LLM, run one English web-search pass,
            # then rescore/recompute once (no loopback).
            post_llm_verdict_now = str(verdict_result.get("verdict", "neutral")).lower()
            non_en_kept = self._count_non_english_evidence(scored_evidence)
            if (
                is_image_mode
                and str(language or "en").lower() != "en"
                and self.image_multi_en_fallback_enable
                and str(language or "").lower() not in self.image_multi_en_fallback_disabled_langs
                and post_llm_verdict_now == "neutral"
                and non_en_kept < self.image_multi_en_fallback_min_non_en_kept
            ):
                llm_status["image_multi_en_fallback_attempted"] = True
                _emit("stage10b_image_multi_en_fallback:start")
                t_fallback = time.perf_counter()
                en_rows = self._image_multi_english_fallback(
                    normalized_claim=normalized_claim,
                    language=language,
                )
                llm_status["image_multi_en_fallback_added"] = len(en_rows)
                if en_rows:
                    combined_raw = list(raw_evidence) + en_rows
                    rescored = self.relevance_scorer.rank_evidence(
                        claim=normalized_claim,
                        evidence_list=combined_raw,
                        language=language,
                    )
                    rescored = self._dedupe_ranked_evidence(rescored)
                    filtered = [
                        ev
                        for ev in rescored
                        if float(ev.get("relevance", 0.0)) >= self.relevance_drop_threshold
                    ]
                    rescored = filtered if filtered else rescored[: min(3, len(rescored))]
                    for ev in rescored:
                        stance_probs = self.stance_detector.detect(normalized_claim, ev["text"])
                        if self.enable_polarity_adjust:
                            stance_probs = self._apply_polarity_adjustment(normalized_claim, ev, stance_probs)
                        ev["stance_probs"] = stance_probs
                        ev["stance"] = max(stance_probs, key=stance_probs.get)
                        ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                            ev, recency_policy=recency_policy
                        )
                        ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                    rescored = self._dedupe_scored_evidence(rescored)
                    rescored = self._cap_image_en_fallback_rows(
                        rescored,
                        max_keep=self.image_multi_en_fallback_max_keep,
                    )
                    llm_status["image_multi_en_fallback_kept"] = sum(
                        1 for ev in rescored if bool(ev.get("_from_image_en_fallback", False))
                    )
                    scored_evidence = rescored
                    verdict_result = self.verdict_engine.compute(scored_evidence, normalized_claim)
                timings["stage10b_image_multi_en_fallback"] = round(time.perf_counter() - t_fallback, 4)
                _emit(
                    "stage10b_image_multi_en_fallback:done "
                    f"sec={timings['stage10b_image_multi_en_fallback']} "
                    f"added={llm_status['image_multi_en_fallback_added']}"
                )
            timings["stage10_llm_verify"] = round(time.perf_counter() - t, 4)
            _emit(
                f"stage10_llm_verify:done sec={timings['stage10_llm_verify']} triggered={llm_status.get('triggered')}"
            )
        else:
            timings["stage10_llm_verify"] = 0.0
        llm_status["post_llm_verdict"] = str(verdict_result.get("verdict", "neutral"))
        llm_status["post_llm_confidence"] = float(verdict_result.get("confidence", 0.0))
        timings["total_pipeline"] = round(time.perf_counter() - t_all, 4)
        _emit(
            f"pipeline:done total_sec={timings['total_pipeline']} verdict={verdict_result.get('verdict')}"
        )
        
        # Build result
        evidence_list = [
            {
                "text": ev["text"],
                "source": ev["source"],
                "url": ev.get("url"),
                "relevance": ev["relevance"],
                "credibility": ev.get("credibility", 0.5),
                "stance": ev["stance"],
                "score": ev["evidence_weight"],
                "llm_adjusted": bool(ev.get("llm_adjusted", False)),
                "llm_adjust_reason": ev.get("llm_adjust_reason"),
            }
            for ev in scored_evidence
        ]
        recency_pref_match_count = sum(
            1 for ev in scored_evidence if float(ev.get("recency_bonus", 1.0) or 1.0) > 1.01
        )
        
        return PipelineResult(
            verdict=verdict_result["verdict"],
            confidence=verdict_result["confidence"],
            evidence=evidence_list,
            reasoning=verdict_result["reasoning"],
            details={
                "checkability": "Checkable",
                "context": f"{l1_label}/{l2_label}",
                "sources_checked": len(evidence_sources),
                "evidence_count": len(scored_evidence),
                "llm_verifier": llm_status,
                "recency_preference": recency_policy,
                "recency_preference_matches": recency_pref_match_count,
                "timings": timings,
            }
        )

    def _image_multi_english_fallback(self, normalized_claim: str, language: str) -> List[Dict]:
        """
        Image-only multi-language fallback:
        translate claim to English, fetch English web evidence, normalize + dedupe.
        """
        try:
            translated = self.normalizer._translate_indic_to_english(  # pylint: disable=protected-access
                text=normalized_claim,
                language=language,
            )
        except Exception:
            translated = ""
        translated = str(translated or "").strip()
        # If translation failed and still mostly non-ASCII, skip EN fallback entirely.
        if not translated or not self.normalizer._is_english_like(translated):  # pylint: disable=protected-access
            return []
        if not translated:
            return []

        en_queries = self.normalizer.rephrase_for_search(translated, language="en")
        en_queries = (en_queries or [translated])[: self.image_multi_en_fallback_max_queries]
        if not en_queries:
            en_queries = [translated]

        try:
            rows = self.evidence_gatherer.web_search.search(
                claim=translated,
                queries=en_queries,
                subtype=None,
                language="en",
                max_results=self.image_multi_en_fallback_max_results,
            )
        except Exception as exc:
            logger.warning("Image multi English fallback failed: %s", exc)
            return []
        if not rows:
            return []

        normalized = self.evidence_gatherer._normalize_evidence_list(rows, default_type="web_search")
        if not normalized:
            return []

        out: List[Dict] = []
        seen = set()
        for row in normalized:
            key = (
                str(row.get("url") or "").strip(),
                str(row.get("text") or "")[:180].strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            row["_from_image_en_fallback"] = True
            row["lang_hint"] = "en"
            out.append(row)
        return out

    def _count_non_english_evidence(self, rows: List[Dict]) -> int:
        count = 0
        for row in rows or []:
            text = str(row.get("text") or "")
            if not text:
                continue
            try:
                is_en = self.normalizer._is_english_like(text)  # pylint: disable=protected-access
            except Exception:
                is_en = False
            if not is_en:
                count += 1
        return count

    def _cap_image_en_fallback_rows(self, rows: List[Dict], max_keep: int) -> List[Dict]:
        if max_keep < 0:
            return rows
        out: List[Dict] = []
        en_added = 0
        for row in rows or []:
            from_en_fallback = bool(row.get("_from_image_en_fallback", False))
            if from_en_fallback and en_added >= max_keep:
                continue
            if from_en_fallback:
                en_added += 1
            out.append(row)
        return out

    def _multi_tavily_boost(self, queries: List[str], language: str) -> List[Dict]:
        """
        Multi-only one-shot Tavily retrieval boost for neutral claims.
        Returns normalized evidence rows ready for relevance ranking.
        """
        try:
            tavily = self.evidence_gatherer.web_search.tavily
        except Exception:
            return []

        if not getattr(tavily, "enabled", False):
            return []

        rows: List[Dict] = []
        query_slice = (queries or [])[: self.multi_tavily_boost_max_queries]
        for q in query_slice:
            try:
                rows.extend(tavily.search(query=q, max_results=self.multi_tavily_boost_max_results))
            except Exception as exc:
                logger.warning("Multi Tavily boost failed for query='%s': %s", q[:80], exc)

        if not rows:
            return []
        # Reuse gatherer's normalization schema for consistency.
        normalized = self.evidence_gatherer._normalize_evidence_list(rows, default_type="web_search")
        if not normalized:
            return []
        # Deduplicate by URL/text key.
        out: List[Dict] = []
        seen = set()
        for r in normalized:
            key = (str(r.get("url") or "").strip(), str(r.get("text") or "")[:160].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _dedupe_ranked_evidence(self, rows: List[Dict]) -> List[Dict]:
        """Deduplicate ranked evidence by stable url/text key while preserving order."""
        out: List[Dict] = []
        seen = set()
        for row in rows or []:
            url_key = self._canonicalize_url(str(row.get("url") or "").strip())
            text_key = " ".join(str(row.get("text") or "").strip().lower().split())[:220]
            key = url_key or text_key
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _dedupe_scored_evidence(self, rows: List[Dict]) -> List[Dict]:
        """
        Deduplicate after stance/scoring so near-identical rows don't carry conflicting stances.
        Keep the row with higher evidence_weight (fallback: relevance).
        """
        chosen: Dict[str, Dict] = {}
        order: List[str] = []
        for row in rows or []:
            url_key = self._canonicalize_url(str(row.get("url") or "").strip())
            text_key = " ".join(str(row.get("text") or "").strip().lower().split())[:220]
            title_key = " ".join(str(row.get("title") or "").strip().lower().split())[:180]
            key = url_key or title_key or text_key
            if not key:
                continue
            current = chosen.get(key)
            if current is None:
                chosen[key] = row
                order.append(key)
                continue
            cur_score = float(current.get("evidence_weight", current.get("relevance", 0.0)) or 0.0)
            new_score = float(row.get("evidence_weight", row.get("relevance", 0.0)) or 0.0)
            if new_score > cur_score:
                chosen[key] = row
        return [chosen[k] for k in order if k in chosen]

    def _apply_polarity_adjustment(self, claim: str, evidence: Dict, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Textual polarity + conflict-aware correction:
        - if text has mixed support/refute cues -> keep neutral as-is
        - if neutral dominates but polarity is clear and evidence is high quality -> shift a bit
        """
        p = {
            "support": float(probs.get("support", 0.0) or 0.0),
            "refute": float(probs.get("refute", 0.0) or 0.0),
            "neutral": float(probs.get("neutral", 0.0) or 0.0),
        }
        txt = str(evidence.get("text") or "")
        title = str(evidence.get("title") or "")
        blob = f"{title} {txt}".lower()
        rel = float(evidence.get("relevance", 0.0) or 0.0)

        support_markers = {
            "confirmed", "confirms", "true", "accurate", "official", "announced",
            "stated", "says", "reported", "verified", "indeed",
        }
        refute_markers = {
            "false", "fake", "hoax", "misleading", "denied", "denies", "not true",
            "did not", "no evidence", "incorrect", "refuted", "contradicts",
        }
        pos_hits = sum(1 for w in support_markers if w in blob)
        neg_hits = sum(1 for w in refute_markers if w in blob)
        conflict = pos_hits > 0 and neg_hits > 0
        evidence["polarity_pos_hits"] = pos_hits
        evidence["polarity_neg_hits"] = neg_hits
        evidence["polarity_conflict"] = bool(conflict)

        # Conflict in same snippet => no directional push.
        if conflict:
            return p

        # Only nudge when neutral dominates and evidence quality is high enough.
        if p["neutral"] < max(p["support"], p["refute"]):
            return p
        if rel < self.polarity_min_relevance:
            return p

        # Claim token overlap guard (avoid off-topic marker matches).
        claim_tokens = {
            t for t in re.findall(r"[A-Za-z\u0900-\u0D7F]{3,}", str(claim or "").lower())
        }
        ev_tokens = {
            t for t in re.findall(r"[A-Za-z\u0900-\u0D7F]{3,}", blob)
        }
        if claim_tokens and len(claim_tokens.intersection(ev_tokens)) == 0:
            return p

        delta = max(0.0, min(0.25, self.polarity_shift))
        if pos_hits > neg_hits and pos_hits > 0:
            take = min(delta, max(0.0, p["neutral"] - self.polarity_neutral_floor))
            p["neutral"] -= take
            p["support"] += take
            evidence["polarity_adjusted"] = True
            evidence["polarity_direction"] = "support"
        elif neg_hits > pos_hits and neg_hits > 0:
            take = min(delta, max(0.0, p["neutral"] - self.polarity_neutral_floor))
            p["neutral"] -= take
            p["refute"] += take
            evidence["polarity_adjusted"] = True
            evidence["polarity_direction"] = "refute"

        # Normalize for safety.
        s = p["support"] + p["refute"] + p["neutral"]
        if s > 0:
            p = {k: max(0.0, v / s) for k, v in p.items()}
        return p

    def _canonicalize_url(self, url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            cleaned = parsed._replace(query="", fragment="")
            return urlunparse(cleaned).rstrip("/").lower()
        except Exception:
            return url.strip().lower()

    def _apply_relevance_noise_guard(self, rows: List[Dict], claim: str) -> List[Dict]:
        """
        Drop obviously noisy evidence rows that pass embedding relevance due to numeric overlap.
        Keeps ultra-high relevance rows untouched.
        """
        claim_tokens = {
            t for t in re.findall(r"[A-Za-z\u0900-\u0D7F]{3,}", str(claim or "").lower())
        }
        if not claim_tokens:
            return rows
        out: List[Dict] = []
        for row in rows:
            rel = float(row.get("relevance", 0.0) or 0.0)
            if rel >= 0.985:
                out.append(row)
                continue
            blob = f"{row.get('title','')} {row.get('text','')}".lower()
            ev_tokens = {t for t in re.findall(r"[A-Za-z\u0900-\u0D7F]{3,}", blob)}
            overlap = len(claim_tokens.intersection(ev_tokens))
            # Relaxed guard: drop only when overlap is zero and relevance is not very high.
            if overlap >= 1 or rel >= 0.92:
                out.append(row)
        return out

    def _apply_context_overrides(self, claim: str, level1: str, level2: str, l1_conf: float) -> tuple[str, str]:
        """Apply lightweight keyword overrides when context model misses obvious domains."""
        if os.getenv("ROUTER_ENABLE_CLAIM_OVERRIDES", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return level1, level2

        text = claim.lower()
        weak_context = (l1_conf < 0.65) or level1 in {"GENERAL_FACTUAL", "TECHNOLOGY"}

        policy_terms = {"gst", "tax", "parliament", "assembly", "government", "policy", "constitution"}
        law_terms = {"act", "section", "fir", "court", "legal", "law", "police"}
        entertainment_terms = {"imdb", "box office", "movie", "film", "cinema", "song", "album"}

        if weak_context and any(t in text for t in law_terms):
            return "LAW_CRIME", "regulation"
        if weak_context and any(t in text for t in policy_terms):
            return "POLITICS_GOVERNMENT", "public_policy"
        if weak_context and any(t in text for t in entertainment_terms):
            return "ENTERTAINMENT", "film"
        return level1, level2

    def _route_with_fallbacks(
        self,
        claim: str,
        level1: str,
        level2: str,
        l1_conf: float,
        l2_conf: float,
    ) -> List[Dict]:
        """Primary route + conservative fallback routes for low-confidence context."""
        routes: List[List[Dict]] = [self.domain_router.route(level1, level2)]
        low_conf = (l1_conf < 0.65) or (l2_conf < 0.45)
        include_general = os.getenv("ROUTER_LOW_CONF_INCLUDE_GENERAL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if low_conf and include_general:
            routes.append(self.domain_router.route("GENERAL_FACTUAL", "encyclopedic"))

        # Legal/policy claims benefit from explicit legal route, even if primary differs.
        text = claim.lower()
        if any(t in text for t in ("act", "section", "fir", "law", "court", "police", "gst", "tax")):
            routes.append(self.domain_router.route("LAW_CRIME", "regulation"))

        merged: List[Dict] = []
        seen = set()
        for group in routes:
            for src in group:
                key = (str(src.get("type")), str(src.get("subtype")))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(src)
        return merged
