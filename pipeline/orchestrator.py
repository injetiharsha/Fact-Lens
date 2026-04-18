"""Main pipeline orchestrator - coordinates all pipeline stages."""

import logging
import os
import time
from typing import Callable
from typing import Dict, List, Optional
from dataclasses import dataclass

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
        self.multi_tavily_boost_enable = os.getenv(
            "MULTI_NEUTRAL_TAVILY_BOOST_ENABLE", "1"
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
        
        logger.info("Pipeline initialized.")
    
    def analyze(
        self,
        claim: str,
        language: str = "en",
        image_path: Optional[str] = None,
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
        
        # Filter low-relevance evidence
        filtered_evidence = [
            ev for ev in scored_evidence if float(ev.get("relevance", 0.0)) >= self.relevance_drop_threshold
        ]
        if filtered_evidence:
            scored_evidence = filtered_evidence
        else:
            # Avoid hard-failing the pipeline if calibration is off.
            scored_evidence = scored_evidence[: min(3, len(scored_evidence))]
        timings["stage6_relevance"] = round(time.perf_counter() - t, 4)
        _emit(
            f"stage6_relevance:done sec={timings['stage6_relevance']} kept={len(scored_evidence)}"
        )
        logger.info(f"After relevance filter: {len(scored_evidence)} items")
        
        # Stage 7: Stance detection
        _emit("stage7_stance:start")
        t = time.perf_counter()
        for ev in scored_evidence:
            stance_probs = self.stance_detector.detect(normalized_claim, ev["text"])
            ev["stance_probs"] = stance_probs  # {support: x, refute: y, neutral: z}
            ev["stance"] = max(stance_probs, key=stance_probs.get)
        timings["stage7_stance"] = round(time.perf_counter() - t, 4)
        _emit(f"stage7_stance:done sec={timings['stage7_stance']}")
        
        # Stage 8: Evidence scoring (relevance + credibility + temporal + stance)
        _emit("stage8_evidence_scoring:start")
        t = time.perf_counter()
        for ev in scored_evidence:
            ev["evidence_weight"] = self.evidence_scorer.calculate_weight(ev)
            ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
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
        }
        if self.enable_llm_verifier:
            _emit("stage10_llm_verify:start")
            t = time.perf_counter()
            # Old-style policy: run verifier only when pre-LLM verdict is neutral.
            should_verify = str(verdict_result.get("verdict", "neutral")).lower() == "neutral"
            if should_verify:
                llm_status["triggered"] = True
                # Multi-only branch: if neutral confidence is already >= threshold,
                # do one Tavily boost pass, recompute verdict once, then call LLM (no loopback).
                is_multi = str(language or "en").lower() != "en"
                pre_conf = float(verdict_result.get("confidence", 0.0))
                if (
                    is_multi
                    and self.multi_tavily_boost_enable
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
                        filtered = [
                            ev
                            for ev in rescored
                            if float(ev.get("relevance", 0.0)) >= self.relevance_drop_threshold
                        ]
                        rescored = filtered if filtered else rescored[: min(3, len(rescored))]
                        for ev in rescored:
                            stance_probs = self.stance_detector.detect(normalized_claim, ev["text"])
                            ev["stance_probs"] = stance_probs
                            ev["stance"] = max(stance_probs, key=stance_probs.get)
                            ev["evidence_weight"] = self.evidence_scorer.calculate_weight(ev)
                            ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                        verdict_result = self.verdict_engine.compute(rescored, normalized_claim)
                        scored_evidence = rescored
                        llm_status["pre_llm_verdict"] = str(verdict_result.get("verdict", "neutral"))
                        llm_status["pre_llm_confidence"] = float(verdict_result.get("confidence", 0.0))
                    timings["stage10a_multi_tavily_boost"] = round(time.perf_counter() - t_boost, 4)
                    _emit(
                        f"stage10a_multi_tavily_boost:done sec={timings['stage10a_multi_tavily_boost']} added={llm_status['multi_tavily_boost_added']}"
                    )
                llm_result = self.llm_verifier.verify(normalized_claim, scored_evidence[:5])
                llm_status["reason"] = str(llm_result.get("reason", "no reason provided"))
                llm_status["verdict"] = str(llm_result.get("verdict", "neutral"))
                llm_status["confidence"] = float(llm_result.get("confidence", 0.5))
                reason_l = llm_status["reason"].lower()
                llm_failed = ("verify call failed" in reason_l) or ("429" in reason_l)
                if llm_failed:
                    llm_status["used"] = False
                    llm_status["reason"] = f"{llm_status['reason']} (ignored; kept pre-LLM verdict)"
                elif llm_result.get("verdict"):
                    verdict_result["verdict"] = llm_result.get("verdict", verdict_result["verdict"])
                    verdict_result["confidence"] = float(llm_result.get("confidence", verdict_result["confidence"]))
                    verdict_result["reasoning"] = (
                        f"{verdict_result['reasoning']} LLM verifier ({self.llm_verifier.provider}) "
                        f"reason: {llm_result.get('reason', 'no reason provided')}."
                    )
                    llm_status["used"] = True
                else:
                    llm_status["used"] = False
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
                "score": ev["evidence_weight"]
            }
            for ev in scored_evidence
        ]
        
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
                "timings": timings,
            }
        )

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
