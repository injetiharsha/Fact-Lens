"""Main pipeline orchestrator - coordinates all pipeline stages."""

import logging
import os
import re
import sys
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
from pipeline.core.sarvam_reranker import SarvamReranker
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
        self.pipeline_language = str(config.get("pipeline_language", "en")).strip().lower()
        
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
        self.sarvam_reranker = SarvamReranker()
        self.enable_sarvam_stance_override = os.getenv(
            "SARVAM_RERANK_USE_STANCE_OVERRIDE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.sarvam_force_advisory = os.getenv(
            "SARVAM_RERANK_FORCE_ADVISORY", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.sarvam_stance_override_min_conf = float(
            os.getenv("SARVAM_RERANK_OVERRIDE_MIN_CONF", "0.8")
        )
        # Optional multi-only mixed-language routing:
        # Route English evidence through EN relevance/stance models while keeping
        # non-English evidence on multi models, then aggregate together.
        self.enable_multi_mixed_evidence_routing = os.getenv(
            "MULTI_MIXED_LANGUAGE_EVIDENCE_ROUTING", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._en_relevance_scorer = None
        self._en_stance_detector = None
        self._en_branch_init_attempted = False
        self.evidence_scorer = EvidenceScorer()
        self.verdict_engine = VerdictEngine()
        self.enable_llm_verifier = bool(config.get("enable_llm_verifier", True))
        self.enable_checkability_stage = bool(config.get("enable_checkability_stage", True))
        self.llm_neutral_only = bool(config.get("llm_neutral_only", True))
        self.relevance_drop_threshold = float(config.get("relevance_drop_threshold", 0.30))
        self.relevance_min_keep = max(1, int(os.getenv("RELEVANCE_MIN_KEEP", "3")))
        self.relevance_min_keep_multi = max(
            self.relevance_min_keep,
            int(os.getenv("RELEVANCE_MIN_KEEP_MULTI", "5")),
        )
        self.domain_diversity_max_per_host = max(
            1, int(os.getenv("EVIDENCE_MAX_PER_DOMAIN", "2"))
        )
        self.neutral_recovery_enable = os.getenv(
            "NEUTRAL_RECOVERY_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.neutral_recovery_max_queries = max(
            1, int(os.getenv("NEUTRAL_RECOVERY_MAX_QUERIES", "2"))
        )
        self.neutral_recovery_max_results = max(
            1, int(os.getenv("NEUTRAL_RECOVERY_MAX_RESULTS", "6"))
        )
        self.neutral_recovery_keep_threshold = max(
            1, int(os.getenv("NEUTRAL_RECOVERY_KEEP_THRESHOLD", "3"))
        )
        self.neutral_recovery_min_non_neutral = max(
            0, int(os.getenv("NEUTRAL_RECOVERY_MIN_NON_NEUTRAL", "1"))
        )
        self.multi_tavily_boost_enable = os.getenv(
            "MULTI_NEUTRAL_TAVILY_BOOST_ENABLE", "0"
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
            "LLM_ONLY_HIGH_QUALITY_NEUTRAL", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.llm_hq_neutral_min_relevance = float(
            os.getenv("LLM_HQ_NEUTRAL_MIN_RELEVANCE", "0.90")
        )
        self.llm_hq_neutral_min_credibility = float(
            os.getenv("LLM_HQ_NEUTRAL_MIN_CREDIBILITY", "0.70")
        )
        self.llm_use_verdict_fallback = os.getenv(
            "LLM_VERIFIER_USE_VERDICT_FALLBACK", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.llm_verdict_fallback_min_conf = float(
            os.getenv("LLM_VERIFIER_VERDICT_FALLBACK_MIN_CONF", "0.55")
        )
        # Neutral quality guard: neutral needs enough strong neutral evidence; else force one recovery pass.
        self.neutral_quality_guard_enable = os.getenv(
            "NEUTRAL_QUALITY_GUARD_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.neutral_quality_min_count = max(1, int(os.getenv("NEUTRAL_QUALITY_MIN_COUNT", "2")))
        self.neutral_quality_min_relevance = float(os.getenv("NEUTRAL_QUALITY_MIN_RELEVANCE", "0.86"))
        self.neutral_quality_min_credibility = float(os.getenv("NEUTRAL_QUALITY_MIN_CREDIBILITY", "0.65"))
        self.neutral_quality_recovery_max_once = max(
            0, int(os.getenv("NEUTRAL_QUALITY_RECOVERY_MAX_ONCE", "1"))
        )
        # Verdict calibration (language-aware): avoid brittle non-neutral when margin/conf is too weak.
        self.stance_decisive_min_conf_en = float(os.getenv("STANCE_DECISIVE_MIN_CONF_EN", "0.55"))
        self.stance_decisive_min_conf_multi = float(os.getenv("STANCE_DECISIVE_MIN_CONF_MULTI", "0.60"))
        self.stance_decisive_min_gap_en = float(os.getenv("STANCE_DECISIVE_MIN_GAP_EN", "0.08"))
        self.stance_decisive_min_gap_multi = float(os.getenv("STANCE_DECISIVE_MIN_GAP_MULTI", "0.12"))
        self.enable_relevance_noise_guard = os.getenv(
            "ENABLE_RELEVANCE_NOISE_GUARD", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Multi-only Phase controls (from MULTI_80_FAIL_PROOF_PLAN.md)
        self.multi_phase1_guard_enable = os.getenv(
            "MULTI_PHASE1_GUARD_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_phase3_tiers_enable = os.getenv(
            "MULTI_PHASE3_TIERS_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_phase4_lane_weight_enable = os.getenv(
            "MULTI_PHASE4_LANE_WEIGHT_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_phase5_gray_zone_enable = os.getenv(
            "MULTI_PHASE5_GRAY_ZONE_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_phase7_decisive_verdict_enable = os.getenv(
            "MULTI_PHASE7_DECISIVE_VERDICT_ENABLE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.multi_phase3_strong_rel = float(os.getenv("MULTI_PHASE3_STRONG_REL", "0.45"))
        self.multi_phase3_strong_quality = float(os.getenv("MULTI_PHASE3_STRONG_QUALITY", "0.40"))
        self.multi_phase3_soft_rel = float(os.getenv("MULTI_PHASE3_SOFT_REL", "0.30"))
        self.multi_phase3_soft_quality = float(os.getenv("MULTI_PHASE3_SOFT_QUALITY", "0.25"))
        self.multi_phase4_lane_weight_structured = float(
            os.getenv("MULTI_PHASE4_LANE_WEIGHT_STRUCTURED", "1.10")
        )
        self.multi_phase4_lane_weight_native = float(
            os.getenv("MULTI_PHASE4_LANE_WEIGHT_NATIVE", "1.00")
        )
        self.multi_phase4_lane_weight_translated = float(
            os.getenv("MULTI_PHASE4_LANE_WEIGHT_TRANSLATED", "0.92")
        )
        self.multi_phase5_gray_min_conf = float(os.getenv("MULTI_PHASE5_GRAY_MIN_CONF", "0.40"))
        self.multi_phase5_gray_max_conf = float(os.getenv("MULTI_PHASE5_GRAY_MAX_CONF", "0.60"))
        self.multi_phase5_gray_evidence_limit = max(
            1, int(os.getenv("MULTI_PHASE5_GRAY_EVIDENCE_LIMIT", "6"))
        )
        self.multi_phase7_min_total = float(os.getenv("MULTI_PHASE7_MIN_TOTAL", "0.35"))
        self.multi_phase7_min_gap = float(os.getenv("MULTI_PHASE7_MIN_GAP", "0.15"))
        self.multi_phase7_soft_factor = float(os.getenv("MULTI_PHASE7_SOFT_FACTOR", "0.60"))
        self.multi_phase7_strong_only_for_decision = os.getenv(
            "MULTI_PHASE7_STRONG_ONLY_DECISION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.pipeline_lock_en = os.getenv("PIPELINE_LOCK_EN", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if self.pipeline_lock_en and self.pipeline_language.startswith("en"):
            self._apply_en_lock_profile()
        
        logger.info("Pipeline initialized.")

    def _apply_en_lock_profile(self) -> None:
        """Force a stable EN runtime profile to avoid accidental config drift."""
        try:
            gatherer = self.evidence_gatherer
            gatherer.source_mode = "staged_fallback"
            stage_order_raw = os.getenv("EN_LOCK_STAGE_ORDER", "structured_api,scraping,web_search")
            gatherer.stage_order = gatherer._parse_stage_order(stage_order_raw)  # pylint: disable=protected-access
            gatherer.stage_min_results = max(1, int(os.getenv("EN_LOCK_STAGE_MIN_RESULTS", "6")))
            gatherer.stage_parallel_within = os.getenv("EN_LOCK_STAGE_PARALLEL_WITHIN", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            gatherer.enable_scraper_enrichment = os.getenv(
                "EN_LOCK_ENABLE_SCRAPER_ENRICHMENT", "1"
            ).strip().lower() in {"1", "true", "yes", "on"}
            gatherer.scraper_enrich_max_results = max(
                1, int(os.getenv("EN_LOCK_SCRAPER_ENRICH_MAX_RESULTS", str(gatherer.scraper_enrich_max_results)))
            )
        except Exception as exc:
            logger.warning("EN lock profile: gatherer override failed: %s", exc)

        try:
            web = self.evidence_gatherer.web_search
            web.max_queries_en = max(1, int(os.getenv("EN_LOCK_WEB_MAX_QUERIES_EN", str(web.max_queries_en))))
            web.min_providers_before_stop = max(
                1, int(os.getenv("EN_LOCK_WEB_MIN_PROVIDERS_BEFORE_STOP", str(web.min_providers_before_stop)))
            )
            web.provider_fail_fast = os.getenv("EN_LOCK_WEB_PROVIDER_FAIL_FAST", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if os.getenv("EN_LOCK_WEB_PROVIDER_ORDER", "").strip():
                web.provider_order = web._resolve_provider_order()  # pylint: disable=protected-access
            if os.getenv("EN_LOCK_WEB_ESCALATION_ORDER", "").strip():
                web.escalation_provider_order = web._resolve_escalation_provider_order()  # pylint: disable=protected-access
        except Exception as exc:
            logger.warning("EN lock profile: web override failed: %s", exc)

        logger.info(
            "EN lock profile applied: source_mode=%s stage_order=%s stage_min_results=%s max_queries_en=%s",
            getattr(self.evidence_gatherer, "source_mode", "unknown"),
            getattr(self.evidence_gatherer, "stage_order", []),
            getattr(self.evidence_gatherer, "stage_min_results", "unknown"),
            getattr(self.evidence_gatherer.web_search, "max_queries_en", "unknown"),
        )
    
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
        sarvam_rerank_status: Dict[str, object] = {
            "enabled": bool(self.sarvam_reranker.enabled),
            "applied": False,
            "reason": "not_run",
            "updated_items": 0,
        }
        claim_head = (claim or "")[:70].replace("\n", " ")

        def _emit(msg: str) -> None:
            if progress_hook is not None:
                try:
                    progress_hook(msg)
                except Exception:
                    pass
            if self.live_progress:
                line = f"[pipeline] {msg}"
                try:
                    print(line, flush=True)
                except UnicodeEncodeError:
                    buf = getattr(sys.stdout, "buffer", None)
                    if buf is not None:
                        buf.write((line + "\n").encode("utf-8", errors="replace"))
                        buf.flush()
                    else:
                        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        
        # Stage 1: Normalize claim
        _emit(f"stage1_normalize:start lang={language} claim='{claim_head}'")
        is_image_mode = bool(image_path)
        t = time.perf_counter()
        normalized_claim = self.normalizer.normalize(claim)
        if is_image_mode:
            normalized_claim = self._normalize_image_claim_for_retrieval(normalized_claim)
        search_queries = self.normalizer.rephrase_for_search(normalized_claim, language=language)
        if is_image_mode:
            search_queries = self._clean_image_queries(search_queries, normalized_claim)
        translated_query_hint = self._extract_translated_query_hint(search_queries, language)
        timings["stage1_normalize"] = round(time.perf_counter() - t, 4)
        _emit(f"stage1_normalize:done sec={timings['stage1_normalize']}")
        logger.info(f"Normalized claim: {normalized_claim}")
        
        # Stage 2: Checkability
        _emit("stage2_checkability:start")
        t = time.perf_counter()
        if self.enable_checkability_stage:
            is_checkable, checkability_reason = self.checkability.classify(normalized_claim)
            timings["stage2_checkability"] = round(time.perf_counter() - t, 4)
            msg = (
                f"stage2_checkability:done sec={timings['stage2_checkability']} "
                f"checkable={is_checkable}"
            )
            if not is_checkable and checkability_reason:
                msg += f" reason='{checkability_reason}'"
            _emit(msg)
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
                        "claim_type_proxy": "checkability_checkpoint",
                        "context": "N/A",
                        "sources_checked": 0,
                        "evidence_count": 0,
                        "timings": timings,
                    }
                )
        else:
            timings["stage2_checkability"] = round(time.perf_counter() - t, 4)
            checkability_reason = "disabled"
            _emit(
                f"stage2_checkability:done sec={timings['stage2_checkability']} checkable=True (disabled)"
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
            is_image_mode=is_image_mode,
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
            is_image_mode=is_image_mode,
            scraper_enrichment_override=(
                self.image_enable_scraper_enrichment if is_image_mode else None
            ),
            # Stage-1 already generates translated query for multi claims.
            # Avoid re-translating in gatherer to keep one translation call per claim.
            queries_already_augmented=True,
        )
        retrieval_telemetry = self.evidence_gatherer.get_last_telemetry()
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
                    "claim_type_proxy": "checkability_checkpoint",
                    "context": f"{l1_label}/{l2_label}",
                    "sources_checked": len(evidence_sources),
                    "evidence_count": 0,
                    "retrieval_telemetry": retrieval_telemetry,
                    "timings": timings,
                }
            )
        
        # Stage 6: Relevance scoring
        _emit("stage6_relevance:start")
        t = time.perf_counter()
        scored_evidence = self._rank_evidence_language_aware(
            claim=normalized_claim,
            evidence_list=raw_evidence,
            language=language,
        )
        scored_evidence = self._dedupe_ranked_evidence(scored_evidence)
        
        # Filter low-relevance evidence
        filtered_evidence = [
            ev for ev in scored_evidence if float(ev.get("relevance", 0.0)) >= self.relevance_drop_threshold
        ]
        if self.enable_relevance_noise_guard:
            filtered_evidence = self._apply_relevance_noise_guard(filtered_evidence, normalized_claim)
        min_keep_target = self._resolve_relevance_min_keep(language=language, is_image_mode=is_image_mode)
        if filtered_evidence:
            # Keep floor of evidence rows to avoid over-pruning in multilingual/OCR cases.
            if len(filtered_evidence) < min_keep_target:
                merged = list(filtered_evidence)
                seen = set(
                    self._canonicalize_url(str(ev.get("url") or "").strip()) or
                    " ".join(str(ev.get("text") or "").strip().lower().split())[:220]
                    for ev in merged
                )
                for ev in scored_evidence:
                    if len(merged) >= min_keep_target:
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
            scored_evidence = scored_evidence[: min(min_keep_target, len(scored_evidence))]
        if not is_image_mode:
            scored_evidence = self._apply_domain_diversity(
                scored_evidence,
                max_per_host=self.domain_diversity_max_per_host,
            )

        # Stage 6b: optional Sarvam final rerank for multi (old-style compatible hook).
        _emit("stage6b_sarvam_rerank:start")
        t_sarvam = time.perf_counter()
        scored_evidence, sarvam_rerank_status = self.sarvam_reranker.rerank(
            claim=normalized_claim,
            evidence_list=scored_evidence,
            language=language,
            provisional_verdict="neutral",
            is_image_mode=is_image_mode,
        )
        timings["stage6b_sarvam_rerank"] = round(time.perf_counter() - t_sarvam, 4)
        _emit(
            "stage6b_sarvam_rerank:done "
            f"sec={timings['stage6b_sarvam_rerank']} "
            f"applied={sarvam_rerank_status.get('applied')} "
            f"updated={sarvam_rerank_status.get('updated_items', 0)}"
        )
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
        self._apply_stance_language_aware(
            claim=normalized_claim,
            evidence_list=scored_evidence,
            language=language,
        )
        timings["stage7_stance"] = round(time.perf_counter() - t, 4)
        _emit(f"stage7_stance:done sec={timings['stage7_stance']}")
        
        # Stage 8: Evidence scoring (relevance + credibility + temporal + stance)
        _emit("stage8_evidence_scoring:start")
        t = time.perf_counter()
        is_multi_lang = str(language or "en").lower() != "en"
        for ev in scored_evidence:
            ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                ev, recency_policy=recency_policy
            )
            if is_multi_lang and self.multi_phase4_lane_weight_enable:
                self._apply_multi_lane_weight(ev)
            if is_multi_lang and self.multi_phase3_tiers_enable:
                self._annotate_multi_evidence_tier(ev)
            ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
        if is_multi_lang and self.multi_phase3_tiers_enable:
            scored_evidence = self._filter_reject_tier_evidence(scored_evidence)
        scored_evidence = self._dedupe_scored_evidence(scored_evidence)
        timings["stage8_evidence_scoring"] = round(time.perf_counter() - t, 4)
        _emit(f"stage8_evidence_scoring:done sec={timings['stage8_evidence_scoring']}")
        
        # Stage 9: Aggregation and verdict
        _emit("stage9_aggregate_verdict:start")
        t = time.perf_counter()
        verdict_result = self._compute_verdict(scored_evidence, normalized_claim, language)
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
            current_verdict = str(verdict_result.get("verdict", "neutral")).lower()
            current_conf = float(verdict_result.get("confidence", 0.0) or 0.0)
            should_verify_neutral = current_verdict == "neutral"
            should_verify_gray = (
                is_multi_lang
                and self.multi_phase5_gray_zone_enable
                and current_verdict in {"support", "refute"}
                and self.multi_phase5_gray_min_conf <= current_conf <= self.multi_phase5_gray_max_conf
            )
            should_verify = should_verify_neutral or should_verify_gray
            if should_verify:
                llm_status["triggered"] = True
                llm_status["verify_mode"] = "neutral" if should_verify_neutral else "gray_zone"
                llm_status["gray_zone_triggered"] = bool(should_verify_gray)
                llm_limit = max(0, int(os.getenv("LLM_VERIFIER_EVIDENCE_LIMIT", "0")))
                llm_min_neutral = max(0, int(os.getenv("LLM_VERIFIER_MIN_NEUTRAL_EVIDENCE", "0")))
                indexed_rows = list(enumerate(scored_evidence))
                neutral_idx = [i for i, ev in indexed_rows if str(ev.get("stance", "neutral")).lower() == "neutral"]
                if should_verify_neutral and self.llm_only_high_quality_neutral:
                    neutral_idx = [
                        i
                        for i in neutral_idx
                        if float(scored_evidence[i].get("relevance", 0.0) or 0.0) >= self.llm_hq_neutral_min_relevance
                        and float(scored_evidence[i].get("credibility", 0.0) or 0.0) >= self.llm_hq_neutral_min_credibility
                    ]
                chosen_idx: List[int] = []
                if should_verify_gray:
                    ranked_idx = sorted(
                        range(len(scored_evidence)),
                        key=lambda i: float(scored_evidence[i].get("relevance", 0.0) or 0.0),
                        reverse=True,
                    )
                    cap = self.multi_phase5_gray_evidence_limit
                    if llm_limit > 0:
                        cap = min(cap, llm_limit)
                    chosen_idx = ranked_idx[:cap]
                else:
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
                is_multi = is_multi_lang
                pre_conf = float(verdict_result.get("confidence", 0.0))
                if (
                    is_multi
                    and should_verify_neutral
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
                        rescored = self._rank_evidence_language_aware(
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
                        self._apply_stance_language_aware(
                            claim=normalized_claim,
                            evidence_list=rescored,
                            language=language,
                        )
                        for ev in rescored:
                            ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                                ev, recency_policy=recency_policy
                            )
                            if is_multi_lang and self.multi_phase4_lane_weight_enable:
                                self._apply_multi_lane_weight(ev)
                            if is_multi_lang and self.multi_phase3_tiers_enable:
                                self._annotate_multi_evidence_tier(ev)
                            ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                        if is_multi_lang and self.multi_phase3_tiers_enable:
                            rescored = self._filter_reject_tier_evidence(rescored)
                        rescored = self._dedupe_scored_evidence(rescored)
                        verdict_result = self._compute_verdict(rescored, normalized_claim, language)
                        scored_evidence = rescored
                        llm_status["pre_llm_verdict"] = str(verdict_result.get("verdict", "neutral"))
                        llm_status["pre_llm_confidence"] = float(verdict_result.get("confidence", 0.0))
                    timings["stage10a_multi_tavily_boost"] = round(time.perf_counter() - t_boost, 4)
                    _emit(
                        f"stage10a_multi_tavily_boost:done sec={timings['stage10a_multi_tavily_boost']} added={llm_status['multi_tavily_boost_added']}"
                    )
                if not llm_input:
                    llm_status["used"] = False
                    llm_status["reason"] = (
                        "no neutral evidence for LLM re-stance"
                        if should_verify_neutral
                        else "no evidence selected for gray-zone LLM verification"
                    )
                    llm_status["verdict"] = str(verdict_result.get("verdict", "neutral"))
                    llm_status["confidence"] = float(verdict_result.get("confidence", 0.0))
                    llm_result = {
                        "verdict": llm_status["verdict"],
                        "confidence": llm_status["confidence"],
                        "reason": llm_status["reason"],
                        "evidence_updates": [],
                    }
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
                        if is_multi_lang and self.multi_phase4_lane_weight_enable:
                            self._apply_multi_lane_weight(ev)
                        if is_multi_lang and self.multi_phase3_tiers_enable:
                            self._annotate_multi_evidence_tier(ev)
                        ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                        applied += 1

                    llm_status["evidence_updates_applied"] = applied
                    if applied > 0:
                        if is_multi_lang and self.multi_phase3_tiers_enable:
                            scored_evidence = self._filter_reject_tier_evidence(scored_evidence)
                        verdict_result = self._compute_verdict(scored_evidence, normalized_claim, language)
                        verdict_result["reasoning"] = (
                            f"{verdict_result['reasoning']} LLM verifier ({self.llm_verifier.provider}) "
                            f"updated {applied} evidence items; reason: {llm_result.get('reason', 'no reason provided')}."
                        )
                        llm_status["used"] = True
                    else:
                        llm_verdict = str(llm_result.get("verdict", "neutral")).lower()
                        llm_conf = float(llm_result.get("confidence", 0.0) or 0.0)
                        if (
                            self.llm_use_verdict_fallback
                            and llm_verdict in {"support", "refute"}
                            and llm_conf >= self.llm_verdict_fallback_min_conf
                        ):
                            verdict_result["verdict"] = llm_verdict
                            verdict_result["confidence"] = llm_conf
                            verdict_result["reasoning"] = (
                                f"{verdict_result.get('reasoning','')} "
                                f"LLM verifier ({self.llm_verifier.provider}) applied verdict fallback: "
                                f"{llm_verdict} ({llm_conf:.2f})."
                            ).strip()
                            llm_status["used"] = True
                            llm_status["reason"] = (
                                f"{llm_status['reason']} (verdict fallback applied)"
                            )
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
                    translated_hint=translated_query_hint,
                )
                llm_status["image_multi_en_fallback_added"] = len(en_rows)
                if en_rows:
                    combined_raw = list(raw_evidence) + en_rows
                    rescored = self._rank_evidence_language_aware(
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
                    self._apply_stance_language_aware(
                        claim=normalized_claim,
                        evidence_list=rescored,
                        language=language,
                    )
                    for ev in rescored:
                        ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                            ev, recency_policy=recency_policy
                        )
                        if is_multi_lang and self.multi_phase4_lane_weight_enable:
                            self._apply_multi_lane_weight(ev)
                        if is_multi_lang and self.multi_phase3_tiers_enable:
                            self._annotate_multi_evidence_tier(ev)
                        ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                    if is_multi_lang and self.multi_phase3_tiers_enable:
                        rescored = self._filter_reject_tier_evidence(rescored)
                    rescored = self._dedupe_scored_evidence(rescored)
                    rescored = self._cap_image_en_fallback_rows(
                        rescored,
                        max_keep=self.image_multi_en_fallback_max_keep,
                    )
                    llm_status["image_multi_en_fallback_kept"] = sum(
                        1 for ev in rescored if bool(ev.get("_from_image_en_fallback", False))
                    )
                    scored_evidence = rescored
                    verdict_result = self._compute_verdict(scored_evidence, normalized_claim, language)
                timings["stage10b_image_multi_en_fallback"] = round(time.perf_counter() - t_fallback, 4)
                _emit(
                    "stage10b_image_multi_en_fallback:done "
                    f"sec={timings['stage10b_image_multi_en_fallback']} "
                    f"added={llm_status['image_multi_en_fallback_added']}"
                )
            # Text-only neutral recovery: one extra retrieval pass for low-evidence neutral outcomes.
            neutral_now = str(verdict_result.get("verdict", "neutral")).lower() == "neutral"
            non_neutral_kept = sum(
                1 for ev in scored_evidence
                if str(ev.get("stance", "neutral")).lower() in {"support", "refute"}
            )
            if (
                self.neutral_recovery_enable
                and (not is_image_mode)
                and str(language or "en").lower() != "en"
                and neutral_now
                and len(scored_evidence) <= self.neutral_recovery_keep_threshold
                and non_neutral_kept <= self.neutral_recovery_min_non_neutral
            ):
                _emit("stage10c_neutral_recovery:start")
                t_recover = time.perf_counter()
                recovery_rows = self._neutral_recovery_boost(
                    normalized_claim=normalized_claim,
                    language=language,
                    queries=search_queries,
                    translated_hint=translated_query_hint,
                )
                llm_status["neutral_recovery_attempted"] = True
                llm_status["neutral_recovery_added"] = len(recovery_rows)
                if recovery_rows:
                    combined_raw = list(raw_evidence) + recovery_rows
                    rescored = self._rank_evidence_language_aware(
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
                    min_keep_target = self._resolve_relevance_min_keep(
                        language=language,
                        is_image_mode=is_image_mode,
                    )
                    rescored = filtered if filtered else rescored[: min(min_keep_target, len(rescored))]
                    rescored = self._apply_domain_diversity(
                        rescored,
                        max_per_host=self.domain_diversity_max_per_host,
                    )
                    self._apply_stance_language_aware(
                        claim=normalized_claim,
                        evidence_list=rescored,
                        language=language,
                    )
                    for ev in rescored:
                        ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                            ev, recency_policy=recency_policy
                        )
                        if is_multi_lang and self.multi_phase4_lane_weight_enable:
                            self._apply_multi_lane_weight(ev)
                        if is_multi_lang and self.multi_phase3_tiers_enable:
                            self._annotate_multi_evidence_tier(ev)
                        ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                    if is_multi_lang and self.multi_phase3_tiers_enable:
                        rescored = self._filter_reject_tier_evidence(rescored)
                    rescored = self._dedupe_scored_evidence(rescored)
                    scored_evidence = rescored
                    verdict_result = self._compute_verdict(scored_evidence, normalized_claim, language)
                timings["stage10c_neutral_recovery"] = round(time.perf_counter() - t_recover, 4)
                _emit(
                    "stage10c_neutral_recovery:done "
                    f"sec={timings['stage10c_neutral_recovery']} "
                    f"added={llm_status.get('neutral_recovery_added', 0)}"
                )
            timings["stage10_llm_verify"] = round(time.perf_counter() - t, 4)
            _emit(
                f"stage10_llm_verify:done sec={timings['stage10_llm_verify']} triggered={llm_status.get('triggered')}"
            )
        else:
            timings["stage10_llm_verify"] = 0.0

        # Neutral-quality recovery (image/text): if final neutral is weak, run one extra retrieval pass.
        neutral_quality_recovery_attempted = 0
        while (
            self.neutral_quality_guard_enable
            and neutral_quality_recovery_attempted < self.neutral_quality_recovery_max_once
            and str(verdict_result.get("verdict", "neutral")).lower() == "neutral"
            and self._neutral_evidence_is_weak(scored_evidence)
        ):
            neutral_quality_recovery_attempted += 1
            _emit("stage10d_neutral_quality_recovery:start")
            t_nq = time.perf_counter()
            recovery_rows = self._neutral_recovery_boost(
                normalized_claim=normalized_claim,
                language=language,
                queries=search_queries,
                translated_hint=translated_query_hint,
            )
            llm_status["neutral_quality_recovery_attempted"] = True
            llm_status["neutral_quality_recovery_added"] = len(recovery_rows)
            if recovery_rows:
                combined_raw = list(raw_evidence) + recovery_rows
                rescored = self._rank_evidence_language_aware(
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
                self._apply_stance_language_aware(
                    claim=normalized_claim,
                    evidence_list=rescored,
                    language=language,
                )
                for ev in rescored:
                    ev["evidence_weight"] = self.evidence_scorer.calculate_weight(
                        ev, recency_policy=recency_policy
                    )
                    if is_multi_lang and self.multi_phase4_lane_weight_enable:
                        self._apply_multi_lane_weight(ev)
                    if is_multi_lang and self.multi_phase3_tiers_enable:
                        self._annotate_multi_evidence_tier(ev)
                    ev["weighted_stance"] = self.evidence_scorer.weight_stance(ev)
                if is_multi_lang and self.multi_phase3_tiers_enable:
                    rescored = self._filter_reject_tier_evidence(rescored)
                rescored = self._dedupe_scored_evidence(rescored)
                scored_evidence = rescored
                verdict_result = self._compute_verdict(scored_evidence, normalized_claim, language)
            timings["stage10d_neutral_quality_recovery"] = round(time.perf_counter() - t_nq, 4)
            _emit(
                "stage10d_neutral_quality_recovery:done "
                f"sec={timings['stage10d_neutral_quality_recovery']} "
                f"added={llm_status.get('neutral_quality_recovery_added', 0)}"
            )
        llm_status["post_recovery_verdict"] = str(verdict_result.get("verdict", "neutral"))
        llm_status["post_recovery_confidence"] = float(verdict_result.get("confidence", 0.0) or 0.0)
        if is_multi_lang and self.multi_phase1_guard_enable:
            llm_v = str(llm_status.get("verdict", "")).lower()
            final_v = str(verdict_result.get("verdict", "neutral")).lower()
            llm_conf = float(llm_status.get("confidence", 0.0) or 0.0)
            overwritten = llm_v in {"support", "refute"} and final_v == "neutral"
            llm_status["llm_non_neutral_overwritten"] = bool(overwritten)
            if overwritten:
                verdict_result["verdict"] = llm_v
                verdict_result["confidence"] = max(
                    float(verdict_result.get("confidence", 0.0) or 0.0),
                    llm_conf,
                )
                verdict_result["reasoning"] = (
                    f"{verdict_result.get('reasoning','')} "
                    "Phase1 guard restored non-neutral LLM decision."
                ).strip()
        llm_status["verdict_transition_path"] = (
            f"aggregate={pre_llm_verdict} -> "
            f"llm={str(llm_status.get('verdict', pre_llm_verdict)).lower()} -> "
            f"post_recovery={str(llm_status.get('post_recovery_verdict', verdict_result.get('verdict','neutral'))).lower()} -> "
            f"final={str(verdict_result.get('verdict','neutral')).lower()}"
        )
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
                "stance_source": ev.get("stance_source", "model"),
                "score": ev["evidence_weight"],
                "evidence_lane": ev.get("evidence_lane", self._assign_multi_lane(ev) if is_multi_lang else ""),
                "evidence_tier": ev.get("evidence_tier", ""),
                "effective_relevance": ev.get("effective_relevance"),
                "llm_adjusted": bool(ev.get("llm_adjusted", False)),
                "llm_adjust_reason": ev.get("llm_adjust_reason"),
            }
            for ev in scored_evidence
        ]
        recency_pref_match_count = sum(
            1 for ev in scored_evidence if float(ev.get("recency_bonus", 1.0) or 1.0) > 1.01
        )
        lane_counts: Dict[str, int] = {}
        tier_counts: Dict[str, int] = {}
        if is_multi_lang:
            for ev in scored_evidence:
                lane = str(ev.get("evidence_lane") or self._assign_multi_lane(ev))
                tier = str(ev.get("evidence_tier") or "unknown")
                lane_counts[lane] = lane_counts.get(lane, 0) + 1
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        return PipelineResult(
            verdict=verdict_result["verdict"],
            confidence=verdict_result["confidence"],
            evidence=evidence_list,
            reasoning=verdict_result["reasoning"],
            details={
                "checkability": "Checkable",
                "claim_type_proxy": "checkability_checkpoint",
                "context": f"{l1_label}/{l2_label}",
                "sources_checked": len(evidence_sources),
                "evidence_count": len(scored_evidence),
                "llm_verifier": llm_status,
                "sarvam_reranker": sarvam_rerank_status,
                "retrieval_telemetry": retrieval_telemetry,
                "recency_preference": recency_policy,
                "recency_preference_matches": recency_pref_match_count,
                "multi_lane_counts": lane_counts,
                "multi_tier_counts": tier_counts,
                "timings": timings,
            }
        )

    def _image_multi_english_fallback(
        self,
        normalized_claim: str,
        language: str,
        translated_hint: str = "",
    ) -> List[Dict]:
        """
        Image-only multi-language fallback:
        translate claim to English, fetch English web evidence, normalize + dedupe.
        """
        translated = str(translated_hint or "").strip()
        if not translated:
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

    def _is_english_evidence(self, row: Dict) -> bool:
        text = str(row.get("text") or "")
        if not text:
            return False
        try:
            return bool(self.normalizer._is_english_like(text))  # pylint: disable=protected-access
        except Exception:
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            return ascii_chars / max(1, len(text)) >= 0.85

    def _ensure_en_branch_models(self) -> bool:
        """Lazy-init EN relevance/stance models for multi mixed-language routing."""
        if self._en_relevance_scorer is not None and self._en_stance_detector is not None:
            return True
        if self._en_branch_init_attempted:
            return self._en_relevance_scorer is not None and self._en_stance_detector is not None
        self._en_branch_init_attempted = True
        try:
            en_relevance_path = os.getenv("RELEVANCE_EN_PATH", "checkpoints/relevance/en")
            self._en_relevance_scorer = RelevanceScorer(
                model_path=en_relevance_path,
                bi_model_name=self.config.get("relevance_bi_encoder_model"),
                enable_two_stage=bool(self.config.get("enable_two_stage_relevance", True)),
                shortlist_k=int(self.config.get("relevance_shortlist_k", 20)),
                top_k=self.config.get("relevance_top_k"),
            )
            en_stance_path = os.getenv("STANCE_EN_PATH", "checkpoints/stance/en")
            self._en_stance_detector = StanceDetector(model_path=en_stance_path)
            return True
        except Exception as exc:
            logger.warning("Failed to initialize EN branch models for multi mixed routing: %s", exc)
            self._en_relevance_scorer = None
            self._en_stance_detector = None
            return False

    def _rank_evidence_language_aware(
        self,
        claim: str,
        evidence_list: List[Dict],
        language: str,
    ) -> List[Dict]:
        """
        Multi-only: route English evidence through EN relevance scorer,
        non-English evidence through multi scorer. EN path remains unchanged.
        """
        lang = str(language or "en").lower()
        if lang == "en" or not self.enable_multi_mixed_evidence_routing:
            return self.relevance_scorer.rank_evidence(
                claim=claim, evidence_list=evidence_list, language=language
            )
        rows = list(evidence_list or [])
        if not rows:
            return []
        en_rows: List[Dict] = []
        non_en_rows: List[Dict] = []
        for ev in rows:
            if self._is_english_evidence(ev):
                en_rows.append(ev)
            else:
                non_en_rows.append(ev)
        # If no mixed language, keep current behavior.
        if not en_rows or not non_en_rows:
            ranked = self.relevance_scorer.rank_evidence(
                claim=claim, evidence_list=rows, language=language
            )
            for ev in ranked:
                ev["_model_branch"] = "multi"
            return ranked

        if not self._ensure_en_branch_models():
            ranked = self.relevance_scorer.rank_evidence(
                claim=claim, evidence_list=rows, language=language
            )
            for ev in ranked:
                ev["_model_branch"] = "multi"
            return ranked

        en_ranked = self._en_relevance_scorer.rank_evidence(
            claim=claim, evidence_list=en_rows, language="en"
        )
        for ev in en_ranked:
            ev["_model_branch"] = "en"
        non_en_ranked = self.relevance_scorer.rank_evidence(
            claim=claim, evidence_list=non_en_rows, language=language
        )
        for ev in non_en_ranked:
            ev["_model_branch"] = "multi"
        merged = en_ranked + non_en_ranked
        merged.sort(key=lambda x: float(x.get("relevance", 0.0) or 0.0), reverse=True)
        return merged

    def _apply_stance_language_aware(
        self,
        claim: str,
        evidence_list: List[Dict],
        language: str,
    ) -> None:
        """Apply stance detection with same language-aware branch routing used in relevance."""
        lang = str(language or "en").lower()
        for ev in evidence_list or []:
            detector = self.stance_detector
            if (
                lang != "en"
                and self.enable_multi_mixed_evidence_routing
                and ev.get("_model_branch") == "en"
                and self._ensure_en_branch_models()
                and self._en_stance_detector is not None
            ):
                detector = self._en_stance_detector
            stance_probs = detector.detect(claim, ev["text"])
            if self.enable_polarity_adjust:
                stance_probs = self._apply_polarity_adjustment(claim, ev, stance_probs)
            ev["stance_probs"] = stance_probs
            ev["stance"] = max(stance_probs, key=stance_probs.get)
            ev["stance_source"] = "model"

            # Sarvam is additive by default. Optional conservative override only when enabled.
            if (self.enable_sarvam_stance_override and (not self.sarvam_force_advisory)) and bool(ev.get("_sarvam_enhanced")):
                sarvam_stance = str(ev.get("sarvam_stance", "neutral")).strip().lower()
                sarvam_conf = float(ev.get("sarvam_stance_confidence", 0.0) or 0.0)
                if (
                    sarvam_stance in {"support", "refute", "neutral"}
                    and sarvam_conf >= self.sarvam_stance_override_min_conf
                ):
                    model_stance = str(ev.get("stance", "neutral")).lower()
                    # Conservative policy: only flip model-neutral to strong Sarvam non-neutral.
                    if model_stance == "neutral" and sarvam_stance in {"support", "refute"}:
                        floor = (1.0 - sarvam_conf) / 2.0
                        if sarvam_stance == "support":
                            ev["stance_probs"] = {"support": sarvam_conf, "neutral": floor, "refute": floor}
                        else:
                            ev["stance_probs"] = {"support": floor, "neutral": floor, "refute": sarvam_conf}
                        ev["stance"] = sarvam_stance
                        ev["stance_source"] = "sarvam_reranker_override"

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

    def _compute_verdict(self, evidence_list: List[Dict], claim: str, language: str) -> Dict:
        """Compute verdict with multi-only Phase7 decisive aggregation override."""
        base = self.verdict_engine.compute(evidence_list, claim)
        lang = str(language or "en").lower()
        if lang != "en" and self.multi_phase7_decisive_verdict_enable:
            base = self._compute_multi_decisive_verdict(evidence_list, claim, base)
        return self._apply_language_verdict_calibration(base, language=lang)

    def _compute_multi_decisive_verdict(self, evidence_list: List[Dict], claim: str, base: Dict) -> Dict:
        """Multi-only decisive-strength aggregation to reduce neutral-overweight artifacts."""
        if not evidence_list:
            return base
        strong_support = 0.0
        strong_refute = 0.0
        soft_support = 0.0
        soft_refute = 0.0
        strong_count = 0
        for ev in evidence_list:
            tier = str(ev.get("evidence_tier") or "soft").lower()
            stance = str(ev.get("stance") or "neutral").lower()
            ew = float(ev.get("evidence_weight", 0.0) or 0.0)
            if stance not in {"support", "refute"}:
                continue
            if tier == "strong":
                strong_count += 1
                if stance == "support":
                    strong_support += ew
                else:
                    strong_refute += ew
            elif tier == "soft":
                if stance == "support":
                    soft_support += ew
                else:
                    soft_refute += ew

        if self.multi_phase7_strong_only_for_decision:
            support_score = strong_support
            refute_score = strong_refute
        else:
            support_score = strong_support + (self.multi_phase7_soft_factor * soft_support)
            refute_score = strong_refute + (self.multi_phase7_soft_factor * soft_refute)

        total = support_score + refute_score
        gap = abs(support_score - refute_score)
        if strong_count == 0 or total < self.multi_phase7_min_total or gap < self.multi_phase7_min_gap:
            out = dict(base)
            out["verdict"] = "neutral"
            out["confidence"] = max(float(base.get("confidence", 0.0) or 0.0), 0.45)
            out["reasoning"] = (
                f"{base.get('reasoning','')} "
                "Phase7 decisive guard: insufficient decisive support/refute evidence."
            ).strip()
            out["multi_decisive"] = {
                "support_score": support_score,
                "refute_score": refute_score,
                "total": total,
                "gap": gap,
                "strong_count": strong_count,
            }
            return out

        verdict = "support" if support_score > refute_score else "refute"
        conf = gap / max(total, 1e-6)
        out = dict(base)
        out["verdict"] = verdict
        out["confidence"] = max(float(base.get("confidence", 0.0) or 0.0), max(0.5, min(0.99, conf)))
        out["reasoning"] = (
            f"{base.get('reasoning','')} "
            "Phase7 decisive aggregation selected non-neutral verdict from strong/soft evidence."
        ).strip()
        out["multi_decisive"] = {
            "support_score": support_score,
            "refute_score": refute_score,
            "total": total,
            "gap": gap,
            "strong_count": strong_count,
        }
        return out

    def _apply_language_verdict_calibration(self, verdict: Dict, language: str) -> Dict:
        """Language-aware abstain calibration to reduce fragile support/refute flips."""
        out = dict(verdict or {})
        v = str(out.get("verdict", "neutral")).lower()
        if v not in {"support", "refute"}:
            return out
        conf = float(out.get("confidence", 0.0) or 0.0)
        agg_support = float(out.get("agg_support", 0.0) or 0.0)
        agg_refute = float(out.get("agg_refute", 0.0) or 0.0)
        gap = abs(agg_support - agg_refute)
        is_en = str(language or "en").lower().startswith("en")
        min_conf = self.stance_decisive_min_conf_en if is_en else self.stance_decisive_min_conf_multi
        min_gap = self.stance_decisive_min_gap_en if is_en else self.stance_decisive_min_gap_multi
        if conf >= min_conf and gap >= min_gap:
            return out
        out["verdict"] = "neutral"
        out["confidence"] = max(0.45, float(out.get("agg_neutral", 0.0) or 0.0), min(conf, min_conf))
        out["reasoning"] = (
            f"{out.get('reasoning','')} "
            "Calibration abstain: support/refute margin or confidence below threshold."
        ).strip()
        out["calibration_abstain"] = {
            "min_conf": min_conf,
            "min_gap": min_gap,
            "conf": conf,
            "gap": gap,
        }
        return out

    def _neutral_evidence_is_weak(self, evidence_list: List[Dict]) -> bool:
        """Neutral should be backed by enough high-quality neutral evidence."""
        if not self.neutral_quality_guard_enable:
            return False
        neutral_rows = [
            ev for ev in (evidence_list or [])
            if str(ev.get("stance", "neutral")).lower() == "neutral"
            and float(ev.get("relevance", 0.0) or 0.0) >= self.neutral_quality_min_relevance
            and float(ev.get("credibility", 0.0) or 0.0) >= self.neutral_quality_min_credibility
        ]
        return len(neutral_rows) < self.neutral_quality_min_count

    def _assign_multi_lane(self, ev: Dict) -> str:
        if str(ev.get("type") or "").lower() == "structured_api":
            return "structured_reference"
        branch = str(ev.get("_model_branch") or "").lower()
        if branch == "en":
            return "translated_en"
        return "native_local"

    def _apply_multi_lane_weight(self, ev: Dict) -> None:
        lane = self._assign_multi_lane(ev)
        ev["evidence_lane"] = lane
        lane_w = self.multi_phase4_lane_weight_native
        if lane == "structured_reference":
            lane_w = self.multi_phase4_lane_weight_structured
        elif lane == "translated_en":
            lane_w = self.multi_phase4_lane_weight_translated
        ev["evidence_lane_weight"] = lane_w
        ev["evidence_weight"] = max(0.0, float(ev.get("evidence_weight", 0.0) or 0.0) * lane_w)

    def _annotate_multi_evidence_tier(self, ev: Dict) -> None:
        quality = float(ev.get("credibility", 0.0) or 0.0)
        rel = float(ev.get("relevance", 0.0) or 0.0)
        selector = float(ev.get("quality_score", 50.0) or 50.0) / 100.0
        selector = max(0.0, min(1.0, selector))
        eff_rel = min(1.0, (rel * 0.85) + (selector * 0.15))
        ev["selector_score"] = selector
        ev["effective_relevance"] = eff_rel
        ev["quality_score_norm"] = quality
        if eff_rel >= self.multi_phase3_strong_rel and quality >= self.multi_phase3_strong_quality:
            ev["evidence_tier"] = "strong"
        elif eff_rel >= self.multi_phase3_soft_rel and quality >= self.multi_phase3_soft_quality:
            ev["evidence_tier"] = "soft"
        else:
            ev["evidence_tier"] = "reject"

    def _filter_reject_tier_evidence(self, rows: List[Dict]) -> List[Dict]:
        keep = [ev for ev in rows if str(ev.get("evidence_tier") or "soft").lower() != "reject"]
        if keep:
            return keep
        # Keep at least a tiny fallback to avoid empty-list churn in edge cases.
        return list(rows[: min(2, len(rows))])

    def _resolve_relevance_min_keep(self, language: str, is_image_mode: bool) -> int:
        lang = str(language or "en").lower()
        if (not is_image_mode) and lang != "en":
            return self.relevance_min_keep_multi
        return self.relevance_min_keep

    def _apply_domain_diversity(self, rows: List[Dict], max_per_host: int) -> List[Dict]:
        if max_per_host <= 0:
            return rows
        host_counts: Dict[str, int] = {}
        out: List[Dict] = []
        for row in rows or []:
            host = ""
            try:
                host = (urlparse(str(row.get("url") or "")).netloc or "").lower().split(":")[0].strip(".")
            except Exception:
                host = ""
            if not host:
                out.append(row)
                continue
            count = host_counts.get(host, 0)
            if count >= max_per_host:
                continue
            host_counts[host] = count + 1
            out.append(row)
        return out

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

    def _neutral_recovery_boost(
        self,
        normalized_claim: str,
        language: str,
        queries: List[str],
        translated_hint: str = "",
    ) -> List[Dict]:
        """
        One-shot low-evidence neutral recovery:
        run additional native + translated English web search and normalize rows.
        """
        native_query = str(normalized_claim or "").strip()
        translated = str(translated_hint or "").strip()
        if not translated:
            try:
                translated = self.normalizer._translate_indic_to_english(  # pylint: disable=protected-access
                    text=native_query,
                    language=language,
                )
            except Exception:
                translated = ""
        translated = str(translated or "").strip()

        combined_queries: List[str] = []
        if native_query:
            combined_queries.append(native_query)
        for q in (queries or []):
            qv = str(q or "").strip()
            if qv and qv not in combined_queries:
                combined_queries.append(qv)
        if translated:
            en_probe = f"Is it true that {translated}"
            for qv in (translated, en_probe):
                qv = qv.strip()
                if qv and qv not in combined_queries:
                    combined_queries.append(qv)
        combined_queries = combined_queries[: self.neutral_recovery_max_queries]
        if not combined_queries:
            return []

        rows: List[Dict] = []
        # Native-language search
        try:
            rows.extend(
                self.evidence_gatherer.web_search.search(
                    claim=native_query or normalized_claim,
                    queries=combined_queries,
                    subtype=None,
                    language=language,
                    max_results=self.neutral_recovery_max_results,
                )
            )
        except Exception as exc:
            logger.warning("Neutral recovery native search failed: %s", exc)
        # English translated search
        if translated:
            try:
                rows.extend(
                    self.evidence_gatherer.web_search.search(
                        claim=translated,
                        queries=combined_queries,
                        subtype=None,
                        language="en",
                        max_results=self.neutral_recovery_max_results,
                    )
                )
            except Exception as exc:
                logger.warning("Neutral recovery English search failed: %s", exc)
        if not rows:
            return []
        normalized = self.evidence_gatherer._normalize_evidence_list(rows, default_type="web_search")
        out: List[Dict] = []
        seen = set()
        for row in normalized or []:
            key = (
                str(row.get("url") or "").strip(),
                str(row.get("text") or "")[:180].strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _extract_translated_query_hint(self, queries: List[str], language: str) -> str:
        """Pick one English-like query from Stage-1 query list for multi flows."""
        lang = str(language or "en").lower()
        if lang == "en":
            return ""
        for q in queries or []:
            qv = str(q or "").strip()
            if not qv:
                continue
            try:
                if self.normalizer._is_english_like(qv):  # pylint: disable=protected-access
                    return qv
            except Exception:
                continue
        return ""

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

    def _apply_context_overrides(
        self,
        claim: str,
        level1: str,
        level2: str,
        l1_conf: float,
        is_image_mode: bool = False,
    ) -> tuple[str, str]:
        """Apply lightweight keyword overrides when context model misses obvious domains."""
        if os.getenv("ROUTER_ENABLE_CLAIM_OVERRIDES", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            # Keep image-specific override independent from global router override toggle.
            if not is_image_mode:
                return level1, level2

        text = claim.lower()
        weak_context = (l1_conf < 0.65) or level1 in {"GENERAL_FACTUAL", "TECHNOLOGY"}

        policy_terms = {"gst", "tax", "parliament", "assembly", "government", "policy", "constitution"}
        law_terms = {"act", "section", "fir", "court", "legal", "law", "police"}
        finance_terms = {"gdp", "inflation", "budget", "fiscal", "repo rate", "stock", "market", "rupee", "export", "import"}
        entertainment_terms = {"imdb", "box office", "movie", "film", "cinema", "song", "album"}
        accident_terms = {"accident", "crash", "killed", "injured", "blast", "fire", "murder", "attack", "arrest"}

        if weak_context and any(t in text for t in law_terms):
            return "LAW_CRIME", "regulation"
        if weak_context and any(t in text for t in policy_terms):
            return "POLITICS_GOVERNMENT", "public_policy"
        if weak_context and any(t in text for t in finance_terms):
            return "ECONOMICS_BUSINESS", "macroeconomics"
        if weak_context and any(t in text for t in accident_terms):
            return "LAW_CRIME", "criminal_cases"
        if weak_context and any(t in text for t in entertainment_terms):
            return "ENTERTAINMENT", "film"
        if is_image_mode and os.getenv("IMAGE_CONTEXT_OVERRIDE_ENABLE", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            weather_terms = {
                "weather", "rain", "rainfall", "storm", "cyclone", "flood", "wind",
                "thunder", "lightning", "forecast", "alert", "temperature",
                "వర్ష", "మబ్బు", "மழை", "புயல்", "ಮಳೆ", "ಮಿಂಚು", "बारिश", "आंधी", "മഴ",
            }
            election_terms = {
                "election", "poll", "vote", "candidate", "constituency", "assembly", "nomination",
                "தேர்தல்", "வேட்புமனு", "ఎన్నిక", "నామినేషన్", "ಚುನಾವಣೆ", "ನಾಮಪತ್ರ",
                "चुनाव", "नामांकन", "തിരഞ്ഞെടുപ്പ്",
            }
            if any(t in text for t in weather_terms) and weak_context:
                return "ENVIRONMENT_CLIMATE", "disasters_weather"
            if any(t in text for t in election_terms) and weak_context:
                return "POLITICS_GOVERNMENT", "elections"
        return level1, level2

    def _normalize_image_claim_for_retrieval(self, claim: str) -> str:
        """Reduce OCR framing noise without changing core claim meaning."""
        text = str(claim or "").strip()
        if not text:
            return text
        # If OCR starts with a question then follows with statement, prefer statement segment.
        qidx = text.find("?")
        if 0 <= qidx < 140 and (len(text) - qidx) > 40:
            tail = text[qidx + 1 :].strip()
            if tail:
                text = tail
        # Collapse repeated whitespace and punctuation runs.
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([.!?])\1{1,}", r"\1", text)
        return text

    def _clean_image_queries(self, queries: List[str], normalized_claim: str) -> List[str]:
        """Image OCR queries can be noisy; keep concise, de-duplicated variants."""
        cap = max(1, int(os.getenv("IMAGE_QUERY_CLEAN_CAP", "3")))
        max_chars = max(80, int(os.getenv("IMAGE_QUERY_MAX_CHARS", "240")))
        out: List[str] = []
        seen = set()
        base = [normalized_claim] + list(queries or [])
        for q in base:
            s = str(q or "").strip()
            if not s:
                continue
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > max_chars:
                clipped = s[:max_chars]
                cut = clipped.rfind(" ")
                s = (clipped[:cut] if cut >= int(max_chars * 0.6) else clipped).strip()
            k = s.lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(s)
            if len(out) >= cap:
                break
        return out or [normalized_claim]

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
        include_general = os.getenv("ROUTER_LOW_CONF_INCLUDE_GENERAL", "1").strip().lower() in {
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
