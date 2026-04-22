# Architecture Root-Cause Analysis (Old vs New)

Date: 2026-04-23

This document replaces earlier planning notes and failed-plan drafts.
It summarizes what changed between old and current pipeline architecture,
what failed in practice, and why.

## 1) Executive Summary

- Current architecture is better than old Stage5/6 on accuracy.
- Old Stage5/6 is faster but over-predicts `neutral` and loses decisive claims.
- The largest engineering risk is retrieval quality and retrieval cost variance, not core model execution.

Validated benchmark comparison:

- EN current: `0.633` (19/30), neutral `0.300`, time `1173.734s`
- EN old Stage5/6: `0.433` (13/30), neutral `0.667`, time `814.695s`
- MULTI current: `0.633` (19/30), neutral `0.067`, time `1226.926s`
- MULTI old Stage5/6: `0.533` (16/30), neutral `0.467`, time `981.602s`

Conclusion: old profile is not a quality baseline; it is only a speed baseline.

## 2) Architecture Comparison by Stage

### Stage 1-4 (Normalize, Checkability, Context, Routing)

Old:
- Simpler flow with fewer downstream correction paths.
- Lower retrieval complexity after routing.

Current:
- Stronger routing and richer downstream instrumentation.
- Better support for multilingual and image/document pathways.

Observed outcome:
- These stages are not the dominant source of regressions.
- Some context misroutes still happen and can poison Stage5 source mix.

### Stage 5 (Evidence Gather) - Main Difference

Old:
- Legacy gather path, no DAG orchestration.
- Lower retrieval expansion and less enrichment.
- Lower latency, lower evidence diversity, lower recall on hard claims.

Current:
- DAG-capable gatherer, source fanout, provider diversity, scrape enrichment.
- Better recall and better final accuracy.
- Higher latency and higher external dependency risk.

Observed outcome:
- Stage5 dominates wall-time and credit burn.
- Retrieval noise is the top root cause of neutral drift and unstable verdict confidence.

### Stage 6 (Relevance + Optional Rerank)

Old:
- Simpler relevance sequence, fewer candidate refinement layers.

Current:
- Two-stage relevance plus optional final rerank hooks.
- Better ranking quality when evidence is good.

Observed outcome:
- Accuracy improves versus old profile.
- Additional complexity can amplify poor retrieval input quality.

### Stage 7-10 (Stance, Score, Aggregate, LLM Verify)

Old:
- Lower intervention and weaker rescue logic.
- Faster but frequently undecided behavior.

Current:
- Better integration of stance evidence and verifier fallback.
- More robust against sparse evidence than old profile.

Observed outcome:
- Current path decisively outperforms old on both EN and MULTI.
- Remaining failures are often retrieval-origin, not verifier-origin.

## 3) What Failed and Why

### Why old Stage5/6 failed

Primary causes:
- Too many claims fall to `neutral` due to weak/insufficient retrieval evidence.
- Lower retrieval breadth misses decisive evidence.
- Reduced enrichment lowers recall for multilingual or niche facts.

Symptoms in benchmark:
- EN neutral rate jumped to `0.667` (from `0.300` current).
- MULTI neutral rate jumped to `0.467` (from `0.067` current).

### Why current pipeline still hurts in production

Primary causes:
- Retrieval dominates runtime and spend.
- Provider instability/rate limits cause run-to-run variance.
- Broad retrieval can introduce noisy pages that dilute decisive evidence.

Symptoms:
- Stage5 remains dominant bottleneck.
- Credits burn mostly in search/scrape, not model inference.

## 4) Main Retrieval Pain Points (Project-Level)

1. External provider dependency:
- Tavily/Serp/DDG variability changes result quality and cost.

2. Query-volume sensitivity:
- More queries improve recall but increase noise and spend.

3. Multilingual translation coupling:
- Retrieval quality depends on translation quality for non-EN claims.

4. Scrape variability:
- Some URLs fail extraction; some succeed with low-quality text.

5. Comparison drift risk:
- If profile/env differs across runs, benchmark conclusions become misleading.

## 5) Root-Cause Tree

Top-level problem: unstable factual accuracy under budget/time constraints.

Branch A: retrieval quality variance
- provider mismatch
- noisy domains
- translation ambiguity

Branch B: retrieval cost/latency variance
- high query fanout
- high enrichment volume
- blocked/failed pages triggering expensive fallbacks

Branch C: configuration drift
- profile mismatch across runs
- hidden env override interactions

Net effect:
- old profile under-retrieves and neutralizes.
- current profile is accurate but expensive and sensitive.

## 6) Final Decision

Adopt:
- Current architecture for core pipeline quality.

Do not adopt:
- Old Stage5/6 as default.

Use old profile only for:
- speed-control experiments and ablation references.

## 7) Practical Next Actions

1. Keep current Stage5/6 as default production baseline.
2. Introduce locked benchmark profiles to prevent run drift.
3. Tune retrieval cost with explicit caps before changing model stages.
4. Track provider-level retrieval telemetry as release criteria.
5. Treat any increase in neutral rate as a hard regression signal.
