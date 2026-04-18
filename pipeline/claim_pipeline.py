"""Claim-level pipeline entrypoint."""

from pipeline.orchestrator import FactCheckingPipeline, PipelineResult


class ClaimPipeline(FactCheckingPipeline):
    """Canonical claim analysis pipeline."""


__all__ = ["ClaimPipeline", "FactCheckingPipeline", "PipelineResult"]

