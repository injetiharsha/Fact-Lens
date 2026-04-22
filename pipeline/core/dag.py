"""Minimal DAG execution utilities for pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Set


@dataclass
class DAGNode:
    """Single DAG node."""

    name: str
    func: Callable[[Dict[str, Any], Dict[str, Any]], Any]
    deps: Set[str] = field(default_factory=set)
    optional: bool = False


class DAGExecutor:
    """Execute DAG nodes with dependency-aware parallel scheduling."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max(1, int(max_workers))

    def run(
        self,
        nodes: List[DAGNode],
        context: Dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        ctx = context or {}
        pending: Dict[str, DAGNode] = {n.name: n for n in nodes}
        done: Dict[str, Any] = {}
        failed: Dict[str, str] = {}

        while pending:
            ready: List[DAGNode] = []
            resolved = set(done.keys()) | set(failed.keys())
            for name, node in list(pending.items()):
                if set(node.deps).issubset(resolved):
                    ready.append(node)

            if not ready:
                unresolved = ", ".join(sorted(pending.keys()))
                raise RuntimeError(f"DAG deadlock/cycle detected. Pending nodes: {unresolved}")

            workers = min(self.max_workers, len(ready))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_map = {}
                for node in ready:
                    dep_results = {k: done.get(k) for k in node.deps if k in done}
                    fut = pool.submit(node.func, ctx, dep_results)
                    fut_map[fut] = node

                for fut in as_completed(fut_map):
                    node = fut_map[fut]
                    try:
                        done[node.name] = fut.result()
                    except Exception as exc:
                        if node.optional:
                            failed[node.name] = str(exc)
                        else:
                            raise

            for node in ready:
                pending.pop(node.name, None)

        return done, failed

