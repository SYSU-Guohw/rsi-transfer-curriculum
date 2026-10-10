"""Deterministic CPU backend for end-to-end orchestration tests."""

import hashlib
import json
import math
from pathlib import Path

from .execution import TransitionCache
from .tasks import FAMILIES
from .trajectory import execute_completion


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ToyBackend:
    """A tabular learner that exercises the real environment and scorer."""

    name = "toy"

    def __init__(self, registry, seed=0):
        self.registry = registry
        self.seed = seed
        self.transition_cache = TransitionCache()

    def initialize(self, destination, warmup_tasks):
        state = {
            "schema_version": 1,
            "abilities": {family: -1.5 for family in FAMILIES},
            "momentum": {family: 0.0 for family in FAMILIES},
            "optimizer_step": 0,
            "warmup_examples": len(tuple(warmup_tasks)),
        }
        _atomic_json(Path(destination) / "state.json", state)
        return {"warmup_examples": state["warmup_examples"]}

    @staticmethod
    def _load(checkpoint):
        return json.loads((Path(checkpoint) / "state.json").read_text(encoding="utf-8"))

    def train_branch(self, source, destination, tasks, branch_seed):
        state = self._load(source)
        counts = {family: 0 for family in FAMILIES}
        for task in tasks:
            counts[task.family] += 1
        transfer = {
            "add": {"add": 1.0, "add_query": 0.7, "replace": 0.4},
            "query": {"query": 1.0, "add_query": 0.6, "query_delete": 0.7},
            "delete": {"delete": 1.0, "query_delete": 0.7, "replace": 0.7},
            "add_query": {"add": 0.5, "query": 0.5, "add_query": 1.0},
            "query_delete": {"query": 0.5, "delete": 0.5, "query_delete": 1.0},
            "replace": {"add": 0.4, "delete": 0.6, "replace": 1.0},
        }
        scale = max(len(tasks), 1)
        for target in FAMILIES:
            signal = (
                sum(
                    counts[source_family] * transfer[source_family].get(target, 0.0)
                    for source_family in FAMILIES
                )
                / scale
            )
            momentum = 0.8 * state["momentum"][target] + 0.2 * signal
            state["momentum"][target] = momentum
            state["abilities"][target] += 0.8 * momentum
        state["optimizer_step"] += 1
        state["branch_seed"] = branch_seed
        state["curriculum_counts"] = counts
        _atomic_json(Path(destination) / "state.json", state)
        return {
            "optimizer_step": state["optimizer_step"],
            "curriculum_counts": counts,
            "mean_ability": sum(state["abilities"].values()) / len(FAMILIES),
        }

    def evaluate(self, checkpoint, tasks, metric, evaluation_seed):
        if metric not in ("reward", "success"):
            raise ValueError("Unknown evaluation metric")
        state = self._load(checkpoint)
        results = {}
        details = {}
        for task in tasks:
            probability = 1.0 / (1.0 + math.exp(-state["abilities"][task.family]))
            digest = hashlib.sha256(
                f"{self.seed}:{evaluation_seed}:{task.task_id}".encode()
            ).digest()
            uniform = int.from_bytes(digest[:8], "big") / 2**64
            completion = (
                task.oracle_completion()
                if uniform < probability
                else '{"tool":"Unknown","arguments":{}}\n{"final":"failed"}'
            )
            score = execute_completion(
                task,
                completion,
                self.registry,
                transition_cache=self.transition_cache,
            )
            results[task.task_id] = getattr(score, metric)
            details[task.task_id] = score.to_dict()
        return results, details
