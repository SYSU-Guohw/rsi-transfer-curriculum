"""Run real pinned agenda tools with independent and cached scripted actions."""

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import random
from pathlib import Path

from .credit import CreditRound
from .execution import Executor, TransitionCache
from .fixtures import TOOL_SEQUENCE, action_text, branches, databases
from .sampling import coupled_sample, positive_residual
from .state import freeze, thaw
from .vendor import DEFAULT_CACHE, DEFAULT_LOCK, load_registry


def execution_smoke(registry):
    independent = branches(registry)
    reused = branches(registry)
    reference_executor = Executor()
    cache = TransitionCache()
    cached_executor = Executor(cache)
    # Illustrative decoder costs, NOT measurements from a model/tokenizer.
    costs = ((4, 4, 4), (6, 2, 4), (4, 4, 4))
    steps = []
    for step, name in enumerate(TOOL_SEQUENCE):
        hits = []
        for index, (left, right) in enumerate(zip(independent, reused)):
            raw = action_text(name, style=index % 2)
            a = reference_executor.step(left, raw, costs[index][step])
            b = cached_executor.step(right, raw, costs[index][step])
            if (
                a.observation != b.observation
                or a.checker_result is not True
                or b.checker_result is not True
                or left.full_snapshot() != right.full_snapshot()
            ):
                raise AssertionError("Independent/cached execution mismatch")
            hits.append(b.cache_hit)
        steps.append({
            "tool": name,
            "cache_hits": hits,
            "tokens_left": [b.tokens_left for b in reused],
            "actions_left": [b.actions_left for b in reused],
        })
    for branch in reused:
        if branch.environment.snapshot() != freeze(databases()):
            raise AssertionError("Final agenda differs from the hand-authored oracle")
    return {
        "fixture_source": "hand-authored; no official benchmark examples",
        "token_cost_source": "synthetic counts for accounting regression only",
        "all_observations_private_histories_states_and_allowances_match": True,
        "all_original_per_call_checks_pass": True,
        "independent_physical_tool_calls": reference_executor.physical_tool_calls,
        "cached_physical_tool_calls": cached_executor.physical_tool_calls,
        "logical_tool_calls_per_arm": cached_executor.logical_tool_calls,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "steps": steps,
        "final_branches": [{
            "id": b.branch_id,
            "tokens_left": b.tokens_left,
            "actions_left": b.actions_left,
            "call_results": b.checker_state["call_results"],
            "private_history_sha256": hashlib.sha256(freeze(b.history)).hexdigest(),
            "environment_sha256": hashlib.sha256(b.environment.snapshot()).hexdigest(),
        } for b in reused],
    }


def sampling_smoke(draws=20000):
    p, q = (0.05, 0.25, 0.70), (0.60, 0.30, 0.10)
    residual = positive_residual(p, q)
    rejected_mass = math.fsum(max(qv - pv, 0.0) for pv, qv in zip(p, q))
    recovered = [
        min(pv, qv) + rejected_mass * rv
        for pv, qv, rv in zip(p, q, residual)
    ]
    rng = random.Random(20261010)
    counts = [0] * len(p)
    accepted = 0
    for _ in range(draws):
        result = coupled_sample(p, q, rng)
        counts[result.token] += 1
        accepted += result.accepted
    frequencies = [count / draws for count in counts]
    if max(abs(a - b) for a, b in zip(recovered, p)) > 1e-12:
        raise AssertionError("Categorical mass identity failed")
    if max(abs(a - b) for a, b in zip(frequencies, p)) > 0.02:
        raise AssertionError("Seeded categorical smoke check failed")
    return {
        "kind": "fixed categorical distributions; not a learned-model experiment",
        "draws": draws,
        "target_p": p,
        "proposal_q": q,
        "algebraic_output_mass": recovered,
        "empirical_output_frequency": frequencies,
        "accepted_proposals": accepted,
    }


def credit_smoke():
    round_ = CreditRound(("candidate-a", "candidate-b"), random.Random(17))
    round_.draw_probes(lambda: ("probe-0", "probe-1", "probe-2", "probe-3"))
    report = round_.compute(
        dict(zip(round_.probe_ids, (0, 1, 0, 1))),
        {
            "candidate-a": dict(zip(round_.probe_ids, (1, 1, 0, 1))),
            "candidate-b": dict(zip(round_.probe_ids, (0, 0, 0, 1))),
        },
        lambda_credit=2.0,
    )
    return {
        "kind": "synthetic supplied scores; no agent or generator training",
        **dataclasses.asdict(report),
    }


def run_smoke(cache_root=DEFAULT_CACHE, lock_path=DEFAULT_LOCK):
    registry = load_registry(cache_root, lock_path)
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "upstream_identity": registry.identity,
        "execution": execution_smoke(registry),
        "sampling": sampling_smoke(),
        "credit": credit_smoke(),
        "limitations": [
            "CPU component checks only; no Qwen3, TRL, GRPO or generator update.",
            "Per-call fixture checks are not an official free-running task success metric.",
            "Categorical checks do not establish learned-model full-path equivalence.",
            "Physical call counts do not establish wall-clock or end-to-end speedup.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_smoke(args.cache_root, args.lock)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
