"""Recoverable transfer-credit curriculum training orchestration."""

import argparse
import dataclasses
import hashlib
import json
import random
import shutil
from pathlib import Path

from .credit import CreditRound
from .curriculum import CurriculumGenerator
from .hf_backend import HFBackend, HFBackendConfig
from .tasks import FAMILIES, AgendaTaskFactory, draw_probes
from .toy_backend import ToyBackend
from .vendor import DEFAULT_CACHE, DEFAULT_LOCK, bootstrap, load_registry


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    backend: str = "toy"
    seed: int = 20261010
    rounds: int = 2
    branches: int = 2
    train_tasks_per_branch: int = 8
    probes_per_round: int = 12
    warmup_tasks: int = 24
    test_tasks: int = 24
    credit_metric: str = "reward"
    lambda_credit: float = 2.0
    generator_learning_rate: float = 0.05
    generator_kl: float = 0.01
    output_dir: str = "outputs/pilot"
    hf: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in ("toy", "hf"):
            raise ValueError("backend must be toy or hf")
        if self.credit_metric not in ("reward", "success"):
            raise ValueError("credit_metric must be reward or success")
        for name in (
            "rounds",
            "train_tasks_per_branch",
            "probes_per_round",
            "warmup_tasks",
            "test_tasks",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if type(self.branches) is not int or self.branches < 2:
            raise ValueError("branches must be at least two")

    @classmethod
    def from_dict(cls, value):
        names = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - names
        if unknown:
            raise ValueError("Unknown config keys: " + ", ".join(sorted(unknown)))
        return cls(**value)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _config_digest(config):
    encoded = json.dumps(
        dataclasses.asdict(config),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed(root, *parts):
    encoded = ":".join(map(str, (root,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _backend(config, registry):
    if config.backend == "toy":
        return ToyBackend(registry, config.seed)
    return HFBackend(registry, HFBackendConfig(**config.hf), config.seed)


def _rebuild_metrics(output, completed_round):
    rows = []
    for round_index in range(completed_round + 1):
        path = output / "rounds" / f"round-{round_index:04d}" / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                key: report[key]
                for key in (
                    "round",
                    "committed_branch",
                    "credits",
                    "credit_weights",
                    "generator_probabilities",
                )
            }
        )
    path = output / "metrics.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_experiment(config, registry):
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    digest = _config_digest(config)
    factory = AgendaTaskFactory(config.seed)
    backend = _backend(config, registry)

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != digest:
            raise ValueError("Refusing to resume with a different experiment config")
        generator = CurriculumGenerator.from_state_dict(state["generator"])
        incumbent = output / state["incumbent"]
        if not incumbent.is_dir():
            raise FileNotFoundError("Committed incumbent checkpoint is missing")
        start_round = state["completed_round"] + 1
        _rebuild_metrics(output, state["completed_round"])
    else:
        generator = CurriculumGenerator()
        incumbent = output / "checkpoints" / "initial"
        if incumbent.exists():
            shutil.rmtree(incumbent)
        _atomic_json(output / "config.json", dataclasses.asdict(config))
        warmup = factory.balanced_pool("train", config.warmup_tasks)
        initialization = backend.initialize(incumbent, warmup)
        state = {
            "schema_version": 1,
            "config_sha256": digest,
            "completed_round": -1,
            "incumbent": str(incumbent.relative_to(output)),
            "generator": generator.state_dict(),
            "initialization": initialization,
        }
        _atomic_json(state_path, state)
        start_round = 0

    for round_index in range(start_round, config.rounds):
        round_dir = output / "rounds" / f"round-{round_index:04d}"
        if round_dir.exists():
            shutil.rmtree(round_dir)
        round_dir.mkdir(parents=True)
        cache_before = (
            backend.transition_cache.hits,
            backend.transition_cache.misses,
        )
        batches = []
        branch_metrics = {}
        branch_paths = {}
        candidate_ids = tuple(
            f"round-{round_index:04d}-branch-{index:02d}"
            for index in range(config.branches)
        )
        for index, branch_id in enumerate(candidate_ids):
            rng = random.Random(_seed(config.seed, "curriculum", round_index, index))
            batch = generator.sample_batch(
                branch_id,
                config.train_tasks_per_branch,
                rng,
                factory,
                round_index,
            )
            batches.append(batch)
            destination = round_dir / branch_id
            branch_metrics[branch_id] = backend.train_branch(
                incumbent,
                destination,
                batch.tasks,
                _seed(config.seed, "branch", round_index),
            )
            branch_paths[branch_id] = destination

        credit_round = CreditRound(
            candidate_ids,
            random.Random(_seed(config.seed, "commitment", round_index)),
        )
        probes = draw_probes(factory, round_index, config.probes_per_round, config.seed)
        credit_round.draw_probes(
            lambda probes=probes: tuple(task.task_id for task in probes)
        )
        evaluation_seed = _seed(config.seed, "evaluation", round_index)
        incumbent_scores, incumbent_details = backend.evaluate(
            incumbent, probes, config.credit_metric, evaluation_seed
        )
        candidate_scores = {}
        candidate_details = {}
        for branch_id in candidate_ids:
            scores, details = backend.evaluate(
                branch_paths[branch_id],
                probes,
                config.credit_metric,
                evaluation_seed,
            )
            candidate_scores[branch_id] = scores
            candidate_details[branch_id] = details
        credit = credit_round.compute(
            incumbent_scores,
            candidate_scores,
            lambda_credit=config.lambda_credit,
        )
        generator_update = generator.update(
            batches,
            credit.credits,
            learning_rate=config.generator_learning_rate,
            kl_coefficient=config.generator_kl,
        )
        incumbent = branch_paths[credit.committed_branch]
        report = {
            "schema_version": 1,
            "round": round_index,
            "incumbent_before": state["incumbent"],
            "committed_branch": credit.committed_branch,
            "incumbent_after": str(incumbent.relative_to(output)),
            "probe_ids": list(credit.probe_ids),
            "credits": credit.credits,
            "credit_weights": credit.weights,
            "incumbent_scores": incumbent_scores,
            "candidate_scores": candidate_scores,
            "curricula": {
                batch.branch_id: {
                    "families": list(batch.families),
                    "task_ids": [task.task_id for task in batch.tasks],
                }
                for batch in batches
            },
            "branch_metrics": branch_metrics,
            "tool_transition_cache": {
                "hits": backend.transition_cache.hits - cache_before[0],
                "misses": backend.transition_cache.misses - cache_before[1],
            },
            "generator_update": generator_update,
            "generator_probabilities": generator.probabilities,
            "details": {
                "incumbent": incumbent_details,
                "candidates": candidate_details,
            },
        }
        _atomic_json(round_dir / "report.json", report)
        state = {
            "schema_version": 1,
            "config_sha256": digest,
            "completed_round": round_index,
            "incumbent": str(incumbent.relative_to(output)),
            "generator": generator.state_dict(),
            "initialization": state["initialization"],
        }
        _atomic_json(state_path, state)
        _rebuild_metrics(output, round_index)

    test_tasks = factory.balanced_pool("test", config.test_tasks)
    test_scores, test_details = backend.evaluate(
        incumbent,
        test_tasks,
        config.credit_metric,
        _seed(config.seed, "final-test"),
    )
    result = {
        "schema_version": 1,
        "backend": config.backend,
        "completed_rounds": config.rounds,
        "incumbent": str(incumbent.relative_to(output)),
        "credit_metric": config.credit_metric,
        "test_mean": sum(test_scores.values()) / len(test_scores),
        "test_scores": test_scores,
        "test_details": test_details,
        "generator_probabilities": generator.probabilities,
        "task_families": list(FAMILIES),
    }
    _atomic_json(output / "final_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-api-bank", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output is not None:
        raw["output_dir"] = str(args.output)
    config = ExperimentConfig.from_dict(raw)
    if args.bootstrap_api_bank:
        bootstrap(args.cache_root, args.lock)
    registry = load_registry(args.cache_root, args.lock)
    result = run_experiment(config, registry)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
