import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from rsi_curriculum.curriculum import CurriculumGenerator
from rsi_curriculum.tasks import FAMILIES, AgendaTaskFactory
from rsi_curriculum.training import ExperimentConfig, run_experiment
from rsi_curriculum.trajectory import execute_completion
from rsi_curriculum.vendor import load_registry


class TaskAndTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.factory = AgendaTaskFactory(71)

    def test_oracle_completions_execute_and_score_for_every_family(self):
        for index, family in enumerate(FAMILIES):
            with self.subTest(family=family):
                task = self.factory.task("train", family, index)
                score = execute_completion(
                    task, task.oracle_completion(), self.registry
                )
                self.assertEqual(score.success, 1.0)
                self.assertEqual(score.reward, 1.0)
                self.assertEqual(score.correct_calls, len(task.calls))

    def test_split_entities_and_task_ids_are_disjoint(self):
        tasks = [
            self.factory.task(split, family, index)
            for split in ("train", "development", "test")
            for index, family in enumerate(FAMILIES)
        ]
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))
        tokens = {
            split: {
                next(iter(task.databases["Account"].values()))["token"]
                for task in tasks
                if task.split == split
            }
            for split in ("train", "development", "test")
        }
        self.assertTrue(tokens["train"].isdisjoint(tokens["development"]))
        self.assertTrue(tokens["train"].isdisjoint(tokens["test"]))
        self.assertTrue(tokens["development"].isdisjoint(tokens["test"]))

    def test_pretty_json_is_allowed_but_missing_extra_and_post_final_text_fail(self):
        task = self.factory.task("development", "add_query", 0)
        pretty = "\n".join(
            json.dumps(json.loads(line), indent=2)
            for line in task.oracle_completion().splitlines()
        )
        self.assertEqual(execute_completion(task, pretty, self.registry).success, 1.0)
        missing = task.calls[0].action_text() + '\n{"final":"done"}'
        self.assertEqual(execute_completion(task, missing, self.registry).success, 0.0)
        extra = (
            task.oracle_completion().rsplit("\n", 1)[0]
            + "\n"
            + task.calls[-1].action_text()
            + '\n{"final":"done"}'
        )
        self.assertEqual(execute_completion(task, extra, self.registry).success, 0.0)
        post_final = task.oracle_completion() + '\n{"final":"again"}'
        self.assertEqual(
            execute_completion(task, post_final, self.registry).success, 0.0
        )


class CurriculumTests(unittest.TestCase):
    def test_positive_credit_increases_sampled_family_probability(self):
        factory = AgendaTaskFactory(3)
        generator = CurriculumGenerator()
        before = generator.probabilities["add"]
        good = generator.sample_batch(
            "good", 10, random.Random(1), factory, round_index=0
        )
        bad = generator.sample_batch(
            "bad", 10, random.Random(2), factory, round_index=0
        )
        # Isolate the intended family count in a transparent synthetic trace.
        good = type(good)("good", ("add",) * 10, good.tasks)
        bad = type(bad)("bad", ("query",) * 10, bad.tasks)
        report = generator.update((good, bad), {"good": 0.5, "bad": -0.5})
        self.assertGreater(generator.probabilities["add"], before)
        self.assertTrue(math.isclose(sum(report["probabilities"].values()), 1.0))
        restored = CurriculumGenerator.from_state_dict(generator.state_dict())
        self.assertEqual(restored.state_dict(), generator.state_dict())


class OrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()

    def test_toy_loop_runs_and_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                backend="toy",
                seed=19,
                rounds=2,
                branches=2,
                train_tasks_per_branch=4,
                probes_per_round=6,
                warmup_tasks=6,
                test_tasks=6,
                output_dir=directory,
            )
            first = run_experiment(config, self.registry)
            metrics_path = Path(directory) / "metrics.jsonl"
            metrics_before = metrics_path.read_text()
            metrics_path.write_text(metrics_before + '{"stale":"partial"}\n')
            second = run_experiment(config, self.registry)
            self.assertEqual(first, second)
            self.assertEqual(metrics_path.read_text(), metrics_before)
            state = json.loads((Path(directory) / "state.json").read_text())
            self.assertEqual(state["completed_round"], 1)
            self.assertTrue(
                (Path(directory) / state["incumbent"] / "state.json").is_file()
            )

    def test_resume_rejects_configuration_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                rounds=1,
                branches=2,
                train_tasks_per_branch=2,
                probes_per_round=2,
                warmup_tasks=2,
                test_tasks=2,
                output_dir=directory,
            )
            run_experiment(config, self.registry)
            changed = dataclass_replace(config, rounds=2)
            with self.assertRaisesRegex(ValueError, "different experiment config"):
                run_experiment(changed, self.registry)


def dataclass_replace(instance, **changes):
    values = {
        field.name: getattr(instance, field.name)
        for field in instance.__dataclass_fields__.values()
    }
    values.update(changes)
    return type(instance)(**values)


if __name__ == "__main__":
    unittest.main()
