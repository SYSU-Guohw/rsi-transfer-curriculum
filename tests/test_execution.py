import copy
import sys
import tempfile
import unittest
from pathlib import Path

from rsi_curriculum.environment import APIBankEnvironment, InvalidAction, parse_action
from rsi_curriculum.execution import Branch, Executor, Transition, TransitionCache
from rsi_curriculum.fixtures import (
    ARGUMENTS, NEW_RECORD, action_text, branches, databases, references,
)
from rsi_curriculum.smoke import execution_smoke
from rsi_curriculum.state import freeze, thaw
from rsi_curriculum.vendor import DEFAULT_CACHE, PROJECT_ROOT, bootstrap, load_registry


class SnapshotTests(unittest.TestCase):
    def test_mixed_keys_order_and_scalar_types_survive_restore(self):
        state = {"1": [False, None, b"x"], 1: (1.0, -0.0, "中文")}
        restored = thaw(freeze(state))
        self.assertEqual(list(restored), ["1", 1])
        self.assertIs(type(restored[1]), tuple)
        self.assertEqual(freeze(restored), freeze(state))
        self.assertNotEqual(freeze(state), freeze(dict(reversed(list(state.items())))))
        self.assertNotEqual(freeze([1]), freeze([True]))
        self.assertNotEqual(freeze(0.0), freeze(-0.0))

    def test_mutable_aliases_cycles_and_unsupported_values_fail_closed(self):
        shared = {}
        cycle = []
        cycle.append(cycle)
        for state in ([shared, shared], cycle, {"x": float("nan")}, {"x": {1, 2}}):
            with self.subTest(state_type=type(state)):
                with self.assertRaises((ValueError, TypeError)):
                    freeze(state)


class AgendaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()

    def test_original_tools_and_cached_execution_agree_at_every_step(self):
        report = execution_smoke(self.registry)
        self.assertEqual(report["independent_physical_tool_calls"], 9)
        self.assertEqual(report["cached_physical_tool_calls"], 4)
        self.assertEqual(report["cache_hits"], 5)
        self.assertEqual(report["steps"][0]["tokens_left"], [26, 24, 26])
        self.assertEqual(report["steps"][1]["cache_hits"], [False, False, True])
        self.assertEqual(report["steps"][2]["cache_hits"], [False, True, True])
        histories = {b["private_history_sha256"] for b in report["final_branches"]}
        self.assertEqual(len(histories), 3)

    def test_real_write_query_delete_and_restore_preserve_aliases(self):
        original = APIBankEnvironment(databases(), self.registry)
        clone = original.fork()
        before = original.snapshot()
        clone.execute(parse_action(action_text("AddAgenda")))
        self.assertEqual(original.snapshot(), before)
        self.assertEqual(list(thaw(clone.snapshot())["Agenda"]), ["1", 2])
        restored = APIBankEnvironment.restore(clone.snapshot(), self.registry)
        result = restored.execute(parse_action(action_text("QueryAgenda")))
        self.assertEqual(result["output"], NEW_RECORD)
        # The upstream query returns a live record. The adapter must detach it.
        result["output"]["content"] = "mutated observation"
        self.assertEqual(restored.execute(parse_action(action_text("QueryAgenda")))["output"], NEW_RECORD)
        restored.execute(parse_action(action_text("DeleteAgenda")))
        self.assertEqual(restored.snapshot(), before)
        self.assertNotEqual(restored.snapshot(), clone.snapshot())

    def test_dictionary_order_changes_real_query_output(self):
        data = databases()
        first = dict(NEW_RECORD, location="first")
        second = dict(NEW_RECORD, location="second")
        data["Agenda"] = {1: first, "1": second}
        a = APIBankEnvironment(data, self.registry)
        data["Agenda"] = {"1": second, 1: first}
        b = APIBankEnvironment(data, self.registry)
        self.assertNotEqual(a.snapshot(), b.snapshot())
        action = parse_action(action_text("QueryAgenda"))
        self.assertEqual(a.execute(action)["output"]["location"], "first")
        self.assertEqual(b.execute(action)["output"]["location"], "second")

    def test_canonical_calls_ignore_only_json_format_and_keyword_order(self):
        a = parse_action(action_text("AddAgenda"))
        b = parse_action(action_text("AddAgenda", style=1))
        self.assertEqual(a, b)
        c = parse_action(action_text("AddAgenda", content=ARGUMENTS["content"] + " "))
        self.assertNotEqual(a, c)

    def test_invalid_calls_do_not_silently_coerce_or_drop_parameters(self):
        invalid = [
            '{"tool":"AddAgenda","tool":"DeleteAgenda","arguments":{}}',
            '{"tool":"Unknown","arguments":{}}',
            '{"final": 42}',
            '[]',
            '{"final":NaN}',
            action_text("AddAgenda", location=3),
            action_text("AddAgenda", extra="x"),
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidAction):
                    parse_action(raw)

    def test_key_captures_task_hidden_state_allowances_and_checker(self):
        parent = branches(self.registry)[0]
        mutations = (
            lambda b: setattr(b, "task_id", "other-task"),
            lambda b: setattr(b, "tokens_left", b.tokens_left - 1),
            lambda b: setattr(b, "actions_left", b.actions_left - 1),
            lambda b: b.checker_state["call_results"].append(False),
            lambda b: setattr(b, "termination", "final"),
            lambda b: setattr(b, "references", ()),
        )
        for mutate in mutations:
            child = parent.fork("child")
            mutate(child)
            self.assertNotEqual(parent.execution_key(), child.execution_key())
        changed = databases()
        changed["Account"]["demo"]["token"] = "different-synthetic-token"
        child = parent.fork("child")
        child.environment = APIBankEnvironment(changed, self.registry)
        self.assertNotEqual(parent.execution_key(), child.execution_key())
        private = parent.fork("child")
        private.history.append({"role": "assistant", "content": "private history"})
        self.assertEqual(parent.execution_key(), private.execution_key())
        self.assertNotEqual(parent.full_snapshot(), private.full_snapshot())

    def test_cache_hit_does_not_copy_history_counts_or_mutable_observations(self):
        a, b, _ = branches(self.registry)
        executor = Executor(TransitionCache())
        executor.step(a, action_text("AddAgenda"), 2)
        self.assertTrue(executor.step(b, action_text("AddAgenda"), 5).cache_hit)
        self.assertEqual((a.tokens_left, b.tokens_left), (28, 25))
        b.history[-1]["content"]["output"] = "locally edited"
        self.assertEqual(a.history[-1]["content"]["output"], "success")
        self.assertNotEqual(a.execution_key(), b.execution_key())
        self.assertNotEqual(a.history[0], b.history[0])

    def test_original_checker_semantics_are_used_including_its_looseness(self):
        env = APIBankEnvironment(databases(), self.registry)
        response = env.execute(parse_action(action_text("AddAgenda")))
        self.assertTrue(env.check_call(response, references()[0]))
        # AddAgenda uses Jaccard > .5, not exact content equality.
        loose = copy.deepcopy(references()[0])
        loose["input"]["content"] = "write meeting"
        self.assertTrue(env.check_call(response, loose))
        wrong = copy.deepcopy(loose)
        wrong["input"]["location"] = "wrong room"
        self.assertFalse(env.check_call(response, wrong))
        self.assertFalse(env.check_call(response, references()[1]))

    def test_failed_tool_is_cached_with_original_exception_and_no_mutation(self):
        a, b, _ = branches(self.registry)
        executor = Executor(TransitionCache())
        raw = action_text("AddAgenda", token="invalid-fixture-token")
        before = a.environment.snapshot()
        first = executor.step(a, raw, 3)
        second = executor.step(b, raw, 3)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.observation, second.observation)
        self.assertEqual(thaw(first.observation)["exception"], "The token is invalid.")
        self.assertFalse(first.checker_result)
        self.assertEqual(a.environment.snapshot(), before)
        self.assertEqual(b.environment.snapshot(), before)

    def test_invalid_final_and_truncated_actions_have_explicit_local_semantics(self):
        for raw, completed, tokens, expected, actions in (
            ("invalid", True, 30, "token_limit", 4),
            ('{"final":"done"}', True, 2, "final", 4),
            (action_text("AddAgenda"), False, 30, "token_limit", 5),
        ):
            branch = branches(self.registry)[0]
            before = branch.environment.snapshot()
            executor = Executor(TransitionCache())
            result = executor.step(branch, raw, tokens, completed=completed)
            self.assertEqual(result.termination, expected)
            self.assertEqual(branch.actions_left, actions)
            self.assertEqual(executor.physical_tool_calls, 0)
            self.assertEqual(branch.environment.snapshot(), before)
            with self.assertRaises(ValueError):
                executor.step(branch, action_text("AddAgenda"), 1)

    def test_call_completed_at_token_limit_executes_then_terminates(self):
        branch = branches(self.registry)[0]
        result = Executor().step(branch, action_text("AddAgenda"), branch.tokens_left)
        self.assertTrue(result.checker_result)
        self.assertEqual(result.termination, "token_limit")
        self.assertIn(2, thaw(branch.environment.snapshot())["Agenda"])

    def test_invalid_budget_requests_leave_branch_unchanged(self):
        branch = branches(self.registry)[0]
        snapshot = branch.full_snapshot()
        for tokens, completed in ((0, True), (31, True), (2, False), (True, True)):
            with self.assertRaises(ValueError):
                Executor().step(branch, action_text("AddAgenda"), tokens, completed)
            self.assertEqual(branch.full_snapshot(), snapshot)

    def test_zero_action_allowance_cannot_run(self):
        branch = Branch("b", "t", APIBankEnvironment(databases(), self.registry), 4, 0)
        self.assertEqual(branch.termination, "action_limit")
        with self.assertRaises(ValueError):
            Executor().step(branch, action_text("AddAgenda"), 1)


class CacheAndVendorTests(unittest.TestCase):
    def test_hash_collisions_require_full_byte_equality(self):
        cache = TransitionCache(digest=lambda _: b"forced-collision")
        one, two = Transition(b"state-a", b"obs-a"), Transition(b"state-b", b"obs-b")
        cache.put(b"key-a", one)
        cache.put(b"key-b", two)
        self.assertEqual(cache.get(b"key-a"), one)
        self.assertEqual(cache.get(b"key-b"), two)
        self.assertIsNone(cache.get(b"key-c"))
        with self.assertRaises(ValueError):
            cache.put(b"key-a", two)

    def test_loader_does_not_import_upstream_import_all_package(self):
        before = set(sys.modules)
        registry = load_registry()
        self.assertEqual(set(registry.tools), {"AddAgenda", "QueryAgenda", "DeleteAgenda"})
        new = set(sys.modules) - before
        self.assertFalse(any(name == "apis" or name.startswith("apis.") for name in new))

    def test_tampered_source_is_rejected_even_if_modules_were_previously_loaded(self):
        load_registry()
        # Bootstrap from already verified originals, with no network.
        roots = [p for p in DEFAULT_CACHE.iterdir() if (p / "LICENSE").is_file()]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".cache") as temp:
            destination = bootstrap(cache_root=Path(temp), checkout=roots[0])
            source = destination / "api-bank/apis/query_agenda.py"
            source.write_bytes(source.read_bytes() + b"\n# tampered\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_registry(cache_root=Path(temp))


if __name__ == "__main__":
    unittest.main()
