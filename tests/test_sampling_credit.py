import math
import random
import unittest
from fractions import Fraction

from rsi_curriculum.credit import CreditRound
from rsi_curriculum.sampling import (
    acceptance_probability, correct_proposal, coupled_sample, distribution,
    positive_residual, sample_categorical, softmax,
)


class FixedRandom:
    def __init__(self, *values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


class SamplingTests(unittest.TestCase):
    def test_probability_mass_matches_rational_oracle_on_small_simplex(self):
        vectors = [
            tuple(Fraction(v, 4) for v in (a, b, 4 - a - b))
            for a in range(5) for b in range(5 - a)
        ]
        for p in vectors:
            for q in vectors:
                pf, qf = tuple(map(float, p)), tuple(map(float, q))
                residual = positive_residual(pf, qf)
                rejected = sum(max(qv - pv, 0) for pv, qv in zip(p, q))
                if rejected == 0:
                    self.assertIsNone(residual)
                    continue
                for i in range(3):
                    accepted = (
                        q[i] * acceptance_probability(i, pf, qf) if q[i] else 0
                    )
                    output_mass = float(accepted) + float(rejected) * residual[i]
                    self.assertAlmostEqual(output_mass, float(p[i]), places=14)
                    expected_residual = max(p[i] - q[i], 0) / rejected
                    self.assertAlmostEqual(residual[i], float(expected_residual), places=14)

    def test_identical_distributions_never_draw_rejection_or_residual(self):
        p = (0.0, 0.4, 0.6)
        for token in (1, 2):
            result = correct_proposal(token, p, p, FixedRandom())
            self.assertTrue(result.accepted)
            self.assertEqual(result.token, token)
        self.assertIsNone(positive_residual(p, p))

    def test_disjoint_support_and_zero_probabilities(self):
        result = correct_proposal(0, (0, 0, 1), (1, 0, 0), FixedRandom(0, 0))
        self.assertFalse(result.accepted)
        self.assertEqual(result.token, 2)
        with self.assertRaises(ValueError):
            correct_proposal(1, (0, 1), (1, 0), FixedRandom())
        for token in (-1, 2, True):
            with self.assertRaises(ValueError):
                correct_proposal(token, (0.5, 0.5), (0.5, 0.5), FixedRandom())

    def test_acceptance_threshold_and_residual_paths(self):
        p, q = (0.25, 0.75), (0.5, 0.5)
        self.assertTrue(correct_proposal(0, p, q, FixedRandom(0.4999)).accepted)
        rejected = correct_proposal(0, p, q, FixedRandom(0.5, 0.99))
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.token, 1)

    def test_seeded_actual_sampling_preserves_target_marginal(self):
        p, q = (0.1, 0.2, 0.7, 0), (0.6, 0, 0.1, 0.3)
        rng = random.Random(113)
        counts = [0] * len(p)
        for _ in range(20000):
            counts[coupled_sample(p, q, rng).token] += 1
        self.assertEqual(counts[-1], 0)
        for expected, count in zip(p, counts):
            self.assertLess(abs(count / 20000 - expected), 0.02)

    def test_sampler_handles_boundaries_without_selecting_zero_mass(self):
        self.assertEqual(sample_categorical((0, 1, 0), FixedRandom(0)), 1)
        self.assertEqual(sample_categorical((0.4, 0.6, 0), FixedRandom(0.9999999999999999)), 1)
        self.assertEqual(sample_categorical((0.4, 0.6), FixedRandom(0.4)), 1)

    def test_softmax_is_stable_and_uses_complete_vector(self):
        result = softmax((10000, 10001, -math.inf), temperature=2)
        self.assertAlmostEqual(math.fsum(result), 1)
        self.assertAlmostEqual(result[1] / result[0], math.exp(0.5))
        self.assertEqual(result[2], 0)
        self.assertEqual(len(result), 3)

    def test_invalid_probability_vectors_and_logits_are_rejected(self):
        for values in ((), (0, 0), (0.2, 0.3), (-0.1, 1.1), (math.nan, 0), (math.inf, 0)):
            with self.assertRaises(ValueError):
                distribution(values)
        with self.assertRaises(ValueError):
            positive_residual((1,), (0.5, 0.5))
        for logits in ((), (math.nan, 0), (math.inf, 0), (-math.inf, -math.inf)):
            with self.assertRaises(ValueError):
                softmax(logits)
        for temperature in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                softmax((0, 1), temperature)


class CreditTests(unittest.TestCase):
    def new_round(self):
        round_ = CreditRound(("a", "b"), random.Random(17))
        round_.draw_probes(lambda: ("p0", "p1", "p2", "p3"))
        return round_

    def scores(self):
        return (
            {"p0": 0, "p1": 1, "p2": 0, "p3": 1},
            {
                "a": {"p0": 1, "p1": 1, "p2": 0, "p3": 1},
                "b": {"p0": 0, "p1": 0, "p2": 0, "p3": 1},
            },
        )

    def test_commitment_precedes_probe_draw_and_does_not_select_best_score(self):
        events = []

        class CommitLast:
            def randrange(self, n):
                events.append("commit")
                return n - 1

        round_ = CreditRound(("a", "b"), CommitLast())
        round_.draw_probes(lambda: events.append("draw") or ("p",))
        result = round_.compute({"p": 0}, {"a": {"p": 1}, "b": {"p": 0}})
        self.assertEqual(events, ["commit", "draw"])
        self.assertEqual(result.committed_branch, "b")
        self.assertGreater(result.weights["a"], result.weights["b"])

    def test_paired_gain_and_stable_exponential_weights(self):
        baseline, candidates = self.scores()
        result = self.new_round().compute(baseline, candidates, lambda_credit=2)
        self.assertEqual(result.credits, {"a": 0.25, "b": -0.25})
        self.assertAlmostEqual(result.weights["a"], 1 / (1 + math.exp(-1)))
        extreme = self.new_round().compute(baseline, candidates, lambda_credit=1e308)
        self.assertEqual(extreme.weights, {"a": 1.0, "b": 0.0})
        reversed_baseline = dict(reversed(list(baseline.items())))
        self.assertEqual(self.new_round().compute(reversed_baseline, candidates).credits, result.credits)

    def test_scores_cannot_be_submitted_before_probes_or_twice(self):
        round_ = CreditRound(("a", "b"), random.Random(1))
        with self.assertRaisesRegex(ValueError, "draw probes"):
            round_.compute(*self.scores())
        round_ = self.new_round()
        round_.compute(*self.scores())
        with self.assertRaisesRegex(ValueError, "finalized"):
            round_.compute(*self.scores())

    def test_test_split_cannot_be_used_for_credit(self):
        round_ = CreditRound(("a", "b"), random.Random(1))
        calls = []
        with self.assertRaises(ValueError):
            round_.draw_probes(lambda: calls.append("called"), split="test")
        self.assertEqual(calls, [])

    def test_missing_failed_branches_and_misaligned_probes_are_rejected(self):
        baseline, candidates = self.scores()
        with self.assertRaisesRegex(ValueError, "every candidate"):
            self.new_round().compute(baseline, {"a": candidates["a"]})
        for changed in (
            {"p0": 0, "p1": 0, "p2": 0},
            {"p0": 0, "p1": 0, "p2": 0, "different-task": 1},
            {"p0": 0, "p1": math.nan, "p2": 0, "p3": 1},
            {"p0": 0, "p1": 2, "p2": 0, "p3": 1},
        ):
            with self.assertRaises(ValueError):
                self.new_round().compute(baseline, dict(candidates, b=changed))

    def test_probe_occurrences_and_round_configuration_are_validated(self):
        for ids in (("a",), ("a", "a"), ("a", "")):
            with self.assertRaises(ValueError):
                CreditRound(ids, random.Random(1))
        for probes in ((), ("p", "p")):
            with self.assertRaises(ValueError):
                CreditRound(("a", "b"), random.Random(1)).draw_probes(lambda: probes)
        with self.assertRaises(ValueError):
            self.new_round().draw_probes(lambda: ("p",))
        for strength in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.new_round().compute(*self.scores(), lambda_credit=strength)


if __name__ == "__main__":
    unittest.main()
