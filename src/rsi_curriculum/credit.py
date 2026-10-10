"""Staged precommitment and paired credit arithmetic; no model optimization."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CreditReport:
    committed_branch: str
    probe_ids: tuple
    credits: dict
    weights: dict


class CreditRound:
    """Declare frozen candidate IDs and draw commitment before requesting probes.

    IDs are audit labels, not evidence that model/optimizer snapshots were
    restored or frozen. The future training backend must enforce that contract.
    """

    def __init__(self, frozen_candidate_ids, rng):
        candidates = tuple(frozen_candidate_ids)
        if (
            len(candidates) < 2
            or any(type(v) is not str or not v for v in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("At least two unique frozen candidate IDs are required")
        self.candidates = candidates
        self.committed_branch = candidates[rng.randrange(len(candidates))]
        self.probe_ids = None
        self._computed = False

    def draw_probes(self, draw, split="development"):
        if split != "development":
            raise ValueError("Credit feedback is restricted to development probes")
        if self.probe_ids is not None:
            raise ValueError("Probes already drawn")
        probes = tuple(draw())
        # IDs identify draw occurrences; repeated underlying tasks are allowed.
        if (
            not probes or any(type(v) is not str or not v for v in probes)
            or len(set(probes)) != len(probes)
        ):
            raise ValueError("Use nonempty unique probe occurrence IDs")
        self.probe_ids = probes
        return probes

    def compute(self, incumbent_scores, candidate_scores, lambda_credit=1.0):
        if self.probe_ids is None:
            raise ValueError("Precommit and draw probes before submitting scores")
        if self._computed:
            raise ValueError("Credit round already finalized")
        if not math.isfinite(lambda_credit) or lambda_credit <= 0:
            raise ValueError("lambda_credit must be finite and positive")
        if set(candidate_scores) != set(self.candidates):
            raise ValueError("Keep scores from every candidate, including failures")

        def aligned(scores):
            if set(scores) != set(self.probe_ids):
                raise ValueError("All branches must score the same probe occurrences")
            values = tuple(float(scores[k]) for k in self.probe_ids)
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Scores must be finite and in [0, 1]")
            return values

        incumbent = aligned(incumbent_scores)
        credits = {}
        for candidate in self.candidates:
            updated = aligned(candidate_scores[candidate])
            credits[candidate] = math.fsum(
                new - old for new, old in zip(updated, incumbent)
            ) / len(self.probe_ids)
        maximum = max(credits.values())
        unnormalized = {
            key: math.exp(lambda_credit * (value - maximum))
            for key, value in credits.items()
        }
        total = math.fsum(unnormalized.values())
        weights = {key: value / total for key, value in unnormalized.items()}
        self._computed = True
        return CreditReport(self.committed_branch, self.probe_ids, credits, weights)
