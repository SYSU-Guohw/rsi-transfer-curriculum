"""A small auditable curriculum generator trained from transfer credit."""

import dataclasses
import math

from .sampling import sample_categorical, softmax
from .tasks import FAMILIES


@dataclasses.dataclass(frozen=True)
class CurriculumBatch:
    branch_id: str
    families: tuple
    tasks: tuple

    @property
    def counts(self):
        return {family: self.families.count(family) for family in FAMILIES}


class CurriculumGenerator:
    """Categorical task-family generator with an Adam policy-gradient update."""

    def __init__(
        self,
        logits=None,
        initial_probabilities=None,
        first_moment=None,
        second_moment=None,
        step=0,
    ):
        self.logits = {
            family: float((logits or {}).get(family, 0.0)) for family in FAMILIES
        }
        uniform = 1.0 / len(FAMILIES)
        self.initial_probabilities = {
            family: float((initial_probabilities or {}).get(family, uniform))
            for family in FAMILIES
        }
        if not math.isclose(
            math.fsum(self.initial_probabilities.values()), 1.0, abs_tol=1e-12
        ):
            raise ValueError("Initial probabilities must sum to one")
        if any(value <= 0 for value in self.initial_probabilities.values()):
            raise ValueError("Initial probabilities must have full support")
        self.first_moment = {
            family: float((first_moment or {}).get(family, 0.0)) for family in FAMILIES
        }
        self.second_moment = {
            family: float((second_moment or {}).get(family, 0.0)) for family in FAMILIES
        }
        self.step = int(step)
        if self.step < 0:
            raise ValueError("step must be nonnegative")

    @property
    def probabilities(self):
        values = softmax(tuple(self.logits[family] for family in FAMILIES))
        return dict(zip(FAMILIES, values))

    def sample_batch(self, branch_id, count, rng, factory, round_index):
        probabilities = tuple(self.probabilities[family] for family in FAMILIES)
        families = tuple(
            FAMILIES[sample_categorical(probabilities, rng)] for _ in range(count)
        )
        tasks = factory.sample_mixture(
            "train",
            families,
            count,
            rng,
            f"round-{round_index}:{branch_id}",
        )
        return CurriculumBatch(branch_id, families, tuple(tasks))

    def update(
        self,
        batches,
        credits,
        learning_rate=0.05,
        kl_coefficient=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        batches = tuple(batches)
        if not batches or set(credits) != {batch.branch_id for batch in batches}:
            raise ValueError("Credits must align with every curriculum batch")
        for value in (learning_rate, beta1, beta2, epsilon):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Optimizer values must be finite and positive")
        if beta1 >= 1 or beta2 >= 1:
            raise ValueError("Adam decay values must be less than one")
        if not math.isfinite(kl_coefficient) or kl_coefficient < 0:
            raise ValueError("kl_coefficient must be finite and nonnegative")
        rewards = {key: float(value) for key, value in credits.items()}
        if any(not math.isfinite(value) for value in rewards.values()):
            raise ValueError("Credits must be finite")

        baseline = math.fsum(rewards.values()) / len(rewards)
        probabilities = self.probabilities
        gradient = {family: 0.0 for family in FAMILIES}
        for batch in batches:
            centered = rewards[batch.branch_id] - baseline
            count = len(batch.families)
            counts = batch.counts
            for family in FAMILIES:
                gradient[family] += (
                    centered
                    * (counts[family] - count * probabilities[family])
                    / len(batches)
                )

        kl = math.fsum(
            probabilities[family]
            * math.log(probabilities[family] / self.initial_probabilities[family])
            for family in FAMILIES
        )
        for family in FAMILIES:
            kl_gradient = probabilities[family] * (
                math.log(probabilities[family] / self.initial_probabilities[family])
                - kl
            )
            gradient[family] -= kl_coefficient * kl_gradient

        self.step += 1
        for family in FAMILIES:
            value = gradient[family]
            self.first_moment[family] = (
                beta1 * self.first_moment[family] + (1 - beta1) * value
            )
            self.second_moment[family] = (
                beta2 * self.second_moment[family] + (1 - beta2) * value * value
            )
            corrected_first = self.first_moment[family] / (1 - beta1**self.step)
            corrected_second = self.second_moment[family] / (1 - beta2**self.step)
            self.logits[family] += (
                learning_rate
                * corrected_first
                / (math.sqrt(corrected_second) + epsilon)
            )
        # Softmax is shift invariant; centering keeps serialized values bounded.
        mean_logit = math.fsum(self.logits.values()) / len(self.logits)
        self.logits = {
            family: value - mean_logit for family, value in self.logits.items()
        }
        return {
            "credits": rewards,
            "baseline": baseline,
            "gradient": gradient,
            "kl_to_initial": kl,
            "probabilities": self.probabilities,
        }

    def state_dict(self):
        return {
            "logits": dict(self.logits),
            "initial_probabilities": dict(self.initial_probabilities),
            "first_moment": dict(self.first_moment),
            "second_moment": dict(self.second_moment),
            "step": self.step,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            logits=state["logits"],
            initial_probabilities=state["initial_probabilities"],
            first_moment=state["first_moment"],
            second_moment=state["second_moment"],
            step=state["step"],
        )
