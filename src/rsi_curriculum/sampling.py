"""Full categorical acceptance/residual reference, using Python float arithmetic.

The caller must supply the actual proposal q and target p under the target's
private accepted prefix. This module does not implement model scoring or KV
rollback and cannot check whether an input vector omitted vocabulary entries.
"""

import math
from dataclasses import dataclass


def distribution(values):
    values = tuple(float(v) for v in values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Expected finite nonnegative probabilities")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Expected a normalized, full probability vector")
    return tuple(v / total for v in values)


def softmax(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = tuple(float(v) for v in logits)
    if (
        not logits or any(math.isnan(v) or v == math.inf for v in logits)
        or max(logits) == -math.inf
    ):
        raise ValueError("Invalid logits")
    maximum = max(logits)
    weights = tuple(math.exp((v - maximum) / temperature) for v in logits)
    total = math.fsum(weights)
    return tuple(v / total for v in weights)


def _pair(p, q):
    p, q = distribution(p), distribution(q)
    if len(p) != len(q):
        raise ValueError("Vocabulary sizes differ")
    return p, q


def _draw(probabilities, rng):
    threshold = rng.random()
    if not 0 <= threshold < 1:
        raise ValueError("RNG must return a value in [0, 1)")
    cumulative = 0.0
    last_positive = None
    for i, probability in enumerate(probabilities):
        if probability > 0:
            last_positive = i
        cumulative += probability
        if threshold < cumulative:
            return i
    # Roundoff near one must never select a zero-probability trailing token.
    return last_positive


def sample_categorical(probabilities, rng):
    return _draw(distribution(probabilities), rng)


def acceptance_probability(token, p, q):
    p, q = _pair(p, q)
    if type(token) is not int or not 0 <= token < len(q) or q[token] == 0:
        raise ValueError("Proposed token must have positive proposal probability")
    return min(1.0, p[token] / q[token])


def positive_residual(p, q):
    p, q = _pair(p, q)
    weights = tuple(max(pv - qv, 0.0) for pv, qv in zip(p, q))
    mass = math.fsum(weights)
    if mass == 0:
        return None
    return tuple(w / mass for w in weights)


@dataclass(frozen=True)
class Correction:
    proposed: int
    token: int
    accepted: bool


def correct_proposal(token, p, q, rng):
    p, q = _pair(p, q)
    if type(token) is not int or not 0 <= token < len(q) or q[token] == 0:
        raise ValueError("Proposed token must have positive proposal probability")
    # Identical distributions require neither residual division nor an RNG draw.
    if p[token] >= q[token] or rng.random() < p[token] / q[token]:
        return Correction(token, token, True)
    residual = positive_residual(p, q)
    if residual is None:
        raise ArithmeticError("Rejection with zero residual mass")
    return Correction(token, _draw(residual, rng), False)


def coupled_sample(p, q, rng):
    p, q = _pair(p, q)
    return correct_proposal(_draw(q, rng), p, q, rng)
