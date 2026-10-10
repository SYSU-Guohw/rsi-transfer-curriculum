"""Independent execution and collision-checked deterministic transition reuse.

History never enters the tool implementation; it stays private. Only physical
effects and detached observations are cached. This CPU protocol uses caller-
supplied token counts and per-call fixture checks, not a trajectory success score.
"""

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from .environment import APIBankEnvironment, InvalidAction, parse_action
from .state import freeze, thaw


@dataclass
class Branch:
    branch_id: str
    task_id: str
    environment: APIBankEnvironment
    tokens_left: int
    actions_left: int
    references: tuple = ()
    history: list = field(default_factory=list)
    checker_state: dict = field(default_factory=lambda: {"cursor": 0, "call_results": []})
    termination: Optional[str] = None

    def __post_init__(self):
        for allowance in (self.tokens_left, self.actions_left):
            if type(allowance) is not int or allowance < 0:
                raise ValueError("Allowances must be nonnegative integers")
        self.references = copy.deepcopy(tuple(self.references))
        self.history = copy.deepcopy(self.history)
        self.checker_state = copy.deepcopy(self.checker_state)
        if self.termination is None:
            self.termination = self._limit_reason()

    def _limit_reason(self):
        if self.tokens_left == 0:
            return "token_limit"
        if self.actions_left == 0:
            return "action_limit"
        return None

    def execution_key(self):
        return freeze({
            "protocol": "cpu-json-actions-per-call-fixtures-v1",
            "environment_identity": self.environment.identity,
            "task_id": self.task_id,
            "environment": self.environment.snapshot(),
            "tokens_left": self.tokens_left,
            "actions_left": self.actions_left,
            "references": self.references,
            "checker_state": self.checker_state,
            "termination": self.termination,
        })

    def full_snapshot(self):
        return freeze({
            "execution": self.execution_key(),
            "branch_id": self.branch_id,
            "history": self.history,
        })

    def fork(self, branch_id):
        return Branch(
            branch_id, self.task_id, self.environment.fork(),
            self.tokens_left, self.actions_left, self.references,
            self.history, self.checker_state, self.termination,
        )


@dataclass(frozen=True)
class Transition:
    environment: bytes
    observation: bytes


class TransitionCache:
    def __init__(self, digest=None):
        self._digest = digest or (lambda key: hashlib.sha256(key).digest())
        self._buckets = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        for original, transition in self._buckets.get(self._digest(key), ()):
            if original == key:
                self.hits += 1
                return transition
        self.misses += 1
        return None

    def put(self, key, transition):
        bucket = self._buckets.setdefault(self._digest(key), [])
        for original, existing in bucket:
            if original == key:
                if existing != transition:
                    raise ValueError("Nondeterministic transition for identical state/call")
                return
        bucket.append((key, transition))


@dataclass(frozen=True)
class StepResult:
    observation: bytes
    cache_hit: bool
    checker_result: Optional[bool]
    termination: Optional[str]


class Executor:
    def __init__(self, cache=None):
        self.cache = cache
        self.physical_tool_calls = 0
        self.logical_tool_calls = 0

    def step(self, branch, raw_action, generated_tokens, completed=True):
        if branch.termination is not None:
            raise ValueError("A terminated branch cannot execute or rejoin")
        if type(raw_action) is not str or type(completed) is not bool:
            raise TypeError("Expected action text and a completion flag")
        if (
            type(generated_tokens) is not int
            or generated_tokens <= 0
            or generated_tokens > branch.tokens_left
        ):
            raise ValueError("Generated tokens must fit this branch's own allowance")
        if not completed and generated_tokens != branch.tokens_left:
            raise ValueError("A truncated action must exhaust the token allowance")

        cache_hit = False
        correctness = None
        action = None
        if not completed:
            observation = {"output": None, "exception": "token_limit_before_action"}
        else:
            try:
                action = parse_action(raw_action)
            except InvalidAction as exc:
                observation = {"output": None, "exception": "invalid_action: " + str(exc)}
            else:
                if action.kind == "final":
                    observation = {"final": thaw(action.payload)}
                else:
                    key = freeze((branch.execution_key(), action.name, action.payload))
                    transition = self.cache.get(key) if self.cache is not None else None
                    cache_hit = transition is not None
                    self.logical_tool_calls += 1
                    if transition is None:
                        observation = branch.environment.execute(action)
                        self.physical_tool_calls += 1
                        transition = Transition(
                            branch.environment.snapshot(), freeze(observation)
                        )
                        if self.cache is not None:
                            self.cache.put(key, transition)
                    else:
                        branch.environment = APIBankEnvironment.restore(
                            transition.environment, branch.environment.registry
                        )
                        observation = thaw(transition.observation)

        # Rebuild private history, costs and checker state after effect reuse.
        branch.tokens_left -= generated_tokens
        branch.history.append({"role": "assistant", "content": raw_action})
        if completed:
            branch.actions_left -= 1
            if action is None or action.kind == "tool":
                cursor = branch.checker_state["cursor"]
                if branch.references:
                    correctness = (
                        cursor < len(branch.references)
                        and action is not None
                        and branch.environment.check_call(observation, branch.references[cursor])
                    )
                    branch.checker_state["call_results"].append(bool(correctness))
                branch.checker_state["cursor"] += 1
                branch.history.append({"role": "tool", "content": copy.deepcopy(observation)})
        if not completed:
            branch.termination = "token_limit"
        elif action is not None and action.kind == "final":
            branch.termination = "final"
        else:
            branch.termination = branch._limit_reason()
        return StepResult(freeze(observation), cache_hit, correctness, branch.termination)
