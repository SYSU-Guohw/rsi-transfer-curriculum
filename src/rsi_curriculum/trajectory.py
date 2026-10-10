"""Execute free-form model completions and score complete Agenda trajectories."""

import dataclasses
import math

from .environment import APIBankEnvironment
from .execution import Branch, Executor
from .state import thaw
from .tasks import TaskSpec


@dataclasses.dataclass(frozen=True)
class TrajectoryScore:
    reward: float
    success: float
    call_score: float
    state_match: bool
    final_present: bool
    attempted_actions: int
    correct_calls: int
    expected_calls: int
    termination: str
    observations: tuple

    def to_dict(self):
        value = dataclasses.asdict(self)
        value["observations"] = [thaw(observation) for observation in self.observations]
        return value


def _json_objects(text):
    """Split whitespace-separated top-level JSON objects without normalizing them."""
    if type(text) is not str:
        raise TypeError("completion must be text")
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index == length:
            return
        if text[index] != "{":
            yield text[index:].strip()
            return
        start = index
        depth = 0
        quoted = False
        escaped = False
        while index < length:
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    index += 1
                    yield text[start:index]
                    break
                if depth < 0:
                    yield text[start:].strip()
                    return
            index += 1
        else:
            yield text[start:].strip()
            return


def execute_completion(
    task, completion, registry, max_actions=None, transition_cache=None
):
    if not isinstance(task, TaskSpec):
        raise TypeError("task must be a TaskSpec")
    action_limit = max_actions or (len(task.calls) + 2)
    if type(action_limit) is not int or action_limit <= 0:
        raise ValueError("max_actions must be positive")
    branch = Branch(
        branch_id="scoring",
        task_id=task.task_id,
        environment=APIBankEnvironment(task.databases, registry),
        tokens_left=action_limit + 1,
        actions_left=action_limit,
        references=task.references,
        history=task.messages,
    )
    executor = Executor(transition_cache)
    observations = []
    attempted = 0
    malformed = False
    final_present = False

    actions = list(_json_objects(completion))
    if not actions:
        malformed = True
    for index, raw in enumerate(actions):
        if branch.termination is not None:
            malformed = True
            break
        result = executor.step(branch, raw, generated_tokens=1)
        observations.append(result.observation)
        if result.checker_result is not None:
            attempted += 1
        if result.termination == "final":
            final_present = True
            if index != len(actions) - 1:
                malformed = True
            break
        if result.termination in ("action_limit", "token_limit"):
            break
        if result.checker_result is False and raw.lstrip()[:1] != "{":
            malformed = True

    expected = len(task.calls)
    results = tuple(branch.checker_state["call_results"])
    correct = sum(value is True for value in results)
    denominator = max(expected, attempted, 1)
    call_score = correct / denominator
    state_match = branch.environment.snapshot() == task.expected_snapshot(registry)
    reward = math.fsum(
        (
            0.7 * call_score,
            0.2 * float(state_match),
            0.1 * float(final_present),
        )
    )
    success = float(
        not malformed
        and final_present
        and state_match
        and attempted == expected
        and len(results) == expected
        and all(results)
    )
    return TrajectoryScore(
        reward=reward,
        success=success,
        call_score=call_score,
        state_match=state_match,
        final_present=final_present,
        attempted_actions=attempted,
        correct_calls=correct,
        expected_calls=expected,
        termination=branch.termination or "incomplete",
        observations=tuple(observations),
    )


def completion_content(completion):
    """Normalize TRL's standard or conversational completion representation."""
    if type(completion) is str:
        return completion
    if type(completion) is list:
        parts = []
        for message in completion:
            if type(message) is dict and message.get("role") == "assistant":
                content = message.get("content", "")
                if type(content) is str:
                    parts.append(content)
        return "\n".join(parts)
    if type(completion) is dict and type(completion.get("content")) is str:
        return completion["content"]
    return ""


def make_grpo_reward(registry, metric="reward", transition_cache=None):
    if metric not in ("reward", "success"):
        raise ValueError("metric must be reward or success")

    def reward(completions, task_json, **_):
        values = []
        for completion, encoded_task in zip(completions, task_json):
            task = TaskSpec.from_json(encoded_task)
            score = execute_completion(
                task,
                completion_content(completion),
                registry,
                transition_cache=transition_cache,
            )
            values.append(getattr(score, metric))
        return values

    reward.__name__ = "agenda_" + metric
    return reward
