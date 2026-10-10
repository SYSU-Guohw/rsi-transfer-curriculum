"""Deterministic, split-safe curriculum tasks for the Agenda API subset."""

import copy
import hashlib
import json
import random
from dataclasses import dataclass

from .environment import APIBankEnvironment, parse_action

FAMILIES = (
    "add",
    "query",
    "delete",
    "add_query",
    "query_delete",
    "replace",
)
SPLITS = ("train", "development", "test")
_SPLIT_OFFSETS = {"train": 100_000, "development": 200_000, "test": 300_000}


@dataclass(frozen=True)
class ToolCallSpec:
    name: str
    arguments: dict
    output: object

    def reference(self):
        return {
            "api_name": self.name,
            "input": copy.deepcopy(self.arguments),
            "output": copy.deepcopy(self.output),
            "exception": None,
        }

    def action_text(self):
        return json.dumps(
            {"tool": self.name, "arguments": self.arguments},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "arguments": copy.deepcopy(self.arguments),
            "output": copy.deepcopy(self.output),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["name"],
            copy.deepcopy(value["arguments"]),
            copy.deepcopy(value["output"]),
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    family: str
    prompt: str
    databases: dict
    calls: tuple

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError("Unknown task split")
        if self.family not in FAMILIES:
            raise ValueError("Unknown task family")
        if not self.calls:
            raise ValueError("A task must contain at least one expected call")

    @property
    def messages(self):
        return [
            {
                "role": "system",
                "content": (
                    "You are an API agent. Return only consecutive JSON objects. "
                    "Each tool action must contain exactly tool and arguments. "
                    "After all required actions, return one final object with exactly "
                    'the key "final". Do not use Markdown or commentary.'
                ),
            },
            {"role": "user", "content": self.prompt},
        ]

    @property
    def references(self):
        return tuple(call.reference() for call in self.calls)

    def oracle_completion(self):
        actions = [call.action_text() for call in self.calls]
        actions.append(json.dumps({"final": "Task completed."}, separators=(",", ":")))
        return "\n".join(actions)

    def expected_snapshot(self, registry):
        environment = APIBankEnvironment(self.databases, registry)
        for call in self.calls:
            environment.execute(parse_action(call.action_text()))
        return environment.snapshot()

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "split": self.split,
            "family": self.family,
            "prompt": self.prompt,
            "databases": copy.deepcopy(self.databases),
            "calls": [call.to_dict() for call in self.calls],
        }

    def to_json(self):
        return json.dumps(
            self.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_dict(cls, value):
        return cls(
            task_id=value["task_id"],
            split=value["split"],
            family=value["family"],
            prompt=value["prompt"],
            databases=copy.deepcopy(value["databases"]),
            calls=tuple(ToolCallSpec.from_dict(call) for call in value["calls"]),
        )

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))


class AgendaTaskFactory:
    """Create reproducible tasks without sharing entities across data splits."""

    def __init__(self, seed=20261010):
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        self.seed = seed

    def task(self, split, family, ordinal):
        if split not in SPLITS or family not in FAMILIES:
            raise ValueError("Unknown split or family")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        global_id = _SPLIT_OFFSETS[split] + ordinal
        digest = hashlib.sha256(
            f"{self.seed}:{split}:{family}:{ordinal}".encode()
        ).hexdigest()
        username = f"user-{split}-{global_id}"
        token = "tok-" + digest[:20]
        target = {
            "username": username,
            "content": f"{family} agenda {digest[20:30]}",
            "time": f"2027-{1 + global_id % 12:02d}-{1 + global_id % 27:02d} "
            f"{8 + global_id % 10:02d}:{(global_id * 7) % 60:02d}:00",
            "location": f"Room-{digest[30:36]}",
        }
        replacement = {
            "username": username,
            "content": target["content"] + " revised",
            "time": target["time"],
            "location": target["location"],
        }
        distractor = {
            "username": username,
            "content": f"unrelated agenda {digest[36:44]}",
            "time": "2027-12-28 17:30:00",
            "location": "Archive",
        }
        databases = {
            "Account": {username: {"token": token}},
            "Agenda": {"seed": distractor},
        }

        def arguments(record):
            return {
                "token": token,
                "content": record["content"],
                "time": record["time"],
                "location": record["location"],
            }

        add = ToolCallSpec("AddAgenda", arguments(target), "success")
        query = ToolCallSpec("QueryAgenda", arguments(target), target)
        delete = ToolCallSpec("DeleteAgenda", arguments(target), "success")
        if family in ("query", "delete", "query_delete", "replace"):
            databases["Agenda"]["target"] = copy.deepcopy(target)

        if family == "add":
            calls = (add,)
            request = "Add the new agenda item."
        elif family == "query":
            calls = (query,)
            request = "Query the matching agenda item."
        elif family == "delete":
            calls = (delete,)
            request = "Delete the matching agenda item."
        elif family == "add_query":
            calls = (add, query)
            request = "Add the agenda item, then query it to verify the stored record."
        elif family == "query_delete":
            calls = (query, delete)
            request = "Query the agenda item, then delete that same item."
        else:
            replacement_add = ToolCallSpec(
                "AddAgenda", arguments(replacement), "success"
            )
            calls = (delete, replacement_add)
            request = "Delete the existing agenda item, then add the revised item."

        details = (
            target
            if family != "replace"
            else {
                "existing": target,
                "revised": replacement,
            }
        )
        prompt = (
            request
            + " Use token "
            + token
            + ". Task data: "
            + json.dumps(details, ensure_ascii=True, sort_keys=True)
            + ". Preserve every string exactly."
        )
        return TaskSpec(
            task_id=f"{split}-{family}-{global_id}-{digest[:10]}",
            split=split,
            family=family,
            prompt=prompt,
            databases=databases,
            calls=calls,
        )

    def balanced_pool(self, split, size):
        if type(size) is not int or size <= 0:
            raise ValueError("size must be a positive integer")
        return [
            self.task(split, FAMILIES[index % len(FAMILIES)], index)
            for index in range(size)
        ]

    def sample_family(self, split, family, count, rng, namespace):
        if type(count) is not int or count <= 0:
            raise ValueError("count must be positive")
        if type(namespace) is not str or not namespace:
            raise ValueError("namespace must be nonempty")
        base = int(hashlib.sha256(namespace.encode()).hexdigest()[:12], 16)
        ordinals = rng.sample(range(base, base + max(10 * count, count)), count)
        return [self.task(split, family, ordinal) for ordinal in ordinals]

    def sample_mixture(self, split, families, count, rng, namespace):
        families = tuple(families)
        if len(families) != count or any(f not in FAMILIES for f in families):
            raise ValueError("Provide one valid family per requested task")
        base = int(hashlib.sha256(namespace.encode()).hexdigest()[:12], 16)
        return [
            self.task(split, family, base + index + rng.randrange(1_000_000))
            for index, family in enumerate(families)
        ]


def draw_probes(factory, round_index, count, seed):
    rng = random.Random(f"{seed}:probes:{round_index}")
    families = [FAMILIES[rng.randrange(len(FAMILIES))] for _ in range(count)]
    return factory.sample_mixture(
        "development", families, count, rng, f"probe-round-{round_index}"
    )
