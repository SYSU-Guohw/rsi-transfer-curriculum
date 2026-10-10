"""Hand-authored smoke fixtures. No official API-Bank sample is used."""

import copy
import json

from .environment import APIBankEnvironment
from .execution import Branch


FIXTURE_TOKEN = "synthetic-demo-token"
OLD_RECORD = {
    "username": "demo",
    "content": "existing agenda item",
    "time": "2025-01-01 08:00:00",
    "location": "Room B",
}
NEW_RECORD = {
    "username": "demo",
    "content": "write meeting notes",
    "time": "2025-01-02 09:00:00",
    "location": "Room A",
}
ARGUMENTS = {
    "token": FIXTURE_TOKEN,
    "content": NEW_RECORD["content"],
    "time": NEW_RECORD["time"],
    "location": NEW_RECORD["location"],
}
TOOL_SEQUENCE = ("AddAgenda", "QueryAgenda", "DeleteAgenda")


def databases():
    return {
        "Account": {"demo": {"token": FIXTURE_TOKEN}},
        "Agenda": {"1": copy.deepcopy(OLD_RECORD)},
    }


def references():
    return tuple({
        "api_name": name,
        "input": copy.deepcopy(ARGUMENTS),
        "output": copy.deepcopy(NEW_RECORD) if name == "QueryAgenda" else "success",
        "exception": None,
    } for name in TOOL_SEQUENCE)


def action_text(name, style=0, **overrides):
    arguments = dict(ARGUMENTS, **overrides)
    if style == 1:
        arguments = dict(reversed(list(arguments.items())))
    return json.dumps(
        {"tool": name, "arguments": arguments},
        indent=2 if style == 1 else None,
        separators=None if style == 1 else (",", ":"),
    )


def branches(registry):
    return [
        Branch(
            branch_id="candidate-" + str(index),
            task_id="hand-authored-agenda-smoke-v1",
            environment=APIBankEnvironment(databases(), registry),
            tokens_left=30,
            actions_left=5,
            references=references(),
            history=[{"role": "user", "content": "Private fixture history " + str(index)}],
        )
        for index in range(3)
    ]
