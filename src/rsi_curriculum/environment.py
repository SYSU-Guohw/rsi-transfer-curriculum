"""Deterministic agenda-only adapter; this is not a full API-Bank evaluator."""

import copy
import json
from dataclasses import dataclass

from .state import freeze, thaw


class InvalidAction(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    kind: str
    name: str
    payload: bytes


def parse_action(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidAction("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise InvalidAction("non-finite JSON value")

    try:
        value = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=invalid_constant
        )
    except (ValueError, TypeError) as exc:
        raise InvalidAction("malformed action JSON") from exc
    if type(value) is not dict:
        raise InvalidAction("action must be a JSON object")
    if set(value) == {"final"} and type(value["final"]) is str:
        return Action("final", "", freeze(value["final"]))
    if set(value) != {"tool", "arguments"}:
        raise InvalidAction("expected tool/arguments or final")
    name, arguments = value["tool"], value["arguments"]
    if type(name) is not str or name not in ("AddAgenda", "QueryAgenda", "DeleteAgenda"):
        raise InvalidAction("unsupported tool")
    if type(arguments) is not dict or set(arguments) != {"token", "content", "time", "location"}:
        raise InvalidAction("unexpected tool arguments")
    if any(type(v) is not str for v in arguments.values()):
        raise InvalidAction("tool arguments must be strings")
    # Only keyword order is immaterial. Keep parameter values byte-for-byte.
    return Action("tool", name, freeze({k: arguments[k] for k in sorted(arguments)}))


class APIBankEnvironment:
    def __init__(self, databases, registry):
        if type(databases) is not dict or set(databases) != {"Account", "Agenda"}:
            raise ValueError("Only explicit Account and Agenda databases are supported")
        if any(type(db) is not dict for db in databases.values()):
            raise ValueError("Databases must be dictionaries")
        self.registry = registry
        self._databases = thaw(freeze(databases))
        checker = registry.token_checker(init_database=self._databases["Account"])
        self._tools = {
            name: cls(init_database=self._databases["Agenda"], token_checker=checker)
            for name, cls in registry.tools.items()
        }

    @property
    def identity(self):
        return "agenda-adapter-v1:" + self.registry.identity

    def snapshot(self):
        return freeze(self._databases)

    def fork(self):
        return self.restore(self.snapshot(), self.registry)

    @classmethod
    def restore(cls, snapshot, registry):
        return cls(thaw(snapshot), registry)

    def execute(self, action):
        if action.kind != "tool" or action.name not in self._tools:
            raise InvalidAction("not a supported tool call")
        # QueryAgenda returns a database record by reference. Detach it here.
        return copy.deepcopy(self._tools[action.name].call(**thaw(action.payload)))

    def check_call(self, response, reference):
        name = response.get("api_name")
        if name not in self._tools or name != reference.get("api_name"):
            return False
        return bool(self._tools[name].check_api_call_correctness(
            copy.deepcopy(response), copy.deepcopy(reference)
        ))
