"""Typed snapshots preserving dictionary insertion order and key types.

Supported states are trees. Mutable aliasing/cycles are rejected, not silently
flattened. The environment adapter reconstructs its known tool/database aliases.
"""

import json
import math


def freeze(value):
    seen = set()

    def encode(item):
        kind = type(item)
        if kind in (dict, list):
            if id(item) in seen:
                raise ValueError("Mutable aliases and cycles are unsupported")
            seen.add(id(item))
        if item is None:
            return ["none"]
        if kind is bool:
            return ["bool", item]
        if kind is int:
            return ["int", str(item)]
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("Non-finite state")
            return ["float", item.hex()]
        if kind is str:
            return ["str", item]
        if kind is bytes:
            return ["bytes", item.hex()]
        if kind in (list, tuple):
            return ["list" if kind is list else "tuple", [encode(v) for v in item]]
        if kind is dict:
            return ["dict", [[encode(k), encode(v)] for k, v in item.items()]]
        raise TypeError("Unsupported state type: " + kind.__name__)

    return json.dumps(encode(value), ensure_ascii=True, separators=(",", ":")).encode()


def thaw(snapshot):
    def decode(node):
        tag = node[0]
        if tag == "none":
            return None
        if tag == "bool":
            return node[1]
        if tag == "int":
            return int(node[1])
        if tag == "float":
            return float.fromhex(node[1])
        if tag == "str":
            return node[1]
        if tag == "bytes":
            return bytes.fromhex(node[1])
        if tag in ("list", "tuple"):
            values = [decode(v) for v in node[1]]
            return values if tag == "list" else tuple(values)
        if tag == "dict":
            result = {}
            for k, v in node[1]:
                key = decode(k)
                if key in result:
                    raise ValueError("Duplicate snapshot key")
                result[key] = decode(v)
            return result
        raise ValueError("Unknown snapshot tag")

    value = decode(json.loads(snapshot))
    if freeze(value) != snapshot:
        raise ValueError("Non-canonical snapshot")
    return value
