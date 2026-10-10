"""Load a hash-verified, narrow API-Bank subset without importing apis/__init__."""

import argparse
import hashlib
import json
import sys
import types
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = PROJECT_ROOT / "upstream.lock.json"
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "api_bank"
MODULES = ("api", "check_token", "add_agenda", "query_agenda", "delete_agenda")
TOOLS = {
    "AddAgenda": "add_agenda",
    "QueryAgenda": "query_agenda",
    "DeleteAgenda": "delete_agenda",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _spec(lock_path):
    return json.loads(Path(lock_path).read_text())["repositories"]["api_bank"]


def _verified(data, expected, name):
    if sha256(data) != expected:
        raise ValueError("API-Bank source hash mismatch: " + name)
    return data


def _adapt(source, module):
    """The sole source change is a package-relative API base-class import."""
    if module == "api":
        return source
    old = b"from apis.api import API"
    if source.count(old) != 1:
        raise ValueError("Unexpected API-Bank import layout")
    return source.replace(old, b"from .api import API", 1)


def bootstrap(cache_root=DEFAULT_CACHE, lock_path=DEFAULT_LOCK, checkout=None):
    """Copy verified originals from a checkout, or fetch only the pinned files."""
    spec = _spec(lock_path)
    destination = Path(cache_root) / spec["commit"]
    originals = {}
    # Validate the whole set before writing anything.
    for relative, expected in spec["selected_files_sha256"].items():
        if checkout is not None:
            data = (Path(checkout) / relative).read_bytes()
        else:
            url = (
                "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/"
                + spec["commit"] + "/" + relative
            )
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
        originals[relative] = _verified(data, expected, relative)
    provenance = {
        "repository": spec["url"],
        "commit": spec["commit"],
        "license": spec["license"],
        "original_sha256": {},
        "executed_source_sha256": {},
        "adaptation": "Only from apis.api import API -> from .api import API",
    }
    for relative, data in originals.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        provenance["original_sha256"][relative] = sha256(data)
        if relative.endswith(".py"):
            provenance["executed_source_sha256"][relative] = sha256(
                _adapt(data, Path(relative).stem)
            )
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return destination


class Registry:
    def __init__(self, tools, token_checker, identity):
        self.tools = tools
        self.token_checker = token_checker
        self.identity = identity


def load_registry(cache_root=DEFAULT_CACHE, lock_path=DEFAULT_LOCK):
    spec = _spec(lock_path)
    root = Path(cache_root) / spec["commit"]
    sources = {}
    # Recheck originals, including LICENSE, on every load. No network here.
    for relative, expected in spec["selected_files_sha256"].items():
        try:
            source = (root / relative).read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Run python -m rsi_curriculum.vendor bootstrap first: " + str(root)
            ) from exc
        _verified(source, expected, relative)
        if relative.endswith(".py"):
            sources[Path(relative).stem] = _adapt(source, Path(relative).stem)
    identity = spec["commit"] + ":" + sha256(
        b"\0".join(sources[name] for name in MODULES)
    )
    package_name = "_rsi_apibank_" + identity.replace(":", "_")
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
        try:
            for name in MODULES:
                full_name = package_name + "." + name
                module = types.ModuleType(full_name)
                module.__package__ = package_name
                module.__file__ = str(root / "api-bank" / "apis" / (name + ".py"))
                sys.modules[full_name] = module
                # Compile verified bytes directly; never execute stale bytecode
                # or the upstream package's import-all __init__.
                exec(compile(sources[name], module.__file__, "exec"), module.__dict__)
                setattr(package, name, module)
        except Exception:
            for name in list(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    del sys.modules[name]
            raise
    package = sys.modules[package_name]
    return Registry(
        {name: getattr(getattr(package, module), name) for name, module in TOOLS.items()},
        package.check_token.CheckToken,
        identity,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap"])
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    print(bootstrap(args.cache_root, args.lock, args.checkout))


if __name__ == "__main__":
    main()
