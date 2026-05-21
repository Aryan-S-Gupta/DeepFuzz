from __future__ import annotations

"""Library-agnostic Stage 4 execution adapters.

The concrete Stage 4 runner is intentionally small, but these interfaces make
the extension points explicit: resolving an API, materializing inputs, invoking
the callable, normalizing outputs, comparing devices, and classifying/repairing
outcomes.  Library-specific behavior belongs in config or adapter packages
outside the core runner.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence, Tuple

from json2init.deepfuzz_common import import_api, resolve_python_object, summarize_python_value


class APIResolver(Protocol):
    def resolve(self, import_path: str) -> Any:
        ...


class InputMaterializer(Protocol):
    def materialize(self, init_obj: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any], List[str]]:
        ...


class Executor(Protocol):
    def invoke(self, callable_obj: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
        ...


class OutputNormalizer(Protocol):
    def normalize(self, value: Any) -> Dict[str, Any]:
        ...


class DeviceComparator(Protocol):
    def compare(self, callable_obj: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        ...


class BugOracle(Protocol):
    def classify(self, event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        ...


class RepairOracle(Protocol):
    def repair(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class DefaultAPIResolver:
    def resolve(self, import_path: str) -> Any:
        return import_api(import_path)


@dataclass
class DefaultInputMaterializer:
    def materialize(self, init_obj: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any], List[str]]:
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        reasons: List[str] = []
        backend = str(init_obj.get("backend", "python") or "python")
        for name, meta in (init_obj.get("params") or {}).items():
            if not isinstance(meta, dict) or not meta.get("include_in_base_call"):
                continue
            if meta.get("unresolved_reason"):
                reasons.append(f"{name}: {meta.get('unresolved_reason')}")
                continue
            try:
                value = resolve_python_object(meta.get("base_seed_spec", {}), backend=backend)
            except Exception as exc:
                reasons.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            kind = str(meta.get("call_kind", "") or "keyword")
            if kind == "positional_only":
                args.append(value)
            elif kind == "var_positional":
                args.extend(list(value) if isinstance(value, (list, tuple)) else [value])
            elif kind == "var_keyword" and isinstance(value, dict):
                kwargs.update(value)
            else:
                kwargs[name] = value
        return args, kwargs, reasons


@dataclass
class DefaultExecutor:
    def invoke(self, callable_obj: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
        return callable_obj(*args, **kwargs)


@dataclass
class DefaultOutputNormalizer:
    def normalize(self, value: Any) -> Dict[str, Any]:
        return summarize_python_value(value)


@dataclass
class NullDeviceComparator:
    reason: str = "only one compatible device is available"

    def compare(self, callable_obj: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"available": False, "reason": self.reason}
