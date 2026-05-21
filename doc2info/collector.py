#!/usr/bin/env python3
"""
collector.py

Collect public APIs from one or more Python packages by walking their module
object graphs and writing the results as JSONL.

Behavior:
- Only public symbols are collected. Any dotted path segment that starts with
  "_" is skipped.
- Traversal is depth-limited.
- Introspection is defensive: failures are captured as unresolved entries
  instead of stopping execution.
- Internal namespaces can be excluded with --exclude-prefix.
- Callables are not globally deduplicated by object id during collection.
  Only traversed containers are deduplicated to avoid cycles.

Optional:
- --no-classes can reduce noise when targeting primarily function or method
  style APIs.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, asdict
from typing import Any, Iterable, List, Optional, Set, Tuple

try:
    from common.api_policy import default_internal_prefixes, is_internal_api
except Exception:  # pragma: no cover - direct script fallback
    default_internal_prefixes = lambda _library: ()  # type: ignore
    is_internal_api = lambda _api, library=None, extra_prefixes=(): False  # type: ignore


@dataclass
class ApiEntry:
    """Represent one collected API entry or one unresolved lookup failure."""

    api: str
    resolved: bool
    kind: Optional[str] = None
    type: Optional[str] = None
    module: Optional[str] = None
    qualname: Optional[str] = None
    signature: Optional[str] = None
    doc_len: Optional[int] = None
    doc_truncated: Optional[bool] = None
    doc: Optional[str] = None

    # Metadata only. Collection does not globally collapse entries by callable_id.
    callable_id: Optional[int] = None
    canonical_target: Optional[str] = None

    error: Optional[str] = None


# ---------- helpers ----------

def is_public_qualname(qname: str) -> bool:
    """Return True if every segment of a dotted qualified name is public."""
    parts = [p for p in qname.split(".") if p]
    return all(not p.startswith("_") for p in parts)


def safe_getattr(obj: Any, name: str) -> Tuple[bool, Any, Optional[str]]:
    """Safely read an attribute and return success, value, and optional error."""
    try:
        return True, getattr(obj, name), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def kind_of(obj: Any) -> str:
    """Classify an object into a small set of inspection-oriented kinds."""
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    if inspect.isbuiltin(obj):
        return "builtin"
    if inspect.ismethod(obj):
        return "method"
    if inspect.ismethoddescriptor(obj) or inspect.isdatadescriptor(obj):
        return "descriptor"
    if callable(obj):
        return "callable"
    return "object"


def safe_signature(obj: Any) -> Optional[str]:
    """Return an object's signature string when available, otherwise None."""
    try:
        return str(inspect.signature(obj))
    except Exception:
        try:
            ts = inspect.getattr_static(obj, "__text_signature__", None)
        except Exception:
            ts = None
        if isinstance(ts, str) and ts.strip():
            return ts.strip()
        return None

def safe_doc(obj: Any, max_chars: int = 200_000) -> Tuple[str, int, bool]:
    """Return a cleaned docstring, its stored length, and whether it was truncated."""
    doc = ""
    try:
        doc = inspect.getdoc(obj) or ""
    except Exception:
        try:
            doc = (getattr(obj, "__doc__", None) or "") if obj is not None else ""
        except Exception:
            doc = ""
    doc = (doc or "").strip()
    truncated = False
    if len(doc) > max_chars:
        doc = doc[:max_chars]
        truncated = True
    return doc, len(doc), truncated


def should_traverse(obj: Any, kind: str, traverse_classes: bool = True) -> bool:
    """Return True when an object kind should be traversed for child members."""
    if kind == "module":
        return True
    if kind == "class":
        return traverse_classes
    return False


def iter_public_names(obj: Any) -> Iterable[str]:
    """
    Yield public attribute names for an object.

    Names are taken first from __all__ when present, then supplemented with
    public names discovered from dir(). Duplicates are removed while preserving
    encounter order.
    """
    seen = set()
    candidates: List[str] = []

    # Prefer curated export names when available.
    try:
        all_list = getattr(obj, "__all__", None)
        if isinstance(all_list, (list, tuple)):
            candidates.extend(x for x in all_list if isinstance(x, str))
    except Exception:
        pass

    # Also include public names discovered via dir().
    try:
        candidates.extend(dir(obj))
    except Exception:
        pass

    for n in candidates:
        if not n or not isinstance(n, str):
            continue
        if n.startswith("_"):
            continue
        if n in seen:
            continue
        seen.add(n)
        yield n


def is_internal_module_name(modname: str, root_name: str) -> bool:
    """
    Return True for module names that look like internal implementation paths.

    This primarily targets TensorFlow-style internal namespaces such as
    `tensorflow.python` and nested `*.python.*` paths.
    """
    if modname.startswith(f"{root_name}.python"):
        return True
    if f".python." in modname:
        return True
    return False


def collect_apis(
    root_name: str,
    max_depth: int = 5,
    exclude_prefixes: Optional[List[str]] = None,
    include_descriptors: bool = True,
    include_callables: bool = True,
    include_classes: bool = True,
    include_functions: bool = True,
    traverse_classes: bool = True,
) -> List[ApiEntry]:
    """
    Walk a package or module graph and collect public API entries.

    The traversal is breadth-first, depth-limited, and defensive against import
    or attribute access failures. Lookup failures are recorded as unresolved
    ApiEntry records instead of raising.
    """
    entries: List[ApiEntry] = []
    exclude_prefixes = list(default_internal_prefixes(root_name)) + list(exclude_prefixes or [])

    try:
        root = importlib.import_module(root_name)
    except Exception as e:
        entries.append(
            ApiEntry(api=root_name, resolved=False, error=f"{type(e).__name__}: {e}")
        )
        return entries

    def excluded(qname: str) -> bool:
        """Return True when a qualified name matches any excluded prefix."""
        return any(qname == pfx.rstrip(".") or qname.startswith(pfx.rstrip(".") + ".") for pfx in exclude_prefixes)

    queue: List[Tuple[Any, str, int]] = [(root, root_name, 0)]
    seen_names: Set[str] = set()
    seen_container_ids: Set[int] = set()

    while queue:
        obj, qname, depth = queue.pop(0)

        if excluded(qname) or is_internal_api(qname, root_name):
            continue
        if qname in seen_names:
            continue
        seen_names.add(qname)

        if not is_public_qualname(qname):
            continue

        k = kind_of(obj)

        should_collect = False
        if k == "class" and include_classes:
            should_collect = True
        elif k in {"function", "builtin"} and include_functions:
            should_collect = True
        elif k in {"method", "descriptor"} and include_descriptors:
            should_collect = True
        elif k == "callable" and include_callables:
            should_collect = True

        if should_collect and k != "module":
            sig = safe_signature(obj)
            doc, doc_len, truncated = safe_doc(obj)
            mod = getattr(obj, "__module__", None)
            target_qn = getattr(obj, "__qualname__", None) or getattr(
                obj, "__name__", None
            )
            canonical_target = f"{mod}.{target_qn}" if mod and target_qn else None
            tname = type(obj).__name__

            entries.append(
                ApiEntry(
                    api=qname,
                    resolved=True,
                    kind=k,
                    type=tname,
                    module=mod,
                    qualname=qname,
                    signature=sig,
                    doc_len=doc_len,
                    doc_truncated=truncated,
                    doc=doc,
                    callable_id=id(obj) if callable(obj) else None,
                    canonical_target=canonical_target,
                )
            )

        if depth >= max_depth:
            continue
        if not should_traverse(obj, k, traverse_classes=traverse_classes):
            continue

        obj_id = id(obj)
        if obj_id in seen_container_ids:
            continue
        seen_container_ids.add(obj_id)

        for child_name in iter_public_names(obj):
            child_qname = f"{qname}.{child_name}"
            if excluded(child_qname) or is_internal_api(child_qname, root_name) or not is_public_qualname(child_qname):
                continue

            ok, child, err = safe_getattr(obj, child_name)
            if not ok:
                entries.append(ApiEntry(api=child_qname, resolved=False, error=err))
                continue

            child_kind = kind_of(child)
            if child_kind == "module":
                modname = getattr(child, "__name__", "")
                if not modname:
                    continue

                # Skip internal TensorFlow-style public paths that resolve into
                # implementation namespaces.
                if root_name == "tensorflow" and is_internal_module_name(
                    child_qname, root_name
                ):
                    continue

                # Apply exclusions to both the public path and the module name.
                if excluded(child_qname) or excluded(modname):
                    continue

            queue.append((child, child_qname, depth + 1))

    return entries


def write_jsonl(entries: List[ApiEntry], out_path: str) -> None:
    """Write collected API entries to a UTF-8 JSONL file."""
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI arguments, collect APIs for all requested roots, and write output."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        action="append",
        required=True,
        help="Root package/module name (repeatable), e.g. --root torch --root tensorflow",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Skip any API under this prefix (repeatable), e.g. --exclude-prefix tensorflow.python",
    )
    ap.add_argument(
        "--no-descriptors",
        action="store_true",
        help="Do not collect method/descriptors (e.g. torch.Tensor.*)",
    )
    ap.add_argument(
        "--no-callables",
        action="store_true",
        help="Do not collect generic callables",
    )
    ap.add_argument(
        "--no-classes",
        action="store_true",
        help="Do not collect classes",
    )
    ap.add_argument(
        "--no-class-members",
        action="store_true",
        help="Do not descend into classes/types; keeps module-level APIs only",
    )
    args = ap.parse_args(argv)

    all_entries: List[ApiEntry] = []
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        for r in args.root:
            all_entries.extend(
                collect_apis(
                    root_name=r,
                    max_depth=args.max_depth,
                    exclude_prefixes=args.exclude_prefix,
                    include_descriptors=not args.no_descriptors,
                    include_callables=not args.no_callables,
                    include_classes=not args.no_classes,
                    traverse_classes=not args.no_class_members,
                )
            )
    finally:
        sys.argv = original_argv

    write_jsonl(all_entries, args.out)
    print(
        f"[collector] roots={args.root} max_depth={args.max_depth} entries={len(all_entries)} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
