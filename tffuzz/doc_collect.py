# tffuzz/doc_collect.py
from __future__ import annotations
import importlib
import inspect
import pkgutil
import types
import enum
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------
#  HIGH-LEVEL DATA STRUCTURE
# ------------------------------

@dataclass
class ParamInfo:
    name: str
    annotation: Optional[str]
    default: Optional[str]
    kind: str   # POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, VAR_KEYWORD


@dataclass
class FuncInfo:
    qualname: str
    module: str
    callable_type: str
    signature: str
    parameters: List[ParamInfo]
    docstring: str
    source_file: Optional[str]


# ------------------------------
# 1. Resolve fully-qualified name → real object
# ------------------------------

def resolve_callable(qualname: str):
    """
    Resolve: 'tensorflow.math.add' → <function tf.math.add>
    Works for ANY deep learning library.
    """
    parts = qualname.split(".")

    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        attrs = parts[i:]
        try:
            mod = importlib.import_module(mod_name)
            obj = mod
            for a in attrs:
                obj = getattr(obj, a)
            return obj
        except Exception:
            continue

    raise ImportError(f"[doc_collect] Cannot resolve: {qualname}")


# ------------------------------
# 2. Extract structured API information
# ------------------------------

def extract_api_info(qualname: str) -> FuncInfo:
    obj = resolve_callable(qualname)
    module = getattr(obj, "__module__", "") or qualname.rsplit(".", 1)[0]

    # ----------------------------------------------------
    # CONSTANT / ENUM / PSEUDO-ENUM DETECTION
    # ----------------------------------------------------
    import enum
    import inspect

    # 1. Real Enum MEMBER (e.g., torch.dtype.float32, tf.float16)
    if isinstance(obj, enum.Enum):
        return FuncInfo(
            qualname=qualname,
            module=module,
            callable_type="enum_member",
            signature="",
            parameters=[],
            docstring=inspect.getdoc(obj) or str(obj),
            source_file=None,
        )

    # 2. Real Enum CLASS (Python Enum)
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return FuncInfo(
            qualname=qualname,
            module=module,
            callable_type="enum",
            signature="",
            parameters=[],
            docstring=inspect.getdoc(obj) or "",
            source_file=None,
        )

    # 3. Pseudo-enums (Torch, TensorFlow, JAX) — classes with ALLCAPS members
    if inspect.isclass(obj) and looks_like_enum(obj):
        return FuncInfo(
            qualname=qualname,
            module=module,
            callable_type="enum_like",
            signature="",
            parameters=[],
            docstring=inspect.getdoc(obj) or "",
            source_file=None,
        )

    # 4. PURE CONSTANT (not callable, not class)
    if not callable(obj) and not inspect.isclass(obj):
        return FuncInfo(
            qualname=qualname,
            module=module,
            callable_type="constant",
            signature="",
            parameters=[],
            docstring=str(obj),
            source_file=None,
        )

    if isinstance(obj, enum.EnumMeta):
        return FuncInfo(
            qualname=qualname,
            module=module,
            callable_type="enum",
            signature="()",
            parameters=[],
            docstring=inspect.getdoc(obj) or "",
            source_file=None,
        )

    # Callable type classification
    if inspect.isfunction(obj):
        callable_type = "function"
    elif inspect.ismethod(obj):
        callable_type = "method"
    elif inspect.isbuiltin(obj):
        callable_type = "builtin"
    elif inspect.isclass(obj):
        callable_type = "class"
    elif hasattr(obj, "__call__") and not inspect.isclass(obj):
        callable_type = "callable_object"
    else:
        callable_type = "unknown"

    # Signature extraction
    try:
        # If it's a class → use __init__ signature
        if inspect.isclass(obj):
            try:
                target = obj.__init__
                if not callable(target):
                    raise TypeError
            except Exception:
                def _empty(): pass
                target = _empty

        else:
            target = obj

        sig = inspect.signature(target)
        sig_str = str(sig)

    except Exception:
        sig = None
        sig_str = "(...)"

    # Parameters
    params = []
    if sig is not None:
        for p in sig.parameters.values():
            # Handle *args / **kwargs generically
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                params.append(ParamInfo(p.name, None, None, "VAR_POSITIONAL"))
                continue

            if p.kind == inspect.Parameter.VAR_KEYWORD:
                params.append(ParamInfo(p.name, None, None, "VAR_KEYWORD"))
                continue

            # Normal parameters
            params.append(
                ParamInfo(
                    name=p.name,
                    annotation=str(
                        p.annotation) if p.annotation != inspect._empty else None,
                    default=repr(
                        p.default) if p.default != inspect._empty else None,
                    kind=str(p.kind)
                )
            )

    # Docstring
    docstring = inspect.getdoc(obj) or ""

    # Source file (may not exist for TF ops)
    try:
        source_file = inspect.getsourcefile(obj)
    except Exception:
        source_file = None

    return FuncInfo(
        qualname=qualname,
        module=module,
        callable_type=callable_type,
        signature=sig_str,
        parameters=params,
        docstring=docstring,
        source_file=source_file
    )


# ------------------------------
# 3. API enumeration (modules → FQNs)
# ------------------------------

def iter_public_callables(root_modules: List[str], limit: int = 0) -> List[str]:
    """
    Enumerate ALL public callables in modules.
    Generic, works for TensorFlow, Torch, JAX, etc.
    """
    out: List[str] = []
    visited = set()

    def add_from_module(mod):
        for name in dir(mod):
            if name.startswith("_"):
                continue

            # Build fully-qualified name early
            fq = f"{mod.__name__}.{name}"

            # Safely get the attribute; skip if it’s a submodule
            try:
                obj = getattr(mod, name)
            except Exception:
                continue

            if inspect.ismodule(obj):
                # Submodules are handled separately below
                continue

            # Use extract_api_info to classify and filter
            try:
                info = extract_api_info(fq)
            except Exception:
                # Anything we can't introspect cleanly → skip
                continue

            # Drop non-executable things:
            #   - constants
            #   - enums / enum members / enum-like pseudo-enums
            #   - anything still classified as unknown
            if info.callable_type in {
                "enum",
                "enum_member",
                "enum_like",
                "constant",
                "unknown",
            }:
                continue

            # Keep only "real" executable APIs:
            # functions, methods, builtins, classes (constructors),
            # and callable objects
            if info.callable_type not in {
                "function",
                "method",
                "builtin",
                "class",
                "callable_object",
            }:
                continue

            out.append(fq)

    for root_name in root_modules:
        try:
            root = importlib.import_module(root_name)
        except Exception:
            continue

        add_from_module(root)

        for name in dir(root):
            try:
                sub = getattr(root, name)
                if inspect.ismodule(sub) and sub.__name__.startswith(root.__name__):
                    add_from_module(sub)
            except Exception:
                pass

        visited.add(root.__name__)

        pkg_path = getattr(root, "__path__", None)
        if pkg_path is None:
            continue

        for finder, name, ispkg in pkgutil.walk_packages(pkg_path, prefix=root.__name__ + "."):
            if name in visited:
                continue
            visited.add(name)

            try:
                sub = importlib.import_module(name)
            except Exception:
                continue

            add_from_module(sub)

            if limit and len(out) >= limit:
                break
        if limit and len(out) >= limit:
            break

    # remove duplicates (preserve order)
    seen = set()
    uniq = []
    for q in out:
        if q not in seen:
            uniq.append(q)
            seen.add(q)
        if limit and len(uniq) >= limit:
            break

    return uniq


# ------------------------------
# 4. Convenience API (called by doc_build)
# ------------------------------

def collect_doc_context(qualname: str) -> Tuple[str, str]:
    info = extract_api_info(qualname)
    return info.signature, info.docstring


# CLI for quick debugging
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", default=["tensorflow"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    qns = iter_public_callables(args.modules, args.limit)

    # OUTPUT ONLY QUALNAMES (doc_build expects this)
    with open(args.out, "w") as f:
        json.dump(qns, f, indent=2)

    print(f"[doc_collect] Found {len(qns)} APIs → {args.out}")
