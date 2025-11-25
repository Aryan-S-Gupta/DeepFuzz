import argparse
import csv
import importlib
import inspect
import re

# ============================================================
# Step patterns and rules
# ============================================================

PLACEHOLDER_PATTERNS = ["...", "pass", "todo", "tbd", "fixme", "lorem ipsum"]

VAGUE_WORDS = [
    "anything", "some object", "data structure", "stuff",
    "something", "misc", "various", "obj"
]

CONSTRUCTIBLE_TYPES = [
    "int", "float", "bool", "string", "str", "tuple", "list",
    "array", "ndarray", "tensor", "numeric", "number"
]

NON_CONSTRUCTIBLE_TYPES = [
    "file", "filepath", "path", "directory", "folder",
    "url", "uri", "http", "https",
    "network", "socket", "connection",
    "session", "graph", "model", "handle",
    "context manager", "stream", "iterator", "generator",
    "callback", "callable",
    "device", "gpu", "cuda", "hardware",
    "environment", "env"
]

SIDE_EFFECT_KEYWORDS = {
    "io": [
        "file", "save", "load", "write", "read", "open",
        "download", "upload", "serialize", "deserialize"
    ],
    "hardware": [
        "gpu", "tpu", "cuda", "device", "accelerator",
        "camera", "microphone", "sensor"
    ],
    "execution": [
        "session", "scope", "global", "attach", "register",
        "context", "stateful"
    ],
    "training": [
        "train", "fit", "learn", "optimizer", "backward",
        "checkpoint", "gradient descent"
    ],
    "distributed": [
        "cluster", "distributed", "server", "worker",
        "rpc", "remote"
    ],
    "async": [
        "callback", "hook", "listener", "subscribe",
        "event", "async", "await"
    ]
}

# ============================================================
# Step 5 patterns
# ============================================================

RETURN_SECTION_RE = r"(returns?|output|outputs?)\s*[:]"
RETURN_GOOD_KEYWORDS = [
    "tensor", "ndarray", "array", "numeric", "number",
    "tuple", "list", "object", "value", "boolean",
    "float", "int", "string", "str", "index"
]

RETURN_BAD_KEYWORDS = [
    "modifies in place", "in-place", "side effect",
    "updates the object", "mutates", "changes internal state",
    "share the same underlying storage", "view on the original tensor",
    "returns a view", "aliasing", "shares storage",
]

# ============================================================
# Step 1–5 Unified Testability Function
# ============================================================


def is_testable_doc(doc: str, api_name: str = "") -> bool:
    if not doc:
        return False

    d = doc.strip()

    # Reject PyTorch in-place ops universally (protocol: side effects)
    if api_name.endswith("_"):
        return False

    if len(d) < 10:
        return False

    # Reject pure signature-only docs
    if "\n" not in d and re.match(r"^[\w\.]+\([^)]*\)\s*->\s*.+$", d):
        return False

    low = d.lower()

    # Placeholder checks
    if any(p in low for p in PLACEHOLDER_PATTERNS):
        return False

    # Forbidden phrases (cross refs, aliases, etc.)
    FORBIDDEN_PHRASES = [
        # cross refs that actually redirect elsewhere
        "see also", "see class", "see function",
        "for details see", "see documentation for",
        "see above", "see below",
        "see source code", "see implementation",
        "see :", "see:",

        # aliases / wrappers
        "alias for", "alias of", "alias to",
        "same as", "equivalent to",
        "wrapper for", "wrapper around",
        "redirects to", "delegates to", "calls into",
        "thin wrapper", "inherits documentation from",

        # implementation shortcuts
        "implemented in", "defined in",
        "internal use only", "private api",

        # deprecation / legacy / instability
        "deprecated", "will be removed", "legacy",
        "backward compatibility",
        "implementation dependent", "backend dependent",
        "platform dependent", "nondeterministic", "may vary",

        # behavior-copying
        "identical to", "matches the behavior of",
        "follows semantics of", "based on",

        # others
        "out-of-place version of", "version of",
        "same semantics as", "similar to", "equivalent to",
    ]

    for phrase in FORBIDDEN_PHRASES:
        if phrase in low:
            return False

    # ============================================================
    # Step 2 — Required parameter validity
    # ============================================================

    structured = re.search(r"(parameters|args|arguments|inputs?)\s*[:]", low)

    inline_sig = re.search(r"\b\w+\s*\(([^()]*)\)", d)
    inline_valid = False
    params_text = ""

    if inline_sig:
        params_text = inline_sig.group(1)
        parts = [p.strip() for p in params_text.split(",") if p.strip()]
        if len(parts) >= 1:
            inline_valid = True

    if not structured and not inline_valid:
        return False

    # Collect required parameter names from the inline signature
    required_params = []
    if inline_valid:
        parts = [p.strip() for p in params_text.split(",") if p.strip()]
        for p in parts:
            name_default = p.split("=", 1)
            name = name_default[0].split(":", 1)[0].strip().lstrip("*")

            # Ignore self, cls
            if name in ("self", "cls"):
                continue

            # No '=' means required param
            if "=" not in p:
                required_params.append(name)

    # ============================================================
    # Step 3 — Constructible types
    # ============================================================

    if not any(t in low for t in CONSTRUCTIBLE_TYPES):
        return False

    # Allow NON_CONSTRUCTIBLE_TYPES only in optional parameters
    for word in NON_CONSTRUCTIBLE_TYPES:
        if word in low:
            in_required = any(
                f"{req} (" in low and word in low.split(f"{req} (", 1)[1].split(")", 1)[0]
                for req in required_params
            )
            if in_required:
                return False

    # ============================================================
    # Step 4 — Side-effect checks
    # (keep IO strict, but do not globally kill distributed / training)
    # ============================================================

    for w in SIDE_EFFECT_KEYWORDS["io"]:
        if w in low:
            return False

    # ============================================================
    # Step 5 — Return-value clarity
    # ============================================================

    structured_return = re.search(RETURN_SECTION_RE, low)

    inline_return = re.search(
        r"returns?\b[^.\n]{0,80}\b(tensor|ndarray|array|number|float|int|list|tuple|value|object)\b",
        low,
    )

    arrow_return = re.search(
        r"->\s*(tensor|[a-z_]*tensor|ndarray|array|float|int|list|tuple|number)",
        low,
    )

    if not (structured_return or inline_return):
        if not (arrow_return and len(d.split()) > 15):
            return False

    if any(bad in low for bad in RETURN_BAD_KEYWORDS):
        return False

    if not any(g in low for g in RETURN_GOOD_KEYWORDS):
        return False

    return True


# ============================================================
# Library Walker (safe, recursive)
# ============================================================

def collect_docs_recursive(obj, prefix, seen, max_depth=5):
    """
    Recursive walk with:
      - object-identity visited guard (seen)
      - structural skips for private / deep namespaces
      - gentle depth limit to avoid huge internal trees
    """
    results = []
    current_depth = prefix.count(".")

    for name in dir(obj):
        if name.startswith("_"):
            continue

        full = f"{prefix}.{name}" if prefix else name

        # Skip any path containing private segments like torch._C, torch.nn._reduction, etc.
        parts = full.split(".")
        if any(part.startswith("_") for part in parts):
            continue

        try:
            child = getattr(obj, name)
        except Exception:
            continue

        # Skip non-APIs (constants, ints, enums, etc.)
        if not (
            inspect.isroutine(child)
            or inspect.isbuiltin(child)
            or inspect.ismethoddescriptor(child)
            or inspect.isfunction(child)
            or inspect.ismethod(child)
            or inspect.isclass(child)
            or inspect.ismodule(child)
        ):
            continue

        oid = id(child)
        if oid in seen:
            continue
        seen.add(oid)

        doc = getattr(child, "__doc__", None)
        if isinstance(doc, str) and is_testable_doc(doc, full):
            results.append((full, doc))

        # Recurse into modules and classes, but avoid deep explosion
        if inspect.ismodule(child) or inspect.isclass(child):
            # stop going deeper after a certain depth
            if current_depth + 1 > max_depth:
                continue

            # big internal modules tend to have huge dir() and are not needed
            try:
                if inspect.ismodule(child) and len(dir(child)) > 1000:
                    continue
            except Exception:
                pass

            results.extend(collect_docs_recursive(child, full, seen, max_depth))

    return results


# ============================================================
# CSV Export
# ============================================================

def save_testable_csv(lib, out_path="testable_apis.csv"):
    # seed seen with the root library object to avoid cycles like autocast_mode.torch -> torch
    seen = {id(lib)}
    results = collect_docs_recursive(lib, lib.__name__, seen)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["api_full_name", "api_doc_text"])
        for api, doc in results:
            decision, reason = step6_final_decision(doc)
            if decision == "TESTABLE":
                writer.writerow([api, doc])

    print(f"[✓] Saved testable APIs to {out_path}")


# ============================================================
# Step 6: Final Decision Function
# ============================================================

def step6_final_decision(doc: str):
    if not doc or not isinstance(doc, str):
        return "NOT TESTABLE", "missing documentation"

    text = doc.strip()
    if not text:
        return "NOT TESTABLE", "empty documentation"

    try:
        passed = is_testable_doc(doc)
    except Exception:
        return "UNCERTAIN", "evaluation error"

    if passed:
        return "TESTABLE", "all steps passed"

    low = text.lower()

    hard_fail_signals = [
        "modifies in place",
        "in-place",
        "side effect",
        "deprecated",
        "unsafe",
        "do not use",
    ]

    if any(w in low for w in hard_fail_signals):
        return "NOT TESTABLE", "explicit non-testable behavior"

    return "UNCERTAIN", "insufficient information"


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        required=True,
        help="Library name, e.g., tensorflow, torch, numpy",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="testable_apis.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    lib = importlib.import_module(args.lib)
    save_testable_csv(lib, args.out)


if __name__ == "__main__":
    main()