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

    # Reject PyTorch in-place ops universally
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

    FORBIDDEN_PHRASES = [
        "see also", "see class", "see function",
        "for details see", "see documentation for",
        "see above", "see below",
        "see source code", "see implementation",
        "see :", "see:",
        "alias for", "alias of", "alias to",
        "same as", "equivalent to",
        "wrapper for", "wrapper around",
        "redirects to", "delegates to", "calls into",
        "thin wrapper", "inherits documentation from",
        "implemented in", "defined in",
        "internal use only", "private api",
        "deprecated", "will be removed", "legacy",
        "backward compatibility", "implementation dependent",
        "backend dependent", "platform dependent",
        "nondeterministic", "may vary",
        "identical to", "matches the behavior of",
        "follows semantics of", "based on",
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
        if parts:
            inline_valid = True

    if not structured and not inline_valid:
        return False

    required_params = []
    if inline_valid:
        parts = [p.strip() for p in params_text.split(",") if p.strip()]
        for p in parts:
            name_default = p.split("=", 1)
            name = name_default[0].split(":", 1)[0].strip().lstrip("*")
            if name in ("self", "cls"):
                continue
            if "=" not in p:
                required_params.append(name)

    # ============================================================
    # Step 3 — Constructible types
    # ============================================================

    if not any(t in low for t in CONSTRUCTIBLE_TYPES):
        return False

    # Forbid non-constructible types only if they are tied
    # to required parameters (optional-only is allowed)
    for word in NON_CONSTRUCTIBLE_TYPES:
        if word in low:
            in_required = any(
                f"{req} (" in low
                and word in low.split(f"{req} (", 1)[1].split(")", 1)[0]
                for req in required_params
            )
            if in_required:
                return False

    # ============================================================
    # Step 4 — IO side effects only
    # ============================================================

    for w in SIDE_EFFECT_KEYWORDS["io"]:
        if w in low:
            return False

    # ============================================================
    # Step 5 — Return clarity
    # Relaxed for class-like APIs (e.g. torch.nn.Conv2d)
    # ============================================================

    is_classish = False
    if api_name:
        last = api_name.split(".")[-1]
        if last and last[0].isupper():
            is_classish = True

    structured_return = re.search(RETURN_SECTION_RE, low)

    inline_return = re.search(
        r"returns?\b[^.\n]{0,80}\b(tensor|ndarray|array|number|float|int|list|tuple|value|object)\b",
        low,
    )

    arrow_return = re.search(
        r"->\s*(tensor|[a-z_]*tensor|ndarray|array|float|int|list|tuple|number)",
        low,
    )

    if not is_classish:
        if not (structured_return or inline_return):
            if not (arrow_return and len(d.split()) > 15):
                return False

    if any(bad in low for bad in RETURN_BAD_KEYWORDS):
        return False

    if not any(g in low for g in RETURN_GOOD_KEYWORDS):
        return False

    return True


# ============================================================
# Library Walker — Correct recursion
# ============================================================

def collect_docs_recursive(obj, prefix, seen, max_depth=5, root_name=None):
    results = []

    if root_name is None:
        # root_name will be "torch" when starting from torch
        root_name = prefix.split(".")[0]

    depth = prefix.count(".")
    if depth > max_depth:
        return results

    attr_names = set()

    try:
        attr_names.update(dir(obj))
    except Exception:
        pass

    try:
        attr_names.update(getattr(obj, "__dict__", {}).keys())
    except Exception:
        pass

    try:
        attr_names.update(getattr(obj, "__all__", []))
    except Exception:
        pass

    # Filter to string names only
    attr_names = {n for n in attr_names if isinstance(n, str)}

    for name in attr_names:
        if name.startswith("_"):
            continue

        full = f"{prefix}.{name}"

        try:
            child = getattr(obj, name)
        except Exception:
            continue

        # treat ANY Python/C type as class-like
        is_class_like = isinstance(child, type) or inspect.isclass(child)

        # Skip giant internal modules
        if inspect.ismodule(child):
            modname = getattr(child, "__name__", "")
            # Only follow modules that actually belong to the same top-level lib
            if not modname.startswith(root_name):
                continue
            
        # Identify valid API objects
        if not (
            inspect.isroutine(child)
            or inspect.isbuiltin(child)
            or inspect.ismethoddescriptor(child)
            or inspect.ismethod(child)
            or inspect.isfunction(child)
            or is_class_like
            or inspect.ismodule(child)
        ):
            continue

        oid = id(child)
        if oid in seen:
            continue
        seen.add(oid)

        # Collect testable docs
        doc = getattr(child, "__doc__", None)
        if isinstance(doc, str) and is_testable_doc(doc, full):
            results.append((full, doc))

        # Recurse into modules and class-like objects
        if inspect.ismodule(child) or is_class_like:
            results.extend(
                collect_docs_recursive(child, full, seen, max_depth, root_name)
            )

    return results


# ============================================================
# CSV Export
# ============================================================

def save_testable_csv(lib, out_path="testable_apis.csv"):
    seen = {id(lib)}
    results = collect_docs_recursive(lib, lib.__name__, seen)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["api_full_name", "api_doc_text"])

        for api, doc in results:
            decision, _ = step6_final_decision(doc)
            if decision == "TESTABLE":
                writer.writerow([api, doc])

    print(f"[✓] Saved testable APIs to {out_path}")


def step6_final_decision(doc: str):
    if not doc or not isinstance(doc, str):
        return "NOT TESTABLE", "missing doc"

    text = doc.strip()
    if not text:
        return "NOT TESTABLE", "empty doc"

    try:
        passed = is_testable_doc(doc)
    except Exception:
        return "UNCERTAIN", "evaluation error"

    if passed:
        return "TESTABLE", "all steps passed"

    low = text.lower()
    hard_fail_signals = [
        "modifies in place", "in-place", "side effect",
        "deprecated", "unsafe", "do not use"
    ]
    if any(w in low for w in hard_fail_signals):
        return "NOT TESTABLE", "explicit fail"

    return "UNCERTAIN", "insufficient info"


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=str, required=True,
                        help="Library name, e.g., tensorflow, torch, numpy")
    parser.add_argument("--out", type=str, default="testable_apis.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    lib = importlib.import_module(args.lib)
    save_testable_csv(lib, args.out)


if __name__ == "__main__":
    main()