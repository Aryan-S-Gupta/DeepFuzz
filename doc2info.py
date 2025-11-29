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
    "something", "misc", "various", "obj",
]

CONSTRUCTIBLE_TYPES = [
    "int",
    "float",
    "bool",
    "string",
    "str",
    "tuple",
    "list",
    "array",
    "ndarray",
    "tensor",
    "numeric",
    "number",
]

NON_CONSTRUCTIBLE_TYPES = [
    "file", "filepath", "path", "directory", "folder",
    "url", "uri", "http", "https",
    "network", "socket", "connection",
    "session", "graph", "model", "handle",
    "context manager", "stream", "iterator", "generator",
    "callback", "callable",
    "device", "gpu", "cuda", "hardware",
    "environment", "env",
    "iterable", "sequence", "mapping", "collection",
    "parameter", "parameters", "nn.module", "module"
]

SIDE_EFFECT_KEYWORDS = {
    "io": [
        "file",
        "save",
        "load",
        "write",
        "read",
        "open",
        "download",
        "upload",
        "serialize",
        "deserialize",
    ],
    "hardware": [
        "gpu",
        "tpu",
        "cuda",
        "device",
        "accelerator",
        "camera",
        "microphone",
        "sensor",
    ],
    "execution": [
        "session",
        "scope",
        "global",
        "attach",
        "register",
        "context",
        "stateful",
    ],
    "training": [
        "train",
        "fit",
        "learn",
        "optimizer",
        "backward",
        "checkpoint",
        "gradient descent",
    ],
    "distributed": [
        "cluster",
        "distributed",
        "server",
        "worker",
        "rpc",
        "remote",
    ],
    "async": [
        "callback",
        "hook",
        "listener",
        "subscribe",
        "event",
        "async",
        "await",
    ],
}

RETURN_SECTION_RE = r"(returns?|output|outputs?)\s*[:]"

RETURN_GOOD_KEYWORDS = [
    "tensor",
    "ndarray",
    "array",
    "numeric",
    "number",
    "tuple",
    "list",
    "object",
    "value",
    "boolean",
    "float",
    "int",
    "string",
    "str",
    "index",
]

RETURN_BAD_KEYWORDS = [
    "modifies in place",
    "in-place",
    "side effect",
    "updates the object",
    "mutates",
    "changes internal state",
    "share the same underlying storage",
    "view on the original tensor",
    "returns a view",
    "aliasing",
    "shares storage",
]


# ============================================================
# Step 1-5 Unified Testability Function
# (exactly your original criteria)
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
    # Step 2 - Required parameter validity
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

    # Fuzzing requires at least one required parameter to generate tests.
    if len(required_params) == 0:
        if not re.search(RETURN_SECTION_RE, low) and "->" not in low:
            return False

    # ============================================================
    # Step 3 - Constructible types
    # ============================================================

    # If doc does not mention any constructible type at all → reject
    if not any(t in low for t in CONSTRUCTIBLE_TYPES):
        return False

    # If a required parameter has a non-constructible or unknown type → reject.
    for req in required_params:
        # Look for patterns like:
        #   param (TYPE
        #   param: TYPE
        pattern = rf"{re.escape(req)}\s*\(\s*([A-Za-z0-9_\[\]]+)"
        m = re.search(pattern, d)
        if m:
            ptype = m.group(1).lower()

            # Reject if required param type is explicitly non-constructible
            if ptype in NON_CONSTRUCTIBLE_TYPES:
                return False

            # Accept if required param type is constructible
            if ptype in CONSTRUCTIBLE_TYPES:
                continue

            # Otherwise type is unknown, not safely constructible → reject
            continue
        else:
            # No type documented → reject
            return False

    # Forbid non-constructible types when tied to required parameters
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
    # Step 4 - IO side effects only
    # ============================================================

    for w in SIDE_EFFECT_KEYWORDS["io"]:
        if w in low:
            return False

    # ============================================================
    # Step 5 - Return clarity
    # Relaxed for class-like APIs (for example torch.nn.Conv2d)
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
# Step 6 - Final decision (kept for compatibility)
# ============================================================

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
        "deprecated", "unsafe", "do not use",
    ]
    if any(w in low for w in hard_fail_signals):
        return "NOT TESTABLE", "explicit fail"

    return "UNCERTAIN", "insufficient info"


# ============================================================
# New - Reason extractor for rejected docs
# ============================================================

def explain_failure_reason(doc: str, api_name: str = "") -> str:
    """
    Best effort explanation of which rule failed, without changing criteria.
    Assumes doc is a non empty string and is_testable_doc(doc, api_name) is False.
    """
    if not isinstance(doc, str):
        return "missing or non-string __doc__"

    if not doc:
        return "missing doc"

    d = doc.strip()
    if not d:
        return "empty doc"

    low = d.lower()

    # Same checks and order as is_testable_doc
    if api_name.endswith("_"):
        return "in-place op"

    if len(d) < 10:
        return "too short"

    if "\n" not in d and re.match(r"^[\w\.]+\([^)]*\)\s*->\s*.+$", d):
        return "signature only"

    if any(p in low for p in PLACEHOLDER_PATTERNS):
        return "placeholder doc"

    FORBIDDEN_PHRASES = [
        # Redirection / missing semantics
        "see also",
        "see class",
        "see function",
        "for details see",
        "see documentation for",
        "see above",
        "see below",
        "see source code",
        "see implementation",
        "see :",
        "see:",

        # Delegation / aliasing (no independent behavior)
        "alias for",
        "alias of",
        "alias to",
        "wrapper for",
        "wrapper around",
        "redirects to",
        "delegates to",
        "calls into",
        "thin wrapper",
        "inherits documentation from",

        # Documentation that explicitly voids meaningful semantics
        "internal use only",
        "private api",

        # Unsafe / removed / unstable
        "deprecated",
        "will be removed",
        "legacy",

        # Stability / determinism issues fatal for fuzzing
        "undefined behavior",
        "do not use",
        "not intended for direct use",
        "experimental",
        "subject to change",
    ]

    for phrase in FORBIDDEN_PHRASES:
        if phrase in low:
            return f"forbidden phrase: {phrase}"

    structured = re.search(r"(parameters|args|arguments|inputs?)\s*[:]", low)
    lines = d.strip().splitlines()
    first_line = lines[0] if lines else ""
    inline_sig = re.match(r"^\s*\w+\s*\(([^()]*)\)", first_line)

    inline_valid = False
    params_text = ""

    if inline_sig:
        params_text = inline_sig.group(1)
        parts = [p.strip() for p in params_text.split(",") if p.strip()]
        if parts:
            inline_valid = True

    if not structured and not inline_valid:
        return "missing parameter description"

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

    if not any(t in low for t in CONSTRUCTIBLE_TYPES):
        return "no constructible input type"

    for word in NON_CONSTRUCTIBLE_TYPES:
        if word in low:
            in_required = any(
                f"{req} (" in low
                and word in low.split(f"{req} (", 1)[1].split(")", 1)[0]
                for req in required_params
            )
            if in_required:
                return f"non-constructible required type: {word}"

    for w in SIDE_EFFECT_KEYWORDS["io"]:
        if w in low:
            return f"io side effect: {w}"

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
                return "unclear return type"

    if any(bad in low for bad in RETURN_BAD_KEYWORDS):
        return "bad return semantics"

    if not any(g in low for g in RETURN_GOOD_KEYWORDS):
        return "no good return keywords"

    return "unspecified failure"


def collect_docs_recursive(obj, prefix, seen_modules, max_depth=10, root_name=None, seen_classes=None):
    import inspect
    import types

    if root_name is None:
        root_name = prefix.split(".")[0]

    if seen_classes is None:
        seen_classes = set()

    depth = prefix.count(".")
    if depth > max_depth:
        return []

    results = []

    # gather attributes
    attr_names = set()
    try:
        attr_names.update(dir(obj))
    except:
        pass

    try:
        dct = getattr(obj, "__dict__", {})
        if isinstance(dct, dict):
            attr_names.update(dct.keys())
    except:
        pass

    try:
        all_ = getattr(obj, "__all__", None)
        if isinstance(all_, (list, tuple, set)):
            attr_names.update(all_)
    except:
        pass

    attr_names = {x for x in attr_names if isinstance(
        x, str) and not x.startswith("_")}

    for name in sorted(attr_names):
        try:
            child = getattr(obj, name)
        except:
            continue

        is_module = inspect.ismodule(child)
        is_class = inspect.isclass(child)
        is_func = (
            inspect.isroutine(child)
            or inspect.isbuiltin(child)
            or inspect.isfunction(child)
            or inspect.ismethod(child)
            or inspect.ismethoddescriptor(child)
        )

        # skip descriptors (.dtype, .shape)
        if isinstance(child, (types.GetSetDescriptorType, types.MemberDescriptorType)):
            continue

        if not (is_module or is_class or is_func):
            continue

        # doc
        try:
            doc_str = getattr(child, "__doc__", None)
        except:
            doc_str = None

        if isinstance(doc_str, str):
            doc = doc_str
        else:
            doc = None

        full_name = f"{prefix}.{name}"
        results.append((full_name, doc))

        # recursion
        if depth == max_depth:
            continue

        if is_module:
            modname = getattr(child, "__name__", "")
            if not isinstance(modname, str):
                continue

            if not (modname == root_name or modname.startswith(root_name + ".")):
                continue

            if modname in seen_modules:
                continue

            seen_modules.add(modname)

            results.extend(
                collect_docs_recursive(
                    child,
                    modname,          # canonical module prefix
                    seen_modules,
                    max_depth,
                    root_name,
                    seen_classes,
                )
            )

        elif is_class:
            cid = id(child)
            if cid in seen_classes:
                continue
            seen_classes.add(cid)

            results.extend(
                collect_docs_recursive(
                    child,
                    full_name,         # class uses attribute prefix
                    seen_modules,
                    max_depth,
                    root_name,
                    seen_classes,
                )
            )

    return results


# ============================================================
# CSV export - original single file version
# ============================================================

def save_testable_csv(lib, out_path="testable_apis.csv", max_depth=10):
    seen = {id(lib)}
    results = collect_docs_recursive(
        lib, lib.__name__, seen, max_depth=max_depth)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["api_full_name", "api_doc_text"])

        for api, doc in results:
            decision, _ = step6_final_decision(doc)
            if decision == "TESTABLE":
                writer.writerow([api, doc])

    print(f"[✓] Saved testable APIs to {out_path}")


# ============================================================
# New - CSV export with accepted and rejected lists
# ============================================================

def save_two_csvs(lib, out_good="testable_apis.csv", out_bad="rejected_apis.csv", max_depth=10):
    seen = {id(lib)}
    all_results = collect_docs_recursive(
        lib, lib.__name__, seen, max_depth=max_depth)

    with open(out_good, "w", newline="", encoding="utf-8") as g, open(
        out_bad, "w", newline="", encoding="utf-8"
    ) as b:
        good_writer = csv.writer(g)
        bad_writer = csv.writer(b)

        good_writer.writerow(["api_full_name", "api_doc_text"])
        bad_writer.writerow(["api_full_name", "api_doc_text", "reason"])

        for api, doc in all_results:
            ok = False
            reason = "missing doc"

            if not isinstance(doc, str):
                ok = False
                reason = "missing or non-string __doc__"
            else:
                try:
                    ok = is_testable_doc(doc, api)
                except Exception as e:
                    ok = False
                    reason = f"evaluation error: {type(e).__name__}"
                else:
                    if ok:
                        reason = "all steps passed"
                    else:
                        reason = explain_failure_reason(doc, api)

            if ok:
                good_writer.writerow([api, doc])
            else:
                bad_writer.writerow([api, doc, reason])

    total = len(all_results)
    n_good = sum(1 for (api, doc)
                 in all_results if is_testable_doc(doc or "", api))
    n_bad = total - n_good

    print(f"[✓] Saved {n_good} testable APIs to {out_good}")
    print(f"[✓] Saved {n_bad} rejected APIs to {out_bad}")
    print(f"[✓] Total APIs scanned: {total}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        required=True,
        help="Library name, for example tensorflow, torch, numpy",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="testable_apis.csv",
        help="Output CSV file for accepted APIs",
    )
    parser.add_argument(
        "--rej",
        type=str,
        default="rejected_apis.csv",
        help="Output CSV file for rejected APIs",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Max recursion depth for module traversal",
    )
    args = parser.parse_args()

    lib = importlib.import_module(args.lib)
    save_two_csvs(lib, args.out, args.rej, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
