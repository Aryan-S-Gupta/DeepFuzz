# tffuzz/registry.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import re

# -------------------------------------------------------------
# UNIVERSAL DTYPE VOCABULARY
# -------------------------------------------------------------
# Canonical names used internally across *all* libraries.
UNIFIED_DTYPES = {
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float8", "float16", "float32", "float64",
    "bfloat16",
    "complex64", "complex128",
}

# -------------------------------------------------------------
# DTYPE NORMALIZATION TABLE
# Keys here are *patterns* we may see from many libraries.
# All matching is done case-insensitively.
# -------------------------------------------------------------
DTYPE_NORMALIZATION_MAP: Dict[str, str] = {
    # --- TensorFlow ---
    "tf.float32": "float32",
    "tf.float64": "float64",
    "tf.float16": "float16",
    "tf.bfloat16": "bfloat16",
    "tf.float8_e4m3fn": "float8",
    "tf.float8_e5m2": "float8",
    "tf.int8": "int8",
    "tf.int16": "int16",
    "tf.int32": "int32",
    "tf.int64": "int64",
    "tf.uint8": "uint8",
    "tf.bool": "bool",
    "tf.complex64": "complex64",
    "tf.complex128": "complex128",

    # --- PyTorch / torch ---
    "torch.float32": "float32",
    "torch.float": "float32",
    "torch.float64": "float64",
    "torch.double": "float64",
    "torch.float16": "float16",
    "torch.half": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int": "int32",
    "torch.int64": "int64",
    "torch.long": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
    "torch.complex64": "complex64",
    "torch.cfloat": "complex64",
    "torch.complex128": "complex128",
    "torch.cdouble": "complex128",

    # --- NumPy (np / numpy) ---
    "numpy.float16": "float16",
    "numpy.float32": "float32",
    "numpy.float64": "float64",
    "numpy.int8": "int8",
    "numpy.int16": "int16",
    "numpy.int32": "int32",
    "numpy.int64": "int64",
    "numpy.uint8": "uint8",
    "numpy.uint16": "uint16",
    "numpy.uint32": "uint32",
    "numpy.uint64": "uint64",
    "numpy.bool_": "bool",
    "numpy.bool": "bool",
    "numpy.complex64": "complex64",
    "numpy.complex128": "complex128",
    "np.float16": "float16",
    "np.float32": "float32",
    "np.float64": "float64",
    "np.int8": "int8",
    "np.int16": "int16",
    "np.int32": "int32",
    "np.int64": "int64",
    "np.uint8": "uint8",
    "np.uint16": "uint16",
    "np.uint32": "uint32",
    "np.uint64": "uint64",
    "np.bool_": "bool",
    "np.bool": "bool",
    "np.complex64": "complex64",
    "np.complex128": "complex128",

    # --- JAX ---
    "jax.numpy.float16": "float16",
    "jax.numpy.float32": "float32",
    "jax.numpy.float64": "float64",
    "jax.numpy.int8": "int8",
    "jax.numpy.int16": "int16",
    "jax.numpy.int32": "int32",
    "jax.numpy.int64": "int64",
    "jax.numpy.uint8": "uint8",
    "jax.numpy.uint16": "uint16",
    "jax.numpy.uint32": "uint32",
    "jax.numpy.uint64": "uint64",
    "jax.numpy.bool_": "bool",
    "jax.numpy.bool": "bool",
    "jax.numpy.bfloat16": "bfloat16",

    # --- OpenCV (cv2) ---
    "cv2.cv_8u": "uint8",
    "cv2.cv_8s": "int8",
    "cv2.cv_16u": "uint16",
    "cv2.cv_16s": "int16",
    "cv2.cv_32s": "int32",
    "cv2.cv_32f": "float32",
    "cv2.cv_64f": "float64",

    # --- ONNX / misc frameworks sometimes reuse numpy names directly ---

    # --- Generic / human text forms ---
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "int": "int32",
    "long": "int64",
    "bool": "bool",
    "boolean": "bool",
}


def _strip_class_wrapper(v: str) -> str:
    """
    Handle strings like "<class 'numpy.float32'>" or
    "<class 'torch.float32'>".
    """
    m = re.search(r"<class ['\"]([^'\"]+)['\"]>", v)
    if m:
        return m.group(1).lower()
    return v


def normalize_dtype(value: str) -> Optional[str]:
    """
    Convert any dtype string into a canonical unified dtype.

    Returns:
        canonical dtype (e.g. "float32") or None if unknown / unsupported.
    """
    if value is None:
        return None

    v = str(value).strip()
    if not v:
        return None

    v = v.lower()
    v = _strip_class_wrapper(v)

    # Remove common prefixes that are not essential
    for prefix in ("typing.", "builtins."):
        if v.startswith(prefix):
            v = v[len(prefix):]

    # Direct hit on canonical name
    if v in UNIFIED_DTYPES:
        return v

    # Exact or suffix matches from normalization table
    for k, out in DTYPE_NORMALIZATION_MAP.items():
        kl = k.lower()
        if v == kl or v.endswith("." + kl) or v == kl.replace("numpy.", "np."):
            return out

    # Very common aliases / textual forms
    simple_aliases = {
        "u8": "uint8",
        "i8": "int8",
        "u16": "uint16",
        "i16": "int16",
        "u32": "uint32",
        "i32": "int32",
        "u64": "uint64",
        "i64": "int64",
    }
    if v in simple_aliases:
        return simple_aliases[v]

    # Patterns like "int32", "uint8", "float64", "complex128" etc.
    if re.fullmatch(r"u?int(8|16|32|64)", v):
        return v
    if re.fullmatch(r"float(8|16|32|64)", v):
        return v
    if re.fullmatch(r"complex(64|128)", v):
        return v

    # Sometimes appear as "uint8_t" etc.
    if v.endswith("_t"):
        core = v[:-2]
        if core in UNIFIED_DTYPES:
            return core

    return None  # Unknown or non-numeric dtype → ignored


# -------------------------------------------------------------
# PARAMETER RULES (user-overridable)
# -------------------------------------------------------------

def load_param_rules(config_path: Optional[str]) -> List[Dict[str, Any]]:
    if not config_path:
        return []
    p = Path(config_path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


# -------------------------------------------------------------
# GENERALIZED PARAMETER TYPE CLASSIFIER
# Works for TensorFlow, PyTorch, JAX, NumPy, OpenCV, SciPy, etc.
# -------------------------------------------------------------

def apply_param_rules(
    name: str,
    proposed_type: Optional[str],
    rules: List[Dict[str, Any]]
) -> str:
    """
    Decide a coarse parameter "type" category for fuzzing:
        tensor | shape | number | bool | str | any

    The logic is:
    1) Honor explicit user rules (param_rules.json)
    2) Apply general DL heuristics (kernel_size, stride, padding, axis, etc.)
    3) Fall back to the LLM's proposed type if valid
    4) Default to "any"
    """
    nm = (name or "").lower()
    proposed = (proposed_type or "").lower()

    # 1) User-defined rule → highest priority
    for r in rules:
        ty = r.get("type")
        if not ty:
            continue
        if "if_name_is" in r and nm in [x.lower() for x in r["if_name_is"]]:
            return ty
        if "if_name_contains" in r and any(key.lower() in nm for key in r["if_name_contains"]):
            return ty

    # 2) Universal heuristics (framework-agnostic)

    # Tensor-like things: images, arrays, tensors, feature maps, etc.
    if nm in ("x", "y", "input", "inputs", "data", "values", "images", "image", "img", "frame"):
        return "tensor"
    if any(tok in nm for tok in ("tensor", "array", "ndarray", "feature_map", "volume")):
        return "tensor"

    # Shape-like / size / kernel / window parameters
    if any(k in nm for k in (
        "shape", "newshape", "target_shape",
        "size", "ksize", "dsize",
        "kernel", "kernel_size",
        "window", "window_size",
        "pool", "pool_size",
        "padding", "pad", "crop", "dilation", "stride", "strides",
        "height", "width", "rows", "cols", "channels"
    )):
        return "shape"

    # Axis / indices / dimensions
    if any(k in nm for k in (
        "axis", "axes", "dim", "dims",
        "rank", "index", "indices",
        "row", "col", "channel"
    )):
        return "number"

    # Boolean flags
    if (
        nm.startswith("is_") or nm.startswith("has_") or nm.startswith("use_")
        or nm.endswith("_flag") or nm.endswith("_only")
        or nm in ("training", "train", "inplace", "normalize", "center_crop")
    ):
        return "bool"

    # Strings / paths / modes / devices
    if nm in ("path", "file", "filename", "filepath", "name", "mode", "device", "backend", "dtype"):
        # dtype is treated as a "str" placeholder, not "number"
        return "str"

    # 3) Honor LLM suggestion if it's already in our allowed set
    if proposed in {"tensor", "shape", "number", "bool", "str", "any"}:
        return proposed

    # 4) Default
    return "any"


# -------------------------------------------------------------
# UNIVERSAL DTYPE COERCION
# Reduces ANY library dtype list into canonical unified list.
# Safe for *all* frameworks: unknown entries are dropped.
# -------------------------------------------------------------

def coerce_allowed_dtypes(seq, allowed: Optional[set] = None):
    """
    Take a list of arbitrary dtype descriptors (strings from the LLM),
    normalize them, filter them to the UNIFIED_DTYPES set, and
    return a sorted unique list. If nothing survives, return None.

    This function is *safe* across all libraries: weird types
    (objects, strings, enums) will just be ignored.
    """
    if not isinstance(seq, list):
        return None

    allowed = allowed or UNIFIED_DTYPES
    out: List[str] = []

    for s in seq:
        norm = normalize_dtype(s)
        if norm and norm in allowed:
            out.append(norm)

    # Deduplicate and sort for determinism
    out = sorted(set(out))
    return out or None
