#!/usr/bin/env python3
from __future__ import annotations

"""Generalized DL API selector for thesis-scale fuzzing experiments.

Benchmark-focused fixes in this version
--------------------------------------
- preserves the original generalized selector structure
- guarantees at least one slot for each available core namespace before proportional fill
- prefers clean computational namespaces over niche/device/experimental ones
- applies hard overflow stops so a single namespace cannot dominate the benchmark
- keeps family-floor coverage before overflow
"""

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


COMMON_FAMILY_WEIGHTS = {
    "tensor_creation": 1.00,
    "shape_manipulation": 1.00,
    "dtype_casting": 0.85,
    "indexing": 1.00,
    "reduction": 1.00,
    "random": 0.95,
    "linalg": 1.00,
    "neural_layers": 1.00,
    "loss_metric": 0.90,
    "autodiff": 0.95,
    "optimizer": 0.90,
    "data_pipeline": 0.85,
    "io_serialization": 0.70,
    "image_signal_text": 0.75,
    "sparse_ragged": 0.85,
    "control_flow": 0.80,
    "probabilistic": 0.70,
    "device_memory": 0.35,
    "distributed_compilation": 0.35,
    "other": 0.25,
}


@dataclass(frozen=True)
class LibraryProfile:
    name: str
    roots: Tuple[str, ...]
    core_namespaces: Tuple[str, ...]
    deprioritized_namespaces: Tuple[str, ...]
    excluded_namespaces: Tuple[str, ...]
    namespace_weights: Dict[str, float]
    namespace_caps: Dict[str, int]
    preferred_core_budget: float = 0.80
    low_value_namespaces: Tuple[str, ...] = field(default_factory=tuple)
    family_floor_target: Dict[str, int] = field(default_factory=dict)
    namespace_overflow_stops: Dict[str, int] = field(default_factory=dict)


LIBRARY_PROFILES: Dict[str, LibraryProfile] = {
    "torch": LibraryProfile(
        name="torch",
        roots=("torch",),
        core_namespaces=(
            "top_level", "nn", "nn.functional", "functional", "linalg", "fft",
            "special", "autograd", "sparse", "distributions",
        ),
        deprioritized_namespaces=(
            "optim", "utils.data", "masked", "cuda",
            "jit", "fx", "importlib", "futures", "backends", "quantization", "ao", "Tensor",
        ),
        excluded_namespaces=("_C", "_VF", "_ops", "_logging", "return_types"),
        namespace_weights={
            "top_level": 1.00,
            "nn": 1.20,
            "nn.functional": 1.25,
            "functional": 1.12,
            "linalg": 1.10,
            "fft": 0.90,
            "special": 0.85,
            "autograd": 0.90,
            "sparse": 0.85,
            "distributions": 0.65,
            "optim": 0.45,
            "utils.data": 0.40,
            "masked": 0.25,
            "cuda": 0.20,
        },
        namespace_caps={
            "top_level": 14,
            "nn": 24,
            "nn.functional": 18,
            "functional": 48,
            "linalg": 16,
            "fft": 8,
            "special": 8,
            "autograd": 10,
            "sparse": 10,
            "distributions": 6,
            "optim": 4,
            "utils.data": 4,
            "masked": 6,
            "cuda": 2,
            "quantization": 1,
            "jit": 1,
            "fx": 1,
            "backends": 1,
            "Tensor": 0,
        },
        preferred_core_budget=0.92,
        low_value_namespaces=("accelerator", "windows", "mtia", "xpu", "mps", "ipu", "npu"),
        family_floor_target={
            "tensor_creation": 4,
            "shape_manipulation": 4,
            "indexing": 4,
            "reduction": 4,
            "random": 3,
            "linalg": 4,
            "dtype_casting": 3,
            "neural_layers": 3,
            "autodiff": 2,
            "sparse_ragged": 2,
        },
        namespace_overflow_stops={
            "functional": 55,
            "nn": 20,
            "nn.functional": 18,
            "linalg": 16,
            "fft": 10,
            "special": 10,
            "autograd": 10,
            "sparse": 10,
            "masked": 8,
            "optim": 4,
            "utils.data": 4,
        },
    ),
    "tensorflow": LibraryProfile(
        name="tensorflow",
        roots=("tensorflow", "tf"),
        core_namespaces=(
            "top_level", "nn", "linalg", "math", "random", "data", "image", "io",
            "strings", "signal", "lookup", "sparse", "ragged", "losses", "metrics",
            "keras.layers", "keras.losses", "keras.metrics", "keras.optimizers",
            "keras.activations", "keras.initializers",
        ),
        deprioritized_namespaces=("quantization", "saved_model", "distribute", "debugging", "config", "queue", "tpu", "keras.applications", "keras.datasets"),
        excluded_namespaces=("raw_ops", "dtensor", "compiler", "python", "tools", "core", "__internal__"),
        namespace_weights={
            "top_level": 1.00,
            "nn": 0.95,
            "linalg": 0.95,
            "math": 1.00,
            "random": 0.80,
            "data": 0.95,
            "image": 0.75,
            "io": 0.75,
            "strings": 0.65,
            "signal": 0.60,
            "lookup": 0.55,
            "sparse": 0.60,
            "ragged": 0.60,
            "losses": 0.55,
            "metrics": 0.55,
            "keras.layers": 1.20,
            "keras.losses": 0.65,
            "keras.metrics": 0.65,
            "keras.optimizers": 0.60,
            "keras.activations": 0.60,
            "keras.initializers": 0.50,
        },
        namespace_caps={
            "top_level": 12,
            "nn": 10,
            "linalg": 9,
            "math": 10,
            "random": 7,
            "data": 10,
            "image": 7,
            "io": 7,
            "strings": 6,
            "signal": 6,
            "lookup": 5,
            "sparse": 5,
            "ragged": 5,
            "losses": 4,
            "metrics": 4,
            "keras.layers": 12,
            "keras.losses": 5,
            "keras.metrics": 5,
            "keras.optimizers": 4,
            "keras.activations": 4,
            "keras.initializers": 4,
            "keras.ops": 3,
        },
        low_value_namespaces=("keras.preprocessing", "keras.visualization", "keras.utils", "keras.quantizers"),
        family_floor_target={
            "tensor_creation": 4,
            "shape_manipulation": 4,
            "indexing": 3,
            "reduction": 3,
            "random": 3,
            "linalg": 4,
            "dtype_casting": 2,
            "neural_layers": 4,
            "loss_metric": 4,
            "image_signal_text": 4,
            "sparse_ragged": 2,
            "io_serialization": 3,
        },
        namespace_overflow_stops={
            "top_level": 14,
            "math": 12,
            "nn": 12,
            "linalg": 10,
            "image": 8,
            "io": 8,
            "strings": 7,
            "signal": 7,
            "keras.layers": 12,
            "keras.ops": 4,
        },
    ),
    "jax": LibraryProfile(
        name="jax",
        roots=("jax", "jnp"),
        core_namespaces=(
            "top_level", "numpy", "linalg", "lax", "random", "nn",
            "scipy.linalg", "scipy.special", "scipy.signal", "tree", "image",
        ),
        deprioritized_namespaces=("experimental", "config", "debug", "monitoring", "sharding"),
        excluded_namespaces=("_src", "extend", "interpreters"),
        namespace_weights={
            "top_level": 0.90,
            "numpy": 1.20,
            "linalg": 1.00,
            "lax": 1.10,
            "random": 0.95,
            "nn": 0.85,
            "scipy.linalg": 0.75,
            "scipy.special": 0.70,
            "scipy.signal": 0.60,
            "tree": 0.45,
            "image": 0.50,
        },
        namespace_caps={
            "top_level": 10,
            "numpy": 26,
            "linalg": 10,
            "lax": 18,
            "random": 10,
            "nn": 10,
            "scipy.linalg": 6,
            "scipy.special": 6,
            "scipy.signal": 4,
            "tree": 3,
            "image": 4,
            "experimental": 1,
        },
        preferred_core_budget=0.90,
        low_value_namespaces=("tree_util",),
        family_floor_target={
            "tensor_creation": 4,
            "shape_manipulation": 4,
            "indexing": 3,
            "reduction": 3,
            "random": 4,
            "linalg": 4,
            "dtype_casting": 2,
            "neural_layers": 3,
            "autodiff": 3,
        },
        namespace_overflow_stops={"numpy": 40, "lax": 24},
    ),
    "paddle": LibraryProfile(
        name="paddle",
        roots=("paddle",),
        core_namespaces=(
            "top_level", "nn", "nn.functional", "optimizer", "linalg", "fft",
            "sparse", "signal", "vision", "audio", "metric", "dataset",
            "io", "distribution", "static",
        ),
        deprioritized_namespaces=("distributed", "incubate", "device", "amp", "geometric"),
        excluded_namespaces=("base", "fluid", "pir", "cinn", "_C_ops"),
        namespace_weights={
            "top_level": 1.00,
            "nn": 1.15,
            "nn.functional": 1.10,
            "optimizer": 0.75,
            "linalg": 0.95,
            "fft": 0.55,
            "sparse": 0.60,
            "signal": 0.55,
            "vision": 0.75,
            "audio": 0.45,
            "metric": 0.50,
            "dataset": 0.60,
            "io": 0.55,
            "distribution": 0.40,
            "static": 0.45,
        },
        namespace_caps={
            "top_level": 12,
            "nn": 20,
            "nn.functional": 16,
            "optimizer": 6,
            "linalg": 10,
            "fft": 4,
            "sparse": 5,
            "signal": 4,
            "vision": 7,
            "audio": 3,
            "metric": 4,
            "dataset": 4,
            "io": 4,
            "distribution": 3,
            "static": 4,
            "distributed": 1,
            "incubate": 1,
        },
        preferred_core_budget=0.90,
        low_value_namespaces=("device",),
        family_floor_target={
            "tensor_creation": 4,
            "shape_manipulation": 4,
            "indexing": 3,
            "reduction": 3,
            "random": 3,
            "linalg": 4,
            "dtype_casting": 2,
            "neural_layers": 4,
            "loss_metric": 2,
        },
        namespace_overflow_stops={"nn": 22, "nn.functional": 18, "linalg": 12},
    ),
}


GENERIC_BAD_PATTERNS = [
    r"\._",
    r"\.__",
    r"\bType\b",
    r"\bCapsule\b",
    r"\bStorage\b",
    r"\bTypedStorage\b",
    r"\bUntypedStorage\b",
    r"\bClassType\b",
    r"\bDeviceObjType\b",
    r"\bScript\b",
    r"\bTracing\b",
]

GENERIC_EXACT_BAD_NAMES = {
    "torch.ClassType", "torch.TupleType", "torch.OptionalType", "torch.AnyType",
    "torch.EnumType", "torch.StringType", "torch.IntType", "torch.FloatType",
    "torch.BoolType", "torch.ComplexType", "torch.DeviceObjType",
}

GENERIC_STOPWORDS = {
    "from", "to", "with", "without", "get", "set", "is", "as", "of", "for", "the",
    "and", "or", "in", "on", "at", "by", "into", "new", "all", "any", "none",
}

GENERIC_LOW_VALUE_NAMESPACE_PATTERNS = [
    r"^(testing?|debug(?:ging)?|benchmark(?:s)?|example(?:s)?)$",
    r"^(cpu|gpu|xpu|mps|npu|ipu|tpu|mtia)$",
    r"^(accelerator|windows)$",
]

CLASSLIKE_NAMESPACE_SUFFIXES = (
    "Tensor", "Storage", "Type", "Spec", "Module", "Scaler", "Stream", "Event", "Handle",
)

FEATURE_PATTERNS: Dict[str, List[str]] = {
    "tensor_shape": [r"\bshape\b", r"reshape", r"rank", r"dimension", r"\bdims?\b", r"\bsize\b"],
    "dtype_casting": [r"\bdtype\b", r"\bcast\b", r"type conversion", r"numeric type"],
    "broadcasting": [r"broadcast"],
    "indexing_slicing": [r"\bindex\b", r"\bslice\b", r"gather", r"scatter", r"\bmask\b", r"\bwhere\b", r"take"],
    "reduction": [r"reduce", r"\bsum\b", r"\bmean\b", r"\bprod\b", r"\bmax\b", r"\bmin\b", r"argmax", r"argmin", r"norm"],
    "creation_init": [r"create", r"initialize", r"\bfill\b", r"\bzeros?\b", r"\bones?\b", r"\bempty\b", r"\bfull\b", r"\barange\b", r"\brange\b", r"linspace", r"logspace", r"\beye\b", r"constant"],
    "randomness": [r"random", r"\bseed\b", r"sample", r"distribution", r"uniform", r"normal", r"randn?", r"bernoulli", r"multinomial"],
    "linear_algebra": [r"matrix", r"matmul", r"linalg", r"eigen", r"\bsvd\b", r"\bqr\b", r"determin", r"\bsolve\b", r"cholesky", r"inverse"],
    "gradients_autodiff": [r"gradient", r"derivative", r"jacobian", r"hessian", r"backprop", r"autograd", r"grad", r"tape"],
    "control_flow": [r"\bwhile\b", r"\bcond\b", r"branch", r"control flow", r"\bloop\b", r"scan"],
    "neural_networks": [r"activation", r"convolution", r"\bconv\b", r"\bpool\b", r"dropout", r"batch norm", r"softmax", r"relu", r"\bloss\b", r"attention", r"embedding", r"layer norm", r"normalization"],
    "dataset_pipeline": [r"dataset", r"iterator", r"\bbatch\b", r"shuffle", r"prefetch", r"\brepeat\b", r"\bmap\b", r"dataloader", r"sampler", r"cache"],
    "io_parsing": [r"parse", r"decode", r"encode", r"serialize", r"\bread\b", r"\bwrite\b", r"\bload\b", r"\bsave\b", r"checkpoint"],
    "image_ops": [r"image", r"resize", r"\bcrop\b", r"bounding box", r"non_max_suppression", r"pixel"],
    "signal_ops": [r"signal", r"\bfft\b", r"\bstft\b", r"window", r"spectrogram"],
    "strings_text": [r"string", r"regex", r"unicode", r"token"],
    "sparse_ragged": [r"sparse", r"csr", r"csc", r"bsr", r"bsc", r"coo", r"ragged"],
    "quantization": [r"quantiz"],
    "optimizer": [r"optimizer", r"adam", r"sgd", r"momentum", r"adagrad", r"rmsprop", r"weight decay"],
    "distribution_probabilistic": [r"distribution", r"gaussian", r"bernoulli", r"categorical", r"poisson"],
    "stateful_side_effects": [r"stateful", r"resource", r"variable", r"mutable", r"assign", r"in-place", r"inplace"],
    "constraints_validation": [r"\bmust\b", r"requires", r"raises", r"\berror\b", r"invalid", r"constraint"],
    "multi_input_signature": [r"\binputs?\b", r"\barguments?\b", r"\bparameters?\b", r"\btensors?\b", r"list of", r"tuple of", r"sequence of"],
    "distributed_compilation": [r"jit", r"compile", r"xla", r"distributed", r"parallel", r"pmap", r"shard"],
}

IMPORTANT_NAME_TOKENS = {
    "relu", "softmax", "conv", "pool", "dropout", "batch", "matmul", "gather", "scatter",
    "reshape", "cast", "fft", "decode", "parse", "dataset", "random", "sparse", "ragged",
    "quantize", "gradient", "reduce", "argmax", "argmin", "resize", "crop", "serialize",
    "lookup", "segment", "while", "cond", "attention", "embedding", "norm", "adam", "sgd",
    "svd", "solve", "cholesky", "inverse", "bernoulli", "normal", "mean", "sum", "sort",
}

FAMILY_RULES: List[Tuple[str, List[str]]] = [
    ("optimizer", [r"optimizer", r"adam", r"sgd", r"rmsprop", r"adagrad", r"weight_decay"]),
    ("autodiff", [r"autograd", r"grad", r"gradient", r"jacobian", r"hessian", r"vjp", r"jvp"]),
    ("random", [r"random", r"rand", r"bernoulli", r"uniform", r"normal", r"sample"]),
    ("linalg", [r"linalg", r"matmul", r"svd", r"qr", r"solve", r"inverse", r"cholesky", r"eigh"]),
    ("reduction", [r"reduce", r"sum", r"mean", r"prod", r"amax", r"amin", r"argmax", r"argmin", r"norm"]),
    ("indexing", [r"index", r"slice", r"gather", r"scatter", r"take", r"where", r"mask"]),
    ("shape_manipulation", [r"reshape", r"squeeze", r"unsqueeze", r"view", r"transpose", r"permute", r"concat", r"stack", r"split", r"tile", r"repeat"]),
    ("dtype_casting", [r"cast", r"dtype", r"astype"]),
    ("neural_layers", [r"relu", r"gelu", r"sigmoid", r"softmax", r"conv", r"pool", r"dropout", r"embedding", r"attention", r"batch_norm", r"layer_norm", r"dense", r"linear"]),
    ("loss_metric", [r"loss", r"metric", r"accuracy", r"cross_entropy", r"mse", r"mae"]),
    ("data_pipeline", [r"dataset", r"dataloader", r"sampler", r"batch", r"shuffle", r"prefetch", r"cache"]),
    ("io_serialization", [r"load", r"save", r"read", r"write", r"parse", r"decode", r"encode", r"serialize"]),
    ("image_signal_text", [r"image", r"resize", r"crop", r"fft", r"stft", r"signal", r"string", r"token"]),
    ("sparse_ragged", [r"sparse", r"ragged", r"csr", r"coo", r"csc", r"bsr", r"bsc"]),
    ("control_flow", [r"while", r"cond", r"scan", r"fori_loop", r"switch"]),
    ("probabilistic", [r"distribution", r"gaussian", r"bernoulli", r"categorical", r"poisson"]),
    ("distributed_compilation", [r"jit", r"compile", r"distributed", r"pmap", r"xla", r"parallel"]),
    ("device_memory", [r"cuda", r"device", r"stream", r"event", r"memory", r"xpu", r"mtia"]),
    ("tensor_creation", [r"zeros", r"ones", r"empty", r"full", r"arange", r"linspace", r"eye", r"constant", r"tensor"]),
]


def safe_text(x: object) -> str:
    return "" if pd.isna(x) else str(x)


def infer_library(api_name: str) -> str:
    if not isinstance(api_name, str):
        return "unknown"
    name = api_name.strip()
    if name.startswith("torch."):
        return "torch"
    if name.startswith("tensorflow.") or name.startswith("tf."):
        return "tensorflow"
    if name.startswith("jax.") or name.startswith("jnp."):
        return "jax"
    if name.startswith("paddle."):
        return "paddle"
    return "unknown"


def is_classlike_namespace(namespace: str) -> bool:
    if not namespace or "." in namespace:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9_]*", namespace))


def is_low_value_namespace(namespace: str, profile: LibraryProfile) -> bool:
    if not namespace or namespace in {"top_level", "unknown"}:
        return False
    if namespace in profile.low_value_namespaces:
        return True
    if namespace in profile.core_namespaces or namespace in profile.namespace_weights or namespace in profile.namespace_caps:
        return False
    return any(re.fullmatch(pat, namespace) for pat in GENERIC_LOW_VALUE_NAMESPACE_PATTERNS)


def is_structural_noise_namespace(namespace: str, profile: LibraryProfile) -> bool:
    if not namespace or namespace in {"top_level", "unknown"}:
        return False
    if namespace in profile.core_namespaces or namespace in profile.namespace_weights or namespace in profile.namespace_caps:
        return False
    if "." in namespace:
        return False
    if namespace.endswith(CLASSLIKE_NAMESPACE_SUFFIXES):
        return True
    return is_classlike_namespace(namespace)


def is_clean_namespace(namespace: str, profile: LibraryProfile) -> bool:
    return namespace in {"top_level", "unknown"} or (
        namespace not in profile.deprioritized_namespaces
        and not is_low_value_namespace(namespace, profile)
        and not is_structural_noise_namespace(namespace, profile)
    )


def api_leaf(api_name: str) -> str:
    return api_name.rsplit(".", 1)[-1].strip().lower()


def strip_examples_and_code(doc_text: str) -> str:
    doc = doc_text or ""
    doc = re.sub(r"```.*?```", " ", doc, flags=re.DOTALL)
    doc = re.sub(r"(?m)^\s*>>>.*$", " ", doc)
    doc = re.split(r"(?i)(?:\bexamples?:\b|\bexample usage\b)", doc)[0]
    return doc


def doc_fingerprint(doc_text: str) -> str:
    doc = strip_examples_and_code(doc_text or "").lower()
    doc = re.sub(r"\s+", " ", doc)
    doc = re.sub(r"[^a-z0-9 ]+", " ", doc)
    return " ".join(doc.split()[:80])


def duplicate_key(api_name: str, doc_text: str) -> Tuple[str, str]:
    return (api_leaf(api_name), doc_fingerprint(doc_text))


def extract_namespace(api_name: str, library: str, keras_split: bool = True) -> str:
    if not isinstance(api_name, str) or not api_name.strip():
        return "unknown"
    parts = api_name.strip().split(".")
    if not parts:
        return "unknown"

    if library == "torch":
        if parts[0] != "torch":
            return parts[0]
        if len(parts) <= 2:
            return "top_level"
        if parts[1] == "nn":
            if len(parts) >= 4 and parts[2] == "functional":
                return "nn.functional"
            return "nn"
        if parts[1] == "utils" and len(parts) >= 3 and parts[2] == "data":
            return "utils.data"
        if parts[1] == "functional":
            return "functional"
        return parts[1]

    if library == "tensorflow":
        root = parts[0]
        if root not in {"tensorflow", "tf"}:
            return root
        if len(parts) <= 2:
            return "top_level"
        ns1 = parts[1]
        if ns1 == "keras" and keras_split:
            return f"keras.{parts[2]}" if len(parts) >= 3 else "keras"
        if ns1 == "RaggedTensor":
            return "ragged"
        return ns1

    if library == "jax":
        if parts[0] == "jnp":
            return "numpy"
        if parts[0] != "jax":
            return parts[0]
        if len(parts) == 1:
            return "top_level"
        if parts[1] == "numpy":
            if len(parts) >= 3 and parts[2] == "linalg":
                return "linalg"
            return "numpy"
        if parts[1] == "scipy":
            if len(parts) >= 3 and parts[2] in {"linalg", "special", "signal"}:
                return f"scipy.{parts[2]}"
            return "scipy"
        if parts[1] == "tree":
            return "tree"
        return parts[1] if len(parts) > 2 else "top_level"

    if library == "paddle":
        if parts[0] != "paddle":
            return parts[0]
        if len(parts) <= 2:
            return "top_level"
        ns1 = parts[1]
        if ns1 == "nn":
            if len(parts) >= 4 and parts[2] == "functional":
                return "nn.functional"
            return "nn"
        if ns1 in {"vision", "audio", "dataset"}:
            return ns1
        return ns1

    return parts[1] if len(parts) > 2 else "top_level"


def tokenize_api_name(api_name: str) -> Set[str]:
    toks = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", api_name.replace("_", " ").replace(".", " "))
    return {t.lower() for t in toks if t}


def normalized_name_tokens(api_name: str) -> Set[str]:
    return {t for t in tokenize_api_name(api_name) if t not in GENERIC_STOPWORDS and len(t) > 1}


def estimate_doc_quality(doc_text: str) -> int:
    doc = (doc_text or "").lower()
    score = 0
    if re.search(r"\bargs?:\b|\bparameters?:\b", doc):
        score += 2
    if re.search(r"\breturns?:\b", doc):
        score += 1
    if re.search(r"\braises?:\b|\berrors?:\b", doc):
        score += 1
    if re.search(r"\bexample\b|>>>", doc):
        score += 1
    if len(doc) > 400:
        score += 1
    return score


def estimate_param_count(doc_text: str) -> int:
    doc = doc_text or ""
    patterns = [
        r"(?m)^\s{0,4}[A-Za-z_][A-Za-z0-9_]*\s*:\s*",
        r"(?m)^\s{0,4}-\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*",
    ]
    count = 0
    for pat in patterns:
        count = max(count, len(re.findall(pat, doc)))
    return min(count, 25)


def signature_complexity_bucket(param_count: int) -> str:
    if param_count >= 8:
        return "high"
    if param_count >= 4:
        return "medium"
    return "low"


def extract_features(api_name: str, doc_text: str, namespace: str) -> Set[str]:
    semantic_doc = strip_examples_and_code(doc_text)
    text = f"{api_name} {semantic_doc} {namespace}".lower()
    feats: Set[str] = set()
    for feat, patterns in FEATURE_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            feats.add(feat)
    for tok in tokenize_api_name(api_name) & IMPORTANT_NAME_TOKENS:
        feats.add(f"token::{tok}")
    if re.search(r"\bargs?\b|\barguments?\b|\bparameters?\b", text):
        feats.add("doc_has_parameters")
    if re.search(r"\breturns?\b", text):
        feats.add("doc_has_returns")
    if re.search(r"\bexample\b|>>>", text):
        feats.add("doc_has_example")
    if re.search(r"\braises?\b|\berrors?\b", text):
        feats.add("doc_has_errors")
    return feats


def infer_primary_family(api_name: str, doc_text: str, namespace: str) -> str:
    name_text = f"{api_name} {namespace}".lower()
    for family, patterns in FAMILY_RULES:
        if any(re.search(p, name_text) for p in patterns):
            return family
    semantic_doc = strip_examples_and_code(doc_text).lower()[:700]
    for family, patterns in FAMILY_RULES:
        if any(re.search(p, semantic_doc) for p in patterns):
            return family
    return "other"


def is_good_public_api(api_name: str, library: str, namespace: str) -> bool:
    if not isinstance(api_name, str) or not api_name.strip():
        return False
    if api_name in GENERIC_EXACT_BAD_NAMES:
        return False
    if library not in LIBRARY_PROFILES:
        return False
    profile = LIBRARY_PROFILES[library]
    if namespace in profile.excluded_namespaces:
        return False
    for bad_ns in profile.excluded_namespaces:
        if f".{bad_ns}." in api_name or api_name.endswith(f".{bad_ns}"):
            return False
    for pat in GENERIC_BAD_PATTERNS:
        if re.search(pat, api_name):
            return False
    if not api_name.startswith(profile.roots):
        return False
    if is_structural_noise_namespace(namespace, profile):
        return False
    return True


def hamilton_allocate(weights: Dict[str, float], capacities: Dict[str, int], total: int) -> Dict[str, int]:
    if total <= 0 or not capacities:
        return {k: 0 for k in capacities}
    positive_items = {k: v for k, v in capacities.items() if v > 0 and weights.get(k, 0) > 0}
    if not positive_items:
        return {k: 0 for k in capacities}
    total_weight = sum(weights[k] for k in positive_items)
    if total_weight <= 0:
        return {k: 0 for k in capacities}
    raw = {k: total * (weights[k] / total_weight) for k in positive_items}
    alloc = {k: min(positive_items[k], int(math.floor(raw[k]))) for k in positive_items}
    used = sum(alloc.values())
    remainder_order = sorted(positive_items, key=lambda k: (raw[k] - math.floor(raw[k]), weights[k], capacities[k]), reverse=True)
    while used < total:
        progressed = False
        for k in remainder_order:
            if used >= total:
                break
            if alloc[k] < positive_items[k]:
                alloc[k] += 1
                used += 1
                progressed = True
        if not progressed:
            break
    return {k: alloc.get(k, 0) for k in capacities}


def compute_namespace_caps(ns_counts: Counter, library: str, target: int) -> Dict[str, int]:
    profile = LIBRARY_PROFILES[library]
    caps: Dict[str, int] = {}
    default_noncore_cap = max(4, math.ceil(target * 0.08))
    default_core_cap = max(8, math.ceil(target * 0.18))
    low_value_cap = max(1, math.ceil(target * 0.02))
    for ns, count in ns_counts.items():
        if is_structural_noise_namespace(ns, profile):
            cap = 0
        elif ns in profile.namespace_caps:
            cap = profile.namespace_caps[ns]
        elif ns in profile.core_namespaces:
            cap = default_core_cap
        elif ns in profile.deprioritized_namespaces:
            cap = max(1, math.ceil(target * 0.02))
        elif is_low_value_namespace(ns, profile):
            cap = low_value_cap
        else:
            cap = default_noncore_cap
        caps[ns] = min(count, cap)
    return caps


def allocate_by_namespace_group(ns_counts: Counter, caps: Dict[str, int], quotas: Dict[str, int], namespaces: Sequence[str], weight_fn, remaining: int) -> int:
    if remaining <= 0:
        return 0
    weights: Dict[str, float] = {}
    capacities: Dict[str, int] = {}
    for ns in namespaces:
        room = max(0, caps.get(ns, 0) - quotas.get(ns, 0))
        if room <= 0:
            continue
        w = weight_fn(ns)
        if w <= 0:
            continue
        weights[ns] = w
        capacities[ns] = room
    extra = hamilton_allocate(weights, capacities, remaining)
    used = 0
    for ns, add in extra.items():
        quotas[ns] = quotas.get(ns, 0) + add
        used += add
    return used


def compute_namespace_quotas(ns_counts: Counter, caps: Dict[str, int], library: str, target: int) -> Dict[str, int]:
    profile = LIBRARY_PROFILES[library]
    quotas: Dict[str, int] = {}
    available_core = [ns for ns in profile.core_namespaces if ns_counts.get(ns, 0) > 0 and caps.get(ns, 0) > 0]
    core_budget = min(target, int(round(target * profile.preferred_core_budget)))

    # Guarantee at least one representative from every available core namespace.
    for ns in available_core:
        quotas[ns] = 1
    remaining_core_budget = max(0, core_budget - sum(quotas.values()))
    if remaining_core_budget > 0:
        core_weights = {ns: profile.namespace_weights.get(ns, 0.7) for ns in available_core}
        core_capacities = {ns: max(0, caps[ns] - quotas.get(ns, 0)) for ns in available_core}
        extra = hamilton_allocate(core_weights, core_capacities, remaining_core_budget)
        for ns, add in extra.items():
            quotas[ns] = quotas.get(ns, 0) + add

    remaining = target - sum(quotas.values())
    if remaining <= 0:
        return {ns: q for ns, q in quotas.items() if q > 0}

    clean_noncore = [ns for ns in ns_counts if ns not in available_core and is_clean_namespace(ns, profile)]
    used = allocate_by_namespace_group(ns_counts, caps, quotas, clean_noncore, lambda ns: math.sqrt(ns_counts[ns]) * profile.namespace_weights.get(ns, 0.60), remaining)
    remaining -= used
    if remaining > 0:
        low_value = [ns for ns in ns_counts if ns not in available_core and ns not in clean_noncore and is_low_value_namespace(ns, profile)]
        used = allocate_by_namespace_group(ns_counts, caps, quotas, low_value, lambda ns: math.sqrt(ns_counts[ns]) * 0.20, remaining)
        remaining -= used
    if remaining > 0:
        deprioritized = [ns for ns in ns_counts if ns in profile.deprioritized_namespaces]
        allocate_by_namespace_group(ns_counts, caps, quotas, deprioritized, lambda ns: math.sqrt(ns_counts[ns]) * 0.15, remaining)
    return {ns: q for ns, q in quotas.items() if q > 0}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def redundancy_penalty(row: dict, selected_rows: Sequence[dict]) -> float:
    penalty = 0.0
    row_tokens = row["name_tokens_norm"]
    row_ns = row["namespace"]
    for prev in reversed(selected_rows[-20:]):
        sim = jaccard_similarity(row_tokens, prev["name_tokens_norm"])
        if sim >= 0.75:
            penalty = max(penalty, 2.0)
        elif sim >= 0.55:
            penalty = max(penalty, 1.2)
        elif row_ns == prev["namespace"] and sim >= 0.40:
            penalty = max(penalty, 0.6)
    return penalty


def complexity_score(bucket: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(bucket, 1)


def candidate_score(row: dict, covered_features: Set[str], covered_families: Set[str], selected_rows: Sequence[dict], selected_ns_counts: Counter, selected_family_counts: Counter, quotas: Dict[str, int], profile: LibraryProfile) -> Tuple[float, int, int, int, int, int, int, int, str]:
    new_feats = row["features"] - covered_features
    family = row["primary_family"]
    ns = row["namespace"]
    under_ns_quota = int(selected_ns_counts[ns] < quotas.get(ns, 0))
    family_new = int(family not in covered_families)
    family_weight = COMMON_FAMILY_WEIGHTS.get(family, 0.25)
    under_family_pressure = int(selected_family_counts[family] == 0 and family_weight >= 0.75)
    core_bonus = int(ns in profile.core_namespaces)
    redundancy = redundancy_penalty(row, selected_rows)
    clean_bonus = int(is_clean_namespace(ns, profile))
    low_value_penalty = int(is_low_value_namespace(ns, profile))
    deprioritized_penalty = int(ns in profile.deprioritized_namespaces)
    family_soft_cap = max(4, math.ceil(max(1, sum(quotas.values())) * 0.18))
    family_overuse = max(0, selected_family_counts[family] - family_soft_cap)
    same_leaf_seen = int(any(api_leaf(prev["api_full_name"]) == api_leaf(row["api_full_name"]) for prev in selected_rows))
    ns_weight = profile.namespace_weights.get(ns, 0.45)
    score = (
        110.0 * len(new_feats)
        + 30.0 * family_new
        + 26.0 * under_ns_quota
        + 22.0 * under_family_pressure
        + 12.0 * family_weight
        + 8.0 * core_bonus
        + 6.0 * clean_bonus
        + 5.0 * ns_weight
        + 3.0 * row["feature_count"]
        + 2.0 * complexity_score(row["signature_complexity"])
        + 0.5 * row["doc_quality"]
        - 24.0 * redundancy
        - 10.0 * family_overuse
        - 18.0 * same_leaf_seen
        - 10.0 * low_value_penalty
        - 12.0 * deprioritized_penalty
    )
    return (round(score, 4), len(new_feats), family_new, under_ns_quota, int(round(100 * family_weight)), row["feature_count"], -len(row["api_full_name"]), -row["row_id"], family)


def overflow_stop_for_namespace(ns: str, caps: Dict[str, int], quotas: Dict[str, int], profile: LibraryProfile, target: int, margin: int) -> int:
    explicit = profile.namespace_overflow_stops.get(ns)
    if explicit is not None:
        return explicit
    base = max(caps.get(ns, 0), quotas.get(ns, 0)) + margin
    if ns in profile.core_namespaces:
        return min(base, max(caps.get(ns, 0), math.ceil(target * 0.35)))
    if is_clean_namespace(ns, profile):
        return min(base, max(caps.get(ns, 0), math.ceil(target * 0.12)))
    if ns in profile.deprioritized_namespaces:
        return min(base, max(caps.get(ns, 0), math.ceil(target * 0.04)))
    if is_low_value_namespace(ns, profile):
        return min(base, max(caps.get(ns, 0), math.ceil(target * 0.03)))
    return base


def greedy_pick(rows: Sequence[dict], need: int, covered_features: Set[str], covered_families: Set[str], selected_rows: List[dict], selected_ids: Set[int], selected_ns_counts: Counter, selected_family_counts: Counter, quotas: Dict[str, int], caps: Dict[str, int], profile: LibraryProfile, target: int, allowed_namespaces: Optional[Set[str]] = None, allow_cap_overflow: bool = False, overflow_margin: int = 0, family_filter: Optional[str] = None) -> List[dict]:
    chosen: List[dict] = []
    while len(chosen) < need:
        best: Optional[dict] = None
        best_score = None
        for row in rows:
            if row["row_id"] in selected_ids:
                continue
            if allowed_namespaces is not None and row["namespace"] not in allowed_namespaces:
                continue
            if family_filter is not None and row["primary_family"] != family_filter:
                continue
            ns = row["namespace"]
            hard_cap = caps.get(ns, 0)
            quota_cap = quotas.get(ns, 0)
            if allow_cap_overflow:
                if hard_cap == 0 and quota_cap == 0:
                    continue
                effective_cap = overflow_stop_for_namespace(ns, caps, quotas, profile, target, overflow_margin)
            else:
                effective_cap = hard_cap
            if selected_ns_counts[ns] >= effective_cap:
                continue
            score = candidate_score(row, covered_features, covered_families, selected_rows, selected_ns_counts, selected_family_counts, quotas, profile)
            if best is None or score > best_score:
                best = row
                best_score = score
        if best is None:
            break
        chosen.append(best)
        selected_rows.append(best)
        selected_ids.add(best["row_id"])
        selected_ns_counts[best["namespace"]] += 1
        selected_family_counts[best["primary_family"]] += 1
        covered_features |= best["features"]
        covered_families.add(best["primary_family"])
    return chosen


def progressive_overflow_fill(rows: Sequence[dict], target: int, covered_features: Set[str], covered_families: Set[str], selected_rows: List[dict], selected_ids: Set[int], selected_ns_counts: Counter, selected_family_counts: Counter, quotas: Dict[str, int], caps: Dict[str, int], profile: LibraryProfile, allowed_namespaces: Set[str]) -> bool:
    overflow_used = False
    for margin in (1, 2, 4, 8, 16, 1000):
        if len(selected_rows) >= target:
            break
        before = len(selected_rows)
        greedy_pick(rows, target - len(selected_rows), covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, allowed_namespaces=allowed_namespaces, allow_cap_overflow=True, overflow_margin=margin)
        if len(selected_rows) > before:
            overflow_used = True
    return overflow_used


def build_selection_reason(row: dict, selected_ns_counts: Counter, quotas: Dict[str, int], profile: LibraryProfile) -> str:
    feats = sorted([f for f in row["features"] if not f.startswith("token::")])
    feat_preview = ", ".join(feats[:4]) if feats else "doc semantics"
    reason_parts = [f"family={row['primary_family']}", f"namespace={row['namespace']}", f"features={feat_preview}"]
    if row["namespace"] in profile.core_namespaces:
        reason_parts.append("core_namespace")
    if is_clean_namespace(row["namespace"], profile):
        reason_parts.append("clean_namespace")
    if selected_ns_counts[row["namespace"]] <= quotas.get(row["namespace"], 0):
        reason_parts.append("quota_supported")
    if row["doc_quality"] >= 4:
        reason_parts.append("good_docs")
    if row["signature_complexity"] in {"medium", "high"}:
        reason_parts.append(f"signature_{row['signature_complexity']}")
    return "; ".join(reason_parts)


def maybe_fill_family_floors(rows: Sequence[dict], target: int, covered_features: Set[str], covered_families: Set[str], selected_rows: List[dict], selected_ids: Set[int], selected_ns_counts: Counter, selected_family_counts: Counter, quotas: Dict[str, int], caps: Dict[str, int], profile: LibraryProfile) -> None:
    if len(selected_rows) >= target:
        return
    for fam, floor in sorted(profile.family_floor_target.items(), key=lambda kv: (-COMMON_FAMILY_WEIGHTS.get(kv[0], 0.25), -kv[1], kv[0])):
        if len(selected_rows) >= target:
            break
        available = sum(1 for r in rows if r["primary_family"] == fam)
        if available <= 0:
            continue
        want = min(floor, available)
        if selected_family_counts[fam] >= want:
            continue
        greedy_pick(rows, min(target - len(selected_rows), want - selected_family_counts[fam]), covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, allow_cap_overflow=False, family_filter=fam)


def select_subset(df: pd.DataFrame, target: int, library: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    profile = LIBRARY_PROFILES[library]
    rows = df.to_dict("records")
    ns_counts = Counter(df["namespace"])
    family_counts = Counter(df["primary_family"])
    caps = compute_namespace_caps(ns_counts, library, target)
    quotas = compute_namespace_quotas(ns_counts, caps, library, target)

    by_ns: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_ns[row["namespace"]].append(row)
    for ns in by_ns:
        by_ns[ns] = sorted(by_ns[ns], key=lambda r: (r["feature_count"], COMMON_FAMILY_WEIGHTS.get(r["primary_family"], 0.25), complexity_score(r["signature_complexity"]), r["param_count"], r["doc_quality"], -len(r["api_full_name"])), reverse=True)

    selected_rows: List[dict] = []
    selected_ids: Set[int] = set()
    selected_ns_counts: Counter = Counter()
    selected_family_counts: Counter = Counter()
    covered_features: Set[str] = set()
    covered_families: Set[str] = set()

    for ns, q in sorted(quotas.items(), key=lambda kv: (kv[0] not in profile.core_namespaces, -kv[1], kv[0])):
        greedy_pick(by_ns.get(ns, []), q, covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, allow_cap_overflow=False)

    maybe_fill_family_floors(rows, target, covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile)

    major_families = [fam for fam, cnt in sorted(family_counts.items(), key=lambda kv: (-COMMON_FAMILY_WEIGHTS.get(kv[0], 0.3), -kv[1], kv[0])) if cnt > 0 and COMMON_FAMILY_WEIGHTS.get(fam, 0.3) >= 0.7]
    if len(selected_rows) < target:
        for fam in major_families:
            if len(selected_rows) >= target:
                break
            if selected_family_counts[fam] > 0:
                continue
            greedy_pick(rows, 1, covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, family_filter=fam, allow_cap_overflow=False)

    if len(selected_rows) < target:
        greedy_pick(rows, target - len(selected_rows), covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, allowed_namespaces=set(profile.core_namespaces), allow_cap_overflow=False)

    if len(selected_rows) < target:
        allowed = {ns for ns in ns_counts if is_clean_namespace(ns, profile)}
        greedy_pick(rows, target - len(selected_rows), covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, allowed_namespaces=allowed, allow_cap_overflow=False)

    if len(selected_rows) < target:
        low_value_allowed = {ns for ns in ns_counts if is_low_value_namespace(ns, profile) and not is_structural_noise_namespace(ns, profile)}
        greedy_pick(rows, target - len(selected_rows), covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, allowed_namespaces=low_value_allowed, allow_cap_overflow=False)

    if len(selected_rows) < target:
        greedy_pick(rows, target - len(selected_rows), covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, target, allowed_namespaces=set(profile.deprioritized_namespaces), allow_cap_overflow=False)

    overflow_used = False
    if len(selected_rows) < target:
        clean_overflow_allowed = {ns for ns in ns_counts if is_clean_namespace(ns, profile)}
        overflow_used = progressive_overflow_fill(rows, target, covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, clean_overflow_allowed)

    if len(selected_rows) < target:
        broader_overflow_allowed = {ns for ns in ns_counts if not is_structural_noise_namespace(ns, profile)}
        overflow_used = progressive_overflow_fill(rows, target, covered_features, covered_families, selected_rows, selected_ids, selected_ns_counts, selected_family_counts, quotas, caps, profile, broader_overflow_allowed) or overflow_used

    out = pd.DataFrame(selected_rows).drop_duplicates(subset=["api_full_name"]).head(target).copy()
    if len(out) == 0:
        return out, {"caps": caps, "quotas": quotas, "selected_ns_counts": {}, "selected_family_counts": {}, "overflow_used": overflow_used, "missing_core_namespaces": sorted(ns for ns in profile.core_namespaces if ns_counts.get(ns, 0) == 0)}

    selected_counter = Counter(out["namespace"])
    out["selection_reason"] = [build_selection_reason(r, selected_counter, quotas, profile) for r in out.to_dict("records")]
    diagnostics = {
        "caps": caps,
        "quotas": quotas,
        "selected_ns_counts": dict(Counter(out["namespace"])),
        "selected_family_counts": dict(Counter(out["primary_family"])),
        "overflow_used": overflow_used,
        "missing_core_namespaces": sorted(ns for ns in profile.core_namespaces if ns_counts.get(ns, 0) == 0),
    }
    return out, diagnostics


def coverage_metrics(full_df: pd.DataFrame, subset_df: pd.DataFrame, library: str) -> Dict[str, float]:
    profile = LIBRARY_PROFILES[library]
    full_namespaces = set(full_df["namespace"])
    subset_namespaces = set(subset_df["namespace"])
    core_full_namespaces = full_namespaces & set(profile.core_namespaces)
    core_subset_namespaces = subset_namespaces & set(profile.core_namespaces)
    full_features = set().union(*full_df["features"].tolist()) if len(full_df) else set()
    subset_features = set().union(*subset_df["features"].tolist()) if len(subset_df) else set()
    full_families = set(full_df["primary_family"])
    subset_families = set(subset_df["primary_family"])
    return {
        "all_namespace_coverage_pct": 100.0 * len(subset_namespaces) / max(1, len(full_namespaces)),
        "core_namespace_coverage_pct": 100.0 * len(core_subset_namespaces) / max(1, len(core_full_namespaces)),
        "feature_coverage_pct": 100.0 * len(subset_features) / max(1, len(full_features)),
        "family_coverage_pct": 100.0 * len(subset_families) / max(1, len(full_families)),
        "num_full_namespaces": len(full_namespaces),
        "num_subset_namespaces": len(subset_namespaces),
        "num_core_full_namespaces": len(core_full_namespaces),
        "num_core_subset_namespaces": len(core_subset_namespaces),
        "num_full_features": len(full_features),
        "num_subset_features": len(subset_features),
        "num_full_families": len(full_families),
        "num_subset_families": len(subset_families),
    }


def compute_namespace_stats(df: pd.DataFrame, subset: pd.DataFrame, library: str) -> pd.DataFrame:
    profile = LIBRARY_PROFILES[library]
    stats = (
        df.groupby("namespace")
        .agg(total_apis=("api_full_name", "count"), avg_feature_count=("feature_count", "mean"), avg_doc_quality=("doc_quality", "mean"))
        .reset_index()
    )
    stats["avg_feature_count"] = stats["avg_feature_count"].round(2)
    stats["avg_doc_quality"] = stats["avg_doc_quality"].round(2)
    selected = subset["namespace"].value_counts().rename_axis("namespace").reset_index(name="selected_count")
    stats = stats.merge(selected, on="namespace", how="left").fillna({"selected_count": 0})
    stats["selected_count"] = stats["selected_count"].astype(int)
    stats["is_core_namespace"] = stats["namespace"].isin(profile.core_namespaces)
    stats["is_deprioritized"] = stats["namespace"].isin(profile.deprioritized_namespaces)
    stats["is_low_value"] = stats["namespace"].map(lambda ns: is_low_value_namespace(ns, profile))
    return stats.sort_values(["is_core_namespace", "is_low_value", "is_deprioritized", "selected_count", "total_apis", "avg_feature_count"], ascending=[False, True, True, False, False, False])


def build_library_report(library: str, full_df: pd.DataFrame, subset_df: pd.DataFrame, metrics: Dict[str, float], diagnostics: Dict[str, object]) -> str:
    ns_dist = subset_df["namespace"].value_counts().sort_values(ascending=False)
    fam_dist = subset_df["primary_family"].value_counts().sort_values(ascending=False)
    ns_lines = "\n".join(f"- {ns}: {cnt}" for ns, cnt in ns_dist.items())
    fam_lines = "\n".join(f"- {fam}: {cnt}" for fam, cnt in fam_dist.items())
    missing_core = diagnostics.get("missing_core_namespaces", [])
    missing_core_text = ", ".join(missing_core) if missing_core else "None"
    return f"""Library: {library}
{'=' * (10 + len(library))}

Input APIs from accepted.csv: {diagnostics.get('raw_input_count', len(full_df))}
Effective candidates after selector preprocessing: {len(full_df)}
Selected APIs: {len(subset_df)}

Selection policy
----------------
1. Public APIs were grouped by structural namespace.
2. Internal / implementation-heavy namespaces were filtered.
3. Strong computational namespaces were prioritized before niche, device, or experimental namespaces.
4. Selection maximized marginal semantic feature gain, operation-family diversity,
   and namespace balance, using documentation richness only as a tie-breaker.
5. Exact alias-like duplicates were collapsed conservatively and near-duplicates were penalized.
6. Cap overflow was only used after clean computational namespaces were exhausted.

Coverage
--------
- All namespace coverage: {metrics['num_subset_namespaces']}/{metrics['num_full_namespaces']} ({metrics['all_namespace_coverage_pct']:.2f}%)
- Core namespace coverage: {metrics['num_core_subset_namespaces']}/{metrics['num_core_full_namespaces']} ({metrics['core_namespace_coverage_pct']:.2f}%)
- Semantic feature coverage: {metrics['num_subset_features']}/{metrics['num_full_features']} ({metrics['feature_coverage_pct']:.2f}%)
- Operation-family coverage: {metrics['num_subset_families']}/{metrics['num_full_families']} ({metrics['family_coverage_pct']:.2f}%)

Diagnostics
-----------
- Missing core namespaces in accepted set: {missing_core_text}
- Cap overflow required to reach target: {diagnostics.get('overflow_used', False)}

Selected namespace distribution
-------------------------------
{ns_lines}

Selected family distribution
----------------------------
{fam_lines}
"""


def choose_best_duplicate_group(group: pd.DataFrame, library: str) -> pd.DataFrame:
    profile = LIBRARY_PROFILES[library]
    work = group.copy()
    work["_dup_key"] = work.apply(lambda r: duplicate_key(r["api_full_name"], r["api_doc_text"]), axis=1)
    work["_ns_weight"] = work["namespace"].map(lambda ns: profile.namespace_weights.get(ns, 0.4))
    work["_is_core"] = work["namespace"].isin(profile.core_namespaces).astype(int)
    work["_is_deprioritized"] = work["namespace"].isin(profile.deprioritized_namespaces).astype(int)
    work["_is_noise"] = work["namespace"].map(lambda ns: int(is_structural_noise_namespace(ns, profile) or is_low_value_namespace(ns, profile)))
    work = work.sort_values(["_is_noise", "_is_deprioritized", "_is_core", "_ns_weight", "feature_count", "doc_quality", "api_full_name"], ascending=[True, True, False, False, False, False, True])
    work = work.drop_duplicates(subset=["_dup_key"], keep="first").copy()
    return work.drop(columns=["_dup_key", "_ns_weight", "_is_core", "_is_deprioritized", "_is_noise"])


def preprocess_group(df: pd.DataFrame, library: str, keras_split: bool = True) -> pd.DataFrame:
    group = df.copy()
    group["api_doc_text"] = group["api_doc_text"].map(safe_text)
    group = group[["api_full_name", "api_doc_text"]].dropna(subset=["api_full_name"]).copy()
    group["api_full_name"] = group["api_full_name"].astype(str).str.strip()
    group = group.drop_duplicates(subset=["api_full_name"]).reset_index(drop=True)
    group["library"] = library
    group["namespace"] = group["api_full_name"].apply(lambda x: extract_namespace(x, library=library, keras_split=keras_split))
    group = group[group.apply(lambda r: is_good_public_api(r["api_full_name"], library, r["namespace"]), axis=1)].reset_index(drop=True)
    group["name_tokens_norm"] = group["api_full_name"].apply(normalized_name_tokens)
    group["features"] = group.apply(lambda r: extract_features(r["api_full_name"], r["api_doc_text"], r["namespace"]), axis=1)
    group["feature_count"] = group["features"].apply(len)
    group["primary_family"] = group.apply(lambda r: infer_primary_family(r["api_full_name"], r["api_doc_text"], r["namespace"]), axis=1)
    group["doc_quality"] = group["api_doc_text"].apply(estimate_doc_quality)
    group["param_count"] = group["api_doc_text"].apply(estimate_param_count)
    group["signature_complexity"] = group["param_count"].apply(signature_complexity_bucket)
    group = choose_best_duplicate_group(group, library=library).reset_index(drop=True)
    group["row_id"] = range(len(group))
    return group


def process_dataframe(df: pd.DataFrame, target_per_library: int, library_filter: str, keras_split: bool = True) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    required_cols = {"api_full_name", "api_doc_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    work = df[["api_full_name", "api_doc_text"]].copy()
    work["api_full_name"] = work["api_full_name"].astype(str)
    work["library"] = work["api_full_name"].apply(infer_library)
    work = work[work["library"].isin(LIBRARY_PROFILES)].reset_index(drop=True)
    if library_filter != "all":
        if library_filter not in LIBRARY_PROFILES:
            raise ValueError(f"Unsupported library filter: {library_filter}")
        work = work[work["library"] == library_filter].reset_index(drop=True)
    if work.empty:
        raise ValueError("No supported APIs found after library inference/filtering.")

    selected_chunks: List[pd.DataFrame] = []
    report_parts: List[str] = []
    stats_chunks: List[pd.DataFrame] = []
    for library, group in work.groupby("library", sort=True):
        pre = preprocess_group(group, library=library, keras_split=keras_split)
        if pre.empty:
            continue
        subset, diagnostics = select_subset(pre, target=target_per_library, library=library)
        diagnostics["raw_input_count"] = len(group)
        diagnostics["effective_candidate_count"] = len(pre)
        subset = subset.sort_values(["namespace", "primary_family", "feature_count", "api_full_name"], ascending=[True, True, False, True]).reset_index(drop=True)
        metrics = coverage_metrics(pre, subset, library=library)
        stats = compute_namespace_stats(pre, subset, library=library)
        stats.insert(0, "library", library)
        stats_chunks.append(stats)
        report_parts.append(build_library_report(library, pre, subset, metrics, diagnostics))
        selected_chunks.append(subset)
    if not selected_chunks:
        raise ValueError("No APIs remained after preprocessing.")
    selected = pd.concat(selected_chunks, ignore_index=True)
    stats_all = pd.concat(stats_chunks, ignore_index=True) if stats_chunks else pd.DataFrame()
    report_text = "\n\n".join(report_parts)
    return selected, report_text, stats_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a generalized, thesis-defensible subset of DL APIs.")
    ap.add_argument("--input", required=True, help="Input CSV with api_full_name and api_doc_text")
    ap.add_argument("--output", required=True, help="Output CSV for selected subset")
    ap.add_argument("--target-per-library", type=int, default=100, help="Target subset size per library")
    ap.add_argument("--library", default="all", choices=["all", "torch", "tensorflow", "jax", "paddle"], help="Restrict selection to one library or process all found libraries")
    ap.add_argument("--report", default=None, help="Optional report path")
    ap.add_argument("--stats-csv", default=None, help="Optional namespace stats path")
    ap.add_argument("--no-keras-split", action="store_true", help="Do not split tf.keras into sub-namespaces")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(output_path.stem + "_report.txt")
    stats_path = Path(args.stats_csv) if args.stats_csv else output_path.with_name(output_path.stem + "_namespace_stats.csv")
    df = pd.read_csv(input_path)
    selected, report_text, stats_all = process_dataframe(df, target_per_library=args.target_per_library, library_filter=args.library, keras_split=not args.no_keras_split)
    selected[["api_full_name", "api_doc_text"]].to_csv(output_path, index=False)
    report_path.write_text(report_text, encoding="utf-8")
    if not stats_all.empty:
        stats_all.to_csv(stats_path, index=False)
    print(f"Selected {len(selected)} APIs total")
    print(selected.groupby("library").size().to_string())
    print(f"Saved selected subset to: {output_path}")
    print(f"Saved report to: {report_path}")
    if not stats_all.empty:
        print(f"Saved namespace stats to: {stats_path}")


if __name__ == "__main__":
    main()
