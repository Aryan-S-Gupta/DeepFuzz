from __future__ import annotations

"""Stage-4-oriented API filter for documentation-collected entries.

This filter is stricter than a plain doc-quality gate. It keeps APIs that are
plausible one-call fuzz targets for a VistaFuzz-style pipeline:
- docs must be real and non-placeholder
- parameters must be documented and align with the signature
- at least one seedable + modifiable input must exist
- required params must be seedable and not opaque custom objects
- returns must be capturable data-form outputs
- wrapper/bridge, utility, side-effect, context-dependent, and special-harness
  APIs are rejected by default
- obvious alias families are deduplicated

Practical fixes in this version
-------------------------------
- more robust signature/parameter-name sanitization to avoid parser-noise rejections
- less overbroad side-effect / context / utility cues that were suppressing real compute APIs
- still rejects receiver-bound / wrapper / special-harness APIs by default
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.pipeline_contract import (
    API_DOC_TEXT,
    API_FULL_NAME,
    api_collection_json,
    collector_entry_to_api_row,
    write_api_csv,
    write_rejected_csv as write_contract_rejected_csv,
)
from common.api_policy import internal_api_reason, is_internal_api


DOC_MIN_CHARS = 40
VALID_PARAM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BAD_PARAM_NAMES = {
    "optional", "required", "parameter", "parameters", "argument", "arguments",
    "return", "returns", "output", "outputs", "input", "inputs",
    "default", "defaults", "note", "notes", "example", "examples",
}

SIDE_EFFECT_NAME_PATTERNS = [
    r"\b(save|load|open|read|write|download|upload|export|import|serialize|deserialize)\b",
    r"\b(checkpoint|state_dict)\b",
    r"\b(train|fit|learn|backward)\b",
    r"\b(distributed|cluster|server|worker|rpc|remote)\b",
    r"\b(register|attach|subscribe|listener|callback|hook)\b",
    r"\b(camera|microphone|audio|video)\b",
]

SIDE_EFFECT_DOC_PATTERNS = SIDE_EFFECT_NAME_PATTERNS + [
    r"\bin-?place\b",
    r"\bside effect\b",
    r"\bglobal state\b",
    r"\bvolatile state\b",
    r"\bupdates? .* state\b",
    r"\bwrites? to\b",
    r"\bsaves? to\b",
    r"\bloads? from\b",
    r"\bmodifies? .* in place\b",
    r"\breturns?\s+none\b",
]

BAD_REQUIRED_PARAM_PATTERNS = [
    r"\b(path|filepath|filename|file|dir|directory)\b",
    r"\b(url|uri|socket|port|host)\b",
    r"\b(session|handle)\b",
    r"\b(model|module|network|optimizer|dataset|dataloader)\b",
    r"\b(callback|hook|listener|subscriber|observer)\b",
    r"\b(cluster|server|worker|rpc|remote)\b",
]

CONTEXT_DEPENDENCY_DOC_PATTERNS = [
    r"\blookup\b",
    r"\bget by name\b",
    r"\bfetch by name\b",
    r"\bidentified by name\b",
    r"\bresource handle\b",
    r"\bhandle to\b",
    r"\bvariant tensor\b",
    r"\breturns?\s+a\s+tensor\s+of\s+type\s+variant\b",
    r"\biterator\b",
    r"\bdataset\b",
    r"\btable\b",
    r"\bgraph\b",
    r"\breplica context\b",
    r"\bcross-replica\b",
    r"\bstrategy\b",
    r"\bdistributed\b",
    r"\blogical device\b",
    r"\bdevice placement\b",
    r"\bstateful\b",
    r"\bstate maintained\b",
    r"\bpre-existing\b",
    r"\bresource\b",
    r"\bdescriptor\b",
    r"\bglob pattern\b",
    r"\bmatching files\b",
    r"\bsharded filename\b",
    r"\bsharded file(?:name|spec)\b",
]

CONTEXT_DEPENDENCY_PARAM_PATTERNS = [
    r"\bresource\b",
    r"\bhandle\b",
    r"\bresource handle\b",
    r"\biterator\b",
    r"\bdataset\b",
    r"\bdataloader\b",
    r"\btable\b",
    r"\bgraph\b",
    r"\bcontext\b",
    r"\breplica\b",
    r"\bstrategy\b",
    r"\bdistributed\b",
    r"\bvariant\b",
    r"\bpattern\b",
    r"\bfile(?:name|spec)?\b",
]

NON_DATAFORM_RETURN_PATTERNS = [
    r"\bhandle\b",
    r"\bresource\b",
    r"\biterator\b",
    r"\bdataset\b",
    r"\btable\b",
    r"\bgraph\b",
    r"\bcontext\b",
    r"\bstrategy\b",
    r"\breplica\b",
    r"\bdescriptor\b",
    r"\bsymbolic\b",
    r"\bvariant\b",
]

FUZZABLE_TYPE_TOKENS = [
    "int", "integer", "float", "double", "bool", "boolean",
    "str", "string", "bytes",
    "list", "tuple", "dict", "set",
    "sequence", "array", "ndarray", "tensor", "tensors",
    "number", "numeric", "scalar",
    "dtype", "shape", "complex",
]

NONFUZZABLE_REQUIRED_TYPE_TOKENS = [
    "file", "path", "filepath", "filename", "directory",
    "url", "uri", "socket",
    "session", "handle",
    "iterator", "generator", "callable", "function", "callback", "hook", "listener",
    "module", "model", "layer", "optimizer", "dataset", "dataloader",
    "rpc", "remote", "server", "worker", "cluster",
    "environment",
]

SECTION_HEADERS = {
    "args": {"args", "arguments", "parameters", "parameter", "inputs", "input"},
    "returns": {"returns", "return", "output", "outputs", "result", "results", "yields", "yield"},
}

GENERIC_PARAM_DESC_EXACT = {
    "input", "the input", "parameter", "argument", "value", "values",
    "object", "some object", "anything", "stuff", "data structure", "data", "the data",
}

UTILITY_API_NAME_PATTERNS = [
    r"\b(importlib|metadata|inspect|typing|warnings|logging|config|backend|backends)\b",
    r"\b(env|environment|version|build|cudnn|mps)\b",
    r"\b(seed|manual_seed|set_default|get_default|is_available)\b",
    r"\b(state_dict|serialization|pickle)\b",
]

UTILITY_DOC_PATTERNS = [
    r"\bdistribution package\b",
    r"\bpackage metadata\b",
    r"\bmodule metadata\b",
    r"\binspect\b",
    r"\bintrospect\b",
    r"\breflection\b",
    r"\breturns? whether\b",
    r"\bchecks? whether\b",
    r"\bavailability\b",
    r"\bbackend\b",
    r"\benvironment\b",
    r"\bglobal state\b",
    r"\bvolatile state\b",
]

CONSTRAINT_TOKENS = [
    "shape", "size", "dimension", "dimensions", "dim",
    "dtype", "type", "types",
    "range", "between", "greater than", "less than", "non-negative", "positive",
    "must be", "should be", "has to be", "expected to",
    "one of", "either", "enum", "valid values", "allowed values",
    "broadcast", "broadcastable",
    "same shape", "same size", "matching shape",
    "scalar", "vector", "matrix",
    "tensor", "ndarray", "array", "sequence", "list", "tuple",
    "length", "rank",
]

RETURN_DATAFORM_TOKENS = [
    "tensor", "tensors", "ndarray", "array", "arrays",
    "numeric", "number", "scalar", "int", "integer", "float", "double", "bool", "boolean", "complex",
    "tuple", "list", "dict", "sequence", "value", "values", "index", "indices", "shape", "dtype",
]

BAD_RETURN_PATTERNS = [
    r"\breturns?\s+none\b",
    r"\bno return value\b",
    r"\bin-?place\b",
    r"\bmodif(?:y|ies|ied)\s+.*\bin-?place\b",
    r"\bside effect\b",
    r"\bglobal state\b",
    r"\bvolatile state\b",
    r"\bupdates? .* state\b",
    r"\bhandle\b",
    r"\bsession\b",
    r"\bdistribution\b",
]

WRAPPER_BRIDGE_NAMESPACE_PATTERNS = [
    r"\.functional\.F\.np(?:\.|$)",
    r"\.compat(?:\.|$)",
    r"\.experimental\.numpy(?:\.|$)",
    r"(?:^|\.)_api(?:\.|$)",
    r"(?:^|\.)raw_ops(?:\.|$)",
]

EXTERNAL_ALIAS_MODULE_PREFIXES = {
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "matplotlib",
    "PIL",
    "cv2",
}

METHOD_SIGNATURE_PATTERNS = [r"^[A-Za-z_][\w\[\]]*\.[A-Za-z_][\w]*\("]

CLASSISH_OWNER_PATTERNS = [
    r"^[A-Z][A-Za-z0-9_]*$",
    r".*Tensor$",
    r".*Scaler$",
    r".*Optimizer$",
    r".*Module$",
    r".*Dataset$",
    r".*Iterator$",
    r"^matrix$",
    r"^ndarray$",
]

NONMODIFIABLE_PARAM_NAME_PATTERNS = [r"^name$", r"^out$", r"^out_?$"]
NONMODIFIABLE_PARAM_DESC_PATTERNS = [
    r"\boptional name for the operation\b",
    r"\boutput buffer\b",
    r"\bresult will be placed in\b",
    r"\bplace the result in\b",
]

GENERIC_NONOPAQUE_TYPE_WORDS = {
    "tensor", "tensors", "array", "arrays", "ndarray", "scalar", "scalars", "number", "numbers",
    "numeric", "dtype", "dtypes", "device", "shape", "sequence", "sequences", "tuple", "tuples",
    "list", "lists", "dict", "dicts", "mapping", "mappings", "bool", "boolean", "int", "integer",
    "float", "double", "complex", "string", "str", "bytes", "object", "objects", "value", "values",
    "dim", "dims", "axis", "axes", "rank",
}

ALLOWED_CUSTOM_TYPE_TOKENS = {
    "Tensor", "Tensors", "ndarray", "Array", "Scalar", "DType", "StringDType", "Device",
    "Tuple", "List", "Sequence", "Dict", "Optional", "Union", "Any",
}

DTYPE_TENSOR_ALIASES = {
    "BFloat16Tensor", "BoolTensor", "ByteTensor", "CharTensor", "DoubleTensor", "FloatTensor",
    "HalfTensor", "IntTensor", "LongTensor", "ShortTensor",
}


@dataclass
class ParamInfo:
    name: str
    type: Optional[str]
    desc: str
    optional: bool = False
    default: Optional[str] = None


@dataclass
class ReturnInfo:
    type: Optional[str]
    desc: str
    has_return: bool


@dataclass
class Decision:
    accepted: bool
    step: int
    reason: str


@dataclass
class SeedabilityResult:
    ok: bool
    reason: str
    seed_value_repr: str = ""


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _split_lines(doc: str) -> List[str]:
    return (doc or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text or "")


def _match_any(patterns: Sequence[str], text: str) -> Optional[str]:
    t = _strip_urls((text or "").lower())
    for pat in patterns:
        if re.search(pat, t):
            return pat
    return None


def _step_name(step: int) -> str:
    return {1: "1_doc", 2: "2_params", 3: "3_types", 4: "4_return", 5: "5_readiness"}.get(step, f"{step}_unknown")


def _api_leaf(api: str) -> str:
    return (api or "").rsplit(".", 1)[-1]


def _api_parent(api: str) -> str:
    return (api or "").rsplit(".", 1)[0] if "." in (api or "") else ""


def _api_owner(api: str) -> str:
    parent = _api_parent(api)
    return parent.rsplit(".", 1)[-1] if parent else ""


def _base_alias_name(api: str) -> str:
    parent = _api_parent(api)
    leaf = _api_leaf(api)
    if leaf.endswith("_") and len(leaf) > 1:
        leaf = leaf[:-1]
    return f"{parent}.{leaf}" if parent else leaf


def _type_tokens(type_str: Optional[str], desc: str, name: str) -> str:
    combined = " ".join([type_str or "", desc or "", name or ""])
    combined = combined.replace(":class:`", "").replace("`", "").replace("'", "")
    combined = _strip_urls(combined)
    return combined.lower()


def _is_reasonable_param_name(name: str) -> bool:
    n = _normalize_param_name(name)
    if not n or not VALID_PARAM_RE.fullmatch(n):
        return False
    return n.lower() not in BAD_PARAM_NAMES


def _is_exception_like(entry: Dict[str, Any]) -> bool:
    api = entry.get("api") or entry.get("qualname") or ""
    leaf = _api_leaf(api)
    kind = (entry.get("kind") or "").lower()
    if kind == "class":
        return True
    return bool(re.search(r"(Error|Exception|Warning)$", leaf))


def _is_inplace_name(api: str) -> bool:
    leaf = _api_leaf(api)
    return bool(leaf.endswith("_") and not leaf.startswith("__"))


def _normalize_param_name(name: str) -> str:
    name = (name or "").strip()
    name = name.lstrip("*")
    return name.strip()


def _looks_like_new_section(line: str) -> bool:
    s = line.strip().rstrip(":").lower()
    if s in SECTION_HEADERS["args"] or s in SECTION_HEADERS["returns"]:
        return True
    if re.fullmatch(r"-{3,}", s):
        return True
    return bool(re.fullmatch(r"(raises|examples?|note|notes|warning|warnings|see also|references?)", s))


def _parse_google_style_fields(doc_lines: List[str], start_idx: int) -> List[Tuple[str, Optional[str], str, bool]]:
    out: List[Tuple[str, Optional[str], str, bool]] = []
    i = start_idx + 1
    current = None
    while i < len(doc_lines):
        line = doc_lines[i]
        if not line.strip():
            i += 1
            continue
        if _looks_like_new_section(line):
            break
        if len(line) - len(line.lstrip(" ")) < 2 and not line.startswith("\t"):
            break

        m = re.match(r"^\s*(\*{0,2}[A-Za-z_][\w\.]*)\s*(\(([^)]*)\))?\s*:\s*(.*)$", line)
        if m:
            if current:
                name, typ, desc_parts, optmeta = current
                if _is_reasonable_param_name(name):
                    out.append((name, typ, _normalize(" ".join(desc_parts)), optmeta))
            name = _normalize_param_name(m.group(1))
            meta = (m.group(3) or "").strip()
            typ = meta if meta else None
            optmeta = "optional" in meta.lower()
            desc = m.group(4).strip()
            current = (name, typ, [desc] if desc else [], optmeta)
        else:
            if current:
                current[2].append(line.strip())
        i += 1

    if current:
        name, typ, desc_parts, optmeta = current
        if _is_reasonable_param_name(name):
            out.append((name, typ, _normalize(" ".join(desc_parts)), optmeta))
    return out


def _parse_numpy_parameters(doc_lines: List[str], header_idx: int) -> List[Tuple[str, Optional[str], str, bool]]:
    i = header_idx + 1
    if i < len(doc_lines) and re.fullmatch(r"\s*-{3,}\s*", doc_lines[i]):
        i += 1

    out: List[Tuple[str, Optional[str], str, bool]] = []
    current = None
    while i < len(doc_lines):
        line = doc_lines[i]
        if not line.strip():
            i += 1
            continue
        if _looks_like_new_section(line) and current:
            break
        m = re.match(r"^\s*(\*{0,2}[A-Za-z_][\w\.]*)\s*:\s*(.+?)\s*$", line)
        if m:
            if current:
                name, typ, desc_parts, optmeta = current
                if _is_reasonable_param_name(name):
                    out.append((name, typ, _normalize(" ".join(desc_parts)), optmeta))
            name = _normalize_param_name(m.group(1))
            typ = m.group(2).strip()
            optmeta = "optional" in typ.lower()
            current = (name, typ, [], optmeta)
        else:
            if current:
                current[2].append(line.strip())
        i += 1

    if current:
        name, typ, desc_parts, optmeta = current
        if _is_reasonable_param_name(name):
            out.append((name, typ, _normalize(" ".join(desc_parts)), optmeta))
    return out


def _parse_rest_params(doc: str) -> List[Tuple[str, Optional[str], str, bool]]:
    out: List[Tuple[str, Optional[str], str, bool]] = []
    param_desc: Dict[str, str] = {}
    param_type: Dict[str, str] = {}

    for m in re.finditer(r":param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)", doc):
        if _is_reasonable_param_name(m.group(1)):
            param_desc[m.group(1)] = _normalize(m.group(2))
    for m in re.finditer(r":type\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)", doc):
        if _is_reasonable_param_name(m.group(1)):
            param_type[m.group(1)] = _normalize(m.group(2))

    for name, desc in param_desc.items():
        typ = param_type.get(name)
        opt = "optional" in (typ or "").lower() or "optional" in desc.lower()
        out.append((name, typ, desc, opt))
    return out


def _parse_dash_bullet_params(doc_lines: List[str]) -> List[Tuple[str, Optional[str], str, bool]]:
    out: List[Tuple[str, Optional[str], str, bool]] = []
    pat = re.compile(r"^\s*[-*]?\s*(\*{0,2}[A-Za-z_][\w\.]*)\s*(\(([^)]*)\))?\s*[–-]\s*(.+?)\s*$")
    for line in doc_lines:
        m = pat.match(line)
        if not m:
            continue
        name = _normalize_param_name(m.group(1))
        if not _is_reasonable_param_name(name):
            continue
        meta = (m.group(3) or "").strip()
        typ = meta if meta else None
        desc = _normalize(m.group(4))
        opt = "optional" in (meta or "").lower() or "optional" in desc.lower()
        out.append((name, typ, desc, opt))
    return out


def parse_params_from_doc(doc: str) -> List[ParamInfo]:
    lines = _split_lines(doc)

    arg_idx = None
    for i, line in enumerate(lines):
        h = line.strip().strip(":").lower()
        if h in SECTION_HEADERS["args"]:
            arg_idx = i
            break

    params: List[Tuple[str, Optional[str], str, bool]] = []
    if arg_idx is not None:
        params = _parse_google_style_fields(lines, arg_idx)

    if not params:
        for i, line in enumerate(lines):
            if line.strip().lower() == "parameters":
                params = _parse_numpy_parameters(lines, i)
                break

    if not params:
        params = _parse_rest_params(doc)

    if not params:
        params = _parse_dash_bullet_params(lines)

    out: List[ParamInfo] = []
    for name, typ, desc, optmeta in params:
        if not _is_reasonable_param_name(name):
            continue
        out.append(ParamInfo(name=name, type=_normalize(typ) if typ else None, desc=_normalize(desc), optional=bool(optmeta)))
    return out


def _parse_google_style_returns(doc_lines: List[str], start_idx: int) -> ReturnInfo:
    i = start_idx + 1
    parts: List[str] = []
    ret_type: Optional[str] = None

    while i < len(doc_lines):
        line = doc_lines[i]
        if not line.strip():
            i += 1
            continue
        if _looks_like_new_section(line):
            break
        if len(line) - len(line.lstrip(" ")) < 2 and not line.startswith("\t"):
            break

        if ret_type is None:
            m = re.match(r"^\s*([A-Za-z_][\w\[\], \.\-\|]*)\s*:\s*(.*)$", line)
            if m:
                ret_type = _normalize(m.group(1))
                tail = _normalize(m.group(2))
                if tail:
                    parts.append(tail)
                i += 1
                continue
        parts.append(line.strip())
        i += 1

    return ReturnInfo(type=ret_type, desc=_normalize(" ".join(parts)), has_return=bool(parts or ret_type))


def _parse_numpy_returns(doc_lines: List[str], header_idx: int) -> ReturnInfo:
    i = header_idx + 1
    if i < len(doc_lines) and re.fullmatch(r"\s*-{3,}\s*", doc_lines[i]):
        i += 1

    ret_type: Optional[str] = None
    parts: List[str] = []
    while i < len(doc_lines):
        line = doc_lines[i]
        if not line.strip():
            i += 1
            continue
        if _looks_like_new_section(line) and (ret_type or parts):
            break
        if ret_type is None:
            m = re.match(r"^\s*([A-Za-z_][\w\[\], \.\-\|]*)\s*$", line)
            if m:
                ret_type = _normalize(m.group(1))
                i += 1
                continue
        parts.append(line.strip())
        i += 1
    return ReturnInfo(type=ret_type, desc=_normalize(" ".join(parts)), has_return=bool(parts or ret_type))


def _parse_rest_return(doc: str) -> ReturnInfo:
    ret_desc = ""
    ret_type = None
    m_desc = re.search(r":return[s]?\s*:\s*(.+)", doc)
    if m_desc:
        ret_desc = _normalize(m_desc.group(1))
    m_type = re.search(r":rtype\s*:\s*(.+)", doc)
    if m_type:
        ret_type = _normalize(m_type.group(1))
    return ReturnInfo(type=ret_type, desc=ret_desc, has_return=bool(ret_desc or ret_type))


def parse_return_from_doc(doc: str) -> ReturnInfo:
    lines = _split_lines(doc)
    for i, line in enumerate(lines):
        h = line.strip().strip(":").lower()
        if h in SECTION_HEADERS["returns"]:
            info = _parse_google_style_returns(lines, i)
            if info.has_return:
                return info
    for i, line in enumerate(lines):
        if line.strip().lower() in {"returns", "return", "output", "outputs", "result", "results"}:
            info = _parse_numpy_returns(lines, i)
            if info.has_return:
                return info
    info = _parse_rest_return(doc)
    if info.has_return:
        return info
    return ReturnInfo(type=None, desc="", has_return=False)


def parse_signature_params(sig: Optional[str]) -> List[Tuple[str, bool, Optional[str]]]:
    if not sig:
        return []
    s = sig.strip()
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    if "(" in s and ")" in s:
        s = s[s.find("(") + 1 : s.rfind(")")]

    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in s:
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
    if cur:
        parts.append("".join(cur).strip())

    out: List[Tuple[str, bool, Optional[str]]] = []
    for p in parts:
        if not p or p in {"*", "/"}:
            continue
        name = p
        default = None
        has_default = False
        if "=" in p:
            name, default = [x.strip() for x in p.split("=", 1)]
            has_default = True
        if ":" in name:
            name = name.split(":", 1)[0].strip()
        name = name.lstrip("*")
        if not _is_reasonable_param_name(name):
            continue
        out.append((name, has_default, default))
    return out


def parse_return_annotation(sig: Optional[str]) -> Optional[str]:
    if not sig:
        return None
    s = sig.strip()
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    if "->" not in s:
        return None
    ann = s.split("->", 1)[1].strip().rstrip(":").strip()
    return ann or None


def _is_vague_param_desc(desc: str) -> bool:
    d = _normalize(desc).lower()
    if not d:
        return True
    if d in GENERIC_PARAM_DESC_EXACT:
        return True
    if re.fullmatch(r"(the\s+)?(input|parameter|argument|value|object|data)\.?", d):
        return True
    if re.fullmatch(r"(some|any)\s+(object|value|data|thing)\.?", d):
        return True
    return bool(re.search(r"\b(anything|stuff|data structure)\b", d))


def _has_constraint_signal(p: ParamInfo) -> bool:
    txt = _type_tokens(p.type, p.desc, p.name)
    for tok in CONSTRAINT_TOKENS:
        if tok in txt:
            return True
    if re.search(r"\b\d+\b", txt):
        return True
    if re.search(r"\b\[[^\]]+\]", txt):
        return True
    if re.search(r"\([^)]+\)", txt):
        return True
    if re.search(r"\b(one of|either|must be|should be|expected to|between)\b", txt):
        return True
    return bool(re.search(r"[<>]=?|==|!=", txt))


def is_fuzzable_param(p: ParamInfo) -> Tuple[bool, str]:
    tokens = _type_tokens(p.type, p.desc, p.name)
    for bad in NONFUZZABLE_REQUIRED_TYPE_TOKENS:
        if re.search(rf"\b{re.escape(bad)}\b", tokens):
            return False, f"non-fuzzable type cue: {bad}"
    for good in FUZZABLE_TYPE_TOKENS:
        if re.search(rf"\b{re.escape(good)}\b", tokens):
            return True, f"fuzzable type cue: {good}"
    if re.search(r"\b(tensor|ndarray|array|scalar|number|numeric|complex)\b", tokens):
        return True, "fuzzable inferred from description"
    return False, "type unclear / not constructable"


def has_meaningful_return(sig: Optional[str], doc: str) -> Tuple[bool, str]:
    ret_doc = parse_return_from_doc(doc)
    ret_ann = parse_return_annotation(sig)
    if not ret_doc.has_return and not ret_ann:
        return False, "no documented return value"
    ret_type = ret_doc.type or ret_ann
    combined = _type_tokens(ret_type, ret_doc.desc, "").strip()
    if not combined:
        return False, "return behavior unclear or unspecified"
    bad = _match_any(BAD_RETURN_PATTERNS, combined)
    if bad:
        return False, f"return not capturable / side-effect style ({bad})"
    good_hit = any(re.search(rf"\b{re.escape(tok)}\b", combined) for tok in RETURN_DATAFORM_TOKENS)
    tuple_like = bool(re.search(r"\breturns?\s*\([^)]+\)", combined) or re.search(r"\([^)]+,\s*[^)]+\)", combined))
    if not (good_hit or tuple_like):
        return False, "return not meaningful/capturable data-form"
    if re.search(r"\b(whether|if)\b", combined) and not re.search(r"\b(bool|boolean)\b", combined):
        return False, "return description too vague / predicate-like"
    return True, "capturable data-form return"


def has_context_dependent_contract(api_name: str, doc: str, params: List[ParamInfo], sig: Optional[str]) -> Tuple[bool, str]:
    doc_head = "\n".join(_split_lines(doc)[:40]).lower()
    ret_doc = parse_return_from_doc(doc)
    ret_ann = parse_return_annotation(sig)
    ret_text = _type_tokens(ret_doc.type or ret_ann, ret_doc.desc, "")
    required_params = [p for p in params if not p.optional]
    bad_required = sum(1 for p in required_params if _match_any(CONTEXT_DEPENDENCY_PARAM_PATTERNS, _type_tokens(p.type, p.desc, p.name)))
    doc_hit = _match_any(CONTEXT_DEPENDENCY_DOC_PATTERNS, doc_head)
    ret_hit = _match_any(NON_DATAFORM_RETURN_PATTERNS, ret_text)
    if bad_required > 0 and (doc_hit or ret_hit):
        return True, "context-dependent/stateful contract; not a self-contained data API"
    if doc_hit and ret_hit:
        return True, "doc describes lookup/resource/spec-style behavior, not data transformation"
    return False, ""


def is_utility_api(api_name: str, doc: str, params: List[ParamInfo]) -> Tuple[bool, str]:
    name_hit = _match_any(UTILITY_API_NAME_PATTERNS, api_name)
    if name_hit:
        return True, f"name contains utility/metadata keyword ({name_hit})"
    doc_head = "\n".join(_split_lines(doc)[:20])
    doc_hit = _match_any(UTILITY_DOC_PATTERNS, doc_head)
    if doc_hit:
        return True, f"doc indicates utility/metadata behavior ({doc_hit})"
    leaf = _api_leaf(api_name).lower()
    if leaf in {"distribution", "version", "build", "is_available", "get_default_dtype", "set_default_dtype", "manual_seed", "seed"}:
        return True, f"utility API leaf '{leaf}'"
    for p in params:
        joined = f"{p.type or ''} {p.desc or ''} {p.name or ''}".lower()
        if re.search(r"\b(distribution_name|package|module_name|backend)\b", joined):
            return True, f"utility-style param '{p.name}'"
    return False, ""


def has_side_effect(api_name: str, doc: str, params: List[ParamInfo]) -> Tuple[bool, str]:
    if _is_inplace_name(api_name):
        return True, "in-place style API name"
    name_hit = _match_any(SIDE_EFFECT_NAME_PATTERNS, api_name)
    if name_hit:
        return True, f"name contains side-effect keyword ({name_hit})"
    doc_head = "\n".join(_split_lines(doc)[:20])
    doc_hit = _match_any(SIDE_EFFECT_DOC_PATTERNS, doc_head)
    if doc_hit:
        return True, f"doc indicates side-effect/external-dependency keyword ({doc_hit})"
    ret_info = parse_return_from_doc(doc)
    ret_text = _type_tokens(ret_info.type, ret_info.desc, "")
    ret_bad = _match_any(BAD_RETURN_PATTERNS, ret_text)
    if ret_bad:
        return True, f"return doc indicates side effect / uncapturable output ({ret_bad})"
    for p in params:
        joined = f"{p.type or ''} {p.desc or ''}"
        if (_match_any(BAD_REQUIRED_PARAM_PATTERNS, p.name) or _match_any(BAD_REQUIRED_PARAM_PATTERNS, joined)) and not p.optional:
            return True, f"required external-dependency param '{p.name}'"
    return False, ""


def _is_reference_only_doc(doc: str) -> bool:
    lines = [ln.strip() for ln in _split_lines(doc) if ln.strip()]
    if not lines:
        return False
    body_lines = lines[1:] if len(lines) > 1 else lines
    body = _normalize(" ".join(body_lines)).lower()
    if not body:
        return False
    redirect_patterns = [
        r"^see\s+(also\s+)?(?::\w+:)?`?[^`]+`?\.?$",
        r"^(out-?of-?place|in-?place)\s+version\s+of\s+.+\.?$",
        r"^alias\s+for\s+.+\.?$",
        r"^.*\bdeprecated\b.*\balias\s+for\b.*\.?$",
    ]
    return any(re.fullmatch(p, body, re.IGNORECASE) for p in redirect_patterns)


def _is_placeholder_doc(doc: str) -> bool:
    d = _normalize(doc).lower()
    if not d:
        return True
    head_lines = [ln.strip().lower() for ln in _split_lines(doc)[:6] if ln.strip()]
    head = " ".join(head_lines)
    placeholder_head_patterns = [
        r"\btodo\b", r"\btbd\b", r"\bfixme\b", r"\badd doc\b", r"\badd docs\b",
        r"\bno doc(?:string)?\b", r"\bplaceholder\b", r"\bstub\b", r"\bautogenerated\b", r"\bgenerated by\b",
    ]
    if any(re.search(p, head) for p in placeholder_head_patterns):
        return True
    if len(d) < DOC_MIN_CHARS:
        if re.fullmatch(r"(pass|\.\.\.|todo|tbd|none|no docstring\.?)", d):
            return True
        if len(re.findall(r"[a-zA-Z0-9]", d)) < 12:
            return True
    has_args = any(line.strip().strip(":").lower() in SECTION_HEADERS["args"] for line in _split_lines(doc))
    has_returns = any(line.strip().strip(":").lower() in SECTION_HEADERS["returns"] for line in _split_lines(doc))
    if not has_args and not has_returns:
        nonempty = [ln.strip() for ln in _split_lines(doc) if ln.strip()]
        if len(nonempty) <= 2 and len(re.findall(r"[.!?]", d)) <= 1 and len(d.split()) < 10:
            return True
    return d in {"", ".", "..", "..."}


def rejects_wrapper_bridge_namespace(api_name: str) -> Tuple[bool, str]:
    pat = _match_any(WRAPPER_BRIDGE_NAMESPACE_PATTERNS, api_name)
    if pat:
        return True, f"wrapper/bridge namespace ({pat})"
    return False, ""


def rejects_external_alias(api_name: str, entry: Dict[str, Any]) -> Tuple[bool, str]:
    root = (api_name or "").split(".", 1)[0]
    if not root:
        return False, ""

    for field in ("module", "canonical_target"):
        value = str(entry.get(field) or "").strip()
        if not value:
            continue
        prefix = value.split(".", 1)[0]
        if prefix == root:
            continue
        if prefix in EXTERNAL_ALIAS_MODULE_PREFIXES:
            return True, f"external imported alias via {field} '{value}'"
    return False, ""


def requires_receiver_harness(api_name: str, sig: Optional[str]) -> Tuple[bool, str]:
    s = (sig or "").strip()
    first = s.split("\n", 1)[0].strip() if s else ""
    for pat in METHOD_SIGNATURE_PATTERNS:
        if re.search(pat, first):
            return True, "method-style signature / receiver required"
    owner = _api_owner(api_name)
    if owner and any(re.fullmatch(pat, owner) for pat in CLASSISH_OWNER_PATTERNS):
        return True, f"receiver-like owner '{owner}'"
    return False, ""


def is_modifiable_param(p: ParamInfo) -> Tuple[bool, str]:
    name = (p.name or "").lower()
    desc = _normalize(p.desc).lower()
    for pat in NONMODIFIABLE_PARAM_NAME_PATTERNS:
        if re.fullmatch(pat, name):
            return False, f"non-modifiable param name '{p.name}'"
    hit = _match_any(NONMODIFIABLE_PARAM_DESC_PATTERNS, desc)
    if hit:
        return False, f"non-modifiable param description ({hit})"
    return True, "modifiable"


def has_opaque_custom_type(p: ParamInfo) -> Tuple[bool, str]:
    raw = " ".join([p.type or "", p.desc or ""])
    raw = raw.replace(":class:`", " ").replace("`", " ")
    lower = raw.lower()
    fuzzable, _ = is_fuzzable_param(p)
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", raw)
    opaque = []
    for tok in candidates:
        if tok in ALLOWED_CUSTOM_TYPE_TOKENS:
            continue
        if tok.lower() in GENERIC_NONOPAQUE_TYPE_WORDS:
            continue
        opaque.append(tok)
    if not opaque:
        return False, ""
    constructorish = bool(re.search(r"\b(int|integer|float|bool|string|sequence|tuple|list|array|tensor|dtype|shape)\b", lower))
    if opaque and ("object" in lower or "instance" in lower or "class" in lower or not fuzzable) and not constructorish:
        return True, f"opaque custom type(s): {', '.join(sorted(set(opaque))[:4])}"
    return False, ""


def simple_seedability(p: ParamInfo) -> SeedabilityResult:
    fuzzable, why = is_fuzzable_param(p)
    if not fuzzable:
        return SeedabilityResult(False, why)
    opaque, opaque_why = has_opaque_custom_type(p)
    if opaque:
        return SeedabilityResult(False, opaque_why)
    t = _type_tokens(p.type, p.desc, p.name)
    name = (p.name or "").lower()
    if re.search(r"\bdtype\b", t) or name == "dtype":
        return SeedabilityResult(True, "dtype seedable", "float32")
    if re.search(r"\b(bool|boolean)\b", t):
        return SeedabilityResult(True, "bool seedable", "True")
    if re.search(r"\b(int|integer|axis|rank|length|count|size|index|indices)\b", t):
        return SeedabilityResult(True, "int-like seedable", "1")
    if re.search(r"\b(float|double|numeric|number|scalar|complex)\b", t):
        return SeedabilityResult(True, "numeric seedable", "1.0")
    if re.search(r"\b(str|string|bytes)\b", t):
        return SeedabilityResult(True, "string seedable", "'x'")
    if re.search(r"\b(shape|dims?|axes?)\b", t):
        return SeedabilityResult(True, "shape-like seedable", "[2, 2]")
    if re.search(r"\b(sequence|list|tuple|array_like|array-like)\b", t):
        return SeedabilityResult(True, "sequence-like seedable", "[1, 2]")
    if re.search(r"\b(tensor|ndarray|array)\b", t):
        return SeedabilityResult(True, "tensor-like seedable", "tensor([1.0, 2.0])")
    return SeedabilityResult(False, "could not synthesize baseline seed")


def baseline_seedability(params: List[ParamInfo]) -> Tuple[bool, str]:
    required = [p for p in params if not p.optional]
    if required:
        for p in required:
            res = simple_seedability(p)
            if not res.ok:
                return False, f"required param '{p.name}' not baseline-seedable: {res.reason}"
        return True, "all required params seedable"
    for p in params:
        mod, _ = is_modifiable_param(p)
        if not mod:
            continue
        res = simple_seedability(p)
        if res.ok:
            return True, f"optional-only API seedable via '{p.name}'"
    return False, "optional-only API has no seedable modifiable params"


def collapse_aliases(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_api = {r.get("api_full_name", ""): r for r in rows}
    dropped = set()
    for api in list(by_api.keys()):
        leaf = _api_leaf(api)
        if leaf.endswith("_") and _base_alias_name(api) in by_api:
            dropped.add(api)
    kept = [r for r in rows if r.get("api_full_name", "") not in dropped]

    seen_keys = set()
    out: List[Dict[str, str]] = []
    for r in sorted(kept, key=lambda x: x.get("api_full_name", "")):
        api = r.get("api_full_name", "")
        doc = _normalize(r.get("api_doc_text", ""))
        parent = _api_parent(api)
        leaf = _api_leaf(api)
        owner = _api_owner(api)
        if owner in DTYPE_TENSOR_ALIASES:
            parent = parent.rsplit(".", 1)[0] + ".Tensor" if "." in parent else "Tensor"
        doc_hash = hashlib.sha1(doc.encode("utf-8")).hexdigest()[:12]
        key = f"{parent}.{leaf}::{doc_hash}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(r)
    return out


def filter_one(entry: Dict[str, Any], allow_receiver_apis: bool = False, allow_wrapper_bridges: bool = False) -> Tuple[Decision, Dict[str, str]]:
    api = entry.get("api") or entry.get("qualname") or ""
    doc = entry.get("doc") or ""
    sig = entry.get("signature")
    base_row = collector_entry_to_api_row(entry)

    resolved = bool(entry.get("resolved", False))
    if not resolved:
        return Decision(False, 1, f"unresolved: {entry.get('error', 'unresolved')}"), base_row
    if is_internal_api(api):
        return Decision(False, 5, internal_api_reason(api) or "internal implementation namespace excluded"), base_row
    if _is_placeholder_doc(doc):
        return Decision(False, 1, "missing/placeholder/too short doc"), base_row
    if _is_reference_only_doc(doc):
        return Decision(False, 1, "reference-only doc / alias redirect"), base_row
    if _is_exception_like(entry):
        return Decision(False, 2, "classes/exceptions are excluded"), base_row

    if not allow_wrapper_bridges:
        bad_ns, why = rejects_wrapper_bridge_namespace(api)
        if bad_ns:
            return Decision(False, 5, why), base_row

    external_alias, why = rejects_external_alias(api, entry)
    if external_alias:
        return Decision(False, 5, why), base_row

    sig_params = parse_signature_params(sig)
    doc_params = parse_params_from_doc(doc)
    concrete_sig = [p for p in sig_params if p[0] not in {"self", "cls", "args", "kwargs"}]
    if not concrete_sig:
        return Decision(False, 2, "no concrete parameters; zero-arg APIs are excluded"), base_row

    doc_by_name = {p.name: p for p in doc_params}
    required_sig = [n for (n, has_default, _) in concrete_sig if not has_default]
    missing_required = [n for n in required_sig if n not in doc_by_name]
    if missing_required:
        return Decision(False, 2, f"missing required param descriptions for: {', '.join(missing_required[:8])}"), base_row

    vague_required = [n for n in required_sig if n in doc_by_name and _is_vague_param_desc(doc_by_name[n].desc)]
    if vague_required:
        return Decision(False, 2, f"vague required param descriptions for: {', '.join(vague_required[:8])}"), base_row

    merged_params: List[ParamInfo] = []
    sig_order = {name: (has_def, default) for name, has_def, default in sig_params if name not in {"self", "cls"}}
    for name, (has_def, default) in sig_order.items():
        dp = doc_by_name.get(name)
        if not dp:
            continue
        merged_params.append(ParamInfo(name=dp.name, type=dp.type, desc=dp.desc, optional=bool(dp.optional or has_def), default=default if has_def else None))

    if not merged_params:
        return Decision(False, 2, "no parsable documented parameters aligned with signature"), base_row

    if not allow_receiver_apis:
        need_receiver, why = requires_receiver_harness(api, sig)
        if need_receiver:
            return Decision(False, 5, why), base_row

    required_params = [p for p in merged_params if not p.optional]
    modifiable_seedable_count = 0
    for p in merged_params:
        mod, _ = is_modifiable_param(p)
        if not mod:
            continue
        if simple_seedability(p).ok:
            modifiable_seedable_count += 1
    if modifiable_seedable_count == 0:
        return Decision(False, 3, "no modifiable seedable parameters"), base_row

    for p in required_params:
        fuzzable, why = is_fuzzable_param(p)
        if not fuzzable:
            return Decision(False, 3, f"required param '{p.name}' not fuzzable: {why}"), base_row
        opaque, opaque_why = has_opaque_custom_type(p)
        if opaque:
            return Decision(False, 3, f"required param '{p.name}' has {opaque_why}"), base_row
        if not _has_constraint_signal(p):
            return Decision(False, 3, f"required param '{p.name}' lacks constraint/type/shape/value detail"), base_row

    seed_ok, seed_why = baseline_seedability(merged_params)
    if not seed_ok:
        return Decision(False, 3, seed_why), base_row

    ret_ok, ret_why = has_meaningful_return(sig=sig, doc=doc)
    if not ret_ok:
        return Decision(False, 4, ret_why), base_row

    ctx_bad, ctx_why = has_context_dependent_contract(api_name=api, doc=doc, params=merged_params, sig=sig)
    if ctx_bad:
        return Decision(False, 5, ctx_why), base_row

    utility, utility_why = is_utility_api(api_name=api, doc=doc, params=merged_params)
    if utility:
        return Decision(False, 5, utility_why), base_row

    se, why = has_side_effect(api_name=api, doc=doc, params=merged_params)
    if se:
        return Decision(False, 5, why), base_row

    return Decision(True, 0, ""), base_row


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def write_two_col_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["api_full_name", "api_doc_text"])
        w.writeheader()
        for r in rows:
            w.writerow({"api_full_name": r.get("api_full_name", ""), "api_doc_text": r.get("api_doc_text", "")})


def write_rejected_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["api_full_name", "reason", "api_doc_text"])
        w.writeheader()
        for r in rows:
            w.writerow({"api_full_name": r.get("api_full_name", ""), "reason": r.get("reason", ""), "api_doc_text": r.get("api_doc_text", "")})


def infer_library_name(data: List[Dict[str, Any]], fallback: str = "library") -> str:
    for e in data:
        api = e.get("api") or e.get("qualname") or ""
        if api and "." in api:
            return api.split(".", 1)[0]
        if api:
            return api
    return fallback


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (collector output)")
    ap.add_argument("--outdir", required=True, help="Base output directory")
    ap.add_argument("--lib-name", default="", help="Library name for output folder (default: inferred)")
    ap.add_argument("--max", type=int, default=0, help="Max entries to process (0=all)")
    ap.add_argument("--max-examples", type=int, default=50, help="Max example reasons per step in summary.json")
    ap.add_argument("--allow-receiver-apis", action="store_true", help="Keep receiver/method APIs that require special harnesses")
    ap.add_argument("--allow-wrapper-bridges", action="store_true", help="Keep wrapper/bridge namespaces")
    args = ap.parse_args(argv)

    data = read_jsonl(args.inp)
    if args.max > 0:
        data = data[: args.max]

    lib_name = args.lib_name.strip() or infer_library_name(data)
    base = os.path.join(args.outdir, lib_name)

    accepted_raw: List[Dict[str, str]] = []
    rejected_rows: List[Dict[str, str]] = []
    summary: Dict[str, Any] = {
        "input": args.inp,
        "library": lib_name,
        "total": 0,
        "accepted": 0,
        "accepted_before_alias_dedup": 0,
        "duplicate_aliases_removed": 0,
        "rejected": 0,
        "rejected_by_step": {_step_name(i): 0 for i in range(1, 6)},
        "top_reasons": {},
        "examples_by_step": {_step_name(i): [] for i in range(1, 6)},
        "config": {"allow_receiver_apis": bool(args.allow_receiver_apis), "allow_wrapper_bridges": bool(args.allow_wrapper_bridges)},
    }

    for entry in data:
        summary["total"] += 1
        decision, row = filter_one(entry, allow_receiver_apis=args.allow_receiver_apis, allow_wrapper_bridges=args.allow_wrapper_bridges)
        if decision.accepted:
            accepted_raw.append(row)
            continue
        summary["rejected"] += 1
        rejected_row = dict(row)
        rejected_row["reason"] = decision.reason
        rejected_rows.append(rejected_row)
        if 1 <= decision.step <= 5:
            sname = _step_name(decision.step)
            summary["rejected_by_step"][sname] += 1
            summary["top_reasons"][decision.reason] = summary["top_reasons"].get(decision.reason, 0) + 1
            ex = summary["examples_by_step"][sname]
            if len(ex) < args.max_examples:
                ex.append({"api_full_name": row.get("api_full_name", ""), "reason": decision.reason})

    accepted_before_alias_dedup = len(accepted_raw)
    accepted = collapse_aliases(accepted_raw)
    summary["accepted_before_alias_dedup"] = accepted_before_alias_dedup
    summary["accepted"] = len(accepted)
    summary["duplicate_aliases_removed"] = accepted_before_alias_dedup - len(accepted)

    os.makedirs(base, exist_ok=True)
    accepted_path = os.path.join(base, "accepted.csv")
    write_api_csv(accepted_path, accepted)
    with open(os.path.join(base, "accepted.json"), "w", encoding="utf-8") as f:
        json.dump(api_collection_json(accepted), f, ensure_ascii=False, indent=2)
    write_contract_rejected_csv(os.path.join(base, "rejected.csv"), rejected_rows)
    with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[filter] library={lib_name} total={summary['total']} accepted={summary['accepted']} dup_removed={summary['duplicate_aliases_removed']} rejected={summary['rejected']}")
    print(f"[filter] wrote: {accepted_path}")
    print(f"[filter] wrote: {os.path.join(base, 'accepted.json')}")
    print(f"[filter] wrote: {os.path.join(base, 'rejected.csv')}")
    print(f"[filter] wrote: {os.path.join(base, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
