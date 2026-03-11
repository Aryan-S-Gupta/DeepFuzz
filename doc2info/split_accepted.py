from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Pattern, Sequence


# -----------------------------
# Rules
# -----------------------------

# Strong API-name signals: these APIs are inherently GPU/backend specific.
EXPLICIT_GPU_API_PATTERNS = [
    r"^torch\.cuda(?:\.|$)",
    r"^torch\.backends\.cuda(?:\.|$)",
    r"^torch\.backends\.cudnn(?:\.|$)",
    r"^torch\.mps(?:\.|$)",
    r"^torch\.xpu(?:\.|$)",
    r"^torch\.xla(?:\.|$)",
    r"^torch\.nccl(?:\.|$)",
    r"(?:^|\.)(?:cuda|cudnn|mps|xpu|xla|nccl)(?:\.|$)",  # e.g. Tensor.cuda
]

# Optional hard CPU overrides for namespaces you explicitly do not want in GPU.
# You mentioned torch.ao as an example, so I included it.
EXPLICIT_CPU_API_PATTERNS = [
    r"^torch\.ao(?:\.|$)",
    r"^torch\.backends\.mkldnn(?:\.|$)",
    r"^torch\.backends\.mkl(?:\.|$)",
]

# Strong doc-text signals that the API is GPU by default or GPU-only.
GPU_DEFAULT_OR_ONLY_PATTERNS = [
    r"\bdefaults?\s+to\s+(?:the\s+)?(?:current\s+)?(?:cuda|gpu|mps|xpu|xla|tpu)\s+device\b",
    r"\bby\s+default[, ]+(?:the\s+)?(?:current\s+)?(?:cuda|gpu|mps|xpu|xla|tpu)\s+device\b",
    r"\buses?\s+the\s+current\s+(?:cuda|gpu|mps|xpu|xla|tpu)\s+(?:device|stream)\b",
    r"\bcurrent\s+(?:cuda|gpu|mps|xpu|xla|tpu)\s+(?:device|stream)\b",
    r"\breturns?\s+(?:a\s+)?(?:cuda|gpu|mps|xpu|xla|tpu)\s+tensor\b",
    r"\bcreates?\s+(?:a\s+)?(?:cuda|gpu|mps|xpu|xla|tpu)\s+tensor\b",
    r"\b(?:cuda|gpu|mps|xpu|xla|tpu)[- ]only\b",
    r"\bonly\s+available\s+(?:for|on)\s+(?:cuda|gpu|mps|xpu|xla|tpu)\b",
    r"\brequires?\s+(?:a\s+)?(?:cuda|gpu|mps|xpu|xla|tpu)\b",
]

# Strong doc-text signals that the API is CPU-only/default CPU.
CPU_DEFAULT_OR_ONLY_PATTERNS = [
    r"\bdefaults?\s+to\s+(?:the\s+)?cpu\b",
    r"\bby\s+default[, ]+(?:the\s+)?cpu\b",
    r"\bonly\s+available\s+(?:for|on)\s+cpu\b",
    r"\bcpu[- ]only\b",
]

# These patterns mean "mentions device/GPU support", but NOT "this API belongs in GPU".
# Per your rule: if GPU requires an extra device arg / append / opt-in, do NOT place in GPU.
OPT_IN_OR_DEVICE_AGNOSTIC_PATTERNS = [
    r"\b(?:cpu|cuda|gpu|mps|xpu|xla|tpu)\s*(?:/|or)\s*(?:cpu|cuda|gpu|mps|xpu|xla|tpu)\b",
    r"\bdepending\s+on\s+(?:the\s+)?device\b",
    r"\bsame\s+device\s+as\s+(?:the\s+)?(?:input|self|other)\b",
    r"\b(?:input|self|other)\s+device\b",
    r"\bdevice-agnostic\b",
    r"\bon\s+whatever\s+device\b",
    r"\bwhen\s+.*\bon\s+(?:cuda|gpu|mps|xpu|xla|tpu)\b",
    r"\bif\s+.*\bdevice\b.*\b(?:cuda|gpu|mps|xpu|xla|tpu)\b",
    r"\bmove\b.*\bto\b.*\b(?:cuda|gpu|mps|xpu|xla|tpu)\b",
    r"\bpass\b.*\bdevice\b",
    r"\buse\b.*\.(?:cuda|mps|xpu|to)\(",
]

_COMPILED_GPU_API = [re.compile(p, re.IGNORECASE)
                     for p in EXPLICIT_GPU_API_PATTERNS]
_COMPILED_CPU_API = [re.compile(p, re.IGNORECASE)
                     for p in EXPLICIT_CPU_API_PATTERNS]
_COMPILED_GPU_DOC = [re.compile(p, re.IGNORECASE)
                     for p in GPU_DEFAULT_OR_ONLY_PATTERNS]
_COMPILED_CPU_DOC = [re.compile(p, re.IGNORECASE)
                     for p in CPU_DEFAULT_OR_ONLY_PATTERNS]
_COMPILED_OPT_IN = [re.compile(p, re.IGNORECASE)
                    for p in OPT_IN_OR_DEVICE_AGNOSTIC_PATTERNS]


def _matches_any(text: str, patterns: Sequence[Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def classify_backend(row: Dict[str, str]) -> str:
    """
    Policy:
    - Put in GPU only if:
      1) the API name itself is explicitly GPU/backend-specific, OR
      2) the doc strongly says the API is GPU-only or defaults to GPU/current GPU device.
    - If GPU is only optional / opt-in / depends on device, keep it OUT of GPU.
    - Everything else falls back to CPU.
    """
    api = _normalize_text(row.get("api_full_name", ""))
    doc = _normalize_text(row.get("api_doc_text", ""))

    # 1) Hard API-name decisions first.
    if _matches_any(api, _COMPILED_GPU_API):
        return "gpu"

    if _matches_any(api, _COMPILED_CPU_API):
        return "cpu"

    # 2) Strong doc decisions.
    has_gpu_default_or_only = _matches_any(doc, _COMPILED_GPU_DOC)
    has_cpu_default_or_only = _matches_any(doc, _COMPILED_CPU_DOC)
    has_opt_in_or_agnostic = _matches_any(doc, _COMPILED_OPT_IN)

    if has_cpu_default_or_only:
        return "cpu"

    # Only keep in GPU if it's really default/required GPU,
    # not just "can work on GPU" or "depends on device".
    if has_gpu_default_or_only and not has_opt_in_or_agnostic:
        return "gpu"

    # 3) Safe fallback:
    # If it is not explicitly/default GPU, keep it in CPU.
    return "cpu"


def _sniff_csv_dialect(path: str) -> csv.Dialect:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            return csv.excel


def read_two_col_csv(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    dialect = _sniff_csv_dialect(path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            out.append(
                {
                    "api_full_name": (row.get("api_full_name", "") or "").strip(),
                    "api_doc_text": row.get("api_doc_text", "") or "",
                }
            )
    return out


def write_two_col_csv(path: str, rows: List[Dict[str, str]]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    headers = ["api_full_name", "api_doc_text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "api_full_name": r.get("api_full_name", ""),
                    "api_doc_text": r.get("api_doc_text", ""),
                }
            )


def maybe_update_summary(
    summary_path: str,
    accepted_total: int,
    accepted_cpu: List[Dict[str, str]],
    accepted_gpu: List[Dict[str, str]],
) -> None:
    if not os.path.exists(summary_path):
        return

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return

    summary["accepted_total"] = accepted_total
    summary["accepted_cpu"] = len(accepted_cpu)
    summary["accepted_gpu"] = len(accepted_gpu)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--accepted-csv",
        required=True,
        help="accepted.csv from the filter step",
    )
    ap.add_argument(
        "--summary-json",
        default="",
        help="Optional summary.json to update with cpu/gpu counts",
    )
    args = ap.parse_args(argv)

    accepted = read_two_col_csv(args.accepted_csv)

    accepted_cpu: List[Dict[str, str]] = []
    accepted_gpu: List[Dict[str, str]] = []

    # Classify each row once.
    for row in accepted:
        backend = classify_backend(row)
        if backend == "gpu":
            accepted_gpu.append(row)
        else:
            accepted_cpu.append(row)

    base_dir = os.path.dirname(os.path.abspath(args.accepted_csv))
    cpu_path = os.path.join(base_dir, "accepted_cpu.csv")
    gpu_path = os.path.join(base_dir, "accepted_gpu.csv")

    write_two_col_csv(cpu_path, accepted_cpu)
    write_two_col_csv(gpu_path, accepted_gpu)

    summary_path = args.summary_json.strip() or os.path.join(base_dir, "summary.json")
    maybe_update_summary(summary_path, len(accepted),
                         accepted_cpu, accepted_gpu)

    print(
        f"[backend-split] total_accepted={len(accepted)} "
        f"cpu={len(accepted_cpu)} gpu={len(accepted_gpu)}"
    )
    print(f"[backend-split] wrote: {cpu_path}")
    print(f"[backend-split] wrote: {gpu_path}")
    if os.path.exists(summary_path):
        print(f"[backend-split] updated: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
