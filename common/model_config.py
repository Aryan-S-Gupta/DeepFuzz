from __future__ import annotations

"""Central model configuration for DeepFuzz LLM calls."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelConfig:
    extraction_model: str
    fast_repair_model: str
    strong_repair_model: str
    host: str
    backend: str = "ollama"

    def to_summary(self) -> Dict[str, str]:
        return asdict(self)


def load_model_config() -> ModelConfig:
    default_model = os.environ.get("DEEPFUZZ_MODEL") or os.environ.get("MODEL") or "mixtral:8x7b"
    fast = os.environ.get("DEEPFUZZ_FAST_MODEL") or os.environ.get("DEEPFUZZ_EXTRACTION_MODEL") or default_model
    repair = os.environ.get("DEEPFUZZ_REPAIR_MODEL") or os.environ.get("REPAIR_MODEL") or fast
    strong = os.environ.get("DEEPFUZZ_STRONG_REPAIR_MODEL") or os.environ.get("STRONG_REPAIR_MODEL") or repair or default_model
    host = os.environ.get("DEEPFUZZ_LLM_HOST") or os.environ.get("OLLAMA_HOST") or os.environ.get("HOST") or "http://localhost:11434"
    backend = (os.environ.get("DEEPFUZZ_MODEL_BACKEND") or os.environ.get("MODEL_BACKEND") or "ollama").strip().lower()
    return ModelConfig(
        extraction_model=fast,
        fast_repair_model=repair,
        strong_repair_model=strong,
        host=host,
        backend=backend,
    )


def check_model_backend(model: str, host: str, backend: str = "ollama", timeout: float = 5.0) -> Tuple[bool, str]:
    """Return whether the configured model backend is reachable.

    The pipeline currently sends generation/repair requests through Ollama.  We
    still allow an OpenAI backend marker in summaries so configuration is
    explicit, but local Ollama health is the only network check performed here.
    """

    backend = (backend or "ollama").strip().lower()
    model = (model or "").strip()
    if not model:
        return False, "no model configured"
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set"
        return False, "OpenAI backend is configured but this repository entrypoint currently expects an Ollama-compatible generation endpoint"
    if backend != "ollama":
        return False, f"unsupported model backend: {backend}"

    base = (host or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        models = data.get("models", []) if isinstance(data, dict) else []
        names = {str(item.get("name", "")) for item in models if isinstance(item, dict)}
        short_names = {name.split(":", 1)[0] for name in names}
        if names and model not in names and model.split(":", 1)[0] not in short_names:
            return False, f"Ollama is reachable at {base}, but model '{model}' is not listed"
        return True, f"Ollama is reachable at {base}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"Ollama backend unavailable at {base}: {type(exc).__name__}: {exc}"
