from __future__ import annotations

"""Library-specific API eligibility policy shared by pipeline stages."""

from typing import Optional, Sequence, Tuple


INTERNAL_NAMESPACE_PREFIXES = {
    "jax": ("jax._src", "jax.interpreters", "jaxlib"),
    "jaxlib": ("jaxlib",),
}


def library_from_api(api_name: str, fallback: str = "") -> str:
    api = str(api_name or "").strip()
    if "." in api:
        return api.split(".", 1)[0]
    return fallback or api


def default_internal_prefixes(library: str) -> Tuple[str, ...]:
    return tuple(INTERNAL_NAMESPACE_PREFIXES.get(str(library or "").strip(), ()))


def _matches_prefix(api_name: str, prefix: str) -> bool:
    api = str(api_name or "").strip()
    pfx = str(prefix or "").strip().rstrip(".")
    return bool(api and pfx and (api == pfx or api.startswith(pfx + ".")))


def is_internal_api(api_name: str, library: Optional[str] = None, extra_prefixes: Sequence[str] = ()) -> bool:
    lib = str(library or "").strip() or library_from_api(api_name)
    prefixes = default_internal_prefixes(lib) + tuple(str(p) for p in extra_prefixes if str(p).strip())
    return any(_matches_prefix(api_name, prefix) for prefix in prefixes)


def internal_api_reason(api_name: str, library: Optional[str] = None) -> str:
    lib = str(library or "").strip() or library_from_api(api_name)
    for prefix in default_internal_prefixes(lib):
        if _matches_prefix(api_name, prefix):
            return f"internal implementation namespace excluded from public {lib} benchmark: {prefix}.*"
    return ""
