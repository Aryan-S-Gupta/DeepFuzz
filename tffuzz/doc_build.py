# tffuzz/doc_build.py
from __future__ import annotations
import argparse
import json
import os
import time
import enum
from pathlib import Path
from typing import Any, Dict, List
import inspect

from .doc_collect import collect_doc_context, resolve_callable

from .llm_client import LLMClient
from .registry import (
    load_param_rules,
    apply_param_rules,
    coerce_allowed_dtypes,
    UNIFIED_DTYPES,
)

# ---------------------------------------------------------------------
# System Prompt + Few-Shot
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You convert Python API documentation into a compact JSON specification.

You are given:
- QUALNAME
- SIGNATURE
- DOCSTRING

The library may be ANY Python library (not necessarily deep learning).

Your output is a single JSON object with fields:

{
  "name": str,
  "qualname": str,
  "params": [
    {
      "name": str,
      "type": "tensor" | "shape" | "number" | "bool" | "str" | "any",
      "optional": bool,
      "allowed_dtypes": [str],         // optional
      "min_rank": int,                 // optional
      "max_rank": int,                 // optional
      "description": str               // optional
    }
  ],
  "oracles": {
    "invariants": [str]
  }
}

Constraints:
- Output must be valid JSON.
- Do NOT invent parameters not present in the signature.
- Do NOT omit any parameters from the signature.
- If unsure, use type="any" and optional=true.
- Keep descriptions brief.
- Never output comments, explanations, or markdown.
- Do NOT invent invariants not explicitly stated in the docstring.
- If no invariants are mentioned in the documentation, use an empty list [].
- Only include allowed_dtypes, min_rank, or max_rank if the docstring explicitly states such constraints.
- If the docstring does not mention dtypes or shapes, omit those fields for that parameter.
"""

# ---------------------------------------------------------------------
# Utility: Merge fallback + partial LLM output
# ---------------------------------------------------------------------


def merge_llm_fallback(llm_obj: dict, fb_obj: dict) -> dict:
    """
    Keep as much of LLM output as possible, fill missing fields from fallback.
    """
    if not isinstance(llm_obj, dict):
        return fb_obj

    out = fb_obj.copy()

    # name/qualname
    out["qualname"] = fb_obj["qualname"]
    out["name"] = fb_obj["name"]

    # merge params
    if isinstance(llm_obj.get("params"), list):
        good = []
        for p in llm_obj["params"]:
            if isinstance(p, dict) and p.get("name"):
                good.append(p)
        if good:
            out["params"] = good

    # merge invariants
    if isinstance(llm_obj.get("oracles"), dict):
        inv = llm_obj["oracles"].get("invariants")
        if isinstance(inv, list):
            out["oracles"]["invariants"] = inv

    return out


# ---------------------------------------------------------------------
# Fallback spec from signature (general-purpose)
# ---------------------------------------------------------------------

def fallback_spec_from_signature(qualname: str, signature_str: str) -> dict:
    """
    Very conservative fallback: no parameters.
    Used only when doc/LLM/sig pipeline fails.
    """
    from .doc_collect import resolve_callable
    import inspect

    try:
        fn = resolve_callable(qualname)

        if inspect.isclass(fn) and (issubclass(fn, enum.Enum) or looks_like_enum(fn)):
            return {
                "name": qualname.split(".")[-1],
                "qualname": qualname,
                "params": [],
                "oracles": {"invariants": []},
            }

        # For everything else, do NOT guess parameters.
        return {
            "name": qualname.split(".")[-1],
            "qualname": qualname,
            "params": [],
            "oracles": {"invariants": []},
        }

    except Exception:
        return {
            "name": qualname.split(".")[-1],
            "qualname": qualname,
            "params": [],
            "oracles": {"invariants": []},
        }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def spec_valid(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("name"), str):
        return False
    if not isinstance(obj.get("qualname"), str):
        return False
    if not isinstance(obj.get("params"), list):
        return False

    for p in obj["params"]:
        if not isinstance(p, dict) or not isinstance(p.get("name"), str):
            return False
        if p.get("type") not in ("tensor", "shape", "number", "bool", "str", "any"):
            return False
        if "optional" in p and not isinstance(p["optional"], bool):
            return False

    return True


# ---------------------------------------------------------------------
# Interior prompt builder
# ---------------------------------------------------------------------

def build_prompt(qualname: str, signature: str, docstring: str) -> str:
    return (
        f"QUALNAME: {qualname}\n"
        f"SIGNATURE: {signature}\n"
        f"DOCSTRING: {docstring}\n"
        "Return ONLY the JSON object."
    )


def looks_useless_sig(sig: inspect.Signature, doc: str):
    # 1-param positional-only like (x, /) → almost always useless (PyTorch, OpenCV)
    if len(sig.parameters) == 1:
        p = list(sig.parameters.values())[0]
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            return True

    # Signatures like (self, /) → not meaningful
    names = [p.name for p in sig.parameters.values()]
    if names in (["self"], ["args"], ["kwargs"]):
        return True

    # empty doc + almost empty signature → useless
    if len(sig.parameters) <= 1 and (not doc or doc.strip() == ""):
        return True

    return False


def looks_like_enum(cls):
    """
    Detect pseudo-enum classes used in PyTorch, TensorFlow, OpenCV, etc.
    """
    import inspect
    try:
        if not inspect.isclass(cls):
            return False

        if issubclass(cls, enum.Enum):
            return True

        sig = inspect.signature(cls.__init__)
        real = [p for p in sig.parameters.values() if p.name != "self"]
        if real:
            return False

        attrs = [a for a in dir(cls) if not a.startswith("_")]
        upper = [a for a in attrs if a.isupper()]
        return len(upper) >= 3
    except Exception:
        return False

# ---------------------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Build enriched API specs for TF/Torch/JAX/etc.")
    ap.add_argument("--in", dest="infile",
                    default="tffuzz/API/TF_API_min.json")
    ap.add_argument("--out", dest="outfile",
                    default="tffuzz/API/TF_API_enriched.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rate-limit", type=float, default=0.0)
    ap.add_argument("--param-rules", default="tffuzz/API/param_rules.json")
    ap.add_argument("--model", default="gemma3:1b-it-qat")
    ap.add_argument(
        "--llm-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Load existing enriched (cache)
    existing = {}
    if outfile.exists():
        try:
            with open(outfile, "r") as f:
                current = json.load(f)
                for o in current:
                    if isinstance(o, dict) and "qualname" in o:
                        existing[o["qualname"]] = o
        except Exception:
            pass

    # Load list of qualnames
    with open(infile, "r") as f:
        raw = json.load(f)

    qualnames = []
    for item in raw:
        if isinstance(item, str):
            qualnames.append(item)
        elif isinstance(item, dict) and isinstance(item.get("qualname"), str):
            qualnames.append(item["qualname"])

    if args.limit > 0:
        qualnames = qualnames[:args.limit]

    client = LLMClient(
        backend="ollama",
        model=args.model,
        base_url=args.llm_url,
        timeout=120,
        temperature=0.0,
        max_retries=3,
        rate_limit_s=args.rate_limit,
    )

    rules = load_param_rules(args.param_rules)

    enriched = []
    total = len(qualnames)

    print(f"[doc_build] Processing {total} APIs...")

    for i, qn in enumerate(qualnames, start=1):
        print(f"[{i}/{total}] {qn} ... ", end="")

        # Cached?
        if qn in existing:
            enriched.append(existing[qn])
            print("cached")
            continue

        # Extract signature + docstring
        try:
            signature, docstring = collect_doc_context(qn)
        except Exception as e:
            print(f"fallback (collect error: {e})")
            fb = fallback_spec_from_signature(qn, "")
            enriched.append(fb)
            continue

        prompt = build_prompt(qn, signature, docstring)

        # Prepare fallback
        fb = fallback_spec_from_signature(qn, signature)

        try:
            llm_obj = client.complete_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt
            )

            # Normalize core fields
            llm_obj["qualname"] = qn
            llm_obj["name"] = qn.split(".")[-1]

            # Basic sanitization of LLM output
            if not isinstance(llm_obj.get("params"), list):
                llm_obj["params"] = []

            # Normalize oracles.invariants: treat ["None"] / ["none"] as empty
            oracles = llm_obj.get("oracles")
            if isinstance(oracles, dict):
                inv = oracles.get("invariants")
                if isinstance(inv, list):
                    cleaned = [str(s).strip()
                               for s in inv if isinstance(s, str)]
                    if all(str(s).strip().lower() == "none" for s in cleaned):
                        cleaned = []
                    oracles["invariants"] = cleaned
                else:
                    oracles["invariants"] = []
            else:
                llm_obj["oracles"] = {"invariants": []}

            # Apply param rules & dtype coercion on raw LLM params
            for p in llm_obj.get("params", []):
                if not isinstance(p, dict) or "name" not in p:
                    continue
                nm = p["name"]
                proposed = p.get("type", "any")
                p["type"] = apply_param_rules(nm, proposed, rules)

                if nm == "name":
                    p["type"] = "str"
                    if "description" not in p:
                        p["description"] = "A name for the operation."

                if "allowed_dtypes" in p:
                    p["allowed_dtypes"] = coerce_allowed_dtypes(
                        p["allowed_dtypes"], allowed=UNIFIED_DTYPES
                    )

            # Drop clearly invalid / marker params hallucinated by the LLM
            llm_obj["params"] = [
                p for p in llm_obj.get("params", [])
                if isinstance(p, dict)
                and p.get("name")
                and p["name"] not in ("/", "*", "**")
            ]

            # ----- Derive real_params from real Python signature, with fallbacks -----
            use_real_sig = True
            try:
                fn = resolve_callable(qn)

                # Enums -> no params
                if inspect.isclass(fn) and (issubclass(fn, enum.Enum) or looks_like_enum(fn)):
                    enriched.append({
                        "name": qn.split(".")[-1],
                        "qualname": qn,
                        "params": [],
                        "oracles": {"invariants": []},
                    })
                    print("enum")
                    continue

                # Determine callable target (class __init__ or function)
                if inspect.isclass(fn):
                    target = fn.__init__
                else:
                    target = fn

                try:
                    sig = inspect.signature(target)
                except Exception:
                    sig = None

                if sig is None or looks_useless_sig(sig, docstring):
                    real_params = fb["params"]
                    use_real_sig = False
                else:
                    real_params = list(sig.parameters.values())

            except Exception:
                real_params = fb["params"]

            # ------------------------------------------------------------------
            # ENFORCE REAL SIGNATURE PARAM NAMES, KEEP LLM METADATA WHERE VALID
            # ------------------------------------------------------------------
            if not use_real_sig:
                final = llm_obj if spec_valid(
                    llm_obj) else merge_llm_fallback(llm_obj, fb)
                enriched.append(final)
                print("ok")
                continue

            clean_params: List[Dict[str, Any]] = []
            llm_by_name = {
                p.get("name"): p
                for p in llm_obj.get("params", [])
                if isinstance(p, dict) and p.get("name")
            }

            ds = (docstring or "").lower()

            mentions_dtype = any(
                kw in ds
                for kw in [
                    "dtype:",
                    "dtypes:",
                    "data type",
                    "must be of type",
                    "only supports dtype",
                ]
            )

            mentions_shape = any(
                kw in ds
                for kw in [
                    "shape:",
                    "shape :",
                    "of shape",
                    "has shape",
                    "expected shape",
                    "tensor shape",
                ]
            )

            for p in real_params:
                if isinstance(p, inspect.Parameter):
                    name = p.name
                    cand = llm_by_name.get(name)

                    # keyword-only
                    if p.kind == inspect.Parameter.KEYWORD_ONLY:
                        optional = (p.default is not inspect._empty)
                        base = {
                            "name": name,
                            "type": cand.get("type", "any") if cand else "any",
                            "optional": optional,
                        }
                        clean_params.append(base)
                        continue

                    # *args
                    if p.kind == inspect.Parameter.VAR_POSITIONAL:
                        name = p.name.lstrip("*")
                        clean_params.append(
                            {"name": name, "type": "any", "optional": True}
                        )
                        continue

                    # **kwargs
                    if p.kind == inspect.Parameter.VAR_KEYWORD:
                        name = p.name.lstrip("*")
                        clean_params.append(
                            {"name": name, "type": "any", "optional": True}
                        )
                        continue

                    # normal parameters
                    if name == "self":
                        continue
                    if name in ("/", "*", "**"):
                        continue

                    cand = llm_by_name.get(name)
                    optional = (p.default is not inspect._empty)

                elif hasattr(p, "name"):  # ParamInfo from doc_collect
                    name = p.name
                    d = getattr(p, "default", None)
                    # ParamInfo.default is a STRING of the default value, not actual Python default
                    optional = not (d in (None, "None", "inspect._empty"))

                elif isinstance(p, dict):
                    name = p.get("name")
                    if not name:
                        continue
                    optional = bool(p.get("optional", True))

                else:
                    continue

                cand = llm_by_name.get(name)

                base: Dict[str, Any] = {
                    "name": name,
                    "type": cand.get("type", "any") if cand else "any",
                    "optional": optional,
                }

                if cand:
                    # Only keep allowed_dtypes if docstring clearly talks about dtypes
                    if mentions_dtype and "allowed_dtypes" in cand:
                        raw_dt = cand["allowed_dtypes"]
                        if isinstance(raw_dt, str):
                            raw_dt = [raw_dt]
                        norm = coerce_allowed_dtypes(
                            raw_dt, allowed=UNIFIED_DTYPES)
                        if norm:
                            base["allowed_dtypes"] = norm

                clean_params.append(base)

            llm_obj["params"] = clean_params

            # Final invariants clean-up
            inv = llm_obj.get("oracles", {}).get("invariants", [])
            if not isinstance(inv, list) or all(
                str(x).strip().lower() == "none" for x in inv
            ):
                llm_obj["oracles"] = {"invariants": []}

            final = llm_obj if spec_valid(
                llm_obj) else merge_llm_fallback(llm_obj, fb)
            enriched.append(final)
            print("ok")

        except Exception as e:
            print(f"fallback (LLM error: {e})")
            try:
                fn = resolve_callable(qn)
                if inspect.isclass(fn):
                    target = fn.__init__
                else:
                    target = fn
                sig = inspect.signature(target)
                real_params = []
                for p in sig.parameters.values():
                    if p.name == "self":
                        continue
                    if p.kind == p.VAR_POSITIONAL:
                        real_params.append({"name": p.name.lstrip(
                            "*"), "type": "any", "optional": True})
                        continue
                    if p.kind == p.VAR_KEYWORD:
                        real_params.append({"name": p.name.lstrip(
                            "*"), "type": "any", "optional": True})
                        continue
                    if p.name in ("/", "*", "**"):
                        continue
                    real_params.append(p)

                fixed = []
                for p in real_params:
                    fixed.append({
                        "name": p.name,
                        "type": "any",
                        "optional": (p.default is not inspect._empty),
                    })

                enriched.append({
                    "name": qn.split(".")[-1],
                    "qualname": qn,
                    "params": fixed,
                    "oracles": {"invariants": []},
                })
            except:
                enriched.append(fb)

        with open(outfile, "w") as f:
            json.dump(enriched, f, indent=2)

    with open(outfile, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"[doc_build] Finished. Wrote {len(enriched)} specs → {outfile}")


if __name__ == "__main__":
    main()
