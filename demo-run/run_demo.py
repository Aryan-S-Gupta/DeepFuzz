#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo-run"

DEFAULT_APIS = [
    "jax.numpy.sin",
    "tensorflow.nn.relu",
    "torch.masked.argmax",
]


def lib_for_api(api: str) -> str:
    if api.startswith("jax."):
        return "jax"
    if api.startswith("tensorflow."):
        return "tensorflow"
    if api.startswith("torch."):
        return "torch"
    raise ValueError(f"Cannot infer target library from API name: {api}")


def read_api_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(cmd: list[str], *, env: dict[str, str], cwd: Path = ROOT, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(cmd)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {rendered}\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        handle.write(proc.stdout)
    print(rendered)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)


def copy_csv_subset(src: Path, dst: Path, apis: list[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(apis)
    with src.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    matched = [row for row in rows if str(row.get("api_full_name", "")).strip() in wanted]
    missing = sorted(wanted - {str(row.get("api_full_name", "")).strip() for row in matched})
    if missing:
        raise SystemExit(f"Missing accepted doc rows in {src}: {missing}")
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched)


def copy_stage1_specs(lib: str, apis: list[str], spec_dir: Path, *, allow_generate: bool, env: dict[str, str], log: Path) -> None:
    spec_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    copied: list[dict[str, str]] = []
    for api in apis:
        src = ROOT / "info2json" / "results" / lib / f"{api}.json"
        dst = spec_dir / f"{api}.json"
        if src.exists():
            shutil.copy2(src, dst)
            copied.append({"api": api, "source": str(src), "copied_to": str(dst)})
        else:
            missing.append(api)

    if missing and allow_generate:
        api_list = spec_dir.parent / "stage0_selection" / "api_list.txt"
        cmd = [
            env["PYTHON"],
            "info2json/info2json.py",
            "--input",
            str(ROOT / "doc2info" / "results" / lib / "accepted.csv"),
            "--outdir",
            str(spec_dir),
            "--model",
            env.get("DEEPFUZZ_MODEL", "qwen2.5:14b"),
            "--host",
            env.get("OLLAMA_HOST", "http://localhost:11434"),
            "--timeout",
            "300",
            "--limit",
            "0",
            "--retries",
            "1",
            "--sleep",
            "0",
            "--overwrite",
            "--num-predict",
            "320",
            "--num-ctx",
            "2048",
            "--temperature",
            "0",
            "--only-api-list",
            str(api_list),
        ]
        run(cmd, env=env, log=log)
        missing = [api for api in missing if not (spec_dir / f"{api}.json").exists()]

    if missing:
        raise SystemExit(
            "Missing Stage 1 specs for APIs: "
            + ", ".join(missing)
            + ". Choose APIs already present in info2json/results, or rerun with --generate-missing-stage1 while Ollama is running."
        )

    write_json(
        spec_dir / "stage1_manifest.json",
        {
            "stage": "stage1_doc_to_spec",
            "mode": "reused_existing_specs" if copied else "generated_missing_specs",
            "apis": apis,
            "copied_specs": copied,
            "note": "For a short live demo, this stage reuses committed Stage 1 JSON specs when available, then downstream stages are rerun into demo-run.",
        },
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def count_csv(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def pct_text(value: Any) -> str:
    if value is None or value == "":
        return "unavailable"
    return f"{value}%"


def build_demo_report(run_dir: Path, apis_by_lib: dict[str, list[str]], mutation_budget: int, seed: int) -> None:
    libraries: dict[str, Any] = {}
    for lib, apis in apis_by_lib.items():
        lib_dir = run_dir / lib
        stage4 = lib_dir / "stage4_results"
        coverage = load_json(stage4 / "coverage_report.json")
        bugs = load_json(stage4 / "bug_report.json")
        summary = load_json(stage4 / "summary.json")
        libraries[lib] = {
            "apis": apis,
            "stage2_valid": count_csv(lib_dir / "stage2_validator" / "ok.csv"),
            "stage3_ready": count_csv(lib_dir / "stage3_init" / "ok.csv"),
            "generated_tests": len(list((lib_dir / "generated_tests").glob("test_*.py"))),
            "stage4_selected": summary.get("selected_apis", 0),
            "base_valid": summary.get("base_success", 0),
            "api_execution_coverage_percent": coverage.get("api_execution_coverage_percent"),
            "selected_function_line_coverage_percent": coverage.get("function_line_coverage_percent"),
            "package_python_line_coverage_percent": coverage.get("python_line_coverage_percent"),
            "valid_programs": summary.get("total_valid_programs", summary.get("valid_programs", 0)),
            "candidate_bugs": bugs.get("summary", {}).get("total_candidates", 0),
            "device_oracle_mismatches": summary.get("device_oracle_mismatch", 0),
            "device_oracle": coverage.get("oracle_summary", {}),
        }

    payload = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "mutation_budget": mutation_budget,
        "seed": seed,
        "libraries": libraries,
    }
    write_json(run_dir / "demo_report.json", payload)

    lines = [
        "# DeepFuzz Demo Run Report",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Seed: `{seed}`",
        f"- Mutation budget per API: `{mutation_budget}`",
        f"- APIs: `{sum(len(v) for v in apis_by_lib.values())}`",
        "",
        "| Library | APIs | Stage 2 valid | Stage 3 ready | Base-valid | API exec. cov. | Valid programs | Candidate bugs | Device mismatches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lib, row in libraries.items():
        lines.append(
            f"| {lib} | {len(row['apis'])} | {row['stage2_valid']} | {row['stage3_ready']} | "
            f"{row['base_valid']}/{row['stage4_selected']} | {pct_text(row['api_execution_coverage_percent'])} | "
            f"{row['valid_programs']} | "
            f"{row['candidate_bugs']} | {row['device_oracle_mismatches']} |"
        )
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- Stage 2 valid means the documentation-derived JSON spec passed schema/runtime validation.",
            "- Stage 3 ready means DeepFuzz could turn the spec into a deterministic executable seed and smoke-test it.",
            "- Base-valid means the original seed executed successfully in Stage 4.",
            "- Valid programs counts the base execution plus successful intended-valid mutations.",
            "- Candidate bugs are triage records, not confirmed upstream bugs.",
            "- Device mismatches appear only when the CPU plus accelerator oracle is available and observes different outputs.",
        ]
    )
    write_text(run_dir / "demo_report.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a small DeepFuzz live-demo campaign for manually supplied APIs.")
    ap.add_argument("--apis", nargs="*", default=[], help="API names, e.g. jax.numpy.sin tensorflow.nn.relu torch.masked.sum")
    ap.add_argument("--api-file", default="", help="Optional newline-delimited API list.")
    ap.add_argument("--interactive", action="store_true", help="Prompt for API names interactively.")
    ap.add_argument("--out", default="demo-run/results/latest", help="Output directory for all demo artefacts.")
    ap.add_argument("--force", action="store_true", help="Replace the output directory if it already exists.")
    ap.add_argument("--mutation-budget", type=int, default=2, help="Small budget keeps the live demo fast.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--worker-timeout-sec", type=int, default=120)
    ap.add_argument("--case-timeout-sec", type=int, default=30)
    ap.add_argument("--python", default="", help="Python interpreter. Defaults to .venv/bin/python if present.")
    ap.add_argument("--generate-missing-stage1", action="store_true", help="Use info2json+Ollama if a Stage 1 spec is missing.")
    ap.add_argument("--enable-device-oracle", action="store_true", help="Compare CPU vs accelerator when both are available.")
    ap.add_argument(
        "--coverage-scope",
        default="api_only",
        choices=["none", "api_only", "python", "both"],
        help="Use api_only for a fast live demo; use python for slower selected-function coverage experiments.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    apis: list[str] = []
    if args.api_file:
        apis.extend(read_api_list(Path(args.api_file)))
    apis.extend(args.apis)
    if args.interactive:
        print("Enter API names separated by spaces or newlines. Finish with an empty line.")
        while True:
            line = input("api> ").strip()
            if not line:
                break
            apis.extend(part.strip() for part in line.replace(",", " ").split() if part.strip())
    if not apis:
        apis = list(DEFAULT_APIS)
    apis = list(dict.fromkeys(apis))

    py = args.python or (str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHON"] = py
    env.setdefault("PYTHONPATH", str(ROOT))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("DEEPFUZZ_MODEL_BACKEND", "ollama")
    env.setdefault("OLLAMA_HOST", "http://localhost:11434")
    env.setdefault("DEEPFUZZ_MODEL", "qwen2.5:14b")
    env.setdefault("DEEPFUZZ_REPAIR_MODEL", "qwen2.5-coder:14b")
    env.setdefault("DEEPFUZZ_STRONG_REPAIR_MODEL", "qwen3-coder:30b")

    apis_by_lib: dict[str, list[str]] = defaultdict(list)
    for api in apis:
        apis_by_lib[lib_for_api(api)].append(api)

    write_text(out / "all_apis.txt", "\n".join(apis) + "\n")
    log = out / "commands.log"
    write_text(log, f"# DeepFuzz demo run command log\n# output={out}\n")

    for lib, lib_apis in sorted(apis_by_lib.items()):
        lib_dir = out / lib
        stage0 = lib_dir / "stage0_selection"
        stage1 = lib_dir / "stage1_specs"
        stage2 = lib_dir / "stage2_validator"
        stage3 = lib_dir / "stage3_init"
        stage4 = lib_dir / "stage4_results"
        tests = lib_dir / "generated_tests"

        write_text(stage0 / "api_list.txt", "\n".join(lib_apis) + "\n")
        copy_csv_subset(ROOT / "doc2info" / "results" / lib / "accepted.csv", stage0 / "accepted_subset.csv", lib_apis)
        write_json(stage0 / "selection_manifest.json", {"library": lib, "apis": lib_apis, "selection": "manual_live_demo_subset"})

        copy_stage1_specs(lib, lib_apis, stage1, allow_generate=args.generate_missing_stage1, env=env, log=log)

        run(
            [
                py,
                "json_validator/json_validator.py",
                "--spec-dir",
                str(stage1),
                "--api-csv",
                str(ROOT / "doc2info" / "results" / lib / "accepted.csv"),
                "--state-dir",
                str(stage2),
                "--max-rounds",
                "1",
                "--only-apis",
                str(stage0 / "api_list.txt"),
                "--force",
            ],
            env=env,
            log=log,
        )

        run(
            [
                py,
                "json2init/json2init.py",
                "--spec-dir",
                str(stage1),
                "--outdir",
                str(stage3),
                "--ok-csv",
                str(stage2 / "ok.csv"),
                "--smoke-test",
                "--overwrite",
                "--limit",
                "0",
                "--smoke-timeout-sec",
                str(args.case_timeout_sec),
                "--only-api-list",
                str(stage0 / "api_list.txt"),
                "--non-strict-smoke",
            ],
            env=env,
            log=log,
        )

        run(
            [
                py,
                "scripts/export_generated_tests.py",
                "--lib",
                lib,
                "--api-list",
                str(stage0 / "api_list.txt"),
                "--init-dir",
                str(stage3),
                "--out",
                str(tests),
                "--seed",
                str(args.seed),
            ],
            env=env,
            log=log,
        )

        run(
            [
                py,
                "scripts/write_run_manifest.py",
                "--lib",
                lib,
                "--run-id",
                out.name,
                "--api-list",
                str(stage0 / "api_list.txt"),
                "--out",
                str(lib_dir / "run_manifest.json"),
                "--seed",
                str(args.seed),
                "--mutation-budget",
                str(args.mutation_budget),
                "--coverage-scope",
                args.coverage_scope,
                "--native-coverage",
                "unavailable_on_this_run",
            ],
            env={**env, "PYTHON_COV_SOURCE": lib},
            log=log,
        )

        stage4_cmd = [
            py,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(stage3),
            "--results-dir",
            str(stage4),
            "--ok-csv",
            str(stage3 / "ok.csv"),
            "--mutation-budget",
            str(args.mutation_budget),
            "--seed",
            str(args.seed),
            "--run-id",
            out.name,
            "--only-api-list",
            str(stage0 / "api_list.txt"),
            "--coverage-scope",
            args.coverage_scope,
            "--worker-timeout-sec",
            str(args.worker_timeout_sec),
            "--case-timeout-sec",
            str(args.case_timeout_sec),
            "--native-coverage-engine",
            "none",
        ]
        if args.coverage_scope in {"python", "both"}:
            stage4_cmd.extend(["--enable-python-coverage", "--python-cov-source", lib])
        if args.enable_device_oracle:
            stage4_cmd.extend(
                [
                    "--enable-device-oracle",
                    "--device-oracle-device",
                    "cpu",
                    "--device-oracle-device",
                    "accelerator",
                    "--edge-oracle-mutations",
                ]
            )
        run(stage4_cmd, env=env, log=log)

    build_demo_report(out, dict(apis_by_lib), args.mutation_budget, args.seed)
    print(f"\nDemo artefacts written to: {out}")
    print(f"Open report: {out / 'demo_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
