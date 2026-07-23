"""Deterministic experiment-grid construction and execution utilities."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue


@dataclass(frozen=True)
class RunSpec:
    suite: str
    name: str
    parameters: dict[str, Any]
    output_dir: str
    command: tuple[str, ...]

    @property
    def run_id(self) -> str:
        explicit = self.parameters.get("experiment_run_id")
        if explicit:
            return str(explicit)
        payload_parameters = {k: v for k, v in self.parameters.items() if k != "output_dir"}
        payload = json.dumps(payload_parameters, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:10]


def parse_csv_values(text: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def expand_grid(fixed: Mapping[str, Any], grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    values = [list(grid[key]) for key in keys]
    if any(len(items) == 0 for items in values):
        raise ValueError("Every grid dimension must contain at least one value")
    rows = []
    for combination in itertools.product(*values):
        row = dict(fixed)
        row.update(dict(zip(keys, combination)))
        rows.append(row)
    return rows


def one_factor_grid(defaults: Mapping[str, Any], candidates: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Return the default plus one-parameter-at-a-time variants."""
    rows = [dict(defaults)]
    seen = {json.dumps(rows[0], sort_keys=True, default=str)}
    for key in sorted(candidates):
        for value in candidates[key]:
            row = dict(defaults)
            row[key] = value
            token = json.dumps(row, sort_keys=True, default=str)
            if token not in seen:
                seen.add(token)
                rows.append(row)
    return rows


def _argument_tokens(parameters: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in sorted(parameters):
        value = parameters[key]
        if value is None:
            continue
        flag = "--" + key
        if isinstance(value, bool):
            value = "True" if value else "False"
        elif isinstance(value, (list, tuple)):
            value = ",".join(map(str, value))
        tokens.extend((flag, str(value)))
    return tokens


def build_specs(
    *,
    suite: str,
    variants: Iterable[tuple[str, Mapping[str, Any]]],
    common: Mapping[str, Any],
    output_root: str | Path,
    entrypoint: str = "main.py",
    python_executable: str = sys.executable,
) -> list[RunSpec]:
    output_root = Path(output_root)
    specs: list[RunSpec] = []
    for name, variant in variants:
        parameters = dict(common)
        parameters.update(variant)
        parameters.setdefault("experiment_suite", suite)
        parameters.setdefault("experiment_name", name)
        digest = hashlib.sha1(
            json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:10]
        parameters["experiment_run_id"] = digest
        seed = parameters.get("seed", "na")
        output_dir = output_root / suite / name / f"seed_{seed}_{digest}"
        parameters["output_dir"] = str(output_dir)
        command = (python_executable, entrypoint, *_argument_tokens(parameters))
        specs.append(RunSpec(suite, name, parameters, str(output_dir), tuple(command)))
    return specs


def write_manifest(specs: Sequence[RunSpec], path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            **asdict(spec),
            "run_id": spec.run_id,
            "command_text": shlex.join(spec.command),
        }
        for spec in specs
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("suite", "name", "run_id", "output_dir", "command_text", "parameters_json"),
        )
        writer.writeheader()
        for row in payload:
            writer.writerow(
                {
                    "suite": row["suite"],
                    "name": row["name"],
                    "run_id": row["run_id"],
                    "output_dir": row["output_dir"],
                    "command_text": row["command_text"],
                    "parameters_json": json.dumps(row["parameters"], sort_keys=True),
                }
            )
    return path, csv_path


def is_complete(spec: RunSpec) -> bool:
    output = Path(spec.output_dir)
    return any((output / filename).exists() for filename in ("test_summary.json", "eval_summary.json"))


def execute_specs(
    specs: Sequence[RunSpec],
    *,
    execute: bool = False,
    max_runs: int = 0,
    skip_completed: bool = True,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    selected = list(specs[: max_runs or None])
    status_rows = []
    for spec in selected:
        if skip_completed and is_complete(spec):
            status_rows.append({"name": spec.name, "status": "skipped_complete", "output_dir": spec.output_dir})
            continue
        print(shlex.join(spec.command))
        if not execute:
            status_rows.append({"name": spec.name, "status": "planned", "output_dir": spec.output_dir})
            continue
        Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            spec.command,
            check=False,
            env={**os.environ, **dict(env or {})},
        )
        status = "completed" if completed.returncode == 0 else f"failed_{completed.returncode}"
        status_rows.append({"name": spec.name, "status": status, "output_dir": spec.output_dir})
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, spec.command)
    return status_rows


def execute_specs_parallel(
    specs: Sequence[RunSpec],
    *,
    execute: bool = False,
    gpu_ids: Sequence[int] = (0,),
    max_runs: int = 0,
    skip_completed: bool = True,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Execute independent runs concurrently, one process per visible GPU."""
    selected = list(specs[: max_runs or None])
    if not execute:
        return execute_specs(
            selected, execute=False, max_runs=0, skip_completed=skip_completed, env=env
        )
    devices = [int(value) for value in gpu_ids]
    if not devices:
        raise ValueError("gpu_ids must contain at least one GPU")
    pool: Queue[int] = Queue()
    for device in devices:
        pool.put(device)

    def run_one(spec: RunSpec) -> dict[str, Any]:
        if skip_completed and is_complete(spec):
            return {"name": spec.name, "status": "skipped_complete", "output_dir": spec.output_dir}
        gpu = pool.get()
        try:
            output = Path(spec.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            log_path = output / "run.log"
            child_env = {**os.environ, **dict(env or {}), "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
            print(f"[GPU {gpu}] {shlex.join(spec.command)}")
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    spec.command,
                    check=False,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            status = "completed" if completed.returncode == 0 else f"failed_{completed.returncode}"
            return {
                "name": spec.name,
                "status": status,
                "output_dir": spec.output_dir,
                "gpu": gpu,
                "log": str(log_path),
                "returncode": completed.returncode,
            }
        finally:
            pool.put(gpu)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(run_one, spec): spec for spec in selected}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if str(row.get("status", "")).startswith("failed_"):
                failures.append(row)
    if failures:
        details = "; ".join(f"{row['name']} -> {row.get('log')}" for row in failures)
        raise RuntimeError(f"{len(failures)} experiment runs failed: {details}")
    return rows


__all__ = [
    "RunSpec",
    "parse_csv_values",
    "expand_grid",
    "one_factor_grid",
    "build_specs",
    "write_manifest",
    "is_complete",
    "execute_specs",
    "execute_specs_parallel",
]
