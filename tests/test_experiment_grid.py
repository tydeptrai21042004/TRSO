from __future__ import annotations

import json
from pathlib import Path

from tools.experiment_grid import build_specs, expand_grid, one_factor_grid, write_manifest
from tools.run_ablation_suite import ablation_variants
from tools.run_hparam_sweep import DEFAULT_BACKBONES, METHOD_SPACES, build_hparam_rows


def test_grid_and_one_factor_generation_are_deterministic():
    assert expand_grid({"fixed": 1}, {"b": [2, 3], "a": [4]}) == [
        {"fixed": 1, "a": 4, "b": 2},
        {"fixed": 1, "a": 4, "b": 3},
    ]
    rows = one_factor_grid({"x": 1, "y": 2}, {"x": [1, 3], "y": [2, 4]})
    assert rows == [{"x": 1, "y": 2}, {"x": 3, "y": 2}, {"x": 1, "y": 4}]


def test_ablation_suite_contains_scientific_controls():
    variants = dict(ablation_variants(1000))
    required = {
        "proposed",
        "basis_random",
        "basis_dct",
        "allocation_greedy",
        "allocation_uniform",
        "score_noise_adjusted",
        "calibration_1",
        "budget_half",
    }
    assert required.issubset(variants)
    assert variants["proposed"]["trso_basis_source"] == "response"
    assert variants["proposed"]["trso_allocation"] == "exact"


def test_all_baseline_hparam_spaces_have_defaults_and_candidates():
    assert {"full", "linear", "trso", "conv", "prompt", "ssf", "lora", "bam", "residual", "bitfit", "sidetune"}.issubset(METHOD_SPACES)
    for defaults, candidates in METHOD_SPACES.values():
        assert defaults and candidates
        assert set(candidates).issubset(defaults)


def test_manifest_has_unique_output_paths_and_commands(tmp_path):
    specs = build_specs(
        suite="unit",
        variants=[("a", {"seed": 0, "lr": 1e-3}), ("a", {"seed": 1, "lr": 1e-3})],
        common={"dataset": "fake", "tuning_method": "linear"},
        output_root=tmp_path,
    )
    assert len({spec.output_dir for spec in specs}) == 2
    assert all("--output_dir" in spec.command for spec in specs)
    assert all("--experiment_suite" in spec.command for spec in specs)
    assert all("--experiment_name" in spec.command for spec in specs)
    assert all(spec.parameters["experiment_run_id"] == spec.run_id for spec in specs)
    json_path, csv_path = write_manifest(specs, tmp_path / "manifest.json")
    assert json_path.exists() and csv_path.exists()
    payload = json.loads(json_path.read_text())
    assert len(payload) == 2 and all("command_text" in row for row in payload)


def test_hparam_defaults_respect_supported_architecture_domains():
    assert DEFAULT_BACKBONES["conv"] == "resnet50"
    assert DEFAULT_BACKBONES["bam"] == "resnet50"
    assert DEFAULT_BACKBONES["residual"] == "resnet26_adapter"
    assert DEFAULT_BACKBONES["lora"].startswith("vit_")
    assert DEFAULT_BACKBONES["ssf"].startswith("vit_")
    assert all(value == 0.0 for value in METHOD_SPACES["lora"][1].get("lora_dropout", [0.0]))


def test_full_grid_retains_reference_constants_not_in_sweep_axes():
    conv_rows = build_hparam_rows("conv", "grid")
    assert conv_rows and all(row["weight_decay"] == 0.0 for row in conv_rows)
    assert all(row["conv_adapter_mode"] == "conv_parallel" for row in conv_rows)
    prompt_rows = build_hparam_rows("prompt", "grid")
    assert all(row["optimizer"] == "sgd" and row["weight_decay"] == 0.0 for row in prompt_rows)
    full_rows = build_hparam_rows("full", "grid")
    assert all(row["optimizer"] == "sgd" for row in full_rows)
