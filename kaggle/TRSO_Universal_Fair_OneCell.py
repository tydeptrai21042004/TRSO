# One-cell Kaggle runner for the universal TRSO fair framework.
# Kaggle settings: Internet ON; GPU T4 x2 recommended.

import json
import os
import queue
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ----------------------------- editable configuration -----------------------------
REPO_URL = "https://github.com/tydeptrai21042004/TRSO.git"
WORK = Path("/kaggle/working")
REPO = WORK / "TRSO"
DATA_PATH = WORK / "data"
OUTPUT_ROOT = WORK / "trso_universal_results"

# Any canonical key printed by: python main.py --list_backbones
DATASET = "dtd"
TASK = "auto"  # auto | single_label | multilabel | regression
DOWNLOAD = True
DATASET_ARGS = {"dtd_partition": 1}

# Each item is backbone@source. Add/remove groups as needed.
BACKBONES = "resnet50@torchvision,vit_tiny_patch16_224@timm"
METHODS = "auto"  # all unique registered baselines; incompatible rows are explicit skips
SEEDS = "0,1,2"
EPOCHS = 20
BATCH_SIZE = 64
INPUT_SIZE = 0  # native pretrained resolution, resolved before transforms
PEFT_LR = 5e-3
FULL_LR = 1e-4
LINEAR_LR = 1e-1
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
TRSO_BUDGET = 12000
RUN_ABLATION = True
ABLATION_BACKBONE = "resnet50"
ABLATION_SOURCE = "torchvision"
RUN_TESTS = True
KEEP_CHECKPOINTS = False
# ----------------------------------------------------------------------------------


def run(command, *, cwd=None, env=None, check=True):
    print("+", " ".join(map(str, command)))
    result = subprocess.run(
        list(map(str, command)), cwd=str(cwd) if cwd else None,
        env=env, text=True, check=False,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


# Clone/update.
if REPO.exists() and (REPO / ".git").exists():
    run(["git", "-C", REPO, "fetch", "--depth", "1", "origin"])
    run(["git", "-C", REPO, "reset", "--hard", "origin/HEAD"])
else:
    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--depth", "1", REPO_URL, REPO])

# Keep Kaggle's CUDA-enabled torch; install remaining dependencies.
run([
    sys.executable, "-m", "pip", "install", "-q",
    "timm>=0.9,<2", "pandas>=2", "scipy>=1.10", "scikit-learn>=1.3",
    "matplotlib>=3.7", "tqdm>=4.65", "pytest>=8",
])

required = [
    REPO / "tools" / "run_fair_suite.py",
    REPO / "tools" / "verify_fairness.py",
    REPO / "tests" / "test_universal_fair_framework.py",
    REPO / "models" / "tuning_modules" / "adaptformer.py",
    REPO / "models" / "tuning_modules" / "piggyback.py",
]
missing = [str(path.relative_to(REPO)) for path in required if not path.exists()]
if missing:
    raise RuntimeError(
        "GitHub does not contain the universal fair release. Push the supplied corrected ZIP "
        f"to {REPO_URL} first. Missing: {missing}"
    )

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator.")
gpu_count = torch.cuda.device_count()
gpu_ids = ",".join(str(index) for index in range(gpu_count))
parallel_runs = max(1, gpu_count)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(gpu_count)])

DATA_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
manifest = OUTPUT_ROOT / "fair_manifest.json"

if RUN_TESTS:
    run([sys.executable, "-m", "pytest", "-q"], cwd=REPO)

fair_command = [
    sys.executable, "-m", "tools.run_fair_suite",
    "--dataset", DATASET,
    "--task", TASK,
    "--data_path", DATA_PATH,
    "--download", str(DOWNLOAD),
    "--dataset_args_json", json.dumps(DATASET_ARGS),
    "--backbones", BACKBONES,
    "--methods", METHODS,
    "--seeds", SEEDS,
    "--epochs", EPOCHS,
    "--batch_size", BATCH_SIZE,
    "--input_size", INPUT_SIZE,
    "--peft_lr", PEFT_LR,
    "--full_lr", FULL_LR,
    "--linear_lr", LINEAR_LR,
    "--weight_decay", WEIGHT_DECAY,
    "--warmup_epochs", WARMUP_EPOCHS,
    "--trso_budget", TRSO_BUDGET,
    "--output_root", OUTPUT_ROOT / "comparison",
    "--manifest", manifest,
    "--gpu_ids", gpu_ids,
    "--parallel_runs", parallel_runs,
    "--profile_efficiency", "True",
    "--measure_eval_latency", "True",
    "--execute",
]
run(fair_command, cwd=REPO)

compatibility = manifest.with_name(manifest.stem + "_compatibility.json")
run([
    sys.executable, "-m", "tools.verify_fairness",
    "--manifest", manifest,
    "--compatibility", compatibility,
    "--output", OUTPUT_ROOT / "fairness_verification.json",
], cwd=REPO)

if RUN_ABLATION:
    ablation_manifest = OUTPUT_ROOT / "ablation_manifest.json"
    run([
        sys.executable, "-m", "tools.run_ablation_suite",
        "--dataset", DATASET,
        "--task", TASK,
        "--dataset_args_json", json.dumps(DATASET_ARGS),
        "--data_path", DATA_PATH,
        "--download", str(DOWNLOAD),
        "--backbone", ABLATION_BACKBONE,
        "--model_source", ABLATION_SOURCE,
        "--input_size", INPUT_SIZE,
        "--seeds", SEEDS,
        "--epochs", EPOCHS,
        "--batch_size", BATCH_SIZE,
        "--parameter_budget", TRSO_BUDGET,
        "--peft_lr", PEFT_LR,
        "--weight_decay", WEIGHT_DECAY,
        "--warmup_epochs", WARMUP_EPOCHS,
        "--output_root", OUTPUT_ROOT / "ablation",
        "--manifest", ablation_manifest,
        "--gpu_ids", gpu_ids,
        "--parallel_runs", parallel_runs,
        "--execute",
    ], cwd=REPO)

# Aggregate all completed runs and retain every task-appropriate metric.
run([
    sys.executable, "-m", "tools.aggregate_revision_results",
    "--root", OUTPUT_ROOT,
    "--out_csv", OUTPUT_ROOT / "all_results.csv",
], cwd=REPO)

# Keep summary files/logs by default; optionally remove large checkpoints.
if not KEEP_CHECKPOINTS:
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        for path in OUTPUT_ROOT.rglob(pattern):
            path.unlink(missing_ok=True)

zip_path = WORK / "trso_universal_results_summary.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT_ROOT.rglob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT_ROOT.parent))

print("\nCompleted universal fair experiments.")
print("Results:", OUTPUT_ROOT)
print("Summary ZIP:", zip_path)
