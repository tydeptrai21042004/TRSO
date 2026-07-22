from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_one_epoch_cli_training_smoke(tmp_path: Path):
    command = [
        sys.executable,
        "main.py",
        "--tuning_method", "linear",
        "--backbone", "resnet18",
        "--weights", "none",
        "--dataset", "fake",
        "--fake_train_size", "4",
        "--fake_val_size", "2",
        "--fake_test_size", "2",
        "--device", "cpu",
        "--epochs", "1",
        "--batch_size", "2",
        "--num_workers", "0",
        "--input_size", "32",
        "--nb_classes", "2",
        "--use_amp", "False",
        "--profile_efficiency", "False",
        "--save_ckpt", "False",
        "--final_test", "False",
        "--output_dir", str(tmp_path / "run"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "Number of training steps per epoch = 2" in completed.stdout
    assert "Training time" in completed.stdout
