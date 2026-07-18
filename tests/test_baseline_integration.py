# tests/test_baseline_integration.py
"""Integration tests for all CNN PEFT baselines exposed by main.py.

The goal is not to reproduce paper accuracies. These tests verify that each
baseline follows the intended fair protocol:

* Full fine-tuning: backbone and head train.
* Linear probe: backbone frozen, head trains.
* PEFT baselines: backbone frozen, baseline module + head train.
* Forward pass is runnable without dataset download or pretrained weights.
"""
from __future__ import annotations

import unittest

import torch

from main import canonicalize_args, get_args_parser, build_model_for_experiment, set_trainability_policy
from models.tuning_modules.bam_adapter import BAMAdapter
from models.tuning_modules.lora_conv import LoRAConv2d
from models.tuning_modules.residual_adapter import ParallelResidualAdapter, SeriesResidualAdapter
from models.tuning_modules.side_tuning import SideTuningClassifier
from models.tuning_modules.ssf import SSF
from models.task_response_adapter import TaskResponseSpatialAdapter


METHODS = [
    "full",
    "linear",
    "bitfit",
    "conv",
    "trso",
    "bam",
    "residual",
    "ssf",
    "lora_conv",
    "sidetune",
]


class BaselineIntegrationFaithfulnessTest(unittest.TestCase):
    @staticmethod
    def make_args(method: str):
        parser = get_args_parser()
        args = parser.parse_args([
            "--backbone", "resnet18",
            "--weights", "none",
            "--pretrained", "False",
            "--tuning_method", method,
            "--nb_classes", "5",
            "--input_size", "64",
            "--device", "cpu",
            "--use_amp", "False",
            "--profile_efficiency", "False",
            "--save_ckpt", "False",
            "--final_test", "False",
            "--batch_size", "2",
            "--num_workers", "0",
            "--bam_insert", "stage",
            "--bam_stages", "1,2,3,4",
            "--trso_calibration", "False",
            "--trso_channel_ratio", "16",
        ])
        return canonicalize_args(args)

    def build(self, method: str):
        args = self.make_args(method)
        model, adapter_ids = build_model_for_experiment(args)
        model = set_trainability_policy(model, args, extra_adapter_param_ids=adapter_ids)
        model.eval()
        return args, model

    def assert_forward_runs(self, model):
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 5))

    def trainable_names(self, model):
        return [n for n, p in model.named_parameters() if p.requires_grad]

    def test_all_baselines_forward_and_trainability(self):
        for method in METHODS:
            with self.subTest(method=method):
                args, model = self.build(method)
                self.assert_forward_runs(model)
                trainable = self.trainable_names(model)
                self.assertGreater(len(trainable), 0, f"{method} should have trainable parameters.")

                if method == "full":
                    self.assertTrue(any(n.startswith("conv1.") for n in trainable), "Full FT must train backbone conv1.")
                    self.assertTrue(any(n.startswith("fc.") for n in trainable), "Full FT must train classifier head.")

                elif method == "linear":
                    self.assertTrue(all(n.startswith("fc.") for n in trainable), f"Linear probe should train only fc, got {trainable[:10]}")

                elif method == "bitfit":
                    self.assertTrue(any(n.startswith("fc.") for n in trainable), "BitFit should train classifier head.")
                    self.assertFalse(any(n == "conv1.weight" for n in trainable), "BitFit must not train backbone conv weights.")
                    self.assertTrue(all(n.endswith(".bias") or n.startswith("fc.") for n in trainable), "BitFit should train only biases + head.")

                elif method == "sidetune":
                    self.assertIsInstance(model, SideTuningClassifier)
                    self.assertTrue(any(n.startswith("side_net.") for n in trainable))
                    self.assertTrue(any(n.startswith("head.") for n in trainable))
                    self.assertTrue(any(n == "alpha_logit" for n in trainable))
                    self.assertFalse(any(n.startswith("base.") for n in trainable), "Side-tuning base must stay frozen.")

                else:
                    self.assertTrue(any(n.startswith("fc.") for n in trainable), f"{method} should train classifier head.")
                    self.assertFalse(any(n == "conv1.weight" for n in trainable), f"{method} must freeze backbone conv1.")
                    if method == "trso":
                        self.assertTrue(any(isinstance(m, TaskResponseSpatialAdapter) for m in model.modules()))
                        probes = [p for n, p in model.named_parameters() if n.endswith("probe_kernel")]
                        self.assertTrue(probes)
                        self.assertTrue(all(not p.requires_grad for p in probes))
                    if method == "bam":
                        self.assertTrue(any(isinstance(m, BAMAdapter) for m in model.modules()))
                    if method == "ssf":
                        self.assertTrue(any(isinstance(m, SSF) for m in model.modules()))
                    if method == "lora_conv":
                        self.assertTrue(any(isinstance(m, LoRAConv2d) for m in model.modules()))
                        self.assertTrue(any("lora_down" in n or "lora_up" in n for n in trainable))
                    if method == "residual":
                        self.assertTrue(any(isinstance(m, (ParallelResidualAdapter, SeriesResidualAdapter)) for m in model.modules()))
                        self.assertTrue(any("core." in n or n.endswith("gate") for n in trainable))
                        self.assertFalse(any(".block." in n for n in trainable), "Residual Adapter wrapped backbone block must remain frozen.")

    def test_bam_stage_insertion_count(self):
        _, model = self.build("bam")
        n_bam = sum(1 for m in model.modules() if isinstance(m, BAMAdapter))
        self.assertEqual(n_bam, 4, "BAM-stage should insert one module after each ResNet stage.")

    def test_lora_no_recursive_wrapping_in_integration(self):
        _, model = self.build("lora_conv")
        wrappers = [m for m in model.modules() if isinstance(m, LoRAConv2d)]
        self.assertGreater(len(wrappers), 0)
        for w in wrappers:
            self.assertNotIsInstance(w.base, LoRAConv2d)


if __name__ == "__main__":
    unittest.main()
