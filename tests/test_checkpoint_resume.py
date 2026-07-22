from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import utils
from models.task_response_adapter import TaskResponseSpatialAdapter


class ResumeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = TaskResponseSpatialAdapter(channels=4, kernel_size=3, spatial_rank=3)


def _args(tmp_path, resume=""):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        resume=str(resume),
        auto_resume=False,
        eval=False,
        start_epoch=0,
        model_ema=False,
        save_ckpt_num=2,
        save_ckpt_freq=1,
    )


def _active_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def test_trso_resume_restores_rank_before_optimizer_construction(tmp_path):
    source = ResumeModel()
    source.adapter.set_active_rank(1)
    source_optimizer = torch.optim.AdamW(_active_parameters(source), lr=1e-3)
    checkpoint = {
        "model": source.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "epoch": 2,
        "scaler": None,
        "training_state": {"best_val_metric": 77.0, "best_epoch": 1, "history": [{"epoch": 0}]},
    }
    path = tmp_path / "checkpoint-2.pth"
    torch.save(checkpoint, path)

    resumed = ResumeModel()
    assert resumed.adapter.active_rank == 3
    args = _args(tmp_path, path)
    loaded = utils.load_model_for_resume(args, resumed, strict=True)

    assert resumed.adapter.active_rank == 1
    assert resumed.adapter.enabled
    assert [p.requires_grad for p in resumed.adapter.coefficient_vectors] == [True, False, False]
    resumed_optimizer = torch.optim.AdamW(_active_parameters(resumed), lr=1e-3)
    state = utils.restore_optimizer_state(args, loaded, resumed_optimizer, loss_scaler=None)
    assert args.start_epoch == 3
    assert state["best_val_metric"] == 77.0
    assert state["best_epoch"] == 1
    assert state["history"] == [{"epoch": 0}]


def test_trso_plain_load_state_dict_runs_post_load_trainability_hook():
    source = ResumeModel()
    source.adapter.set_active_rank(2)
    target = ResumeModel()
    target.load_state_dict(source.state_dict(), strict=True)
    assert [p.requires_grad for p in target.adapter.coefficient_vectors] == [True, True, False]


def test_strict_resume_rejects_architecture_mismatch(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"model": {"unknown.weight": torch.randn(2, 2)}}, path)
    with pytest.raises(RuntimeError):
        utils.load_model_for_resume(_args(tmp_path, path), ResumeModel(), strict=True)
