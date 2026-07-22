from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from engine import TASK_REGRESSION, train_one_epoch


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _run(num_batches: int, update_freq: int):
    x = torch.ones(num_batches, 1)
    y = torch.ones(num_batches, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    steps = math.ceil(len(loader) / update_freq)
    train_one_epoch(
        model=model,
        criterion=torch.nn.MSELoss(),
        data_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        loss_scaler=None,
        num_training_steps_per_epoch=steps,
        update_freq=update_freq,
        use_amp=False,
        task_type=TASK_REGRESSION,
        lr_schedule_values=[0.1] * steps,
        wd_schedule_values=[0.0] * steps,
    )
    return model, optimizer


def test_incomplete_single_microbatch_window_still_updates():
    model, optimizer = _run(num_batches=1, update_freq=2)
    assert optimizer.step_calls == 1
    assert model.weight.item() != 0.0


def test_final_partial_accumulation_window_is_not_dropped():
    model, optimizer = _run(num_batches=3, update_freq=2)
    assert optimizer.step_calls == 2
    assert model.weight.item() > 0.0


def test_exact_multiple_keeps_expected_optimizer_step_count():
    _, optimizer = _run(num_batches=4, update_freq=2)
    assert optimizer.step_calls == 2


class FrozenBatchNormClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.head = torch.nn.Linear(2, 1)
        for parameter in self.bn.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        return self.head(self.bn(x))


def test_frozen_batchnorm_statistics_remain_in_eval_mode():
    x = torch.tensor([[10.0, -10.0], [12.0, -12.0], [14.0, -14.0], [16.0, -16.0]])
    y = torch.ones(4, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    model = FrozenBatchNormClassifier()
    initial_mean = model.bn.running_mean.detach().clone()
    optimizer = CountingSGD(model.head.parameters(), lr=0.01)
    train_one_epoch(
        model=model,
        criterion=torch.nn.MSELoss(),
        data_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        loss_scaler=None,
        num_training_steps_per_epoch=2,
        update_freq=1,
        use_amp=False,
        task_type=TASK_REGRESSION,
        lr_schedule_values=[0.01, 0.01],
        wd_schedule_values=[0.0, 0.0],
    )
    assert not model.bn.training
    assert torch.equal(model.bn.running_mean, initial_mean)
