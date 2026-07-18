# engine.py
"""
Training/evaluation loops for revision experiments.

Revision fixes:
1. Evaluation no longer crashes for datasets with fewer than 5 classes.
2. Training returns epoch time and peak CUDA memory for efficiency tables.
3. Evaluation can return latency and peak inference memory when requested.
4. AMP autocast is safely disabled on CPU.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils


def _amp_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda" and torch.cuda.is_available())
    if enabled:
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()



def _set_frozen_batchnorm_eval(model: torch.nn.Module) -> None:
    """Keep frozen BatchNorm statistics fixed after model.train(True)."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            affine_trainable = any(
                parameter is not None and parameter.requires_grad
                for parameter in (module.weight, module.bias)
            )
            if not affine_trainable:
                module.eval()

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    wandb_logger=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    use_amp: bool = False,
):
    model.train(True)
    _set_frozen_batchnorm_eval(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    if update_freq is None:
        update_freq = 1
    if start_steps is None:
        start_steps = 0
    if num_training_steps_per_epoch is None:
        num_training_steps_per_epoch = len(data_loader)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    epoch_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        if (data_iter_step % update_freq) == 0:
            if lr_schedule_values is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_schedule_values[it]
            if wd_schedule_values is not None:
                for param_group in optimizer.param_groups:
                    if param_group.get("weight_decay", 0) > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with _amp_context(device, use_amp):
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Loss is not finite: {loss_value}")

        loss = loss / update_freq
        if use_amp and device.type == "cuda":
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)
        else:
            loss.backward()
            grad_norm = None
            if (data_iter_step + 1) % update_freq == 0:
                if max_norm is not None and max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        if class_acc is not None:
            metric_logger.update(class_acc=class_acc.item())

        min_lr = min(group["lr"] for group in optimizer.param_groups)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group.get("weight_decay", 0) > 0:
                weight_decay_value = group["weight_decay"]
        if weight_decay_value is not None:
            metric_logger.update(weight_decay=weight_decay_value)
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if class_acc is not None:
                log_writer.update(class_acc=class_acc.item(), head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            if weight_decay_value is not None:
                log_writer.update(weight_decay=weight_decay_value, head="opt")
            if grad_norm is not None:
                log_writer.update(grad_norm=float(grad_norm), head="opt")
            log_writer.set_step()

        if wandb_logger:
            payload = {
                "Rank-0 Batch Wise/train_loss": loss_value,
                "Rank-0 Batch Wise/train_max_lr": max_lr,
                "Rank-0 Batch Wise/train_min_lr": min_lr,
                "Rank-0 Batch Wise/global_train_step": it,
            }
            if class_acc is not None:
                payload["Rank-0 Batch Wise/train_class_acc"] = class_acc.item()
            if grad_norm is not None:
                payload["Rank-0 Batch Wise/train_grad_norm"] = float(grad_norm)
            wandb_logger._wandb.log(payload)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    epoch_time = time.time() - epoch_start
    metric_logger.update(epoch_time=epoch_time)

    if device.type == "cuda" and torch.cuda.is_available():
        peak_train_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        metric_logger.update(peak_train_memory_mb=peak_train_memory_mb)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp: bool = False, measure_latency: bool = False):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    n_images = 0
    elapsed_forward = 0.0

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.time()
        with _amp_context(device, use_amp):
            output = model(images)
            loss = criterion(output, target)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed_forward += time.time() - start

        num_classes = output.shape[-1]
        batch_size = images.shape[0]
        if num_classes >= 5:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        else:
            acc1 = accuracy(output, target, topk=(1,))[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        n_images += batch_size

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    if measure_latency and n_images > 0:
        metric_logger.update(latency_ms_per_image=1000.0 * elapsed_forward / n_images)
        metric_logger.update(fps=n_images / max(elapsed_forward, 1e-12))

    if device.type == "cuda" and torch.cuda.is_available():
        peak_inference_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        metric_logger.update(peak_inference_memory_mb=peak_inference_memory_mb)

    metric_logger.synchronize_between_processes()
    if "acc5" in metric_logger.meters:
        print(
            "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
            )
        )
    else:
        print(
            "* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1, losses=metric_logger.loss
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
