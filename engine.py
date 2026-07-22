"""Shared training/evaluation loops for classification, multi-label, regression."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Iterable, Optional

import torch
try:
    from timm.data import Mixup
    from timm.utils import ModelEma, accuracy
except Exception:
    from compat.timm_compat import Mixup, ModelEma, accuracy

import utils


TASK_SINGLE_LABEL = "single_label"
TASK_MULTILABEL = "multilabel"
TASK_REGRESSION = "regression"


def _amp_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda" and torch.cuda.is_available())
    return torch.amp.autocast(device_type="cuda") if enabled else nullcontext()


def _set_frozen_batchnorm_eval(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            affine_trainable = any(
                parameter is not None and parameter.requires_grad
                for parameter in (module.weight, module.bias)
            )
            if not affine_trainable:
                module.eval()


def _extract_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "out", "pred"):
            if isinstance(output.get(key), torch.Tensor):
                return output[key]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _prepare_target(target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == TASK_SINGLE_LABEL:
        return target.long()
    return target.float()


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
    task_type: str = TASK_SINGLE_LABEL,
):
    model.train(True)
    _set_frozen_batchnorm_eval(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header, print_freq = f"Epoch: [{epoch}]", 10

    update_freq = int(update_freq or 1)
    start_steps = int(start_steps or 0)
    total_microbatches = len(data_loader)
    num_training_steps_per_epoch = int(
        num_training_steps_per_epoch
        or math.ceil(total_microbatches / update_freq)
    )

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    epoch_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            break
        iteration = start_steps + step
        window_start = step * update_freq
        window_size = min(update_freq, total_microbatches - window_start)
        is_update_step = (data_iter_step + 1 == total_microbatches) or (
            (data_iter_step + 1) % update_freq == 0
        )

        if data_iter_step % update_freq == 0:
            if lr_schedule_values is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr_schedule_values[iteration]
            if wd_schedule_values is not None:
                for group in optimizer.param_groups:
                    if group.get("weight_decay", 0) > 0:
                        group["weight_decay"] = wd_schedule_values[iteration]

        samples = samples.to(device, non_blocking=True)
        targets = _prepare_target(targets.to(device, non_blocking=True), task_type)
        if mixup_fn is not None:
            if task_type != TASK_SINGLE_LABEL:
                raise ValueError("Mixup/CutMix is supported only for single-label classification.")
            samples, targets = mixup_fn(samples, targets)

        with _amp_context(device, use_amp):
            output = _extract_output(model(samples))
            loss = criterion(output, targets)

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Loss is not finite: {loss_value}")

        # Normalize by the actual window size so a final incomplete
        # accumulation window has the same mean-gradient semantics.
        scaled_loss = loss / max(1, window_size)
        if use_amp and device.type == "cuda":
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            grad_norm = loss_scaler(
                scaled_loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=is_update_step,
            )
            if is_update_step:
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)
        else:
            scaled_loss.backward()
            grad_norm = None
            if is_update_step:
                if max_norm is not None and max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        metric_logger.update(loss=loss_value)
        if task_type == TASK_SINGLE_LABEL and mixup_fn is None:
            batch_acc = (output.argmax(dim=-1) == targets).float().mean()
            metric_logger.update(class_acc=float(batch_acc.item()))
        elif task_type == TASK_REGRESSION:
            metric_logger.update(batch_mae=float((output - targets).abs().mean().item()))

        min_lr = min(group["lr"] for group in optimizer.param_groups)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr, min_lr=min_lr)
        positive_wd = [group["weight_decay"] for group in optimizer.param_groups if group.get("weight_decay", 0) > 0]
        if positive_wd:
            metric_logger.update(weight_decay=positive_wd[-1])
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, min_lr=min_lr, head="opt")
            log_writer.set_step()

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    metric_logger.update(epoch_time=time.time() - epoch_start)
    if device.type == "cuda" and torch.cuda.is_available():
        metric_logger.update(peak_train_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def _distributed_concat(tensor: torch.Tensor) -> torch.Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
    gathered = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, tensor.cpu())
    return torch.cat(gathered, dim=0)


def _binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    positives = float(targets.sum().item())
    if positives <= 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].float()
    precision = sorted_targets.cumsum(0) / torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
    return float((precision * sorted_targets).sum().item() / positives)


def _multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor):
    probabilities = logits.sigmoid()
    predictions = probabilities >= 0.5
    truth = targets >= 0.5
    tp = (predictions & truth).sum().float()
    fp = (predictions & ~truth).sum().float()
    fn = (~predictions & truth).sum().float()
    micro_f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1)).item() * 100.0
    aps = [_binary_average_precision(probabilities[:, c], truth[:, c]) for c in range(probabilities.shape[1])]
    valid = [value for value in aps if not math.isnan(value)]
    return {"map": 100.0 * sum(valid) / max(1, len(valid)), "micro_f1": micro_f1}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    use_amp: bool = False,
    measure_latency: bool = False,
    task_type: str = TASK_SINGLE_LABEL,
    criterion: Optional[torch.nn.Module] = None,
):
    if criterion is None:
        criterion = {
            TASK_SINGLE_LABEL: torch.nn.CrossEntropyLoss(),
            TASK_MULTILABEL: torch.nn.BCEWithLogitsLoss(),
            TASK_REGRESSION: torch.nn.MSELoss(),
        }[task_type]
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    outputs, targets_all = [], []
    n_images, elapsed_forward = 0, 0.0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = _prepare_target(batch[1].to(device, non_blocking=True), task_type)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.time()
        with _amp_context(device, use_amp):
            output = _extract_output(model(images))
            loss = criterion(output, target)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed_forward += time.time() - start
        batch_size = images.shape[0]
        metric_logger.meters["loss"].update(float(loss.item()), n=batch_size)

        if task_type == TASK_SINGLE_LABEL:
            topk = (1, 5) if output.shape[-1] >= 5 else (1,)
            values = accuracy(output, target, topk=topk)
            metric_logger.meters["acc1"].update(float(values[0].item()), n=batch_size)
            if len(values) > 1:
                metric_logger.meters["acc5"].update(float(values[1].item()), n=batch_size)
        else:
            outputs.append(output.detach().cpu())
            targets_all.append(target.detach().cpu())
        n_images += batch_size

    if task_type != TASK_SINGLE_LABEL and outputs:
        prediction_tensor = _distributed_concat(torch.cat(outputs, dim=0))
        target_tensor = _distributed_concat(torch.cat(targets_all, dim=0))
        if task_type == TASK_MULTILABEL:
            for key, value in _multilabel_metrics(prediction_tensor, target_tensor).items():
                metric_logger.update(**{key: value})
        else:
            error = prediction_tensor - target_tensor
            metric_logger.update(mae=float(error.abs().mean().item()))
            metric_logger.update(rmse=float(error.square().mean().sqrt().item()))

    if measure_latency and n_images > 0:
        metric_logger.update(latency_ms_per_image=1000.0 * elapsed_forward / n_images)
        metric_logger.update(fps=n_images / max(elapsed_forward, 1e-12))
    if device.type == "cuda" and torch.cuda.is_available():
        metric_logger.update(peak_inference_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2)

    metric_logger.synchronize_between_processes()
    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    if task_type == TASK_SINGLE_LABEL:
        print(f"* Acc@1 {stats.get('acc1', 0):.3f} loss {stats.get('loss', 0):.3f}")
    elif task_type == TASK_MULTILABEL:
        print(f"* mAP {stats.get('map', 0):.3f} micro-F1 {stats.get('micro_f1', 0):.3f} loss {stats.get('loss', 0):.3f}")
    else:
        print(f"* MAE {stats.get('mae', 0):.5f} RMSE {stats.get('rmse', 0):.5f} loss {stats.get('loss', 0):.5f}")
    return stats
