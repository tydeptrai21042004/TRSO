"""Paper-reproduction tuning modules."""
from .prompter import PadPrompter, VisualPromptingClassifier
from .conv_adapter import ConvAdapter, ConvAdapterBottleneck, apply_conv_adapter_resnet50
from .program_module import ProgramModule
from .ssf import SSF, SSFPost, SSFMultiheadAttention, apply_ssf
from .bam_adapter import BAM, BAMAdapter, BAMResNet50
from .lora_transformer import (
    LoRALinear,
    LoRAQKVLinear,
    LoRAMultiheadAttention,
    apply_lora_transformer,
)
from .residual_adapter import ResidualAdapterResNet26
from .side_tuning import SideTuningClassifier


def set_tuning_config(tuning_method, args):
    method = str(tuning_method).strip().lower().replace("-", "_")
    aliases = {
        "conv_adapter": "conv",
        "adapter": "conv",
        "task_response": "trso",
        "trso_adapter": "trso",
        "bam_adapter": "bam",
        "residual_adapter": "residual",
        "side_tuning": "sidetune",
    }
    method = aliases.get(method, method)
    if method == "prompt":
        return {"method": method, "prompt_size": getattr(args, "prompt_size", 30)}
    if method == "conv":
        return {
            "method": method,
            "kernel_size": getattr(args, "kernel_size", 3),
            "mode": getattr(args, "conv_adapter_mode", "conv_parallel"),
        }
    if method == "trso":
        return {
            "method": method,
            "kernel_size": getattr(args, "trso_kernel_size", 5),
            "spatial_rank": getattr(args, "trso_spatial_rank", 2),
        }
    if method == "bam":
        return {"method": method, "reduction": getattr(args, "bam_reduction", 16), "dilation": getattr(args, "bam_dilation", 4)}
    if method == "residual":
        return {"method": method, "mode": getattr(args, "ra_mode", "parallel")}
    if method == "ssf":
        return {"method": method, "init_std": getattr(args, "ssf_init_std", 0.02)}
    if method == "lora":
        return {"method": method, "rank": getattr(args, "lora_r", 8), "alpha": getattr(args, "lora_alpha", 16.0)}
    if method == "sidetune":
        return {"method": method, "alpha": getattr(args, "sidetune_alpha", 0.5)}
    if method in {"full", "linear", "bitfit"}:
        return {"method": method}
    if method in {"lora_conv", "lora_conv2d"}:
        raise NotImplementedError(
            "LoRA-Conv is an experimental repository control, not an original-paper baseline, "
            "and is excluded from strict reproduction runs."
        )
    raise NotImplementedError(f"Unknown tuning_method: {tuning_method}")


__all__ = [name for name in globals() if not name.startswith("_")]
