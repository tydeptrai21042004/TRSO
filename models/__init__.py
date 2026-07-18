# models/__init__.py

import torch
import torch.nn as nn

from .backbones import *           # resnet50, ...
from .heads import *               # LinearHead
from .tuning_modules import set_tuning_config
from .layers.ws_conv import WSConv2d

from .tuning_modules.side_tuning import SideTuningClassifier

__all__ = ['build_model']


def replace_conv2d_with_my_conv2d(net, ws_eps=None):
    if ws_eps is None:
        return
    for m in net.modules():
        to_update = {}
        for name, sub in m.named_children():
            if isinstance(sub, nn.Conv2d) and sub.bias is None:
                to_update[name] = sub
        for name, sub in to_update.items():
            m._modules[name] = WSConv2d(
                sub.in_channels, sub.out_channels, sub.kernel_size, sub.stride,
                sub.padding, sub.dilation, sub.groups, sub.bias is not None,
            )
            m._modules[name].load_state_dict(sub.state_dict())
            m._modules[name].weight.requires_grad = sub.weight.requires_grad
            if sub.bias is not None:
                m._modules[name].bias.requires_grad = sub.bias.requires_grad
    for m in net.modules():
        if isinstance(m, WSConv2d):
            m.ws_eps = ws_eps


def _normalize_tuning_method(tuning_method: str) -> str:
    alias = {
        "conv": "conv_adapt",
        "conv-adapter": "conv_adapt",
        "conv_adapter": "conv_adapt",

        "trso_adapter": "trso",
        "task_response": "trso",

        "residual_adapter": "residual",
        "residual_adapters": "residual",
        "ra": "residual",

        "side-tuning": "sidetune",
        "sidetuning": "sidetune",
        "side_tune": "sidetune",

        "lora": "lora_conv",
        "lora-conv": "lora_conv",
    }
    return alias.get(str(tuning_method), str(tuning_method))


def _safe_tuning_config(tuning_method, args):
    """
    Some repos don't define a config for 'sidetune'; in that case,
    just return a neutral config so backbones build cleanly.
    """
    try:
        return set_tuning_config(tuning_method, args)
    except NotImplementedError:
        if str(_normalize_tuning_method(tuning_method)) == 'sidetune':
            return {"method": "full"}
        raise


def build_model(model_name, pretrained=True, num_classes=1000, input_size=224,
                tuning_method='full', args=None, **kwargs):
    """
    Build a backbone and apply the requested parameter-efficient tuning method.
    'sidetune' wraps a frozen backbone with a lightweight side network + alpha blending.
    """
    tm = _normalize_tuning_method(tuning_method)

    # 1) Build the base backbone
    tuning_config = _safe_tuning_config(tm, args)
    base = eval(model_name)(
        pretrained=pretrained,
        tuning_config=tuning_config,
        input_resolution=input_size,
        **kwargs
    )

    # 2) Wrap (sidetune) or attach standard head
    if str(tm) == 'sidetune':
        # Be robust to wrapper signature differences
        kw = dict(
            num_classes=num_classes,
            side_width=int(getattr(args, 'sidetune_width', 64)),
            side_depth=int(getattr(args, 'sidetune_depth', 3)),
            learn_alpha=bool(getattr(args, 'sidetune_learn_alpha', True)),
            alpha_init=float(getattr(args, 'sidetune_alpha', 0.5)),
            use_checkpoint=True,
        )
        try:
            model = SideTuningClassifier(base_backbone=base, **kw)
        except TypeError:
            model = SideTuningClassifier(base_model=base, **kw)
    else:
        model = base
        model.head = LinearHead(model.num_features, num_classes, dropout=0.2)


    # 2.5) Apply LoRA replacement (builder-path support)
    if tm == 'lora_conv':
        from .tuning_modules.lora_conv import apply_lora_conv2d
        r = int(getattr(args, 'lora_r', 4))
        alpha = float(getattr(args, 'lora_alpha', 1.0))
        apply_lora_conv2d(model, r=r, alpha=alpha)

    # 3) Freeze/unfreeze according to tuning method
    if tm == 'full':
        pass

    elif tm == 'prompt':
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if name.startswith('norm'):
                continue
            if 'tuning_module' in name:
                continue
            p.requires_grad = False

    elif tm == 'adapter':
        raise NotImplementedError

    elif tm == 'sidetune':
        # Train only side network + alpha + head; freeze the base
        for name, p in model.named_parameters():
            train_ok = (
                name.startswith('side_net.') or
                name.startswith('head.') or
                name == 'alpha_logit'
            )
            p.requires_grad = train_ok

    elif tm == 'linear':
        for name, p in model.named_parameters():
            if name.startswith('head') or name.startswith('norm'):
                continue
            p.requires_grad = False

    elif tm == 'norm':
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if ('bn' in name) or ('gn' in name) or ('norm' in name):
                continue
            if 'before_head' in name:
                continue
            p.requires_grad = False

    elif tm == 'bias':
        # (existing behavior): head + norms + all biases
        for name, p in model.named_parameters():
            if name.startswith('head') or name.startswith('norm') or ('bias' in name):
                continue
            p.requires_grad = False

    elif tm == 'bitfit':
        # strict BitFit: head + biases only (no norms)
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if name.endswith('.bias') or ('.bias' in name):
                continue
            p.requires_grad = False

    elif tm in ('conv_adapt', 'repnet'):
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'tuning_module' in name:
                continue
            if 'norm' in name:
                continue
            p.requires_grad = False

    elif tm == 'conv_adapt_norm':
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'tuning_module' in name:
                continue
            if ('bn' in name) or ('gn' in name) or ('norm' in name):
                continue
            if 'before_head' in name:
                continue
            p.requires_grad = False

    elif tm in ('conv_adapt_bias', 'repnet_bias'):
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'tuning_module' in name:
                continue
            if 'bias' in name:
                continue
            if name.startswith('norm'):
                continue
            p.requires_grad = False

    elif tm == 'ssf':
        # Train head + SSF params (and optionally norms, keep consistent with conv_adapt)
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'ssf' in name:
                continue
            if 'tuning_module' in name:
                continue
            if 'norm' in name:
                continue
            p.requires_grad = False

    elif tm == 'lora_conv':
        # Train head + LoRA params (and optionally norms, keep consistent with conv_adapt)
        for name, p in model.named_parameters():
            if name.startswith('head'):
                continue
            if 'lora_' in name or 'lora_down' in name or 'lora_up' in name or 'LoRA' in name:
                continue
            if 'norm' in name:
                continue
            p.requires_grad = False

    if 'repnet' in str(tm):
        replace_conv2d_with_my_conv2d(model, 1e-5)

    # 4) Debug: list trainable params
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n} is trainable")

    return model
