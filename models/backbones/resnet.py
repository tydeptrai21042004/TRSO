# models/backbones/resnet.py
from typing import Type, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from ..tuning_modules import PadPrompter, ConvAdapter

__all__ = ['resnet50', 'resnet50_mocov3', 'resnet101', 'resnet152']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=bias, dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self, inplanes: int, planes: int, stride: int = 1,
        downsample: Optional[nn.Module] = None, groups: int = 1,
        base_width: int = 64, dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        tuning_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        tc = tuning_config or {}
        method = tc.get('method', '')

        self.conv1 = conv3x3(inplanes, planes, stride, bias=(method == 'bias'))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=(method == 'bias'))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.tuning_config = tc
        if 'conv_adapt' in method:
            width = max(1, inplanes // max(1, tc.get('adapt_size', 8)))
            groups_r = max(1, inplanes // max(1, tc.get('adapt_size', 8)))
            self.tuning_module = ConvAdapter(
                inplanes, planes,
                kernel_size=3, padding=1,
                width=width, stride=stride, groups=groups_r, dilation=1,
                act_layer=nn.ReLU
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        if 'conv_adapt' in self.tuning_config.get('method', ''):
            out = out + self.tuning_module(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self, inplanes: int, planes: int, stride: int = 1,
        downsample: Optional[nn.Module] = None, groups: int = 1,
        base_width: int = 64, dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        tuning_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        tc = tuning_config or {}
        method = tc.get('method', '')

        self.conv1 = conv1x1(inplanes, width, bias=(method == 'bias'))
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, bias=(method == 'bias'))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=(method == 'bias'))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.tuning_config = tc
        if 'conv_adapt' in method:
            width_red = max(1, width // max(1, tc.get('adapt_size', 8)))
            groups_red = max(1, width // max(1, tc.get('adapt_size', 8)))
            self.tuning_module = ConvAdapter(
                width, width,
                kernel_size=3, padding=1,
                width=width_red, stride=stride, groups=groups_red, dilation=1,
                act_layer=nn.ReLU
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)

        out_adapt = None
        if 'conv_adapt' in self.tuning_config.get('method', ''):
            out_adapt = self.tuning_module(out)

        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        if out_adapt is not None:
            out = out + out_adapt

        out = self.conv3(out); out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
        num_classes: int = 1000, zero_init_residual: bool = False,
        groups: int = 1, width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_resolution: Optional[int] = 224,
        tuning_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.tuning_config = tuning_config or {}
        if self.tuning_config.get('method') == 'prompt':
            self.tuning_module = PadPrompter(
                prompt_size=int(self.tuning_config.get('prompt_size', 10)),
                image_size=input_resolution
            )

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must have len=3")
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512 * block.expansion, num_classes)
        self.num_features = 512 * block.expansion  # <- used by SideTuning

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

        # expose features; external builder usually overwrites head
        self.head = nn.Identity()
        self.norm = nn.BatchNorm2d(512 * block.expansion)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups, self.base_width,
            previous_dilation, norm_layer, tuning_config=self.tuning_config
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                dilation=self.dilation, norm_layer=norm_layer, tuning_config=self.tuning_config
            ))
        return nn.Sequential(*layers)

    def forward_features(self, x: Tensor) -> Tensor:
        # Prompt (if used)
        if self.tuning_config.get('method') == 'prompt':
            x = self.tuning_module(x)
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, C]
        return x

    def forward(self, x: Tensor) -> Tensor:
        feats = self.forward_features(x)
        return self.head(feats)


# --- Factory helpers & weights loading (basic) ---
model_urls = {
    "resnet50_1k": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "resnet101_1k": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    "resnet152_1k": "https://download.pytorch.org/models/resnet152-f82ba261.pth",
    "resnet50_mocov3": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
}

def _safe_url(key_1k: str, key_22k: str, in_22k: bool) -> str:
    if in_22k and key_22k in model_urls:
        return model_urls[key_22k]
    return model_urls[key_1k]

def resnet50(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        url = _safe_url("resnet50_1k", "resnet50_22k", in_22k)
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        _ = model.load_state_dict(checkpoint, strict=False)
    return model

def resnet50_mocov3(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        url = model_urls["resnet50_mocov3"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        state_dict = checkpoint['state_dict']
        new_state_dict = { '.'.join(k.split('.')[2:]): v
                           for k, v in state_dict.items()
                           if k.startswith('module.base_encoder') }
        _ = model.load_state_dict(new_state_dict, strict=False)
    return model

def resnet101(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        url = _safe_url("resnet101_1k", "resnet101_22k", in_22k)
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        _ = model.load_state_dict(checkpoint, strict=False)
    return model

def resnet152(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        url = _safe_url("resnet152_1k", "resnet152_22k", in_22k)
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        _ = model.load_state_dict(checkpoint, strict=False)
    return model

if __name__ == '__main__':
    model = resnet50(pretrained=True, tuning_config={'method': 'full', 'adapt_size': 16})
    print(model)
    x = torch.randn((1, 3, 224, 224))
    o = model(x)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', n_parameters)
