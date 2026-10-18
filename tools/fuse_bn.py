#!/usr/bin/env python3
"""Fold BatchNorm parameters and supplied statistics into convolutional layers."""

import argparse
import os
import sys
import logging
import torch
from typing import Dict, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from models.ResNet8 import ResNet8
import torch.nn as nn
from typing import Dict


def fuse_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> torch.nn.Conv2d:
    """Fold BN into a convolution.

    W_fused = W * gamma / sqrt(var + eps)
    b_fused = beta + (b - mean) * gamma / sqrt(var + eps).
    """
    if conv.bias is None:
        conv.bias = torch.nn.Parameter(
            torch.zeros(conv.weight.size(0), device=conv.weight.device, dtype=conv.weight.dtype)
        )

    w = conv.weight
    b = conv.bias

    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    device = conv.weight.device
    dtype = conv.weight.dtype
    gamma = gamma.to(device=device, dtype=dtype)
    beta = beta.to(device=device, dtype=dtype)
    running_mean = running_mean.to(device=device, dtype=dtype)
    running_var = running_var.to(device=device, dtype=dtype)
    eps = torch.tensor(float(eps), device=device, dtype=dtype)

    std = torch.sqrt(running_var + eps)
    scale = gamma / std

    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = beta + (b - running_mean) * scale

    conv.weight = torch.nn.Parameter(w_fused)
    conv.bias = torch.nn.Parameter(b_fused)
    return conv


def fuse_model(model: torch.nn.Module) -> torch.nn.Module:
    """Fuse BN layers into preceding Conv layers for the ResNet8 architecture."""
    # stem
    model.conv1 = fuse_conv_bn(model.conv1, model.bn1)
    model.bn1 = torch.nn.Identity()

    # layer1
    for i, block in enumerate(model.layer1):
        block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
        block.bn1 = torch.nn.Identity()
        block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
        block.bn2 = torch.nn.Identity()
        # shortcut conv if present
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            conv_sc, bn_sc = block.shortcut[0], block.shortcut[1]
            block.shortcut = torch.nn.Sequential(fuse_conv_bn(conv_sc, bn_sc))

    # layer2
    for i, block in enumerate(model.layer2):
        block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
        block.bn1 = torch.nn.Identity()
        block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
        block.bn2 = torch.nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            conv_sc, bn_sc = block.shortcut[0], block.shortcut[1]
            block.shortcut = torch.nn.Sequential(fuse_conv_bn(conv_sc, bn_sc))

    # layer3
    for i, block in enumerate(model.layer3):
        block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
        block.bn1 = torch.nn.Identity()
        block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
        block.bn2 = torch.nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            conv_sc, bn_sc = block.shortcut[0], block.shortcut[1]
            block.shortcut = torch.nn.Sequential(fuse_conv_bn(conv_sc, bn_sc))

    # layer4 (ResNet18)
    if hasattr(model, "layer4"):
        for i, block in enumerate(model.layer4):
            block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
            block.bn1 = torch.nn.Identity()
            block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
            block.bn2 = torch.nn.Identity()
            if (
                getattr(block, "shortcut", None) is not None
                and isinstance(block.shortcut, nn.Sequential)
                and len(block.shortcut) >= 2
                and isinstance(block.shortcut[0], nn.Conv2d)
                and isinstance(block.shortcut[1], nn.BatchNorm2d)
            ):
                conv_sc, bn_sc = block.shortcut[0], block.shortcut[1]
                block.shortcut = torch.nn.Sequential(fuse_conv_bn(conv_sc, bn_sc))

    return model


# Track negative-variance channels encountered during folding.
_NEGATIVE_VARIANCE_COUNTER = {"channels": 0, "layers": 0}


def negative_variance_counts() -> Dict[str, int]:
    return dict(_NEGATIVE_VARIANCE_COUNTER)


def reset_negative_variance_counts() -> None:
    _NEGATIVE_VARIANCE_COUNTER["channels"] = 0
    _NEGATIVE_VARIANCE_COUNTER["layers"] = 0


def fuse_conv_bn_with_stats(
    conv: nn.Conv2d, bn: nn.BatchNorm2d, mean: torch.Tensor, var: torch.Tensor
) -> nn.Conv2d:
    """Fold a convolution and BN using the supplied batch mean and variance."""
    if conv.bias is None:
        conv.bias = torch.nn.Parameter(
            torch.zeros(conv.weight.size(0), device=conv.weight.device, dtype=conv.weight.dtype)
        )

    w = conv.weight
    b = conv.bias

    device = conv.weight.device
    dtype = conv.weight.dtype
    gamma = bn.weight.to(device=device, dtype=dtype)
    beta = bn.bias.to(device=device, dtype=dtype)
    eps = torch.tensor(float(bn.eps), device=device, dtype=dtype)
    mean = mean.to(device=device, dtype=dtype)
    var = var.to(device=device, dtype=dtype)

    # Clamp negative noisy variances before computing the BN scale.
    negative = int(torch.count_nonzero(var < 0).item())
    if negative:
        _NEGATIVE_VARIANCE_COUNTER["channels"] += negative
        _NEGATIVE_VARIANCE_COUNTER["layers"] += 1
        logging.getLogger("logger").warning(
            "[fuse_bn] %d/%d BN channels carry a negative variance "
            "(min %.6g); clamping to zero before the square root",
            negative,
            int(var.numel()),
            float(var.min().item()),
        )
    std = torch.sqrt(var.clamp_min(0.0) + eps)
    scale = gamma / std

    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = beta + (b - mean) * scale

    conv.weight = torch.nn.Parameter(w_fused)
    conv.bias = torch.nn.Parameter(b_fused)
    return conv


def fuse_model_batch(model: nn.Module, x: torch.Tensor) -> nn.Module:
    """Fold ResNet BN layers using statistics measured on the supplied batch."""
    device = next(model.parameters()).device
    x = x.to(device)

    # Measure population moments at BN inputs.
    bn_stats: Dict[int, Dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(bn_mod: nn.BatchNorm2d):
        def hook(mod, inp, out):
            z = inp[0]
            dims = (0, 2, 3)
            mean = z.mean(dim=dims)
            var = z.var(dim=dims, unbiased=False)
            bn_stats[id(mod)] = {"mean": mean.detach(), "var": var.detach()}

        return hook

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(make_hook(m)))

    model.train()  # Use train-mode normalization to measure batch statistics.
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    stem_bn = model.bn1
    stem_stats = bn_stats.get(id(stem_bn), None)
    if stem_stats is not None:
        model.conv1 = fuse_conv_bn_with_stats(
            model.conv1, stem_bn, stem_stats["mean"], stem_stats["var"]
        )
    model.bn1 = nn.Identity()

    # layer1
    for block in model.layer1:
        s1 = bn_stats.get(id(block.bn1), None)
        s2 = bn_stats.get(id(block.bn2), None)
        if s1 is not None:
            block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
        block.bn1 = nn.Identity()
        if s2 is not None:
            block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
        block.bn2 = nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            sc_bn = block.shortcut[1]
            sc_stats = bn_stats.get(id(sc_bn), None)
            if sc_stats is not None:
                sc_conv = fuse_conv_bn_with_stats(
                    block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                )
                block.shortcut = nn.Sequential(sc_conv)

    # layer2
    for block in model.layer2:
        s1 = bn_stats.get(id(block.bn1), None)
        s2 = bn_stats.get(id(block.bn2), None)
        if s1 is not None:
            block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
        block.bn1 = nn.Identity()
        if s2 is not None:
            block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
        block.bn2 = nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            sc_bn = block.shortcut[1]
            sc_stats = bn_stats.get(id(sc_bn), None)
            if sc_stats is not None:
                sc_conv = fuse_conv_bn_with_stats(
                    block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                )
                block.shortcut = nn.Sequential(sc_conv)

    # layer3
    for block in model.layer3:
        s1 = bn_stats.get(id(block.bn1), None)
        s2 = bn_stats.get(id(block.bn2), None)
        if s1 is not None:
            block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
        block.bn1 = nn.Identity()
        if s2 is not None:
            block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
        block.bn2 = nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            sc_bn = block.shortcut[1]
            sc_stats = bn_stats.get(id(sc_bn), None)
            if sc_stats is not None:
                sc_conv = fuse_conv_bn_with_stats(
                    block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                )
                block.shortcut = nn.Sequential(sc_conv)

    # layer4 (ResNet18)
    if hasattr(model, "layer4"):
        for block in model.layer4:
            s1 = bn_stats.get(id(block.bn1), None)
            s2 = bn_stats.get(id(block.bn2), None)
            if s1 is not None:
                block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
            block.bn1 = nn.Identity()
            if s2 is not None:
                block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
            block.bn2 = nn.Identity()
            if (
                getattr(block, "shortcut", None) is not None
                and isinstance(block.shortcut, nn.Sequential)
                and len(block.shortcut) >= 2
                and isinstance(block.shortcut[0], nn.Conv2d)
                and isinstance(block.shortcut[1], nn.BatchNorm2d)
            ):
                sc_bn = block.shortcut[1]
                sc_stats = bn_stats.get(id(sc_bn), None)
                if sc_stats is not None:
                    sc_conv = fuse_conv_bn_with_stats(
                        block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                    )
                    block.shortcut = nn.Sequential(sc_conv)

    return model


def extract_bn_stats_from_model(
    model: nn.Module, x: torch.Tensor
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Measure per-layer BN input means and population variances in one forward pass.

    Layer-name keys allow reuse with another instance of the same architecture.
    """
    device = next(model.parameters()).device
    x = x.to(device)

    # Measure population moments at BN inputs.
    bn_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(bn_name: str):
        def hook(mod, inp, out):
            z = inp[0]
            dims = (0, 2, 3)
            mean = z.mean(dim=dims)
            var = z.var(dim=dims, unbiased=False)
            bn_stats[bn_name] = {"mean": mean.detach(), "var": var.detach()}

        return hook

    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(make_hook(name)))

    model.train()  # Use train-mode normalization to measure batch statistics.
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return bn_stats


def fuse_model_batch_with_external_stats(
    model: nn.Module, bn_stats: Dict[str, Dict[str, torch.Tensor]]
) -> nn.Module:
    """Fold BN using a layer-name mapping of externally measured means and variances."""
    # stem
    stem_bn = model.bn1
    stem_stats = bn_stats.get("bn1", None)
    if stem_stats is not None:
        model.conv1 = fuse_conv_bn_with_stats(
            model.conv1, stem_bn, stem_stats["mean"], stem_stats["var"]
        )
    model.bn1 = nn.Identity()

    # layer1
    for i, block in enumerate(model.layer1):
        bn1_name = f"layer1.{i}.bn1"
        bn2_name = f"layer1.{i}.bn2"
        s1 = bn_stats.get(bn1_name, None)
        s2 = bn_stats.get(bn2_name, None)
        if s1 is not None:
            block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
        block.bn1 = nn.Identity()
        if s2 is not None:
            block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
        block.bn2 = nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            sc_bn = block.shortcut[1]
            sc_name = f"layer1.{i}.shortcut.1"
            sc_stats = bn_stats.get(sc_name, None)
            if sc_stats is not None:
                sc_conv = fuse_conv_bn_with_stats(
                    block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                )
                block.shortcut = nn.Sequential(sc_conv)

    # layer2
    for i, block in enumerate(model.layer2):
        bn1_name = f"layer2.{i}.bn1"
        bn2_name = f"layer2.{i}.bn2"
        s1 = bn_stats.get(bn1_name, None)
        s2 = bn_stats.get(bn2_name, None)
        if s1 is not None:
            block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
        block.bn1 = nn.Identity()
        if s2 is not None:
            block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
        block.bn2 = nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            sc_bn = block.shortcut[1]
            sc_name = f"layer2.{i}.shortcut.1"
            sc_stats = bn_stats.get(sc_name, None)
            if sc_stats is not None:
                sc_conv = fuse_conv_bn_with_stats(
                    block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                )
                block.shortcut = nn.Sequential(sc_conv)

    # layer3
    for i, block in enumerate(model.layer3):
        bn1_name = f"layer3.{i}.bn1"
        bn2_name = f"layer3.{i}.bn2"
        s1 = bn_stats.get(bn1_name, None)
        s2 = bn_stats.get(bn2_name, None)
        if s1 is not None:
            block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
        block.bn1 = nn.Identity()
        if s2 is not None:
            block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
        block.bn2 = nn.Identity()
        if (
            getattr(block, "shortcut", None) is not None
            and isinstance(block.shortcut, nn.Sequential)
            and len(block.shortcut) >= 2
            and isinstance(block.shortcut[0], nn.Conv2d)
            and isinstance(block.shortcut[1], nn.BatchNorm2d)
        ):
            sc_bn = block.shortcut[1]
            sc_name = f"layer3.{i}.shortcut.1"
            sc_stats = bn_stats.get(sc_name, None)
            if sc_stats is not None:
                sc_conv = fuse_conv_bn_with_stats(
                    block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                )
                block.shortcut = nn.Sequential(sc_conv)

    # layer4 (ResNet18)
    if hasattr(model, "layer4"):
        for i, block in enumerate(model.layer4):
            bn1_name = f"layer4.{i}.bn1"
            bn2_name = f"layer4.{i}.bn2"
            s1 = bn_stats.get(bn1_name, None)
            s2 = bn_stats.get(bn2_name, None)
            if s1 is not None:
                block.conv1 = fuse_conv_bn_with_stats(block.conv1, block.bn1, s1["mean"], s1["var"])
            block.bn1 = nn.Identity()
            if s2 is not None:
                block.conv2 = fuse_conv_bn_with_stats(block.conv2, block.bn2, s2["mean"], s2["var"])
            block.bn2 = nn.Identity()
            if (
                getattr(block, "shortcut", None) is not None
                and isinstance(block.shortcut, nn.Sequential)
                and len(block.shortcut) >= 2
                and isinstance(block.shortcut[0], nn.Conv2d)
                and isinstance(block.shortcut[1], nn.BatchNorm2d)
            ):
                sc_bn = block.shortcut[1]
                sc_name = f"layer4.{i}.shortcut.1"
                sc_stats = bn_stats.get(sc_name, None)
                if sc_stats is not None:
                    sc_conv = fuse_conv_bn_with_stats(
                        block.shortcut[0], sc_bn, sc_stats["mean"], sc_stats["var"]
                    )
                    block.shortcut = nn.Sequential(sc_conv)

    return model


def fuse_cnnmnist_with_external_stats(
    model: nn.Module, bn_stats: Dict[str, Dict[str, torch.Tensor]]
) -> nn.Module:
    stem_stats = bn_stats.get("bn1", None)
    if stem_stats is not None and isinstance(model.bn1, nn.BatchNorm2d):
        model.conv1 = fuse_conv_bn_with_stats(
            model.conv1, model.bn1, stem_stats["mean"], stem_stats["var"]
        )
    model.bn1 = nn.Identity()
    bn2_stats = bn_stats.get("bn2", None)
    if bn2_stats is not None and isinstance(model.bn2, nn.BatchNorm2d):
        model.conv2 = fuse_conv_bn_with_stats(
            model.conv2, model.bn2, bn2_stats["mean"], bn2_stats["var"]
        )
    model.bn2 = nn.Identity()
    return model


def fuse_cnnmnist_batch(model: nn.Module, x: torch.Tensor) -> nn.Module:
    device = next(model.parameters()).device
    x = x.to(device)
    bn_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(mod, inp, out):
            z = inp[0]
            dims = (0, 2, 3)
            mean = z.mean(dim=dims)
            var = z.var(dim=dims, unbiased=False)
            bn_stats[name] = {"mean": mean.detach(), "var": var.detach()}

        return hook

    if isinstance(model.bn1, nn.BatchNorm2d):
        hooks.append(model.bn1.register_forward_hook(make_hook("bn1")))
    if isinstance(model.bn2, nn.BatchNorm2d):
        hooks.append(model.bn2.register_forward_hook(make_hook("bn2")))

    model.train()
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()

    return fuse_cnnmnist_with_external_stats(model, bn_stats)


def fuse_model_arch(
    model: nn.Module,
    arch: str,
    x: Optional[torch.Tensor] = None,
    bn_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> nn.Module:
    arch = str(arch).lower()
    if arch in ["resnet8", "resnet18"]:
        if bn_stats is not None:
            return fuse_model_batch_with_external_stats(model, bn_stats)
        if x is not None:
            return fuse_model_batch(model, x)
        return fuse_model(model)
    elif arch in ["cnnmnist", "mnist", "cnn_mnist"]:
        if bn_stats is not None:
            return fuse_cnnmnist_with_external_stats(model, bn_stats)
        if x is not None:
            return fuse_cnnmnist_batch(model, x)

        # Without supplied moments, remove the CNN BatchNorm modules.
        model.bn1 = nn.Identity() if hasattr(model, "bn1") else model
        model.bn2 = nn.Identity() if hasattr(model, "bn2") else model
        return model
    else:
        # Unrecognized architectures are returned unchanged.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                pass
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original ResNet8 .pth")
    parser.add_argument("--output", required=True, help="Path to fused output .pth")
    args = parser.parse_args()

    model = ResNet8(num_classes=10)
    sd = torch.load(args.input, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)

    model.eval()
    model = fuse_model(model)
    fused_sd: Dict[str, torch.Tensor] = model.state_dict()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(fused_sd, args.output)
    print(f"Fused weights saved to {args.output}")


if __name__ == "__main__":
    main()
