"""
model.py
========

This module defines the neural network used to classify radio galaxy
morphologies.  The provided skeleton uses a `ResNet‑18` backbone from
torchvision and adapts it for single‑channel input and multi‑label
output.  The final fully connected layer is replaced with a linear
layer of size `num_classes`, and we do not apply a softmax activation
so that you can use `BCEWithLogitsLoss` during training.

You may wish to experiment with different architectures (e.g.
rotation‑equivariant CNNs, DenseNets, or deeper ResNets) depending on
your dataset size and hardware constraints.  Feel free to modify
`build_model()` accordingly.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Construct a ResNet‑18 model for multi‑label classification.

    The first convolution is modified to accept single‑channel (grayscale)
    input.  The final fully connected layer is replaced to output
    `num_classes` logits.  A sigmoid activation should be applied
    externally during evaluation or via `BCEWithLogitsLoss` during
    training.

    Args:
        num_classes: The number of output labels in the multi‑label
            classification task.
        pretrained: Whether to load ImageNet pre‑trained weights.  For
            this assignment you will typically set this to `False` to
            adhere to the requirement of training from scratch.

    Returns:
        A `torch.nn.Module` implementing the modified ResNet‑18.
    """
    # Load a ResNet‑18 backbone.  Set `pretrained=False` for training
    # from scratch.
    model = models.resnet18(pretrained=pretrained)

    # Modify the first convolution to take a single input channel.  The
    # default conv1 expects 3 channels (RGB).  We initialise the new
    # convolution weights by averaging across the RGB channels of the
    # original weights if `pretrained` is True; otherwise use Kaiming
    # initialisation.
    orig_conv = model.conv1
    model.conv1 = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                            stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            # Average the weights across the input channel dimension to
            # produce a single channel kernel.  This is a common trick
            # when adapting pre‑trained models to grayscale input.
            model.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

    # Replace the fully connected layer to output `num_classes` logits
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Factory function for model creation.

    You can replace this function with your own architecture (e.g.
    DenseNet, custom CNN, rotation‑equivariant network) by swapping
    calls here.  The default is to construct a ResNet‑18 via
    `build_resnet18()`.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretraining.

    Returns:
        A PyTorch module implementing the classifier.
    """
    return build_resnet18(num_classes, pretrained)
