from typing import List
import logging

import torch
from torch import Tensor
from torch.nn import Module

from ..hooks import ForwardIOHook
from ..utils import get_layer


def get_featmap_single_layer(model: Module,
                             image: Tensor,
                             target_layer: str,
                             average: bool=True,
                             input: bool=False) -> Tensor:
    """
    This function gets feature map from a specified layer in a model
    Args:
        model (Module): deep learning model
        image (Tensor): input image to the model
        target_layer (str): layer to visualize
        average (bool): whether to average feature maps on every location, default to True
    """
    # Type checking
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be of type 'Tensor', current type {type(image)}")
    if not isinstance(target_layer, str):
        raise TypeError(f"target_layer must be of type 'str', current type {type(target_layer)}")
    model.eval()
    layer = get_layer(model, target_layer)
    hook = ForwardIOHook(layer)
    with torch.no_grad():
        _ = model(image)
        if not input:
            feat_maps = hook.output
        else:
            feat_maps = hook.input[0]
        if average:
            feat_maps = torch.mean(feat_maps, dim=1, keepdim=True)
    return feat_maps


def get_featmap_multi_layer(model: Module,
                            image: Tensor,
                            target_layers: List[str],
                            average: bool=True,
                            input: bool=False) -> List[Tensor]:
    # Type checking
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be of type 'Tensor', current type {type(image)}")
    if not isinstance(target_layers, list):
        raise TypeError(f"target_layer must be of type 'list(str)', current type {type(target_layers)}")
    assert len(target_layers) > 1, f"len of 'target_layers' must be greater than 1," \
                                   f"else use `get_featmap_single_layer`"
    model.eval()

    layers = []
    for target_layer in target_layers:
        try:
            layers.append(get_layer(
                model, target_layer
            ))
        except AttributeError:
            logging.warning(f"model does not contain layer {target_layer}, "
                            f"skipping that layer")
            pass

    hooks = []
    for layer in layers:
        hooks.append(ForwardIOHook(layer))

    with torch.no_grad():
        _ = model(image)
        if not input:
            feat_maps = [hook.output for hook in hooks]
        else:
            feat_maps = [hook.input[0] for hook in hooks]
        if average:
            feat_maps = [torch.mean(feat_map, dim=1, keepdim=True) \
                         for feat_map in feat_maps]
    return feat_maps
