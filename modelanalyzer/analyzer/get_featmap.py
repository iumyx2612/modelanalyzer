from typing import List

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
    # list of layers in model
    model_children = list(model.children())
    layer = get_layer(model, target_layer)
    hook = ForwardIOHook(layer)
    with torch.no_grad():
        _ = model(image)
        if not input:
            feat_maps = hook.output
        else:
            feat_maps = hook.input[0]
        # checking dimension of feature map
        if feat_maps.dim() == 3:
            index = model_children.index(layer)
            hook = ForwardIOHook(model_children[index-1])
            _ = model(image)
            # get feature map of previous layer
            feat_map_previous = hook.output
            if feat_map_previous.dim() == 4:
                B,C,W,H = feat_map_previous.size()
                # reshape feature map from (B,L,C) to (B,C,W,H) with L = W*H
                feature = feat_maps.reshape(B, W, H, C)
                feat_maps = feature.permute(0, 3, 1, 2)
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
    feat_maps = []
    for target_layer in target_layers:
        #get feature map of each layer
        feat_map = get_featmap_single_layer(model, image, target_layer,average,input)
        feat_maps.append(feat_map)
    return feat_maps
