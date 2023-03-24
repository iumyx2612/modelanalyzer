import os
import logging
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import Tensor

from .hooks import ForwardIOHook
from .utils import is_perfect_square


def vis_featmap_single_layer(image: Tensor,
                             model: nn.Module,
                             target_layer: str,
                             input: bool=False,
                             num_channels: int=None,
                             average: bool=False,
                             show: bool=True,
                             save_path: str=None) -> None:
    """
    This function visualize input/output feature maps of a layer in model
    Args:
        image (Tensor): input image of type Tensor
        model (nn.Module): the model
        target_layer (str): layer to visualize
        input (bool): gets input feature maps to a layer instead of output, default to False
        num_channels (int): number of channels to plot, default to None
        average (bool): whether to average feature maps on every location, default to False
        show (bool): whether to show the visualization, default to True
        save_path (str): path to save the visualization, default to None. The saved visualization
        will have the file name being the target_layer
    """
    model.eval()
    layer = eval(target_layer)
    hook = ForwardIOHook(layer)
    with torch.no_grad():
        _ = model(image)
        if not input:
            feat_maps = hook.output
        else:
            feat_maps = hook.input[0]
        if average:
            feat_maps = torch.mean(feat_maps, dim=1, keepdim=True)

    feat_maps = feat_maps.cpu().numpy()
    _, c, _, _ = feat_maps.shape
    logging.info(f"Feature maps have shape: {feat_maps.shape}")

    if not is_perfect_square(c) or \
        (num_channels is not None and not is_perfect_square(num_channels)):
        logging.warning(f"Number of channels to visualize is {num_channels}, "
                        f"which is not a perfect square. Visualization might be "
                        f"missing some channels")

    nrows, ncols = int(math.sqrt(c)), int(math.sqrt(c)) # number of rows and cols to plot
    if num_channels is not None:
        nrows, ncols = int(math.sqrt(num_channels)), int(math.sqrt(num_channels))

    if not average:
        c_i = 0
        fig_size = nrows ** 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size, fig_size))
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].imshow(feat_maps[0, c_i, :, :])
                axes[i, j].axis('off')
                c_i += 1
    else:
        plt.imshow(feat_maps[0, 0])
        plt.axis('off')

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if not average:
            if not input:
                plt.savefig(f"{save_path}/{target_layer}.jpg")
            else:
                plt.savefig(f"{save_path}/{target_layer}_inp.jpg")
        else:
            if not input:
                plt.savefig(f"{save_path}/{target_layer}_avg.jpg")
            else:
                plt.savefig(f"{save_path}/{target_layer}_inp_avg.jpg")

    if show:
        plt.show()
