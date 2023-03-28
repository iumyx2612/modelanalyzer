from typing import Union
import os
import logging
import math
import matplotlib.pyplot as plt

from torch import Tensor
from numpy import ndarray

from .misc import is_perfect_square

def plot_single_featmap(feat_maps: Union[Tensor, ndarray],
                        num_channels: int=None,
                        show: bool=True,
                        save_path: str=None,
                        target_layer: str='temp') -> None:
    """
    Function to visualize feature maps under 2D representation
    Args:
        feat_maps (Union[Tensor, ndarray]): feature maps of shape (B, C, H, W) to visualize
        num_channels (int): number of channel to visualize, visualize all if None
            default to None
        show (bool): show visualization, default to True
        save_path (str): path to save the visualization, default to None
        target_layer (str): save name, default to 'temp'
    """
    if not isinstance(feat_maps, (Tensor, ndarray)):
        raise TypeError(f"featmaps must be of type 'Tensor' or numpy array,"
                        f" current type {type(feat_maps)}")

    if isinstance(feat_maps, Tensor):
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

    if c != 1:
        c_i = 0
        fig_size = nrows ** 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size, fig_size))
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].imshow(feat_maps[0, c_i, :, :])
                axes[i, j].axis('off')
                c_i += 1
    else: # has been averaged
        plt.imshow(feat_maps[0, 0])
        plt.axis('off')

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if c != 1:
            plt.savefig(f"{save_path}/{target_layer}.jpg")
        else:
            plt.savefig(f"{save_path}/{target_layer}_avg.jpg")

    if show:
        plt.show()
