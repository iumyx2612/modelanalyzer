from typing import Union, List, Optional
import os
import logging
import math
import matplotlib.pyplot as plt

import pdb

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
        raise TypeError(f"feat_maps must be of type 'Tensor' or numpy array,"
                        f" current type {type(feat_maps)}")

    if isinstance(feat_maps, Tensor):
        feat_maps = feat_maps.cpu().numpy()
    _, c, _, _ = feat_maps.shape
    logging.info(f"Feature maps have shape: {feat_maps.shape}")
    if num_channels is not None:
        assert num_channels < c, f"num_channels must be smaller than {c}"

    if not is_perfect_square(c) or \
        (num_channels is not None and not is_perfect_square(num_channels)):
        logging.warning(f"Number of channels to visualize is {c or num_channels}, "
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


def plot_multi_featmap(feat_maps: Union[List[Tensor], List[ndarray]],
                       num_channels: Optional[List[int]]=None,
                       show: bool=True,
                       save_path: str=None,
                       target_layers: Optional[List[str]]=None) -> None:
    """
    Function to visualize feature maps of multiple layers
        under 2D representation
    Args:
        feat_maps (Union[List[Tensor], List[ndarray]):
            list of feature maps of shape (B, C, H, W) to visualize
        num_channels (List[int]): number of channel to visualize, visualize all if None
            default to None
        show (bool): show visualization, default to True
        save_path (str): path to save the visualization, default to None
        target_layers (List[str]): save name, default to 'temp'
    """
    # Type checking
    if not isinstance(feat_maps, list):
        raise TypeError(f"feat_map must be a list of Tensor of numpy array")

    if not all(isinstance(feat_map, (Tensor, ndarray)) for \
           feat_map in feat_maps):
        raise TypeError(f"feat_maps must be list of Tensor or numpy array,"
                        f"got type {type(feat_maps[0])}")
    if target_layers is not None:
        if not all(isinstance(target_layer, str) for \
               target_layer in target_layers):
            raise TypeError(f"target_layers must be list of str")
    if num_channels is not None:
        if not all(isinstance(num_channel, int) for \
               num_channel in num_channels):
            raise TypeError(f"num_channels must be list of int")

    # Length checking
    if target_layers is not None:
        assert len(target_layers) == len(feat_maps)
    if num_channels is not None:
        assert len(num_channels) == len(feat_maps)
    # Assign default value
    if target_layers is None:
        target_layers = ['temp'] * len(feat_maps)

    if all(isinstance(feat_map, Tensor) for feat_map
           in feat_maps):
        feat_maps = [feat_map.cpu().numpy() for feat_map
                     in feat_maps]

    # shape checking
    for feat_map in feat_maps:
        assert feat_map.ndim == 4, f"feat_map must have dim of 4," \
                                   f"current shape: {feat_map.shape}"

    cs = [] # all channels of feature maps
    for feat_map in feat_maps:
        _, c, _, _ = feat_map.shape
        cs.append(c)
    if num_channels is not None: # remapping num_channels
        for i, num_channel in enumerate(num_channels):
            assert num_channel < cs[i], f"num_channels[{i}] must be smaller " \
                                        f"than {cs[i]}"
            cs[i] = num_channel
    for c in cs:
        if not is_perfect_square(c):
            logging.warning(f"Number of channels to visualize is {c}, "
                            f"which is not a perfect square. Visualization might be "
                            f"missing some channels")

    # plot multiple figures
    for i, c in enumerate(cs):
        print(c)
        if c == 1: # has been averaged
            plt.imshow(feat_maps[i][0, 0])
            plt.axis("off")
            plt.title(f"{target_layers[i]}")
        else:
            nrows, ncols = int(math.sqrt(c)), int(math.sqrt(c))
            c_i = 0
            fig, axes = plt.subplots(nrows, ncols)
            plt.suptitle(f"{target_layers[i]}")
            for row_i in range(nrows):
                for col_i in range(ncols):
                    axes[row_i, col_i].imshow(feat_maps[i][0, c_i, :, :])
                    axes[row_i, col_i].axis('off')
                    c_i += 1

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            if c != 1:
                plt.savefig(f"{save_path}/{target_layers[i]}.jpg")
            else:
                plt.savefig(f"{save_path}/{target_layers[i]}_avg.jpg")
    if show:
        plt.show()
    plt.close()
