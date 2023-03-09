import torch.nn as nn

from .hooks import ForwardIOHook


def vis_featmap(model: nn.Module,
                target_layer: str,
                input: bool=False,
                num_channels: int=None,
                average: bool=False):
    """
    This function visualize input/output feature maps of a layer in model

    :param model:
    :param target_layer:
    :param num_channels:
    :param average:
    :return:
    """
