from typing import Union, List
from torch.nn import Module
from torch import Tensor

from .analyzer.get_featmap import get_featmap_single_layer, get_featmap_multi_layer
from .utils.plot import plot_single_featmap, plot_multi_featmap


def vis_featmap(model: Module,
                image: Tensor,
                target_layer: Union[str, List[str]],
                input: bool=False,
                average: bool=False,
                num_channels: Union[int, List[int]]=None,
                show: bool=True,
                save_path: str=None) -> None:
    """
    This function visualize input/output feature maps of layer(s) in model
    Args:
        model (nn.Module): the model
        image (Tensor): input image of type Tensor
        target_layer (Union[str, List[str]]): layer to visualize
        input (bool): gets input feature maps to a layer instead of output, default to False
        average (bool): whether to average feature maps on every location, default to False
        num_channels (Union[int, List[int]]): number of channels to plot, default to None
        show (bool): whether to show the visualization, default to True
        save_path (str): path to save the visualization, default to None. The saved visualization
            will have the file name being the target_layer
    """
    if isinstance(target_layer, str):
        feat_maps = get_featmap_single_layer(
            model=model,
            image=image,
            target_layer=target_layer,
            input=input,
            average=average
        )

        plot_single_featmap(
            feat_maps=feat_maps,
            num_channels=num_channels,
            show=show,
            save_path=save_path,
            target_layer=target_layer
        )
    elif isinstance(target_layer, list):
        feat_maps = get_featmap_multi_layer(
            model=model,
            image=image,
            target_layers=target_layer,
            input=input,
            average=average
        )

        plot_multi_featmap(
            feat_maps=feat_maps,
            num_channels=num_channels,
            show=show,
            save_path=save_path,
            target_layers=target_layer
        )
