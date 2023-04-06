import os
import pytest
import numpy as np
from PIL import Image

import torch

from modelanalyzer.analyzer import get_featmap_single_layer, get_featmap_multi_layer
from modelanalyzer.utils import plot_single_featmap, plot_multi_featmap
from tests.data.model import CNNModel


@pytest.mark.parametrize("average",
                         [True,
                          False])
@pytest.mark.parametrize("input",
                         [True,
                          False])
def test_plot_single_featmap(average,
                             input):
    # prepare data and model to test
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "../data/test_car.jpg"))
    image = np.asarray(image)
    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float()

    model = CNNModel()

    # specify correct target layer
    target_layer = "model.layer_1[2]"

    # get featmap to plot
    torch_featmap = get_featmap_single_layer(
        model=model,
        image=image,
        target_layer=target_layer,
        average=average,
        input=input
    )

    # test plot with wrong featmap type
    with pytest.raises(TypeError):
        featmap = [1, 1, 1]
        plot_single_featmap(featmap,
                            show=False)

    # test plot with featmap of type Tensor
    plot_single_featmap(
        torch_featmap,
        show=False
    )

    # test plot with featmap of type ndarray
    numpy_featmap = torch_featmap.cpu().numpy()
    plot_single_featmap(
        numpy_featmap,
        show=False
    )

    # test num_channels bigger than channels
    with pytest.raises(AssertionError):
        plot_single_featmap(torch_featmap,
                            num_channels=100,
                            show=False)


@pytest.mark.parametrize("average",
                         [True,
                          False])
@pytest.mark.parametrize("input",
                         [True,
                          False])
def test_plot_multi_featmap(average,
                            input):
    # prepare data and model to test
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "../data/test_car.jpg"))
    image = np.asarray(image)
    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float()

    model = CNNModel()

    # test featmap not a list
    with pytest.raises(TypeError):
        target_layer = "model.layer_1[2]"
        torch_featmap = get_featmap_single_layer(
            model=model,
            image=image,
            target_layer=target_layer
        )
        plot_multi_featmap(torch_featmap)

    # specify correct target layer
    target_layers = ["model.layer_1[2]",
                    "layer_1.2"]

    # get featmap to plot
    torch_featmaps = get_featmap_multi_layer(
        model=model,
        image=image,
        target_layers=target_layers,
        average=average,
        input=input
    )

    # test target_layers not list of str
    with pytest.raises(TypeError):
        plot_multi_featmap(
            torch_featmaps,
            target_layers=[1, 2, "a"]
        )

    # test length of target_layers
    with pytest.raises(AssertionError):
        plot_multi_featmap(
            torch_featmaps,
            target_layers=["a", "a", "a"]
        )

    # test num_channels not list of int
    with pytest.raises(TypeError):
        plot_multi_featmap(
            torch_featmaps,
            num_channels=[1.0, 10]
        )

    # test length of num_channels
    with pytest.raises(AssertionError):
        plot_multi_featmap(
            torch_featmaps,
            num_channels=[10, 10, 10]
        )

    # test num_channels larger than channels of featmap
    with pytest.raises(AssertionError):
        plot_multi_featmap(
            torch_featmaps,
            num_channels=[100, 100]
        )

    if average:
        for featmap in torch_featmaps:
            assert featmap.shape[1] == 1

    # test correct input of type Tensor
    plot_multi_featmap(torch_featmaps,
                       show=False)
    # test correct input of type numpy array
    numpy_featmaps = [featmap.cpu().numpy() for featmap in \
                      torch_featmaps]
    plot_multi_featmap(numpy_featmaps,
                       show=False)
