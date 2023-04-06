import pytest
import os
import numpy as np
from PIL import Image

import torch
from torch import Tensor

from modelanalyzer.analyzer.get_featmap import get_featmap_single_layer, \
    get_featmap_multi_layer
from tests.data.model import CNNModel


@pytest.mark.parametrize("target_layer",
                         ["model.layer_1[0]",
                          "layer_1.0"])
@pytest.mark.parametrize("average",
                         [True,
                          False])
@pytest.mark.parametrize("input",
                         [True,
                          False])
def test_get_featmap_single(target_layer,
                            average,
                            input):
    # prepare data and model to test
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "../data/test_car.jpg"))
    image = np.asarray(image)

    model = CNNModel()

    # test false input
    with pytest.raises(TypeError):
        # input to func must be a tensor
        get_featmap_single_layer(model,
                                 image,
                                 target_layer)

    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float() # B, C, H, W

    # test false target_layer
    _target_layer = [target_layer]
    with pytest.raises(TypeError):
        # target_layer must be a string
        get_featmap_single_layer(model,
                                 image,
                                 _target_layer)

    # normal input
    output = get_featmap_single_layer(model,
                                      image,
                                      target_layer,
                                      average=average,
                                      input=input)
    assert isinstance(output, Tensor)
    assert output.dim() == 4
    if average:
        assert output.shape[1] == 1


@pytest.mark.parametrize("target_layers",
                         [["model.layer_1[0]", "layer_1.0"],
                          ["layer_1.0", "model.layer_1[2]"]])
@pytest.mark.parametrize("average",
                         [True,
                          False])
@pytest.mark.parametrize("input",
                         [True,
                          False])
def test_get_featmap_multi(target_layers,
                           average,
                           input):
    # prepare data and model to test
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "../data/test_car.jpg"))
    image = np.asarray(image)

    model = CNNModel()

    # test false input
    with pytest.raises(TypeError):
        # input to func must be a tensor
        get_featmap_multi_layer(model,
                                image,
                                target_layers)

    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float()  # B, C, H, W
    # test false target layers
    _target_layers = "model.layer_1[0]"
    with pytest.raises(TypeError):
        # target_layers must be a list of string
        get_featmap_multi_layer(model,
                                image,
                                _target_layers)

    with pytest.raises(AssertionError):
        # target_layers is list of len 1
        get_featmap_multi_layer(model,
                                image,
                                [_target_layers])

    # normal input
    output = get_featmap_multi_layer(model,
                                     image,
                                     target_layers,
                                     average=average,
                                     input=input)
    assert isinstance(output, list)
    for _output in output:
        assert _output.dim() == 4
    if average:
        for _output in output:
            assert _output.shape[1] == 1
