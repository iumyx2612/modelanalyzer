import os
import numpy as np
from PIL import Image
import pytest

import torch
from torch import Tensor
import torchvision.transforms as transforms

from tests.data.model import CNN_AttentionModel
from modelanalyzer.analyzer.get_featmap import get_featmap_multi_layer, get_featmap_single_layer



@pytest.mark.parametrize("average",
                         [True,
                          False])
@pytest.mark.parametrize("target_layer",["model.layer_3",
                                         'model.layer_1[2]'])

def test_get_featmap_attention( target_layer, average):
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "../data/test_car.jpg"))
    transform = transforms.ToTensor()
    model = CNN_AttentionModel()
    image = transform(image).unsqueeze(dim=0)
    image_np = np.asarray(image)
    with pytest.raises(TypeError):
        get_featmap_single_layer(image_np, target_layer=target_layer)

    output = get_featmap_single_layer(model,
                                      image,
                                target_layer=target_layer,
                                average=average,
                                      )
    assert isinstance(output, Tensor)
    assert output.dim() == 4
    if average:
        assert output.shape[1] == 1


@pytest.mark.parametrize("target_layers",
                         [["layer_1.0", "model.layer_3"]])
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

    model = CNN_AttentionModel()

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

    with pytest.raises(AttributeError):
        get_featmap_multi_layer(model,
                                image,
                                ["model.layer_1",
                                 "model.layer1"])

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


