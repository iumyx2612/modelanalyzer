import pytest
import os
import numpy as np
from PIL import Image

import torch
from torch import Tensor

from modelanalyzer.analyzer.get_featmap import get_featmap_single_layer
from tests.data.model import CNNModel


def test_get_featmap_single():
    # prepare data and model to test
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "../data/test_car.jpg"))
    image = np.asarray(image)

    model = CNNModel()

    # specify correct target layer
    target_layer = "model.layer_1[2]"

    # test false input
    with pytest.raises(TypeError):
        # input to func must be a tensor
        get_featmap_single_layer(model,
                                 image,
                                 target_layer)

    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float() # B, C, H, W
    print(image.shape)

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
                                      target_layer)
    assert isinstance(output, Tensor)
