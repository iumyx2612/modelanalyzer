import pytest
import os
import numpy as np
from PIL import Image

import torch

from modelanalyzer.visualize_featmap import vis_featmap_single_layer
from .data.model import CNNModel


@pytest.mark.parametrize("input",
                         [True,
                          False])
@pytest.mark.parametrize("average",
                         [True,
                          False])
def test_vis_featmap_single_layer(input,
                                  average):
    # prepare data and model to test
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "./data/test_car.jpg"))
    image = np.asarray(image)
    image = torch.tensor(image.transpose(2, 0, 1))[None].float() # B, C, H, W

    model = CNNModel()

    # specify correct target layer
    target_layer = "model.layer_1[2]"

    vis_featmap_single_layer(
        model=model,
        image=image,
        target_layer=target_layer,
        input=input,
        average=average,
        show=False
    )
