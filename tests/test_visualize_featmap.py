import pytest
import os
import numpy as np
from PIL import Image

import torch

from modelanalyzer.visualize_featmap import vis_featmap
from .data.model import CNNModel


@pytest.mark.parametrize("input",
                         [True,
                          False])
@pytest.mark.parametrize("average",
                         [True,
                          False])
def vis_featmap(input,
                average):
    # prepare data and model to test
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "./data/test_car.jpg"))
    image = np.asarray(image)
    image = torch.tensor(image.transpose(2, 0, 1))[None].float() # B, C, H, W

    model = CNNModel()

    # specify correct target layer
    target_layer = "model.layer_1[2]"
    target_layers = ["model.layer_1[2]",
                     "layer_2.0"]

    vis_featmap(
        model=model,
        image=image,
        target_layer=target_layer,
        input=input,
        average=average,
        show=False
    )
    vis_featmap(
        model,
        image,
        target_layers,
        input=input,
        average=average,
        show=False
    )
