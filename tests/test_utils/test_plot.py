import os
import pytest
import numpy as np
from PIL import Image

import torch

from modelanalyzer.analyzer import get_featmap_single_layer
from modelanalyzer.utils import plot_single_featmap
from ..data.model import CNNModel


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
    print(type(torch_featmap))

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
