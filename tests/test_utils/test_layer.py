import pytest

import torch.nn as nn

from modelanalyzer.utils import has_layer, get_layer
from ..data.model import CNNModel


pytestmark = pytest.mark.parametrize("layer",
                                     ["model.layer_1[0]",
                                      "layer_2.0"])


def test_has_layer(layer):
    model = CNNModel()

    # check input not str
    with pytest.raises(TypeError):
        has_layer(model, model.layer_1)

    # check input not in model, "model" in input
    with pytest.raises(AttributeError):
        layer_f = "model.layer"
        has_layer(model, layer_f)

    # check input not in model, "model" not in input
    with pytest.raises(AttributeError):
        layer_f = "layer_3"
        has_layer(model, layer_f)

    # normal input
    output = has_layer(model, layer)
    assert output == True


def test_get_layer(layer):
    model = CNNModel()
    layer = get_layer(model, layer)
    assert isinstance(layer, nn.Module)
