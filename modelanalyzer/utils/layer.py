import logging

import torch.nn as nn


def has_layer(model: nn.Module,
              layer: str) -> bool:
    """
    This function check if layer exist inside model
    Args:
        model (nn.Module): the model
        layer (str): the specified layer
    """
    model = model
    if not isinstance(layer, str):
        raise TypeError(f"layer must be of type 'str', current type {type(layer)}")

    if "model" in layer:
        try:
            eval(layer)
            return True
        except AttributeError as attr_err:
            logging.warning(
                f"model does not contain {layer}, please check again"
            )
            raise AttributeError(f"model does not contain {layer},"
                                 f" please check again") from attr_err
    else:
        try:
            model.get_submodule(layer)
            return True
        except AttributeError as attr_err:
            raise AttributeError(f"model does not contain {layer},"
                                 f" please check again") from attr_err


def get_layer(model: nn.Module,
              layer: str) -> nn.Module:
    """
    This function returns a specified if it exists in model
    Args:
        model (nn.Module): the model
        layer (str): the specified layer
    """
    model = model
    has_layer(model, layer)
    if "model" in layer:
        return eval(layer)
    else:
        return model.get_submodule(layer)
