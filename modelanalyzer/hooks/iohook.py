import torch.nn as nn


class ForwardIOHook:
    """
    This hook is attached to a nn.Module to get input/output feature maps of that nn.Module
    in a forward pass
    Args:
        module (nn.Module): the module to be attached to
    """
    def __init__(self,
                 module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output) -> None:
        """
        This function registers the input feature maps of attached nn.Module to self.input
        the same for output feature maps
        In order to access feature maps, users can call hook.input or hook.output
        """
        self.input = input
        self.output = output

    def close(self) -> None:
        self.hook.remove()
