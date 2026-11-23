import importlib

import torch


def get_loss(loss_fn_path, loss_class_name=None):
    if loss_fn_path is None:
        return torch.nn.CrossEntropyLoss()
    # Extract the module name from the file path (removing the ".py" extension)
    loss_fn_path = loss_fn_path.replace("/", ".")
    module_name = loss_fn_path.removesuffix(".py")

    # Import the module dynamically
    module = importlib.import_module(module_name)

    if loss_class_name is not None:
        # Get the specified class from the module
        loss_class = getattr(module, loss_class_name)
    else:
        # If no class name is given, return the first class found in the module
        loss_class = None
        for _, obj in module.__dict__.items():
            if isinstance(obj, type):
                loss_class = obj
                break
    return loss_class()
