import copy
import torch
from typing import Dict, Any


def laplace_mechanism_output_perturb(
    model: torch.nn.Module, sensitivity: float, epsilon: float
) -> Dict[str, Any]:
    """
    Differential privacy for output perturbation based on Laplacian distribution.
    The output perturbation adds Laplacian noise with zero mean to ``model.named_parameters()``.
    Variance is equal to `2*(scale_value)^2`, and `scale_value = sensitivity / epsilon`,
    where `sensitivity` is determined by data and algorithm.
    :param model: torch.nn.Module
    :param sensitivity: sensitivity
    :param epsilon: privacy budget
    :return: model state dictionary with Laplacian noise added
    """
    scale_value = sensitivity / epsilon
    state_dict = copy.deepcopy(model.state_dict())
    for name, param in model.named_parameters():
        mean = torch.zeros_like(param.data)
        scale = torch.zeros_like(param.data) + scale_value
        m = torch.distributions.laplace.Laplace(mean, scale)
        state_dict[name] += m.sample()
    return state_dict
