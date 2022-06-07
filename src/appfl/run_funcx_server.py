import torch.nn as nn
from .misc import *
from .algorithm import *
def run_server(
    cfg: DictConfig,
    model: nn.Module,
    loss_fn: nn.Module,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    dataset_name: str = "appfl",
):
    """Run PPFL server that aggregates the updates the global parameters of model using FuncX
    """
    pass
