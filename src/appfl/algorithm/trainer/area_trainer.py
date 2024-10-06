import copy
from torch.nn import Module
from omegaconf import DictConfig
from typing import Optional, Any
from torch.utils.data import Dataset
from appfl.algorithm.trainer import VanillaTrainer

class AREATrainer(VanillaTrainer):
    """
    Trainer for the AREA algorithm: https://arxiv.org/abs/2405.10123
    """
    def __init__(
        self,
        model: Optional[Module]=None,
        loss_fn: Optional[Module]=None,
        metric: Optional[Any]=None,
        train_dataset: Optional[Dataset]=None,
        val_dataset: Optional[Dataset]=None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs
        )
        self.memory = copy.deepcopy(self.model)
        
    def train(self):
        new_memory = copy.deepcopy(self.model)
        self.train_configs.send_gradient = False
        super().train()
        # Compute the model state (model parameters to be sent) 
        # as the differnce between the current model and the memory
        if not hasattr(self, 'named_parameters'):
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        for name in self.named_parameters:
            self.model_state[name] = self.model_state[name] - self.memory.state_dict()[name].cpu()   
        # Update the memory
        self.memory = new_memory