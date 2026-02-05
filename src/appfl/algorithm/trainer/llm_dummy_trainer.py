import gc
import torch
from torch.nn import Module
from typing import Optional, Any
from appfl.algorithm.trainer import BaseTrainer


class LLMDummyTrainer(BaseTrainer):
    def __init__(
        self,
        model: Optional[Module] = None,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        self.model = model
        self.logger = logger
        
    def get_parameters(self):
        return self.model.state_dict()
    
    def load_parameters(
        self,
        parameters
    ):
        """Load model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])
                    del parameters[name]
            gc.collect()