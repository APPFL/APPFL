import os
import json
import torch
import wandb
import numpy as np
from datetime import datetime
from typing import Any, Optional
from omegaconf import DictConfig, OmegaConf
from appfl.algorithm.trainer import BaseTrainer


class FedSBTrainer(BaseTrainer):
    def __init__(
        self, 
        train_configs: DictConfig,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        self.logger = logger
        self.train_configs = train_configs
        self._set_seed()
        
        self.train_configs.agg_type = self.train_configs.get("agg_type", "fed-sb")
        if self.train_configs.agg_type == "fed-sb":
            self.train_configs.lora_alpha = self.train_configs.lora_r
        
        self.run_dir = self._create_run_directory()
        self._init_wandb()
        
        
        
        
    def _set_seed(self):
        """
        Set the random seed for reproducibility.
        """
        seed = self.train_configs.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def _create_run_directory(self):
        """
        Create a directory for saving the model and logs.
        """
        base_dir = self.train_configs.get("run_dir", "experiments")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.train_configs.model.split('/')[-1]
        run_name = f"{model_name}__" + \
            f"r{self.train_configs.lora_r}__" + \
            f"lr{self.train_configs.lr}__" + \
            f"train_{self.train_configs.dataset_split.replace('[:','').replace(']','')}"
        run_dir = os.path.join(base_dir, model_name, self.train_configs.agg_type, f"{timestamp}_{run_name}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        
        config_dict = dict(self.train_configs)
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        return run_dir
    
    def _init_wandb(self):
        """
        Initialize Weights and Biases for logging.
        """
        self.enable_wandb = self.train_configs.get("enable_wandb", False)
        if self.enable_wandb:
            wandb_run = wandb.init(
                entity=self.train_configs.wandb_configs.get("entity", None),
                project=self.train_configs.wandb_configs.get("project", None),
                dir=os.path.join(self.run_dir, "logs"),
                name=self.train_configs.wandb_configs.get("exp_name", None),
                config=OmegaConf.to_container(self.train_configs, resolve=True),
            )
            with open(os.path.join(self.run_dir, "wandb_run_id.txt"), "w") as f:
                f.write(wandb_run.id)