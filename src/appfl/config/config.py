from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf


@dataclass
class ServerAgentConfig:
    client_configs: DictConfig = field(
        default_factory=lambda: OmegaConf.create(
            {
                "train_configs": OmegaConf.create({}),
                "model_configs": OmegaConf.create({}),
            }
        )
    )
    server_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))


@dataclass
class ClientAgentConfig:
    """
    ClientAgentConfig is a dataclass that holds the configurations for the client agent.
    It basically holds the following types of configurations:
    - train_configs: Configurations for local training, such as trainer, device, optimizer, loss function, etc.
    - model_configs: Configurations for the AI model
    - data_configs: Configurations for the data loader
    - comm_configs: Configurations for communication, such as compression, etc.
    - additional_configs: Additional configurations that are not covered by the above categories.
    """

    train_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    model_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    data_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    comm_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    additional_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
