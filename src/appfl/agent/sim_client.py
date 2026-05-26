import uuid
import warnings

from omegaconf import OmegaConf

from .client import ClientAgent


class SimClientAgent(ClientAgent):
    """Simulation-oriented client agent backed by the standard ClientAgent."""

    def __init__(
        self,
        client_agent_config=None,
        client_id=None,
        client_dataset=None,
        **kwargs,
    ):
        config_input = (
            client_agent_config
            if client_agent_config is not None
            else kwargs.pop("config", {})
        )
        config = OmegaConf.create(config_input)
        self.sim_config = OmegaConf.create(config)
        is_unified_config = "train_configs" not in config

        if is_unified_config:
            algorithm = config.get("algorithm_configs", {})
            log_configs = config.get("log_configs", {})
            scheduler_kwargs = algorithm.get("scheduler_kwargs", {})
            trainer_kwargs = algorithm.get("trainer_kwargs", {})
            self.num_clients = scheduler_kwargs.get("num_clients")
            self.num_clients_per_round = scheduler_kwargs.get("num_clients_per_round")
            config = OmegaConf.create(
                {
                    "client_id": str(client_id) if client_id is not None else "0",
                    "model_configs": config.get("model_configs", {}),
                    "train_configs": OmegaConf.merge(
                        trainer_kwargs,
                        {
                            "trainer": algorithm.get("trainer", "VanillaTrainer"),
                            "device": algorithm.get("device", "cpu"),
                            "logging_output_dirname": log_configs.get(
                                "logging_output_dirname", "./output"
                            ),
                            "logging_output_filename": log_configs.get(
                                "client_logging_output_filename", ""
                            ),
                        },
                    ),
                    "comm_configs": config.get("comm_configs", {}),
                }
            )
        elif client_id is not None:
            config.client_id = str(client_id)
        else:
            warnings.warn(
                    message="Client ID (client_id) not specified. Generating a random one.",
                    category=UserWarning,
                )
            config.client_id = str(uuid.uuid4())
        self.client_id = config.client_id

        if client_dataset is not None:
            self._dataset = client_dataset
        else:
            raise ValueError("Client dataset must be provided for SimClientAgent.")
        super().__init__(client_agent_config=config, **kwargs)

    def _load_data(self):
        if self._dataset is not None:
            train_dataset, val_dataset = self._dataset[0], self._dataset[1] if len(self._dataset) > 1 else None
            self.train_dataset, self.val_dataset = train_dataset, val_dataset
            return
        super()._load_data()
