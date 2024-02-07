import sys
from .default import *
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from omegaconf import DictConfig, OmegaConf

from .fed.federated import *
from .fed.fedasync import *
from .fed.iceadmm import *  ## TODO: combine iceadmm and iiadmm under the name of ADMM.
from .fed.iiadmm import *

@dataclass
class CompressorConfig:
    lossy_compressor: str = "SZ2"
    lossless_compressor: str = "blosc"

    # Lossy compression path configuration
    ext = ".dylib" if sys.platform.startswith("darwin") else ".so"
    compressor_sz2_path: str = "../.compressor/SZ/build/sz/libSZ" + ext
    compressor_sz3_path: str = "../.compressor/SZ3/build/tools/sz3c/libSZ3c" + ext
    compressor_szx_path: str = "../.compressor/SZx-main/build/lib/libSZx" + ext

    # Compressor parameters
    error_bounding_mode: str = ""
    error_bound: float = 0.0

@dataclass
class ServerAgentConfig:
    client_configs: DictConfig = OmegaConf.create({
        "train_configs": OmegaConf.create({
            # Trainer
            "trainer": "NaiveTrainer",
            # Training mode
            "mode": "epoch",
            "num_local_epochs": 1,
            ## Alternatively
            # "mode": "step",
            # "num_local_steps": 100,
            # Optimizer
            "optim": "SGD",
            "optim_args": {
                "lr": 0.001,
            },
            # Loss function
            "loss_fn": "CrossEntropyLoss",
            "loss_fn_kwargs": {},
            "loss_fn_path": "",
            "loss_fn_name": "",
            # Evaluation
            "do_validation": True,
            "metric_path": "../../examples/metric/acc.py",
            "metric_name": "accuracy",
            # Differential Privacy
            "use_dp": False,
            "epsilon": 1,
            # Gradient Clipping
            "clip_grad": False,
            "clip_value": 1,
            "clip_norm": 1,
            # Checkpointing
            "save_model_state_dict": False,
            "checkpoints_interval": 2,
        }),
        "model_configs": OmegaConf.create({
            "model_path": "../../examples/models/cnn.py",
            "model_name": "CNN",
            "model_kwargs": {
                "num_channel": 1,
                "num_classes": 10,
                "num_pixel": 28,
            },
        }),
        "comm_configs": OmegaConf.create({
            "enable_compression": False,
            "compression_configs": CompressorConfig(),
        }),
    })
    server_configs: DictConfig = OmegaConf.create({
        "scheduler": "SyncScheduler",
        "scheduler_args": {
            "num_clients": 3,
        },
        "aggregator": "FedAvgAggregator",
        "aggregator_args": {
            "weights": "equal",
        },
        "device": "cpu",
        "num_epochs": 2,
        "server_validation": False,
        "logging_output_dirname": "./output",
        "logging_output_filename": "result",
    })

@dataclass
class ClientAgentConfig:
    """
    ClientAgentConfig is a dataclass that holds the configurations for the client agent. 
    It basically holds the following types of configurations:
    - train_configs: Configurations for local training, such as trainer, device, optimizer, loss function, etc.
    - model_configs: Configurations for the AI model
    - comm_configs: Configurations for communication, such as compression, etc.
    - data_configs: Configurations for the data loader
    """
    train_configs: DictConfig = field(default_factory=default_train_config)
    model_configs: DictConfig = field(default_factory=default_model_config)
    data_configs: DictConfig = field(default_factory=default_data_config)
    comm_configs: DictConfig = OmegaConf.create({
        "enable_compression": False,
        "compression_configs": CompressorConfig(),
    })


@dataclass
class Config:
    fed: Any = field(default_factory=Federated)

    # Compute device
    device: str = "cpu"
    device_server: str = "cpu"

    # Number of training epochs
    num_clients: int = 1

    # Number of training epochs
    num_epochs: int = 2

    # Number of workers in DataLoader
    num_workers: int = 0

    # Train data batch info
    batch_training: bool = True  ## TODO: revisit
    train_data_batch_size: int = 64
    train_data_shuffle: bool = True

    # Indication of whether to validate or not using testing data
    validation: bool = True
    test_data_batch_size: int = 64
    test_data_shuffle: bool = False

    # Checking data sanity
    data_sanity: bool = False

    # Reproducibility
    reproduce: bool = True

    # PCA on Trajectory
    pca_dir: str = ""
    params_start: int = 0
    params_end: int = 49
    ncomponents: int = 40

    # Tensorboard
    use_tensorboard: bool = False

    # Loading models
    load_model: bool = False
    load_model_dirname: str = ""
    load_model_filename: str = ""

    # Saving models (server)
    save_model: bool = False
    save_model_dirname: str = ""
    save_model_filename: str = ""
    checkpoints_interval: int = 2

    # Saving state_dict (clients)
    save_model_state_dict: bool = False
    send_final_model: bool = False

    # Logging and recording outputs
    output_dirname: str = "output"
    output_filename: str = "result"

    logginginfo: DictConfig = OmegaConf.create({})
    summary_file: str = ""

    # Personalization options
    personalization: bool = False
    p_layers: List[str] = field(default_factory=lambda: [])
    config_name: str = ""

    #
    # gRPC configutations
    #

    # 100 MB for gRPC maximum message size
    max_message_size: int = 104857600

    operator: DictConfig = OmegaConf.create({"id": 1})
    server: DictConfig = OmegaConf.create(
        {"id": 1, "host": "localhost", "port": 50051, "use_tls": False, "api_key": None}
    )
    client: DictConfig = OmegaConf.create({"id": 1})

    # Lossy compression enabling
    enable_compression: bool = False
    lossy_compressor: str = "SZ2"
    lossless_compressor: str = "blosc"

    # Lossy compression path configuration
    ext = ".dylib" if sys.platform.startswith("darwin") else ".so"
    compressor_sz2_path: str = "../.compressor/SZ/build/sz/libSZ" + ext
    compressor_sz3_path: str = "../.compressor/SZ3/build/tools/sz3c/libSZ3c" + ext
    compressor_szx_path: str = "../.compressor/SZx-main/build/lib/libSZx" + ext

    # Compressor parameters
    error_bounding_mode: str = ""
    error_bound: float = 0.0

    # Default data type
    flat_model_dtype: str = "np.float32"
    param_cutoff: int = 1024


@dataclass
class GlobusComputeServerConfig:
    device: str = "cpu"
    output_dir: str = "./"
    data_dir: str = "./"
    s3_bucket: Any = None
    s3_creds: str = ""


@dataclass
class GlobusComputeClientConfig:
    name        : str = ""
    endpoint_id : str = ""
    device      : str = "cpu"
    output_dir  : str = "./output"
    data_dir    : str = "./datasets"
    get_data    :  DictConfig = OmegaConf.create({})
    data_pipeline: DictConfig = OmegaConf.create({})


@dataclass
class ExecutableFunc:
    module: str = ""
    call: str = ""
    script_file: str = ""
    source: str = ""


@dataclass
class ClientTask:
    task_id: str = ""
    task_name: str = ""
    client_idx: int = ""
    pending: bool = True
    success: bool = False
    start_time: float = -1
    end_time: float = -1
    log: Optional[Dict] = field(default_factory=dict)


@dataclass
class GlobusComputeConfig(Config):
    get_data: ExecutableFunc = field(default_factory=ExecutableFunc)
    get_model: ExecutableFunc = field(default_factory=ExecutableFunc)
    get_loss: ExecutableFunc = field(default_factory=ExecutableFunc)
    val_metric: ExecutableFunc = field(default_factory=ExecutableFunc)
    clients: List[GlobusComputeClientConfig] = field(default_factory=list)
    dataset: str = ""
    loss: str = "CrossEntropy"
    model_kwargs: Dict = field(default_factory=dict)
    server: GlobusComputeServerConfig
    logging_tasks: List = field(default_factory=list)
    hf_model_arc: str = ""
    hf_model_weights: str = ""

    # Testing and validation params
    client_do_validation: bool = True
    client_do_testing: bool = True
    server_do_validation: bool = True
    server_do_testing: bool = True

    # Testing and validation frequency
    client_validation_step: int = 1
    server_validation_step: int = 1

    # Cloud storage
    use_cloud_transfer: bool = True
