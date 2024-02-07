from omegaconf import OmegaConf

def default_train_config():
    return OmegaConf.create({
        # Trainer
        "trainer": "NaiveTrainer",
        "device": "cpu",
        # Training mode
        "mode": "epoch",
        "num_local_epochs": 1,
        ## Alternatively
        # "mode": "step",
        # "num_local_steps": 100,
        # Optimizer
        "optim": "SGD",
        "optim_args": {
            "lr": 0.01,
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
        # Output and logging
        "logging_id": "Client-0",
        "logging_output_dirname": "./output",
        "logging_output_filename": "result",
        # Checkpointing
        "save_model_state_dict": False,
        "checkpoints_interval": 2,
        "save_model_dirname": "./output/models",
        "save_model_filename": "model",

    })

def default_model_config():
    return OmegaConf.create({
        "model_path": "../../examples/models/cnn.py",
        "model_name": "CNN",
        "model_kwargs": {
            "num_channel": 1,
            "num_classes": 10,
            "num_pixel": 28,
        },
    })

def default_data_config():
    return OmegaConf.create({
        "dataloader_path": "../../examples/dataloader/mnist_dataloader_test.py",
        "dataloader_name": "get_mnist",
        "dataloader_kwargs": {
            "comm": None, 
            "num_clients": 1,
            "client_id": 0,
            "train_batch_size": 64,
            "test_batch_size": 64,
        },
    })

